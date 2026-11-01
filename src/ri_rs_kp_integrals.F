!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Numerical integrals d_lP(R) for periodic RI-RS fitting.
!> \author Jan Wilhelm (with help of ChatGPT, GPT-6 Sol and GPT-6 Astra)
! **************************************************************************************************
MODULE ri_rs_kp_integrals
   USE basis_set_types,                 ONLY: gto_basis_set_type
   USE cell_types,                      ONLY: cell_type,&
                                              pbc
   USE ieee_arithmetic,                 ONLY: ieee_is_finite
   USE input_constants,                 ONLY: do_potential_coulomb,&
                                              do_potential_short,&
                                              do_potential_truncated
   USE kinds,                           ONLY: dp,&
                                              int_8
   USE machine,                         ONLY: m_walltime
   USE mathconstants,                   ONLY: pi,&
                                              twopi
   USE mathlib,                         ONLY: diamat_all
   USE message_passing,                 ONLY: mp_comm_type,&
                                              mp_mem_avail_per_rank_GB
   USE omp_lib,                         ONLY: omp_get_max_threads,&
                                              omp_in_parallel
   USE orbital_pointers,                ONLY: indco,&
                                              ncoset,&
                                              nso
   USE particle_types,                  ONLY: particle_type
   USE physcon,                         ONLY: angstrom
   USE post_scf_bandstructure_types,    ONLY: post_scf_bandstructure_type
   USE ri_rs_utils,                     ONLY: evaluate_ao_basis_on_points
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'ri_rs_kp_integrals'

   PUBLIC :: compute_d_lP_R, &
             compute_d_lP_cutoffs, &
             get_local_l_range, &
             write_ri_rs_progress

   TYPE                                 :: uniform_settings_type
      REAL(dp)                          :: accuracy = 0.0_dp, &
                                           spacing = 0.0_dp, padding = 0.0_dp, &
                                           l_block_edge = 0.0_dp
      INTEGER                           :: counts(3) = 0
      REAL(dp)                          :: lower(3) = 0.0_dp, &
                                           origin(3) = 0.0_dp, &
                                           step(3, 3) = 0.0_dp, &
                                           weight = 0.0_dp
   END TYPE uniform_settings_type

   TYPE                                 :: l_block_partition_type
      INTEGER                           :: nblocks = 0
      INTEGER, ALLOCATABLE              :: first(:), indices(:)
      REAL(dp), ALLOCATABLE             :: lower(:, :), upper(:, :)
   END TYPE l_block_partition_type

   TYPE                                 :: ao_image_type
      TYPE(gto_basis_set_type), POINTER :: basis => NULL()
      REAL(dp)                          :: center(3) = 0.0_dp
      REAL(dp)                          :: bound = 0.0_dp, decay = 0.0_dp, &
                                           radius2 = HUGE(0.0_dp)
      REAL(dp), ALLOCATABLE             :: primary(:, :)
      INTEGER, ALLOCATABLE              :: primary_l(:)
      REAL(dp), ALLOCATABLE             :: envelope_exponents(:), &
                                           radial_envelope(:, :)
   END TYPE ao_image_type

   ! Per-atom interpolation tables for the existing Coulomb/erfc kernel.
   TYPE                                 :: radial_interpolation_type
      REAL(dp), ALLOCATABLE             :: short_abscissae(:)
      REAL(dp), ALLOCATABLE             :: coefficients(:, :), &
                                           series(:, :, :), &
                                           short_series(:, :, :), &
                                           short_moments(:, :, :, :)
      ! Radial value and derivative, indexed (2, primitive/angular channel,
      ! squared-radius index).
      REAL(dp), ALLOCATABLE             :: mesh_moments(:, :, :)
      INTEGER, ALLOCATABLE              :: mesh_first(:, :), &
                                           spherical_mesh_first(:, :)
      ! An optional exact factorization of primitive Cartesian rows through
      ! spherical RI functions.
      REAL(dp), ALLOCATABLE             :: spherical_selection(:, :), &
                                           spherical_angular(:, :, :, :), &
                                           spherical_primitive(:, :, :)
      LOGICAL                           :: spherical_factorized = .FALSE.
      REAL(dp)                          :: mesh_spacing = 0.0_dp, &
                                           mesh_offset = 0.0_dp, &
                                           mesh_exponent = 0.0_dp
      REAL(dp)                          :: cutoff = 0.0_dp, &
                                           attenuation = 0.0_dp
   END TYPE radial_interpolation_type

   TYPE                                 :: radial_work_type
      REAL(dp), ALLOCATABLE             :: jet(:), gaussian(:), derivative(:), &
                                           bracket(:)
   END TYPE radial_work_type

CONTAINS

! **************************************************************************************************
!> \brief Allocate d_lP(R), prepare its integration grid, and evaluate the integrals.
!> \param bs_env ...
!> \param grid_points ...
!> \param cells_R_d_lP Retained integer cell vectors R, indexed (3,R).
!> \param d_lP_R Numerical integrals indexed by local point l, RI function P and cell R.
!> \param qpoints ...
!> \param charge_lq ...
! **************************************************************************************************
   SUBROUTINE compute_d_lP_R(bs_env, grid_points, cells_R_d_lP, d_lP_R, &
                             qpoints, charge_lq)
      TYPE(post_scf_bandstructure_type), POINTER      :: bs_env
      REAL(dp), INTENT(IN)                            :: grid_points(:, :)
      INTEGER, INTENT(IN)                             :: cells_R_d_lP(:, :)
      REAL(dp), ALLOCATABLE, INTENT(OUT)              :: d_lP_R(:, :, :)
      REAL(dp), INTENT(IN), OPTIONAL                  :: qpoints(:, :)
      COMPLEX(dp), ALLOCATABLE, INTENT(OUT), OPTIONAL :: charge_lq(:, :)

      INTEGER                                         :: first_l, handle, last_l, n_P
      TYPE(uniform_settings_type)                     :: settings

      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,A9)') 'd_lP method', 'NUMERICAL'
      END IF
      CALL get_local_l_range(SIZE(grid_points, 2), bs_env%para_env%num_pe, &
                             bs_env%para_env%mepos, first_l, last_l)
      n_P = bs_env%n_RI
      IF (bs_env%auto_ri%enabled) n_P = SUM(bs_env%auto_ri%sizes_opt_RI)
      ALLOCATE (d_lP_R(last_l - first_l + 1, n_P, SIZE(cells_R_d_lP, 2)))

      CALL timeset("d_lP_uniform_grid", handle)

      CALL prepare_d_lP_integration_grid(bs_env, grid_points, settings)
      IF (PRESENT(charge_lq)) THEN
         CPASSERT(PRESENT(qpoints))
         ALLOCATE (charge_lq(SIZE(grid_points, 2), SIZE(qpoints, 2)))
         CALL integrate_d_lP_R(bs_env, grid_points, cells_R_d_lP, settings, d_lP_R, first_l, &
                               qpoints, charge_lq)
      ELSE
         CALL integrate_d_lP_R(bs_env, grid_points, cells_R_d_lP, settings, d_lP_R, first_l)
      END IF

      IF (.NOT. ALL(ieee_is_finite(d_lP_R))) CPABORT("Nonfinite DLP cell value")

      CALL timestop(handle)

   END SUBROUTINE compute_d_lP_R

! **************************************************************************************************
!> \brief Automatic l/P cutoff from the overlap of the diffuse AO-field square and RI potential.
!>
!> Outside a truncated Coulomb sphere, conservative Gaussian envelopes are
!> f_l(r)^2 ~ exp[-alpha_AO |r-r_l|^2] and
!> v_PT(r) ~ exp[-alpha_RI (|r-R_PT|-r_c)^2/2]. Maximizing their product over r gives
!> exp[-gamma (|r_l-R_PT|-r_c)^2], with 1/gamma=1/alpha_AO+2/alpha_RI.
!> Both methods use the same EPS_FILTER-dependent lattice domain.
!> \param bs_env ...
!> \param cutoffs ...
! **************************************************************************************************
   SUBROUTINE compute_d_lP_cutoffs(bs_env, cutoffs)
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      REAL(dp), ALLOCATABLE, INTENT(OUT)         :: cutoffs(:)

      INTEGER                                    :: iatom, ikind
      REAL(dp)                                   :: accuracy, ao_max, ao_min, ri_max, ri_min
      TYPE(gto_basis_set_type), POINTER          :: basis
      TYPE(particle_type), POINTER               :: particles(:)

      particles => bs_env%ri_rs%particle_set
      accuracy = d_lP_integration_accuracy(bs_env)
      ALLOCATE (cutoffs(bs_env%n_atom))
      DO iatom = 1, bs_env%n_atom
         IF (bs_env%ri_metric%potential_type == do_potential_coulomb) THEN
            cutoffs(iatom) = 2.0_dp*MAXVAL(bs_env%ri_rs%radius_ao_per_atom) + &
                             bs_env%ri_rs%radius_ri_per_atom(iatom)
            CYCLE
         END IF
         ikind = particles(iatom)%atomic_kind%kind_number
         basis => bs_env%basis_set_AO(ikind)%gto_basis_set
         CALL basis_exponent_range(basis, ao_min, ao_max)
         basis => bs_env%basis_set_RI(ikind)%gto_basis_set
         CALL basis_exponent_range(basis, ri_min, ri_max)
         cutoffs(iatom) = MAX(0.0_dp, bs_env%ri_metric%cutoff_radius) + &
                          SQRT(LOG(1.0_dp/accuracy)*(1.0_dp/ao_min + 2.0_dp/ri_min))
      END DO

   END SUBROUTINE compute_d_lP_cutoffs

! **************************************************************************************************
!> \brief Distribute individual l rows evenly across MPI ranks; empty ranges are valid.
!> \param n_grid Number of RI-RS grid points.
!> \param n_ranks Number of MPI ranks.
!> \param i_rank MPI rank.
!> \param first_l First locally owned grid row.
!> \param last_l Last locally owned grid row.
! **************************************************************************************************
   SUBROUTINE get_local_l_range(n_grid, n_ranks, i_rank, first_l, last_l)
      INTEGER, INTENT(IN)  :: n_grid, n_ranks, i_rank
      INTEGER, INTENT(OUT) :: first_l, last_l

      CPASSERT(i_rank >= 0 .AND. i_rank < n_ranks)
      first_l = INT(INT(n_grid, int_8)*INT(i_rank, int_8)/INT(n_ranks, int_8)) + 1
      last_l = INT(INT(n_grid, int_8)*INT(i_rank + 1, int_8)/INT(n_ranks, int_8))

   END SUBROUTINE get_local_l_range

! **************************************************************************************************
!> \brief Print one rank-local RI-RS work milestone without synchronizing MPI ranks.
!> Percentages count completed local units; time is the elapsed wall time since the phase began.
!> \param unit_nr ...
!> \param phase ...
!> \param n_done ...
!> \param n_total ...
!> \param next_progress ...
!> \param phase_start ...
! **************************************************************************************************
   SUBROUTINE write_ri_rs_progress(unit_nr, phase, n_done, n_total, next_progress, phase_start)
      INTEGER, INTENT(IN)             :: unit_nr
      CHARACTER(LEN=*), INTENT(IN)    :: phase
      INTEGER(KIND=int_8), INTENT(IN) :: n_done, n_total
      INTEGER, INTENT(INOUT)          :: next_progress
      REAL(KIND=dp), INTENT(IN)       :: phase_start

      REAL(KIND=dp), PARAMETER        :: milestones(9) = [0.1_dp, 0.3_dp, 1.0_dp, 3.0_dp, 10.0_dp, 30.0_dp , 60.0_dp, 90.0_dp, &
                                         100.0_dp]

      CHARACTER(LEN=100)              :: progress_line
      CHARACTER(LEN=20)               :: count_text
      CHARACTER(LEN=32)               :: phase_text
      CHARACTER(LEN=64)               :: progress_format
      CHARACTER(LEN=8)                :: time_text
      INTEGER                         :: count_width, previous_progress
      REAL(KIND=dp)                   :: percent_done, phase_time

      IF (unit_nr <= 0 .OR. n_total <= 0) RETURN
      percent_done = 100.0_dp*REAL(n_done, dp)/REAL(n_total, dp)
      previous_progress = next_progress
      DO WHILE (next_progress <= SIZE(milestones))
         IF (percent_done < milestones(next_progress)) EXIT
         next_progress = next_progress + 1
      END DO
      IF (next_progress == previous_progress) RETURN
      CPASSERT(LEN_TRIM(phase) <= 32)
      WRITE (count_text, '(I0)') n_total
      count_width = LEN_TRIM(count_text)
      WRITE (progress_format, '(A,I0,A,I0,A)') &
         '(A32,1X,I', count_width, ',A,I', count_width, ',1X,F5.1,A)'
      phase_text = phase
      WRITE (progress_line, progress_format) &
         phase_text, n_done, ' / ', n_total, percent_done, ' %'
      phase_time = m_walltime() - phase_start
      WRITE (time_text, '(F8.1)') phase_time
      IF (INDEX(time_text, '*') > 0) WRITE (time_text, '(ES8.1E3)') phase_time
      IF (LEN_TRIM(progress_line) > 56) THEN
         WRITE (unit_nr, '(T2,A)') TRIM(phase)
         WRITE (unit_nr, '(T2,A)') TRIM(ADJUSTL(progress_line(34:)))
         progress_line = ''
      END IF
      WRITE (unit_nr, '(T2,A,T60,A,T71,A,A)') TRIM(progress_line), 'Total time:', time_text, ' s'
      FLUSH (unit_nr)

   END SUBROUTINE write_ri_rs_progress

! **************************************************************************************************
!> \brief Choose the smallest fitting divisor of MPI ranks with 30 percent reserve.
!> A memory group has ngroups ranks and jointly holds all T. Corresponding T slices
!> across memory groups divide g. Equal integration ranks
!> share their uniquely evaluated fields across T groups. The caller's P result is already live.
!> Use the existing node-memory estimate to bound temporary workspaces.
!> Budget the maximum simultaneous integration, reduction and redistribution storage.
!> The final redistribution needs an extra receive buffer as large as the live result.
!> Selection masks remain live throughout; block masks have at most nl rows.
!> \param comm ...
!> \param nl ...
!> \param bs_env ...
!> \param nt ...
!> \param nresult ...
!> \param integration_comm ...
!> \param field_comm ...
!> \param ngroups ...
! **************************************************************************************************
   SUBROUTINE prepare_d_lP_mpi_groups(comm, nl, bs_env, nt, nresult, integration_comm, field_comm, ngroups)
      CLASS(mp_comm_type), INTENT(IN)            :: comm
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      INTEGER(int_8), INTENT(IN)                 :: nresult
      INTEGER, INTENT(IN)                        :: nl, nt
      TYPE(mp_comm_type), INTENT(OUT)            :: integration_comm, field_comm
      INTEGER, INTENT(OUT)                       :: ngroups

      INTEGER                                    :: maxnp
      REAL(dp)                                   :: contraction_bytes, selection_bytes, local_bytes, reduction_bytes
      INTEGER                                    :: np, igroup, integration_rank, nlocal_T
      REAL(dp)                                   :: result_bytes
      REAL(dp)                                   :: available_GB, cell_bytes, required_bytes, cyclic_bytes, workspace_bytes

      np = bs_env%n_RI
      IF (bs_env%auto_ri%enabled) np = SUM(bs_env%auto_ri%sizes_opt_RI)
      CALL mp_mem_avail_per_rank_GB(comm, available_GB)
      result_bytes = 8.0_dp*REAL(nresult, dp)
      CALL comm%max(result_bytes)
      cell_bytes = 8.0_dp*REAL(nl, dp)*REAL(np, dp)
      cyclic_bytes = cell_bytes*REAL((nt + comm%num_pe - 1)/comm%num_pe, dp)
      maxnp = MAXVAL(bs_env%i_RI_end_from_atom - bs_env%i_RI_start_from_atom + 1)
      IF (bs_env%auto_ri%enabled) maxnp = MAXVAL(bs_env%auto_ri%sizes_opt_RI)
      contraction_bytes = 8.0_dp*(REAL(nl, dp)*REAL(maxnp + 512, dp) + 512.0_dp*REAL(maxnp, dp))
      workspace_bytes = 96.0_dp*1024.0_dp**2 + 40.0_dp*REAL(nl + 3, dp)*512.0_dp + &
                        MAX(128.0_dp*1024.0_dp**2, contraction_bytes)
      DO ngroups = 1, comm%num_pe
         IF (MODULO(comm%num_pe, ngroups) /= 0) CYCLE
         nlocal_T = (nt + ngroups - 1)/ngroups
         ! The caller allocated d_lP_R before this call. Its bytes are already
         ! reflected in the measured free memory and must not be charged twice.
         local_bytes = cell_bytes*REAL(nlocal_T, dp)
         selection_bytes = 2.0_dp*REAL(STORAGE_SIZE(.TRUE.)/8, dp)* &
                           REAL(nl, dp)*REAL(bs_env%n_atom, dp)*REAL(nlocal_T, dp)
         selection_bytes = selection_bytes + 3.0_dp*REAL(STORAGE_SIZE(0)/8, dp)*REAL(nlocal_T, dp)
         reduction_bytes = MAX(32.0_dp*1024.0_dp**2, cell_bytes)
         required_bytes = 1.3_dp*(selection_bytes + &
                                  MAX(local_bytes + workspace_bytes, &
                                      local_bytes + cyclic_bytes + reduction_bytes, &
                                      2.0_dp*cyclic_bytes + result_bytes + reduction_bytes))
         IF (required_bytes <= available_GB*1.0E9_dp) EXIT
      END DO
      IF (ngroups > comm%num_pe) THEN
         ngroups = comm%num_pe
         CPWARN('d_lP workspaces exceed memory with all MPI ranks')
      END IF
      igroup = MODULO(comm%mepos, ngroups)
      integration_rank = comm%mepos/ngroups
      CALL integration_comm%from_split(comm, igroup, key=integration_rank)
      CALL field_comm%from_split(comm, integration_rank, key=igroup)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T66,F12.1,A)') 'd_lP PSS-aware rank headroom:', available_GB, ' GB'
         WRITE (bs_env%unit_nr, '(T2,A,T61,I20)') 'd_lP memory group size:', ngroups
         WRITE (bs_env%unit_nr, '(T2,A,T61,I20)') 'd_lP independent integration groups:', integration_comm%num_pe
         WRITE (bs_env%unit_nr, '(T2,A,T66,F12.1,A)') 'd_lP existing output storage:', result_bytes*1.0E-9_dp, ' GB'
         WRITE (bs_env%unit_nr, '(T2,A,T66,F12.1,A)') 'd_lP additional memory budget:', required_bytes*1.0E-9_dp, ' GB'
         FLUSH (bs_env%unit_nr)
      END IF

   END SUBROUTINE prepare_d_lP_mpi_groups

! **************************************************************************************************
!> \brief Sum Eq. (2) directly onto ranks owning cyclic T cells, T=r+1+j*N.
!> The first two indices are (L,P); contiguous T blocks are pointer views, with no packing copy.
!> Reduce-scatter retains only each rank's output cells instead of replicating the full sum.
!> \param comm ...
!> \param partial ...
!> \param cells ...
! **************************************************************************************************
   SUBROUTINE sum_d_lP_R_over_mpi_ranks(comm, partial, cells)
      CLASS(mp_comm_type), INTENT(IN)            :: comm
      REAL(dp), CONTIGUOUS, TARGET, INTENT(IN)   :: partial(:, :, :)
      REAL(dp), ALLOCATABLE, TARGET, INTENT(OUT) :: cells(:, :, :)

      INTEGER                                    :: nl, np, nt, nranks, rank, nlocal, icycle, first_T, last_T, round_first, &
                                                    round_last, first_rank, last_rank, block_cells
      INTEGER(int_8)                             :: plane_count, iround
      INTEGER                                    :: recv_counts(comm%num_pe)
      REAL(dp), POINTER, CONTIGUOUS              :: send_block(:, :), receive_block(:)
      REAL(dp), TARGET                           :: empty(0)

      nl = SIZE(partial, 1)
      np = SIZE(partial, 2)
      nt = SIZE(partial, 3)
      nranks = comm%num_pe
      rank = comm%mepos
      nlocal = 0
      IF (nt > rank) nlocal = 1 + (nt - 1 - rank)/nranks
      ALLOCATE (cells(nl, np, nlocal))
      IF (nl == 0 .OR. np == 0 .OR. nt == 0) RETURN
      IF (nranks == 1) THEN
         cells(:, :, :) = partial(:, :, :)
         RETURN
      END IF
      plane_count = INT(nl, int_8)*INT(np, int_8)
      IF (plane_count > HUGE(0)) THEN
         CPABORT('DLP integration block exceeds the MPI count range')
      END IF
      block_cells = INT(MAX(1_int_8, (32_int_8*1024_int_8**2/8_int_8)/plane_count))
      block_cells = MIN(block_cells, INT(INT(HUGE(0), int_8)/plane_count))
      icycle = 0
      DO iround = 1_int_8, INT(nt, int_8), INT(nranks, int_8)
         icycle = icycle + 1
         round_first = INT(iround)
         round_last = INT(MIN(INT(nt, int_8), iround + INT(nranks, int_8) - 1_int_8))
         first_T = round_first
         DO WHILE (first_T <= round_last)
            last_T = INT(MIN(INT(round_last, int_8), INT(first_T, int_8) + INT(block_cells, int_8) - 1_int_8))
            first_rank = first_T - round_first
            last_rank = last_T - round_first
            recv_counts = 0
            recv_counts(first_rank + 1:last_rank + 1) = INT(plane_count)
            send_block(1:INT(plane_count), 1:last_T - first_T + 1) => partial(:, :, first_T:last_T)
            receive_block => empty
            IF (rank >= first_rank .AND. rank <= last_rank) THEN
               receive_block(1:INT(plane_count)) => cells(:, :, icycle)
            END IF
            CALL comm%sum_scatter(send_block, receive_block, recv_counts)
            IF (last_T == round_last) EXIT
            first_T = last_T + 1
         END DO
      END DO

   END SUBROUTINE sum_d_lP_R_over_mpi_ranks

! **************************************************************************************************
!> \brief Redistribute Eq. (2) from cyclic T cells to disjoint contiguous l rows.
!> Rank r initially owns T=r+1+j*N. The caller owns first_l:first_l+size(d_lP_R,1)-1.
!> Only the local T slice and local l slice are materialized, never a global DLP replica.
!> \param comm ...
!> \param cell_values ...
!> \param first_l ...
!> \param d_lP_R Numerical integrals indexed by local point l, RI function P and cell R.
! **************************************************************************************************
   SUBROUTINE distribute_d_lP_R_by_l(comm, cell_values, first_l, d_lP_R)
      CLASS(mp_comm_type), INTENT(IN) :: comm
      REAL(dp), INTENT(IN)            :: cell_values(:, :, :)
      INTEGER, INTENT(IN)             :: first_l
      REAL(dp), INTENT(OUT)           :: d_lP_R(:, :, :)

      INTEGER                         :: il, ip, irank, it, local_it, nl, np, nlocal_l, nlocal_T, nranks, nt, position
      INTEGER(int_8)                  :: count_wide
      INTEGER, ALLOCATABLE            :: first_rows(:), row_counts(:), cell_counts(:), send_counts(:), send_offsets(:), &
                                         recv_counts(:), recv_offsets(:)
      REAL(dp), ALLOCATABLE           :: send_values(:), receive_values(:)

      CHARACTER(LEN=*), PARAMETER     :: routineN = 'distribute_d_lP_R_by_l'
      INTEGER                         :: handle

      CALL timeset(routineN, handle)

      nranks = comm%num_pe
      nl = SIZE(cell_values, 1)
      np = SIZE(d_lP_R, 2)
      nlocal_l = SIZE(d_lP_R, 1)
      nt = SIZE(d_lP_R, 3)
      nlocal_T = SIZE(cell_values, 3)
      CPASSERT(SIZE(cell_values, 2) == np)
      ALLOCATE (first_rows(nranks), row_counts(nranks), cell_counts(nranks), &
                send_counts(nranks), send_offsets(nranks), recv_counts(nranks), recv_offsets(nranks))
      CALL comm%allgather(first_l, first_rows)
      CALL comm%allgather(nlocal_l, row_counts)
      CALL comm%allgather(nlocal_T, cell_counts)
      CPASSERT(SUM(cell_counts) == nt)
      CPASSERT(SUM(row_counts) == nl)
      DO irank = 1, nranks
         count_wide = INT(np, int_8)*INT(row_counts(irank), int_8)*INT(nlocal_T, int_8)
         CPASSERT(count_wide <= HUGE(0))
         count_wide = INT(np, int_8)*INT(nlocal_l, int_8)*INT(cell_counts(irank), int_8)
         CPASSERT(count_wide <= HUGE(0))
         send_counts(irank) = np*row_counts(irank)*nlocal_T
         recv_counts(irank) = np*nlocal_l*cell_counts(irank)
      END DO
      CPASSERT(SUM(INT(send_counts, int_8)) <= HUGE(0))
      CPASSERT(SUM(INT(recv_counts, int_8)) <= HUGE(0))
      position = 0
      DO irank = 1, nranks
         send_offsets(irank) = position
         position = position + send_counts(irank)
      END DO
      ALLOCATE (send_values(position))
      position = 0
      DO irank = 1, nranks
         recv_offsets(irank) = position
         position = position + recv_counts(irank)
      END DO
      ALLOCATE (receive_values(position))
      DO irank = 1, nranks
         position = send_offsets(irank)
         DO local_it = 1, nlocal_T
            DO ip = 1, np
               DO il = first_rows(irank), first_rows(irank) + row_counts(irank) - 1
                  position = position + 1
                  send_values(position) = cell_values(il, ip, local_it)
               END DO
            END DO
         END DO
         CPASSERT(position == send_offsets(irank) + send_counts(irank))
      END DO
      CALL comm%alltoall(send_values, send_counts, send_offsets, &
                         receive_values, recv_counts, recv_offsets)
      DO irank = 1, nranks
         position = recv_offsets(irank)
         DO local_it = 1, cell_counts(irank)
            it = irank + (local_it - 1)*nranks
            CPASSERT(it <= nt)
            DO ip = 1, np
               DO il = 1, nlocal_l
                  position = position + 1
                  d_lP_R(il, ip, it) = receive_values(position)
               END DO
            END DO
         END DO
         CPASSERT(position == recv_offsets(irank) + recv_counts(irank))
      END DO

      CALL timestop(handle)

   END SUBROUTINE distribute_d_lP_R_by_l

! **************************************************************************************************
!> \brief Accumulate Q_lq += w SUM_g f_l(r_g)^2 exp(i q.r_g) with real BLAS products.
!> The real field is the existing f_l(r_g)^2 on one quadrature integration-point block. Cosine and sine
!> columns share a DGEMM; OpenMP partitions disjoint l rows without a reduction.
!> The grid, phase convention, weights, and retained quadrature points are unchanged.
!> \param points ...
!> \param field ...
!> \param q_cart ...
!> \param weight ...
!> \param charge_lq ...
! **************************************************************************************************
   SUBROUTINE accumulate_fourier_charge(points, field, q_cart, weight, charge_lq)
      REAL(dp), INTENT(IN)             :: points(:, :)
      REAL(dp), CONTIGUOUS, INTENT(IN) :: field(:, :)
      REAL(dp), INTENT(IN)             :: q_cart(:, :), weight
      COMPLEX(dp), INTENT(INOUT)       :: charge_lq(:, :)

      INTEGER, PARAMETER               :: row_block = 64

      COMPLEX(dp)                      :: phase
      INTEGER                          :: first_l, ig, il, iq, nblock, ng, nl, nq
      REAL(dp), ALLOCATABLE            :: charge_parts(:, :), phase_parts(:, :)

      ng = SIZE(points, 2)
      nl = SIZE(field, 2)
      nq = SIZE(q_cart, 2)
      CPASSERT(SIZE(points, 1) == 3 .AND. SIZE(q_cart, 1) == 3)
      CPASSERT(SIZE(field, 1) >= ng)
      CPASSERT(SIZE(charge_lq, 1) == nl .AND. SIZE(charge_lq, 2) == nq)
      IF (ng == 0 .OR. nl == 0 .OR. nq == 0) RETURN
      ALLOCATE (phase_parts(ng, 2*nq))
      DO iq = 1, nq
         DO ig = 1, ng
            phase = EXP(CMPLX(0.0_dp, SUM(q_cart(:, iq)*points(:, ig)), dp))
            phase_parts(ig, iq) = REAL(phase, dp)
            phase_parts(ig, nq + iq) = AIMAG(phase)
         END DO
      END DO
      !$OMP PARALLEL DEFAULT(NONE) IF(nl > row_block) &
      !$OMP SHARED(field, phase_parts, charge_lq, ng, nl, nq, weight) &
      !$OMP PRIVATE(charge_parts, first_l, il, iq, nblock)
      ALLOCATE (charge_parts(row_block, 2*nq))
      !$OMP DO SCHEDULE(STATIC)
      DO first_l = 1, nl, row_block
         nblock = MIN(row_block, nl - first_l + 1)
         CALL dgemm('T', 'N', nblock, 2*nq, ng, weight, &
                    field(:, first_l:first_l + nblock - 1), SIZE(field, 1), &
                    phase_parts, ng, 0.0_dp, charge_parts, row_block)
         DO iq = 1, nq
            DO il = 1, nblock
               charge_lq(first_l + il - 1, iq) = charge_lq(first_l + il - 1, iq) + &
                                                 CMPLX(charge_parts(il, iq), charge_parts(il, nq + iq), dp)
            END DO
         END DO
      END DO
      !$OMP END DO
      DEALLOCATE (charge_parts)
      !$OMP END PARALLEL
      DEALLOCATE (phase_parts)

   END SUBROUTINE accumulate_fourier_charge

! **************************************************************************************************
!> \brief Validate the operator and define the integration box and mesh in Eq. (3).
!> \param bs_env ...
!> \param grid_points ...
!> \param settings ...
!> \param report ...
! **************************************************************************************************
   SUBROUTINE prepare_d_lP_integration_grid(bs_env, grid_points, settings, report)
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      REAL(dp), INTENT(IN)                       :: grid_points(:, :)
      TYPE(uniform_settings_type), INTENT(OUT)   :: settings
      LOGICAL, INTENT(IN), OPTIONAL              :: report

      INTEGER(int_8)                             :: npoints
      LOGICAL                                    :: report_settings
      REAL(dp)                                   :: exponent_max, exponent_min, padding_safety, spacing_prefactor

      settings%accuracy = d_lP_integration_accuracy(bs_env)
      IF (bs_env%ri_metric%potential_type /= do_potential_coulomb .AND. &
          bs_env%ri_metric%potential_type /= do_potential_truncated .AND. &
          bs_env%ri_metric%potential_type /= do_potential_short) THEN
         CPABORT('Uniform DLP requires a Coulomb, truncated or short-range RI metric')
      END IF
      CALL ao_exponent_range(bs_env, exponent_min, exponent_max)
      ! The AO exponent calibration includes the matched reference RI basis.
      spacing_prefactor = 0.4_dp*SQRT(2.693604_dp*LOG(1.0E8_dp))
      padding_safety = 1.1_dp
      settings%l_block_edge = 1.0_dp/angstrom
      settings%spacing = spacing_prefactor/ &
                         SQRT(exponent_max*LOG(1.0_dp/settings%accuracy))
      settings%padding = padding_safety*SQRT(LOG(1.0_dp/settings%accuracy)/exponent_min)
      CALL build_d_lP_integration_grid(bs_env, grid_points, settings)
      npoints = PRODUCT(INT(settings%counts, int_8))
      report_settings = .TRUE.
      IF (PRESENT(report)) report_settings = report
      IF (bs_env%unit_nr > 0 .AND. report_settings) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,ES9.1E2)') 'd_lP uniform quadrature points', REAL(npoints, dp)
         WRITE (bs_env%unit_nr, '(T2,A,T72,ES9.1)') 'd_lP integration threshold', settings%accuracy
         WRITE (bs_env%unit_nr, '(T2,A,T69,F12.3)') 'd_lP largest AO exponent [bohr^-2]', exponent_max
         WRITE (bs_env%unit_nr, '(T2,A,T69,F12.3)') 'd_lP uniform spacing [bohr]', settings%spacing
      END IF

   END SUBROUTINE prepare_d_lP_integration_grid

! **************************************************************************************************
!> \brief Build Eq. (3) on a cell-commensurate lattice without coarsening its largest spacing.
!> The integration parallelepiped contains the original padded Cartesian box.
!> \param bs_env Band-structure environment with RI-RS cell and atoms.
!> \param grid_points ...
!> \param settings ...
! **************************************************************************************************
   SUBROUTINE build_d_lP_integration_grid(bs_env, grid_points, settings)
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      REAL(dp), INTENT(IN)                       :: grid_points(:, :)
      TYPE(uniform_settings_type), INTENT(INOUT) :: settings

      INTEGER                                    :: idir, subdivisions(3)
      REAL(dp)                                   :: box_center(3), box_extent(3), eigenvalues(3), inverse_step(3, 3), &
                                                    metric(3, 3), scale, upper(3)
      TYPE(cell_type), POINTER                   :: cell
      TYPE(particle_type), POINTER               :: particles(:)

      CPASSERT(SIZE(grid_points, 1) == 3 .AND. SIZE(grid_points, 2) > 0)
      CPASSERT(settings%spacing > 0.0_dp)
      cell => bs_env%ri_rs%cell
      particles => bs_env%ri_rs%particle_set
      settings%origin = 0.0_dp
      settings%step = 0.0_dp
      inverse_step = 0.0_dp
      settings%weight = settings%spacing**3
      DO idir = 1, 3
         settings%step(idir, idir) = settings%spacing
         inverse_step(idir, idir) = 1.0_dp/settings%spacing
      END DO
      IF (ANY(cell%perd /= 0)) THEN
         CPASSERT(ASSOCIATED(particles))
         CPASSERT(SIZE(particles) > 0 .AND. ABS(cell%deth) > TINY(1.0_dp))
         settings%origin = pbc(particles(1)%r, cell)
         DO idir = 1, 3
            subdivisions(idir) = MAX(1, CEILING(NORM2(cell%hmat(:, idir))/settings%spacing))
            settings%step(:, idir) = cell%hmat(:, idir)/REAL(subdivisions(idir), dp)
         END DO
         metric = MATMUL(TRANSPOSE(settings%step), settings%step)
         CALL diamat_all(metric, eigenvalues, dac=.TRUE.)
         scale = MAX(1.0_dp, SQRT(MAXVAL(eigenvalues))/settings%spacing)
         subdivisions = CEILING(scale*REAL(subdivisions, dp))
         DO idir = 1, 3
            settings%step(:, idir) = cell%hmat(:, idir)/REAL(subdivisions(idir), dp)
            inverse_step(idir, :) = REAL(subdivisions(idir), dp)*cell%h_inv(idir, :)
         END DO
         settings%weight = ABS(cell%deth)/PRODUCT(REAL(subdivisions, dp))
      ELSE
         settings%lower = REAL(CEILING((MINVAL(grid_points, DIM=2) - settings%padding)/ &
                                       settings%spacing), dp)
         upper = REAL(FLOOR((MAXVAL(grid_points, DIM=2) + settings%padding)/settings%spacing), dp)
         settings%counts = INT(upper - settings%lower) + 1
         CPASSERT(ALL(settings%counts > 0))
         RETURN
      END IF
      box_center = 0.5_dp*(MINVAL(grid_points, DIM=2) + MAXVAL(grid_points, DIM=2)) - settings%origin
      box_extent = 0.5_dp*(MAXVAL(grid_points, DIM=2) - MINVAL(grid_points, DIM=2)) + settings%padding
      settings%lower = REAL(CEILING(MATMUL(inverse_step, box_center) - &
                                    MATMUL(ABS(inverse_step), box_extent)), dp)
      upper = REAL(FLOOR(MATMUL(inverse_step, box_center) + &
                         MATMUL(ABS(inverse_step), box_extent)), dp)
      settings%counts = INT(upper - settings%lower) + 1
      CPASSERT(ALL(settings%counts > 0))

   END SUBROUTINE build_d_lP_integration_grid

! **************************************************************************************************
!> \brief Partition the RI-RS points r_l into Cartesian cubes used by Eq. (4).
!> \param grid_points ...
!> \param edge ...
!> \param blocks ...
! **************************************************************************************************
   SUBROUTINE group_points_l_by_position(grid_points, edge, blocks)
      REAL(dp), INTENT(IN)                      :: grid_points(:, :), edge
      TYPE(l_block_partition_type), INTENT(OUT) :: blocks

      INTEGER                                   :: cube(3), iblock, idir, il, insert
      INTEGER, ALLOCATABLE                      :: block_of_l(:), counts(:), cubes(:, :), next(:)
      REAL(dp)                                  :: origin(3)

      CPASSERT(edge > 0.0_dp)
      origin = MINVAL(grid_points, DIM=2)
      ALLOCATE (block_of_l(SIZE(grid_points, 2)), cubes(3, SIZE(grid_points, 2)))
      blocks%nblocks = 0
      DO il = 1, SIZE(grid_points, 2)
         cube = FLOOR((grid_points(:, il) - origin)/edge)
         iblock = 0
         DO insert = 1, blocks%nblocks
            IF (ALL(cubes(:, insert) == cube)) THEN
               iblock = insert
               EXIT
            END IF
         END DO
         IF (iblock == 0) THEN
            blocks%nblocks = blocks%nblocks + 1
            iblock = blocks%nblocks
            cubes(:, iblock) = cube
         END IF
         block_of_l(il) = iblock
      END DO
      ALLOCATE (counts(blocks%nblocks), next(blocks%nblocks), &
                blocks%first(blocks%nblocks + 1), blocks%indices(SIZE(grid_points, 2)), &
                blocks%lower(3, blocks%nblocks), blocks%upper(3, blocks%nblocks))
      counts = 0
      DO il = 1, SIZE(grid_points, 2)
         counts(block_of_l(il)) = counts(block_of_l(il)) + 1
      END DO
      blocks%first(1) = 1
      DO iblock = 1, blocks%nblocks
         blocks%first(iblock + 1) = blocks%first(iblock) + counts(iblock)
      END DO
      next(:) = blocks%first(1:blocks%nblocks)
      blocks%lower = HUGE(1.0_dp)
      blocks%upper = -HUGE(1.0_dp)
      DO il = 1, SIZE(grid_points, 2)
         iblock = block_of_l(il)
         blocks%indices(next(iblock)) = il
         next(iblock) = next(iblock) + 1
         DO idir = 1, 3
            blocks%lower(idir, iblock) = MIN(blocks%lower(idir, iblock), grid_points(idir, il))
            blocks%upper(idir, iblock) = MAX(blocks%upper(idir, iblock), grid_points(idir, il))
         END DO
      END DO

   END SUBROUTINE group_points_l_by_position

! **************************************************************************************************
!> \brief Return the smallest and largest positive AO primitive exponents for the grid heuristic.
!> \param bs_env ...
!> \param exponent_min ...
!> \param exponent_max ...
! **************************************************************************************************
   SUBROUTINE ao_exponent_range(bs_env, exponent_min, exponent_max)
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      REAL(dp), INTENT(OUT)                      :: exponent_min, exponent_max

      INTEGER                                    :: ikind, ipgf, iset
      REAL(dp)                                   :: exponent
      TYPE(gto_basis_set_type), POINTER          :: basis

      exponent_min = HUGE(1.0_dp)
      exponent_max = 0.0_dp
      DO ikind = 1, SIZE(bs_env%basis_set_AO)
         basis => bs_env%basis_set_AO(ikind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis)) CYCLE
         DO iset = 1, basis%nset
            DO ipgf = 1, basis%npgf(iset)
               exponent = basis%zet(ipgf, iset)
               IF (exponent <= 0.0_dp) CYCLE
               exponent_min = MIN(exponent_min, exponent)
               exponent_max = MAX(exponent_max, exponent)
            END DO
         END DO
      END DO
      IF (exponent_max <= 0.0_dp .OR. exponent_min == HUGE(1.0_dp)) THEN
         CPABORT('Uniform DLP found no positive AO primitive exponent')
      END IF

   END SUBROUTINE ao_exponent_range

! **************************************************************************************************
!> \brief Set the DLP threshold below the sparse-filter threshold.
!> \param bs_env ...
!> \return ...
! **************************************************************************************************
   FUNCTION d_lP_integration_accuracy(bs_env) RESULT(accuracy)
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      REAL(dp)                                   :: accuracy

      IF (.NOT. ieee_is_finite(bs_env%eps_filter) .OR. &
          bs_env%eps_filter <= 0.0_dp .OR. bs_env%eps_filter >= 1.0_dp) THEN
         CPABORT('EPS_FILTER must be between zero and one for periodic RI-RS DLP')
      END IF
      accuracy = bs_env%eps_filter/10.0_dp
      IF (accuracy <= 0.0_dp) CPABORT('EPS_FILTER is too small for DLP quadrature')
   END FUNCTION d_lP_integration_accuracy

! **************************************************************************************************
!> \brief Return the smallest and largest positive primitive exponent of one Gaussian basis.
!> \param basis ...
!> \param exponent_min ...
!> \param exponent_max ...
! **************************************************************************************************
   SUBROUTINE basis_exponent_range(basis, exponent_min, exponent_max)
      TYPE(gto_basis_set_type), POINTER :: basis
      REAL(dp), INTENT(OUT)             :: exponent_min, exponent_max

      INTEGER                           :: ipgf, iset
      REAL(dp)                          :: exponent

      CPASSERT(ASSOCIATED(basis))
      exponent_min = HUGE(1.0_dp)
      exponent_max = 0.0_dp
      DO iset = 1, basis%nset
         DO ipgf = 1, basis%npgf(iset)
            exponent = basis%zet(ipgf, iset)
            IF (exponent <= 0.0_dp) CYCLE
            exponent_min = MIN(exponent_min, exponent)
            exponent_max = MAX(exponent_max, exponent)
         END DO
      END DO
      IF (exponent_max <= 0.0_dp .OR. exponent_min == HUGE(1.0_dp)) THEN
         CPABORT('Automatic DLP cutoff found no positive primitive exponent')
      END IF

   END SUBROUTINE basis_exponent_range

! **************************************************************************************************
!> \brief Construct d_lP(T) by shared uniform quadrature with analytic auxiliary potentials.
!> (1) f_l(r)=sum_mu,R ϕ_mu,R(r_l) ϕ_mu,R(r), using the explicit AO cell set.
!> (2) d_lP(T)=integral f_l(r)^2 v_P,T(r) dr is approximated by |det S| sum_g f_l(r_g)^2 v_P,T(r_g).
!> (3) r_g=r_0+S n covers the bounding box of r_l enlarged by padding. The spacing follows
!> h=C/[sqrt(alpha_max log(1/epsilon))], calibrated to h=0.4 bohr for the validated Si basis
!> at epsilon=1e-8. The padding is 1.1 sqrt[log(1/epsilon)/alpha_min].
!> Periodic cells use r_0 at atom 1 and S=H diag(1/N_i), with integer N_i and ||S||_2<=h.
!> Translation and cell wrapping then only shift the quadrature lattice or relabel its points.
!> Nonperiodic systems retain r_0=0 and S=h I. Finite-box and screening errors still converge.
!> (4) Spatial blocks of r_l are screened separately for every P-atom/T pair immediately before
!> contraction. A triangle-inequality product bound controls discarded discrete terms, not
!> quadrature error. For a truncated kernel, it decays outside the Coulomb cutoff.
!> The usual l/P/T cutoff removes individual l rows before contraction.
!> Unlike the analytic screened reference,
!> Eq. (1) does not omit individual AO pairs; cell-domain and screening differences must converge.
!> Integration-point blocks are MPI distributed and each AO field is evaluated once. Bounded batches
!> share these fields within memory groups whose ranks divide T; independent groups divide g.
!> Group sums assemble disjoint cyclic T cells. A final all-to-all
!> redistributes the completed d(T) from T ownership to the caller's l ownership.
!> Each memory group holds one full partial d(T), subject to the memory budget.
!> No global AO-pair matrix is formed.
!> EPS_FILTER/10 controls the grid, finite box and screening threshold together.
!> \param bs_env ...
!> \param grid_points ...
!> \param cells_R_d_lP Retained integer cell vectors R, indexed (3,R).
!> \param settings ...
!> \param d_lP_R Numerical integrals indexed by local point l, RI function P and cell R.
!> \param first_l ...
!> \param qpoints ...
!> \param charge_lq ...
! **************************************************************************************************
   SUBROUTINE integrate_d_lP_R(bs_env, grid_points, cells_R_d_lP, settings, d_lP_R, first_l, &
                               qpoints, charge_lq)
      TYPE(post_scf_bandstructure_type), POINTER   :: bs_env
      REAL(dp), INTENT(IN)                         :: grid_points(:, :)
      INTEGER, INTENT(IN)                          :: cells_R_d_lP(:, :)
      TYPE(uniform_settings_type), INTENT(IN)      :: settings
      REAL(dp), INTENT(OUT)                        :: d_lP_R(:, :, :)
      INTEGER, INTENT(IN)                          :: first_l
      REAL(dp), INTENT(IN), OPTIONAL               :: qpoints(:, :)
      COMPLEX(dp), INTENT(OUT), OPTIONAL           :: charge_lq(:, :)

      INTEGER, PARAMETER                           :: quadrature_block_edge = 8, &
                                                      quadrature_block_size = quadrature_block_edge**3

      INTEGER                                      :: counts(3), handle, iatom, ibatch, iblock, idir, ig, igroup, il, &
                                                      indices(quadrature_block_size), integration_rank, irank, it, last_rank, &
                                                      next_progress, ngroups, nkeep, nl, np, npoints, nranks, nreceived, &
                                                      position, quadrature_block_counts(3), quadrature_block_first(3), &
                                                      quadrature_block_points(3), rank, ranks_per_batch
      INTEGER(int_8)                               :: block_contractions, coalesced_contractions, contraction_products, &
                                                      index_grid, iquadrature_block, iround, n_completed_quadrature_blocks, &
                                                      n_local_quadrature_blocks, nquadrature_blocks, nrounds, ntotal, &
                                                      potential_points, screened_blocks
      INTEGER, ALLOCATABLE                         :: kept_per_rank(:), local_cells(:, :), recv_counts(:), recv_offsets(:)
      LOGICAL, ALLOCATABLE                         :: include_l_blocks_P_R(:, :, :), include_lP_R(:, :, :)
      REAL(dp)                                     :: bound, discarded, elapsed, field_bound(quadrature_block_size), lower(3), &
                                                      mesh_index(3), omitted, points(3, quadrature_block_size), prep_start, &
                                                      RI_center_R(3), threshold, weight
      REAL(dp), ALLOCATABLE                        :: cutoffs_lP(:), cyclic_d_lP_R(:, :, :), envelopes(:, :), &
                                                      local_d_lP_R(:, :, :), q_cart(:, :), receive_fields(:), send_fields(:)
      REAL(dp)                                     :: field(quadrature_block_size, SIZE(grid_points, 2))
      TYPE(ao_image_type), ALLOCATABLE             :: images(:)
      TYPE(cell_type), POINTER                     :: cell
      TYPE(l_block_partition_type)                 :: blocks
      TYPE(mp_comm_type)                           :: field_comm, integration_comm
      TYPE(particle_type), POINTER                 :: particles(:)
      TYPE(radial_interpolation_type), ALLOCATABLE :: aux_potentials(:)

      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A)') 'dLP timing guide (details in final timing table):'
         WRITE (bs_env%unit_nr, '(T2,A)') 'd_lP_allocate_thread_buffers: Allocate thread work arrays'
         WRITE (bs_env%unit_nr, '(T2,A)') 'd_lP_screen_atom_image: Screen atom images and grid points'
         WRITE (bs_env%unit_nr, '(T2,A)') 'd_lP_wait_for_threads: Wait for other OpenMP threads'
         WRITE (bs_env%unit_nr, '(T2,A)') 'd_lP_release_thread_buffers: Release thread work arrays'
         WRITE (bs_env%unit_nr, '(T2,A)') 'd_lP_uniform_exchange: Exchange grid fields between MPI ranks'
         WRITE (bs_env%unit_nr, '(T2,A)') 'd_lP_prepare_field_bounds: Bound fields for screening'
         WRITE (bs_env%unit_nr, '(T2,A)') 'd_lP_sum_g_f2_v: Contract potential with squared AO fields'
      END IF
      nl = SIZE(grid_points, 2)
      IF (PRESENT(qpoints) .NEQV. PRESENT(charge_lq)) THEN
         CPABORT('Request DLP q points and Fourier charges together')
      END IF
      IF (PRESENT(qpoints)) THEN
         CPASSERT(SIZE(qpoints, 1) == 3)
         CPASSERT(SIZE(charge_lq, 1) == nl)
         CPASSERT(SIZE(charge_lq, 2) == SIZE(qpoints, 2))
         cell => bs_env%ri_rs%cell
         ALLOCATE (q_cart(3, SIZE(qpoints, 2)))
         q_cart(:, :) = MATMUL(twopi*TRANSPOSE(cell%h_inv), qpoints)
         charge_lq = CMPLX(0.0_dp, 0.0_dp, dp)
      END IF
      elapsed = m_walltime()
      prep_start = elapsed
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A)') 'd_lP quadrature: evaluating AO image values'
         FLUSH (bs_env%unit_nr)
      END IF
      CALL prepare_ao_images(bs_env, grid_points, images)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T67,F12.1,A)') &
            'd_lP AO image values, rank 0 time:', m_walltime() - prep_start, ' s'
         FLUSH (bs_env%unit_nr)
      END IF
      CALL prepare_auto_ri_potentials(bs_env, settings, aux_potentials)
      CALL auxiliary_potential_bound(bs_env, aux_potentials, envelopes)
      CALL group_points_l_by_position(grid_points, settings%l_block_edge, blocks)
      bound = MAXVAL(envelopes(1, :))
      d_lP_R = 0.0_dp
      CALL prepare_d_lP_mpi_groups(bs_env%para_env, nl, bs_env, SIZE(cells_R_d_lP, 2), SIZE(d_lP_R, KIND=int_8), &
                                   integration_comm, field_comm, ngroups)
      igroup = field_comm%mepos
      integration_rank = integration_comm%mepos
      nranks = field_comm%num_pe
      rank = field_comm%mepos
      np = bs_env%n_RI
      IF (bs_env%auto_ri%enabled) np = SUM(bs_env%auto_ri%sizes_opt_RI)
      ALLOCATE (local_cells(3, SIZE(cells_R_d_lP(:, igroup + 1::ngroups), 2)))
      local_cells(:, :) = cells_R_d_lP(:, igroup + 1::ngroups)

      CALL timeset('d_lP_prepare_RI_geometry', handle)

      cell => bs_env%ri_rs%cell
      particles => bs_env%ri_rs%particle_set
      CALL compute_d_lP_cutoffs(bs_env, cutoffs_lP)
      ALLOCATE (include_lP_R(nl, bs_env%n_atom, SIZE(local_cells, 2)))
      ALLOCATE (include_l_blocks_P_R(blocks%nblocks, bs_env%n_atom, SIZE(local_cells, 2)), source=.FALSE.)
      DO it = 1, SIZE(local_cells, 2)
         DO iatom = 1, bs_env%n_atom
            RI_center_R = pbc(particles(iatom)%r, cell) + MATMUL(bs_env%hmat, REAL(local_cells(:, it), dp))
            DO il = 1, nl
               include_lP_R(il, iatom, it) = SUM((grid_points(:, il) - RI_center_R)**2) <= cutoffs_lP(iatom)**2
            END DO
            DO iblock = 1, blocks%nblocks
               IF (distance_to_box_squared(RI_center_R, blocks%lower(:, iblock), &
                                           blocks%upper(:, iblock)) > cutoffs_lP(iatom)**2) CYCLE
               include_l_blocks_P_R(iblock, iatom, it) = &
                  ANY(include_lP_R(blocks%indices(blocks%first(iblock):blocks%first(iblock + 1) - 1), iatom, it))
            END DO
         END DO
      END DO
      DEALLOCATE (cutoffs_lP)
      IF (bs_env%unit_nr > 0) THEN
         FLUSH (bs_env%unit_nr)
      END IF

      CALL timestop(handle)

      ALLOCATE (local_d_lP_R(nl, np, SIZE(local_cells, 2)), source=0.0_dp)
      ALLOCATE (kept_per_rank(nranks), recv_counts(nranks), recv_offsets(nranks))
      ! At most 32 MiB of shared fields per rank, independent of the MPI size.
      ranks_per_batch = MAX(1, INT(32.0_dp*1024.0_dp**2/ &
                                   (8.0_dp*REAL(nl + 3, dp)*REAL(quadrature_block_size, dp))))
      ranks_per_batch = MIN(nranks, ranks_per_batch)
      ALLOCATE (send_fields((nl + 3)*quadrature_block_size), &
                receive_fields((nl + 3)*quadrature_block_size*ranks_per_batch))
      omitted = 0.0_dp
      block_contractions = 0_int_8
      coalesced_contractions = 0_int_8
      contraction_products = 0_int_8
      potential_points = 0_int_8
      screened_blocks = 0_int_8
      weight = settings%weight
      lower = settings%lower
      counts = settings%counts
      ntotal = PRODUCT(INT(counts, int_8))
      quadrature_block_counts = (counts + quadrature_block_edge - 1)/quadrature_block_edge
      nquadrature_blocks = PRODUCT(INT(quadrature_block_counts, int_8))
      nrounds = (nquadrature_blocks + INT(bs_env%para_env%num_pe, int_8) - 1_int_8)/ &
                INT(bs_env%para_env%num_pe, int_8)
      next_progress = 1
      n_local_quadrature_blocks = nrounds
      threshold = 0.5_dp*settings%accuracy/(REAL(ntotal, dp)*weight)
      CALL screen_ao_images(images, settings%accuracy/ &
                            (8.0_dp*weight*REAL(ntotal, dp)*MAX(1.0_dp, bound)))
      DO iround = 0_int_8, nrounds - 1_int_8
         iquadrature_block = iround*INT(bs_env%para_env%num_pe, int_8) + &
                             INT(bs_env%para_env%mepos, int_8)
         nkeep = 0
         IF (iquadrature_block < nquadrature_blocks) THEN
            ! Compact integration-point blocks preserve the quadrature while tightening all spatial bounds.
            index_grid = iquadrature_block
            DO idir = 1, 3
             quadrature_block_first(idir) = quadrature_block_edge*INT(MODULO(index_grid, INT(quadrature_block_counts(idir), int_8)))
               quadrature_block_points(idir) = MIN(quadrature_block_edge, counts(idir) - quadrature_block_first(idir))
               index_grid = index_grid/INT(quadrature_block_counts(idir), int_8)
            END DO
            npoints = PRODUCT(quadrature_block_points)
            DO ig = 1, npoints
               index_grid = INT(ig - 1, int_8)
               DO idir = 1, 3
                  mesh_index(idir) = lower(idir) + &
                         REAL(quadrature_block_first(idir) + INT(MODULO(index_grid, INT(quadrature_block_points(idir), int_8))), dp)
                  index_grid = index_grid/INT(quadrature_block_points(idir), int_8)
               END DO
               points(:, ig) = settings%origin + MATMUL(settings%step, mesh_index)
            END DO
            CALL timeset('d_lP_f_l_squared', handle)
            CALL evaluate_rho_l_squared(images, points(:, 1:npoints), field(1:npoints, :), &
                                        settings%accuracy/(8.0_dp*weight*REAL(ntotal, dp)*MAX(1.0_dp, bound)))
            CALL timestop(handle)
            IF (PRESENT(charge_lq)) THEN
               CALL timeset('d_lP_charge_fourier', handle)
               CALL accumulate_fourier_charge(points(:, 1:npoints), field, q_cart, weight, charge_lq)
               CALL timestop(handle)
            END IF
            CALL timeset('d_lP_screen_pack_fields', handle)
            nkeep = 0
            field_bound(1:npoints) = 0.0_dp
            DO il = 1, nl
               !$OMP SIMD
               DO ig = 1, npoints
                  field_bound(ig) = MAX(field_bound(ig), ABS(field(ig, il)))
               END DO
            END DO
            DO ig = 1, npoints
               discarded = field_bound(ig)
               IF (discarded*bound > threshold) THEN
                  nkeep = nkeep + 1
                  indices(nkeep) = ig
               ELSE
                  omitted = omitted + weight*bound*discarded
               END IF
            END DO
            DO ig = 1, nkeep
               send_fields(3*ig - 2:3*ig) = points(:, indices(ig))
            END DO
            DO il = 1, nl
               DO ig = 1, nkeep
                  send_fields(3*nkeep + (il - 1)*nkeep + ig) = field(indices(ig), il)
               END DO
            END DO
            CALL timestop(handle)
         END IF
         CALL field_comm%allgather(nkeep, kept_per_rank)
         DO ibatch = 1, nranks, ranks_per_batch
            last_rank = MIN(nranks, ibatch + ranks_per_batch - 1)
            recv_counts = 0
            recv_counts(ibatch:last_rank) = (nl + 3)*kept_per_rank(ibatch:last_rank)
            nreceived = 0
            DO irank = 1, nranks
               recv_offsets(irank) = nreceived
               nreceived = nreceived + recv_counts(irank)
            END DO
            IF (nreceived == 0) CYCLE
            CPASSERT(nreceived <= SIZE(receive_fields))
            CALL timeset('d_lP_uniform_exchange', handle)
            CALL field_comm%allgatherv(send_fields(1:recv_counts(rank + 1)), &
                                       receive_fields(1:nreceived), recv_counts, recv_offsets)
            CALL timestop(handle)
            DO irank = ibatch, last_rank
               nkeep = kept_per_rank(irank)
               IF (nkeep == 0 .OR. SIZE(local_cells, 2) == 0) CYCLE
               CALL timeset('d_lP_unpack_fields', handle)
               position = recv_offsets(irank)
               DO ig = 1, nkeep
                  points(:, ig) = receive_fields(position + 3*ig - 2:position + 3*ig)
               END DO
               DO il = 1, nl
                  DO ig = 1, nkeep
                     field(ig, il) = receive_fields(position + 3*nkeep + (il - 1)*nkeep + ig)
                  END DO
               END DO
               CALL timestop(handle)
               CALL timeset('d_lP_integrate_quadrature_blocks', handle)
               CALL integrate_potentials(bs_env, grid_points, local_cells, include_lP_R, include_l_blocks_P_R, &
                                         points(:, 1:nkeep), field(1:nkeep, :), &
                                         weight, envelopes, blocks, threshold, omitted, potential_points, &
                                         block_contractions, screened_blocks, coalesced_contractions, &
                                         contraction_products, local_d_lP_R, aux_potentials)
               CALL timestop(handle)
            END DO
         END DO
         n_completed_quadrature_blocks = iround + 1_int_8
         CALL write_ri_rs_progress(bs_env%unit_nr, 'd_lP Integration batch done', &
                                   n_completed_quadrature_blocks, n_local_quadrature_blocks, next_progress, elapsed)
      END DO

      CALL timeset('d_lP_uniform_reduce', handle)

      DEALLOCATE (send_fields, receive_fields, kept_per_rank, recv_counts, recv_offsets)
      CALL apply_d_lP_distance_cutoff(bs_env, grid_points, local_cells, local_d_lP_R)
      CALL sum_d_lP_R_over_mpi_ranks(integration_comm, local_d_lP_R, cyclic_d_lP_R)
      DEALLOCATE (local_d_lP_R, local_cells)
      CALL distribute_d_lP_R_by_l(bs_env%para_env, cyclic_d_lP_R, first_l, d_lP_R)
      DEALLOCATE (cyclic_d_lP_R)
      CALL integration_comm%free()
      CALL field_comm%free()
      IF (PRESENT(charge_lq)) CALL bs_env%para_env%sum(charge_lq)
      CALL bs_env%para_env%sum(omitted)
      CALL bs_env%para_env%sum(potential_points)
      CALL bs_env%para_env%sum(block_contractions)
      CALL bs_env%para_env%sum(screened_blocks)
      CALL bs_env%para_env%sum(coalesced_contractions)
      CALL bs_env%para_env%sum(contraction_products)

      CALL timestop(handle)

      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,ES9.1E2)') &
            'Potential evaluations V_P^R(r_g) (point, atom image):', REAL(potential_points, dp)
         WRITE (bs_env%unit_nr, '(T2,A,T72,ES9.1E2)') &
            'd_lP products f_l(r_g)^2 V_P^R(r_g):', REAL(contraction_products, dp)
         WRITE (bs_env%unit_nr, '(T2,A)') &
            'f_l(r) = SUM_(μ,R) ϕ_μ^R(r_l) ϕ_μ^R(r)'
         WRITE (bs_env%unit_nr, '(T2,A)') &
            'd_lP^R = SUM_g w_g f_l(r_g)^2 V_P^R(r_g)'
      END IF
      elapsed = m_walltime() - elapsed
      CALL bs_env%para_env%max(elapsed)
      IF (bs_env%unit_nr > 0) WRITE (bs_env%unit_nr, '(T2,A,T69,F10.1,A)') &
         'd_lP completed numerical integration, Exec. time:', elapsed, ' s'
      IF (bs_env%unit_nr > 0) FLUSH (bs_env%unit_nr)

   END SUBROUTINE integrate_d_lP_R

! **************************************************************************************************
!> \brief Form each atom-local auxiliary potential's spherical coefficients just once.
!> \param bs_env ...
!> \param settings ...
!> \param aux_potentials ...
! **************************************************************************************************
   SUBROUTINE prepare_auto_ri_potentials(bs_env, settings, aux_potentials)
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      TYPE(uniform_settings_type), INTENT(IN)    :: settings
      TYPE(radial_interpolation_type), ALLOCATABLE, &
         INTENT(OUT)                                     :: aux_potentials(:)

      CHARACTER(LEN=12)                          :: spacing_text
      INTEGER                                    :: first_U, iatom, ikind, nmesh, np, nref
      REAL(dp)                                   :: cutoff, mesh_bytes, mesh_displacement(3), mesh_spacing
      REAL(dp), ALLOCATABLE                      :: U_Pp(:, :)
      TYPE(cell_type), POINTER                   :: cell
      TYPE(gto_basis_set_type), POINTER          :: basis
      TYPE(particle_type), POINTER               :: particles(:)

      IF (.NOT. bs_env%auto_ri%enabled) RETURN
      particles => bs_env%ri_rs%particle_set
      cell => bs_env%ri_rs%cell
      cutoff = 0.0_dp
      IF (bs_env%ri_metric%potential_type == do_potential_truncated) cutoff = bs_env%ri_metric%cutoff_radius
      ALLOCATE (aux_potentials(bs_env%n_atom))
      nmesh = 0
      mesh_bytes = 0.0_dp
      mesh_spacing = 0.0_dp
      DO iatom = 1, bs_env%n_atom
         ikind = particles(iatom)%atomic_kind%kind_number
         basis => bs_env%basis_set_RI(ikind)%gto_basis_set
         nref = bs_env%auto_ri%sizes_ref_RI(iatom)
         np = bs_env%auto_ri%sizes_opt_RI(iatom)
         first_U = bs_env%auto_ri%U_Pp_atom_offset(iatom)
         ALLOCATE (U_Pp(nref, np))
         U_Pp(:, :) = RESHAPE(bs_env%auto_ri%U_Pp_atom(first_U:first_U + nref*np - 1), [nref, np])
         mesh_displacement = settings%origin - pbc(particles(iatom)%r, cell)
         CALL prepare_contracted_potential(basis, U_Pp, aux_potentials(iatom), cutoff, &
                                           MAX(0.0_dp, bs_env%ri_metric%omega), &
                                           radial_table_bytes=256.0_dp*1024.0_dp**2/REAL(MAX(1, bs_env%n_atom), dp), &
                                           mesh_step=settings%step, &
                                           mesh_displacement=mesh_displacement)
         IF (ALLOCATED(aux_potentials(iatom)%mesh_moments)) THEN
            nmesh = nmesh + 1
            mesh_bytes = mesh_bytes + 8.0_dp*REAL(SIZE(aux_potentials(iatom)%mesh_moments, KIND=int_8), dp)
            mesh_spacing = aux_potentials(iatom)%mesh_spacing
         END IF
         DEALLOCATE (U_Pp)
      END DO
      IF (bs_env%unit_nr > 0) THEN
         IF (nmesh > 0) THEN
            WRITE (bs_env%unit_nr, '(T2,A,T69,F12.1)') 'd_lP radial mesh storage [MiB/rank]', mesh_bytes/1024.0_dp**2
            WRITE (spacing_text, '(ES12.1E1)') mesh_spacing
            IF (INDEX(spacing_text, '*') > 0) WRITE (spacing_text, '(ES12.1E2)') mesh_spacing
            IF (INDEX(spacing_text, '*') > 0) WRITE (spacing_text, '(ES12.1E3)') mesh_spacing
            WRITE (bs_env%unit_nr, '(T2,A,T69,A12)') 'd_lP radial mesh squared spacing [bohr^2]', spacing_text
         END IF
      END IF

   END SUBROUTINE prepare_auto_ri_potentials

! **************************************************************************************************
!> \brief Store nonzero atom-image coefficients ϕ_mu,R(r_l) in Eq. (1) once.
!> \param bs_env ...
!> \param grid_points ...
!> \param images ...
! **************************************************************************************************
   SUBROUTINE prepare_ao_images(bs_env, grid_points, images)
      TYPE(post_scf_bandstructure_type), POINTER    :: bs_env
      REAL(dp), INTENT(IN)                          :: grid_points(:, :)
      TYPE(ao_image_type), ALLOCATABLE, INTENT(OUT) :: images(:)

      INTEGER                                       :: iatom, icell, iimage, ikind, maxao, nao
      REAL(dp)                                      :: center(3), cutoff2
      REAL(dp), ALLOCATABLE                         :: primary(:, :)
      TYPE(cell_type), POINTER                      :: cell
      TYPE(gto_basis_set_type), POINTER             :: basis
      TYPE(particle_type), POINTER                  :: particles(:)

      cell => bs_env%ri_rs%cell
      particles => bs_env%ri_rs%particle_set
      maxao = MAXVAL(bs_env%i_ao_end_from_atom - bs_env%i_ao_start_from_atom + 1)
      ALLOCATE (primary(SIZE(grid_points, 2), maxao))
      ALLOCATE (images(bs_env%n_atom*SIZE(bs_env%index_to_cell_3c, 2)))
      iimage = 0
      DO icell = 1, SIZE(bs_env%index_to_cell_3c, 2)
         DO iatom = 1, bs_env%n_atom
            ikind = particles(iatom)%atomic_kind%kind_number
            basis => bs_env%basis_set_AO(ikind)%gto_basis_set
            nao = bs_env%i_ao_end_from_atom(iatom) - bs_env%i_ao_start_from_atom(iatom) + 1
            center = pbc(particles(iatom)%r, cell) + MATMUL(bs_env%hmat, REAL(bs_env%index_to_cell_3c(:, icell), dp))
            cutoff2 = bs_env%ri_rs%radius_ao_per_atom(iatom)**2
            IF (bs_env%ri_rs%cutoff_radius_ri_ao > 0.0_dp) cutoff2 = bs_env%ri_rs%cutoff_radius_ri_ao**2
            primary = 0.0_dp
            CALL evaluate_ao_basis_on_points(primary(:, 1:nao), grid_points, basis, center, cell, &
                                             cutoff_squared=cutoff2, minimum_image=.FALSE.)
            IF (.NOT. ANY(primary(:, 1:nao) /= 0.0_dp)) CYCLE
            iimage = iimage + 1
            images(iimage)%basis => basis
            images(iimage)%center = center
            ALLOCATE (images(iimage)%primary(SIZE(grid_points, 2), nao))
            images(iimage)%primary(:, :) = primary(:, 1:nao)
            CALL pack_nonzero_ao_rows(images(iimage))
            CALL bound_ao_image(images(iimage))
         END DO
      END DO

   END SUBROUTINE prepare_ao_images

! **************************************************************************************************
!> \brief Store only exactly nonzero l rows of Eq. (1) when at least half the rows are zero.
!> primary_l maps packed rows back to the original r_l; no finite AO coefficient is discarded.
!> \param image ...
! **************************************************************************************************
   SUBROUTINE pack_nonzero_ao_rows(image)
      TYPE(ao_image_type), INTENT(INOUT) :: image

      INTEGER                            :: il, ipacked, nkeep, nl
      LOGICAL, ALLOCATABLE               :: nonzero(:)
      REAL(dp), ALLOCATABLE              :: primary(:, :)

      CPASSERT(ALLOCATED(image%primary))
      CPASSERT(.NOT. ALLOCATED(image%primary_l))
      nl = SIZE(image%primary, 1)
      ALLOCATE (nonzero(nl))
      DO il = 1, nl
         nonzero(il) = ANY(ABS(image%primary(il, :)) > 0.0_dp)
      END DO
      nkeep = COUNT(nonzero)
      IF (nkeep == 0 .OR. nkeep > nl/2) RETURN
      ALLOCATE (primary(nkeep, SIZE(image%primary, 2)), image%primary_l(nkeep))
      ipacked = 0
      DO il = 1, nl
         IF (.NOT. nonzero(il)) CYCLE
         ipacked = ipacked + 1
         primary(ipacked, :) = image%primary(il, :)
         image%primary_l(ipacked) = il
      END DO
      CALL MOVE_ALLOC(primary, image%primary)

   END SUBROUTINE pack_nonzero_ao_rows

! **************************************************************************************************
!> \brief Bound each image contribution to f_l(r) in Eq. (1) by B exp(-a_min r^2/2).
!> |x^n exp(-a x^2/2)| <= (n/(a e))^(n/2) bounds the remaining polynomial and Gaussian.
!> \param image ...
! **************************************************************************************************
   SUBROUTINE bound_ao_image(image)
      TYPE(ao_image_type), INTENT(INOUT) :: image

      INTEGER                            :: ico, idir, il, ipgf, irow, iset, isgf, ncart, power
      REAL(dp)                           :: amplitude, exponent, factor
      REAL(dp), ALLOCATABLE              :: ao_bound(:)
      TYPE(gto_basis_set_type), POINTER  :: basis

      basis => image%basis
      image%decay = HUGE(0.0_dp)
      ALLOCATE (ao_bound(SIZE(image%primary, 2)), source=0.0_dp)
      DO iset = 1, basis%nset
         ncart = ncoset(basis%lmax(iset))
         DO ipgf = 1, basis%npgf(iset)
            exponent = basis%zet(ipgf, iset)
            CPASSERT(exponent > 0.0_dp)
            image%decay = MIN(image%decay, 0.5_dp*exponent)
            DO ico = 1, ncart
               factor = 1.0_dp
               DO idir = 1, 3
                  power = indco(idir, ico)
                  IF (power > 0) factor = factor*(REAL(power, dp)/(exponent*EXP(1.0_dp)))** &
                                          (0.5_dp*REAL(power, dp))
               END DO
               irow = (ipgf - 1)*ncart + ico
               DO isgf = basis%first_sgf(1, iset), basis%last_sgf(basis%nshell(iset), iset)
                  ao_bound(isgf) = ao_bound(isgf) + ABS(basis%sphi(irow, isgf))*factor
               END DO
            END DO
         END DO
      END DO
      image%bound = 0.0_dp
      DO il = 1, SIZE(image%primary, 1)
         amplitude = 0.0_dp
         DO isgf = 1, SIZE(ao_bound)
            amplitude = amplitude + ABS(image%primary(il, isgf))*ao_bound(isgf)
         END DO
         image%bound = MAX(image%bound, amplitude)
      END DO
      CALL prepare_ao_radial_envelope(image)

   END SUBROUTINE bound_ao_image

! **************************************************************************************************
!> \brief Bound each Cartesian primitive in Eq. (1) by its exact angular maximum at fixed radius.
!> For degree n, |x^i y^j z^k| <= r^n product_d (n_d/n)^(n_d/2).
!> Positive coefficient sums bound every l row without changing the field error budget.
!> \param image ...
! **************************************************************************************************
   SUBROUTINE prepare_ao_radial_envelope(image)
      TYPE(ao_image_type), INTENT(INOUT) :: image

      INTEGER                            :: ico, idegree, idir, iexponent, il, ipgf, irow, iset, isgf, ncart, power
      REAL(dp)                           :: amplitude, angular_factor
      REAL(dp), ALLOCATABLE              :: ao_weight(:)
      TYPE(gto_basis_set_type), POINTER  :: basis

      CPASSERT(ASSOCIATED(image%basis))
      CPASSERT(ALLOCATED(image%primary))
      basis => image%basis
      IF (ALLOCATED(image%envelope_exponents)) DEALLOCATE (image%envelope_exponents, image%radial_envelope)
      ALLOCATE (image%envelope_exponents(SUM(basis%npgf(:basis%nset))), &
                image%radial_envelope(0:MAXVAL(basis%lmax(:basis%nset)), SUM(basis%npgf(:basis%nset))), &
                ao_weight(SIZE(image%primary, 2)))
      ao_weight = 0.0_dp
      DO isgf = 1, SIZE(ao_weight)
         DO il = 1, SIZE(image%primary, 1)
            ao_weight(isgf) = MAX(ao_weight(isgf), ABS(image%primary(il, isgf)))
         END DO
      END DO
      image%radial_envelope = 0.0_dp
      iexponent = 0
      DO iset = 1, basis%nset
         ncart = ncoset(basis%lmax(iset))
         DO ipgf = 1, basis%npgf(iset)
            iexponent = iexponent + 1
            image%envelope_exponents(iexponent) = basis%zet(ipgf, iset)
            DO ico = 1, ncart
               idegree = SUM(indco(:, ico))
               angular_factor = 1.0_dp
               DO idir = 1, 3
                  power = indco(idir, ico)
                  IF (power > 0) angular_factor = angular_factor* &
                                                  (REAL(power, dp)/REAL(idegree, dp))**(0.5_dp*REAL(power, dp))
               END DO
               irow = (ipgf - 1)*ncart + ico
               amplitude = 0.0_dp
               DO isgf = basis%first_sgf(1, iset), basis%last_sgf(basis%nshell(iset), iset)
                  amplitude = amplitude + ao_weight(isgf)*ABS(basis%sphi(irow, isgf))
               END DO
               image%radial_envelope(idegree, iexponent) = &
                  image%radial_envelope(idegree, iexponent) + angular_factor*amplitude
            END DO
         END DO
      END DO
      image%radial_envelope(:, :) = image%radial_envelope* &
                                    (1.0_dp + EPSILON(1.0_dp)*(64.0_dp + &
                                                            4.0_dp*REAL(SIZE(image%radial_envelope), dp)*REAL(SIZE(ao_weight), dp)))

   END SUBROUTINE prepare_ao_radial_envelope

! **************************************************************************************************
!> \brief Bound Eq. (1) outside a sphere using max_{r>=r_min} r^n exp(-alpha*r²).
!> The maximum is at r²=max(r_min²,n/(2*alpha)); far-field polynomials share one exponential.
!> \param image ...
!> \param distance_squared ...
!> \return ...
! **************************************************************************************************
   PURE FUNCTION bound_ao_image_distance(image, distance_squared) RESULT(bound)
      TYPE(ao_image_type), INTENT(IN) :: image
      REAL(dp), INTENT(IN)            :: distance_squared
      REAL(dp)                        :: bound

      INTEGER                         :: idegree, iexponent, lmax
      REAL(dp)                        :: exponent, polynomial, radius, radius_squared

      IF (.NOT. ALLOCATED(image%radial_envelope)) THEN
         bound = image%bound*EXP(-image%decay*distance_squared)
         RETURN
      END IF
      bound = 0.0_dp
      lmax = UBOUND(image%radial_envelope, 1)
      radius = SQRT(distance_squared)
      DO iexponent = 1, SIZE(image%envelope_exponents)
         exponent = image%envelope_exponents(iexponent)
         IF (exponent*distance_squared >= 0.5_dp*REAL(lmax, dp)) THEN
            polynomial = image%radial_envelope(lmax, iexponent)
            DO idegree = lmax - 1, 0, -1
               polynomial = image%radial_envelope(idegree, iexponent) + radius*polynomial
            END DO
            bound = bound + polynomial*EXP(-exponent*distance_squared)
         ELSE
            DO idegree = 0, lmax
               radius_squared = MAX(distance_squared, 0.5_dp*REAL(idegree, dp)/exponent)
               bound = bound + image%radial_envelope(idegree, iexponent)* &
                       radius_squared**(0.5_dp*REAL(idegree, dp))*EXP(-exponent*radius_squared)
            END DO
         END IF
      END DO
   END FUNCTION bound_ao_image_distance

! **************************************************************************************************
!> \brief Bound the omitted squared field in Eq. (2) by tolerance at every point.
!> With A=sum_i B_i and delta<=min(1,tolerance/(2A+1)), 2A delta+delta^2<=tolerance.
!> Each omitted image contributes at most delta/N; box distances bound every point in a integration-point block.
!> \param images ...
!> \param tolerance ...
! **************************************************************************************************
   SUBROUTINE screen_ao_images(images, tolerance)
      TYPE(ao_image_type), INTENT(INOUT) :: images(:)
      REAL(dp), INTENT(IN)               :: tolerance

      INTEGER                            :: iimage, nimages
      REAL(dp)                           :: amplitude, delta

      nimages = 0
      amplitude = 0.0_dp
      DO iimage = 1, SIZE(images)
         IF (.NOT. ALLOCATED(images(iimage)%primary)) EXIT
         nimages = nimages + 1
         amplitude = amplitude + images(iimage)%bound
      END DO
      IF (nimages == 0) RETURN
      delta = MIN(1.0_dp, tolerance/(2.0_dp*amplitude + 1.0_dp))/REAL(nimages, dp)
      IF (delta <= TINY(0.0_dp)) RETURN
      DO iimage = 1, nimages
         IF (images(iimage)%bound <= delta) THEN
            images(iimage)%radius2 = -1.0_dp
         ELSE
            images(iimage)%radius2 = (LOG(images(iimage)%bound) - LOG(delta))/images(iimage)%decay
         END IF
      END DO

   END SUBROUTINE screen_ao_images

! **************************************************************************************************
!> \brief Evaluate the complete squared AO field in Eq. (2) on one Cartesian integration-point block.
!> \param images ...
!> \param points ...
!> \param field_squared ...
!> \param field_accuracy ...
! **************************************************************************************************
   SUBROUTINE evaluate_rho_l_squared(images, points, field_squared, field_accuracy)
      TYPE(ao_image_type), INTENT(IN) :: images(:)
      REAL(dp), INTENT(IN)            :: points(:, :)
      REAL(dp), INTENT(OUT)           :: field_squared(:, :)
      REAL(dp), INTENT(IN), OPTIONAL  :: field_accuracy

      INTEGER                         :: first, last, nblock, ng, nthreads
      REAL(dp)                        :: tolerance

      tolerance = 0.0_dp
      IF (PRESENT(field_accuracy)) tolerance = field_accuracy
      ng = SIZE(points, 2)
      nthreads = 1
!$    nthreads = omp_get_max_threads()
      nblock = MAX(1, (ng + nthreads - 1)/nthreads)
      !$OMP PARALLEL DO DEFAULT(NONE) SCHEDULE(STATIC) &
      !$OMP SHARED(images, points, field_squared, ng, nblock, tolerance) PRIVATE(first, last)
      DO first = 1, ng, nblock
         last = MIN(ng, first + nblock - 1)
         CALL evaluate_rho_l_squared_block(images, points(:, first:last), field_squared(first:last, :), tolerance)
      END DO
      !$OMP END PARALLEL DO

   END SUBROUTINE evaluate_rho_l_squared

! **************************************************************************************************
!> \brief Evaluate Eq. (1), then square, with a integration-point block-local bound on the squared-field error.
!> For image bounds b_i and A=sum_i b_i, omit at most delta=sqrt(A²+epsilon)-A in amplitude;
!> then 2A*delta+delta²<=epsilon. Whole integration-point blocks with A²<=epsilon need no AO evaluation.
!> The legacy radius bound applies when no positive squared-field accuracy is supplied.
!> \param images ...
!> \param points ...
!> \param field_squared ...
!> \param field_accuracy ...
! **************************************************************************************************
   SUBROUTINE evaluate_rho_l_squared_block(images, points, field_squared, field_accuracy)
      TYPE(ao_image_type), INTENT(IN) :: images(:)
      REAL(dp), INTENT(IN)            :: points(:, :)
      REAL(dp), INTENT(OUT)           :: field_squared(:, :)
      REAL(dp), INTENT(IN), OPTIONAL  :: field_accuracy

      INTEGER                         :: handle, iimage, il, nao, ng, nimages, nl, nprimary
      REAL(dp)                        :: amplitude, delta, lower(3), root_tolerance, tolerance, upper(3)
      REAL(dp), ALLOCATABLE           :: field(:, :), image_field(:, :), values(:, :)
      REAL(dp)                        :: image_bounds(SIZE(images))

      tolerance = 0.0_dp
      IF (PRESENT(field_accuracy)) tolerance = field_accuracy
      ng = SIZE(points, 2)
      nl = SIZE(field_squared, 2)
      ALLOCATE (field(ng, nl))
      field = 0.0_dp
      lower = MINVAL(points, DIM=2)
      upper = MAXVAL(points, DIM=2)
      delta = 0.0_dp
      IF (tolerance > 0.0_dp) THEN
         amplitude = 0.0_dp
         nimages = 0
         DO iimage = 1, SIZE(images)
            IF (.NOT. ALLOCATED(images(iimage)%primary)) EXIT
            image_bounds(iimage) = images(iimage)%bound* &
                                   EXP(-images(iimage)%decay*distance_to_box_squared(images(iimage)%center, lower, upper))
            amplitude = amplitude + image_bounds(iimage)
            nimages = nimages + 1
         END DO
         amplitude = amplitude*(1.0_dp + REAL(nimages, dp)*EPSILON(1.0_dp))
         root_tolerance = SQRT(tolerance)
         IF (amplitude <= root_tolerance) THEN
            field_squared = 0.0_dp
            RETURN
         END IF
         delta = (tolerance/amplitude)/(SQRT(1.0_dp + (root_tolerance/amplitude)**2) + 1.0_dp)
         delta = delta/REAL(MAX(1, nimages), dp)
      END IF
      DO iimage = 1, SIZE(images)
         IF (.NOT. ALLOCATED(images(iimage)%primary)) EXIT
         IF (tolerance > 0.0_dp) THEN
            IF (image_bounds(iimage) <= delta) CYCLE
            IF (bound_ao_image_distance(images(iimage), &
                                        distance_to_box_squared(images(iimage)%center, lower, upper)) <= delta) CYCLE
         ELSE
            IF (distance_to_box_squared(images(iimage)%center, lower, upper) > images(iimage)%radius2) CYCLE
         END IF
         nao = SIZE(images(iimage)%primary, 2)
         ALLOCATE (values(ng, nao), SOURCE=0.0_dp)
         CALL timeset('d_lP_AO_values', handle)
         CALL evaluate_mu_r_values(values, points, images(iimage)%basis, images(iimage)%center)
         CALL timestop(handle)
         CALL timeset('d_lP_f_l_sum_AO', handle)
         nprimary = SIZE(images(iimage)%primary, 1)
         IF (ALLOCATED(images(iimage)%primary_l)) THEN
            IF (.NOT. ALLOCATED(image_field)) ALLOCATE (image_field(ng, nl))
            CALL dgemm('N', 'T', ng, nprimary, nao, 1.0_dp, values, ng, &
                       images(iimage)%primary, nprimary, 0.0_dp, image_field, ng)
            DO il = 1, nprimary
               field(:, images(iimage)%primary_l(il)) = &
                  field(:, images(iimage)%primary_l(il)) + image_field(:, il)
            END DO
         ELSE
            CALL dgemm('N', 'T', ng, nl, nao, 1.0_dp, values, ng, &
                       images(iimage)%primary, nl, 1.0_dp, field, ng)
         END IF
         CALL timestop(handle)
         DEALLOCATE (values)
      END DO
      field_squared = field**2

   END SUBROUTINE evaluate_rho_l_squared_block

! **************************************************************************************************
!> \brief Evaluate the AO values in Eq. (1), sharing primitive factors before contraction.
!> Columns of primitive are Cartesian Gaussian primitives, grouped by basis set and exponent.
!> The original spherical contraction coefficients define exactly the same unscreened AO values.
!> \param values ...
!> \param points ...
!> \param basis ...
!> \param center ...
! **************************************************************************************************
   SUBROUTINE evaluate_mu_r_values(values, points, basis, center)
      REAL(dp), INTENT(OUT)             :: values(:, :)
      REAL(dp), INTENT(IN)              :: points(:, :)
      TYPE(gto_basis_set_type), POINTER :: basis
      REAL(dp), INTENT(IN)              :: center(3)

      INTEGER                           :: first_sgf, ico, ig, ipgf, ipower, iset, last_sgf, ncart, ng, nprimitive
      INTEGER, ALLOCATABLE              :: offsets(:)
      LOGICAL                           :: parallel_points
      REAL(dp)                          :: relative(3)
      REAL(dp), ALLOCATABLE             :: coordinate_power(:, :), exponential(:), monomial(:, :), primitive(:, :), radius2(:)

      CPASSERT(ASSOCIATED(basis))
      ng = SIZE(points, 2)
      CPASSERT(SIZE(values, 1) == ng)
      IF (ng == 0) RETURN
      CPASSERT(basis%nset > 0)
      ALLOCATE (offsets(basis%nset), monomial(ng, ncoset(MAXVAL(basis%lmax))), &
                coordinate_power(3, 0:MAXVAL(basis%lmax)), radius2(ng), exponential(ng))
      nprimitive = 0
      DO iset = 1, basis%nset
         offsets(iset) = nprimitive
         nprimitive = nprimitive + basis%npgf(iset)*ncoset(basis%lmax(iset))
      END DO
      ALLOCATE (primitive(ng, nprimitive))
      parallel_points = .TRUE.
!$    parallel_points = .NOT. omp_in_parallel()
      !$OMP PARALLEL DEFAULT(NONE) IF(parallel_points) &
      !$OMP SHARED(ng, points, center, basis, primitive, offsets, ncoset, indco, radius2, monomial) &
      !$OMP PRIVATE(ig, relative, ico, coordinate_power, ipower, iset, ipgf, ncart, exponential)
      !$OMP DO SCHEDULE(STATIC)
      DO ig = 1, ng
         relative = points(:, ig) - center
         radius2(ig) = DOT_PRODUCT(relative, relative)
         coordinate_power(:, 0) = 1.0_dp
         DO ipower = 1, UBOUND(coordinate_power, 2)
            coordinate_power(:, ipower) = coordinate_power(:, ipower - 1)*relative
         END DO
         DO ico = 1, SIZE(monomial, 2)
            monomial(ig, ico) = coordinate_power(1, indco(1, ico))* &
                                coordinate_power(2, indco(2, ico))*coordinate_power(3, indco(3, ico))
         END DO
      END DO
      !$OMP END DO
      ! Fill primitive columns contiguously in g, reusing each Cartesian monomial.
      DO iset = 1, basis%nset
         ncart = ncoset(basis%lmax(iset))
         !$OMP DO SCHEDULE(STATIC)
         DO ipgf = 1, basis%npgf(iset)
            !$OMP SIMD
            DO ig = 1, ng
               exponential(ig) = EXP(-basis%zet(ipgf, iset)*radius2(ig))
            END DO
            DO ico = 1, ncart
               !$OMP SIMD
               DO ig = 1, ng
                  primitive(ig, offsets(iset) + (ipgf - 1)*ncart + ico) = monomial(ig, ico)*exponential(ig)
               END DO
            END DO
         END DO
         !$OMP END DO
      END DO
      !$OMP END PARALLEL
      values = 0.0_dp
      DO iset = 1, basis%nset
         nprimitive = basis%npgf(iset)*ncoset(basis%lmax(iset))
         first_sgf = basis%first_sgf(1, iset)
         last_sgf = basis%last_sgf(basis%nshell(iset), iset)
         CALL dgemm('N', 'N', ng, last_sgf - first_sgf + 1, nprimitive, &
                    1.0_dp, primitive(:, offsets(iset) + 1:), ng, &
                    basis%sphi(:, first_sgf:last_sgf), SIZE(basis%sphi, 1), &
                    1.0_dp, values(:, first_sgf:last_sgf), ng)
      END DO

   END SUBROUTINE evaluate_mu_r_values

! **************************************************************************************************
!> \brief Bound every auxiliary potential in Eq. (2), including contracted angular functions.
!> |x^n exp(-a x²/2)| <= (n/(a e))^(n/2); the remaining Gaussian potential is <=4 pi/a.
!> The triangle inequality uses the transformed primitive coefficients for AUTO_RI functions.
!> Truncating the positive kernel preserves the bound; envelopes are indexed (bound/exponents,atom).
!> \param bs_env ...
!> \param aux_potentials ...
!> \param envelopes ...
! **************************************************************************************************
   SUBROUTINE auxiliary_potential_bound(bs_env, aux_potentials, envelopes)
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      TYPE(radial_interpolation_type), ALLOCATABLE, &
         INTENT(IN)                                      :: aux_potentials(:)
      REAL(dp), ALLOCATABLE, INTENT(OUT)         :: envelopes(:, :)

      INTEGER                                    :: iatom, ico, idir, ikind, ipgf, irow, irow_packed, iset, isgf, ishell, l, &
                                                    ncart, power
      LOGICAL, ALLOCATABLE                       :: angular_used(:)
      REAL(dp)                                   :: exponent, factor
      REAL(dp), ALLOCATABLE                      :: bounds(:)
      TYPE(gto_basis_set_type), POINTER          :: basis
      TYPE(particle_type), POINTER               :: particles(:)

      particles => bs_env%ri_rs%particle_set
      ALLOCATE (envelopes(3, bs_env%n_atom))
      envelopes = 0.0_dp
      DO iatom = 1, bs_env%n_atom
         ikind = particles(iatom)%atomic_kind%kind_number
         basis => bs_env%basis_set_RI(ikind)%gto_basis_set
         CPASSERT(ASSOCIATED(basis))
         IF (bs_env%auto_ri%enabled) THEN
            CPASSERT(ALLOCATED(aux_potentials))
            ALLOCATE (bounds(SIZE(aux_potentials(iatom)%coefficients, 2)))
         ELSE
            ALLOCATE (bounds(SIZE(basis%sphi, 2)))
         END IF
         ALLOCATE (angular_used(0:MAXVAL(basis%lmax)))
         bounds = 0.0_dp
         irow_packed = 0
         envelopes(2, iatom) = HUGE(1.0_dp)
         DO iset = 1, basis%nset
            ncart = ncoset(basis%lmax(iset))
            angular_used = .FALSE.
            DO ishell = 1, basis%nshell(iset)
               angular_used(basis%l(ishell, iset)) = .TRUE.
            END DO
            DO ipgf = 1, basis%npgf(iset)
               exponent = basis%zet(ipgf, iset)
               envelopes(2, iatom) = MIN(envelopes(2, iatom), exponent)
               envelopes(3, iatom) = MAX(envelopes(3, iatom), exponent)
               DO l = 0, basis%lmax(iset)
                  IF (.NOT. angular_used(l)) CYCLE
                  DO ico = ncoset(l - 1) + 1, ncoset(l)
                     irow = (ipgf - 1)*ncart + ico
                     irow_packed = irow_packed + 1
                     factor = 4.0_dp*pi/exponent
                     DO idir = 1, 3
                        power = indco(idir, ico)
                        IF (power > 0) factor = factor*(REAL(power, dp)/(exponent*EXP(1.0_dp)))** &
                                                (0.5_dp*REAL(power, dp))
                     END DO
                     IF (bs_env%auto_ri%enabled) THEN
                        bounds(:) = bounds(:) + ABS(aux_potentials(iatom)%coefficients(irow_packed, :))*factor
                     ELSE
                        DO ishell = 1, basis%nshell(iset)
                           IF (basis%l(ishell, iset) /= l) CYCLE
                           DO isgf = basis%first_sgf(ishell, iset), basis%last_sgf(ishell, iset)
                              bounds(isgf) = bounds(isgf) + ABS(basis%sphi(irow, isgf))*factor
                           END DO
                        END DO
                     END IF
                  END DO
               END DO
            END DO
         END DO
         IF (bs_env%auto_ri%enabled) THEN
            IF (.NOT. (irow_packed == SIZE(aux_potentials(iatom)%coefficients, 1))) THEN
               CPABORT('Inconsistent packed auxiliary rows')
            END IF
         END IF
         envelopes(1, iatom) = MAXVAL(bounds)
         DEALLOCATE (bounds, angular_used)
      END DO

   END SUBROUTINE auxiliary_potential_bound

! **************************************************************************************************
!> \brief Contract Eqs. (2),(4), screening every spatial l/P-atom/T block separately.
!> For d=max(0,|r-center|-rc), |v_P(r)|<=B min(1,a_max rc² exp(-a_min d²/2)/2).
!> B bounds every selected potential on this atom; a_min/max bound its primitive exponents.
!> For erfc(ωr)/r, |v_P(r)|<=B exp[-a_min ω² |r-center|²/(a_min+2ω²)].
!> This follows from erfc(x)<=exp(-x²), completing the square, and the Gaussian Coulomb bound.
!> Half the absolute budget screens common points. The other half is applied independently to
!> every d_lP(T): a point is contracted for one l block only when its local product bound exceeds
!> the per-point threshold. The exact l/P/T distance criterion is applied before each GEMM.
!> \param bs_env ...
!> \param grid_points ...
!> \param cells_R_d_lP Retained integer cell vectors R, indexed (3,R).
!> \param include_lP_R ...
!> \param include_l_blocks_P_R ...
!> \param points ...
!> \param field_squared ...
!> \param weight ...
!> \param envelopes ...
!> \param blocks ...
!> \param threshold ...
!> \param omitted ...
!> \param potential_points ...
!> \param block_contractions ...
!> \param screened_blocks ...
!> \param coalesced_contractions ...
!> \param contraction_products ...
!> \param d_lP_R Numerical integrals indexed by local point l, RI function P and cell R.
!> \param aux_potentials ...
! **************************************************************************************************
   SUBROUTINE integrate_potentials(bs_env, grid_points, cells_R_d_lP, include_lP_R, include_l_blocks_P_R, &
                                   points, field_squared, weight, &
                                   envelopes, blocks, threshold, omitted, potential_points, block_contractions, &
                                   screened_blocks, coalesced_contractions, contraction_products, d_lP_R, &
                                   aux_potentials)
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      REAL(dp), INTENT(IN)                       :: grid_points(:, :)
      INTEGER, INTENT(IN)                        :: cells_R_d_lP(:, :)
      LOGICAL, INTENT(IN)                        :: include_lP_R(:, :, :), include_l_blocks_P_R(:, :, :)
      REAL(dp), INTENT(IN)                       :: points(:, :), field_squared(:, :), weight, envelopes(:, :)
      TYPE(l_block_partition_type), INTENT(IN)   :: blocks
      REAL(dp), INTENT(IN)                       :: threshold
      REAL(dp), INTENT(INOUT)                    :: omitted
      INTEGER(int_8), INTENT(INOUT)              :: potential_points, block_contractions, screened_blocks, &
                                                    coalesced_contractions, contraction_products
      REAL(dp), INTENT(INOUT)                    :: d_lP_R(:, :, :)
      TYPE(radial_interpolation_type), ALLOCATABLE, &
         INTENT(IN)                                      :: aux_potentials(:)

      INTEGER                                    :: firstp, handle, iatom, iblock, ig, ikind, ilocal, it, lastp, maxnp, &
                                                    nactive, nkeep, np, nthreads, nunion, row
      REAL(dp)                                   :: center(3), cutoff, decay, distance, estimate
      REAL(dp), ALLOCATABLE                      :: block_field_bound(:, :), block_RI(:, :), field_work(:, :), point_bound(:), &
                                                    potential(:, :)
      LOGICAL                                    :: point_needed(SIZE(points, 2))
      INTEGER                                    :: union_indices(SIZE(points, 2)), active_l(SIZE(grid_points, 2))
      TYPE(cell_type), POINTER                   :: cell
      TYPE(gto_basis_set_type), POINTER          :: basis
      LOGICAL                                    :: block_needed(blocks%nblocks)
      INTEGER                                    :: block_active_last(blocks%nblocks), block_active_first(blocks%nblocks)
      TYPE(particle_type), POINTER               :: particles(:)

      cell => bs_env%ri_rs%cell
      particles => bs_env%ri_rs%particle_set
      cutoff = 0.0_dp
      IF (bs_env%ri_metric%potential_type == do_potential_truncated) cutoff = bs_env%ri_metric%cutoff_radius
      maxnp = MAXVAL(bs_env%i_RI_end_from_atom - bs_env%i_RI_start_from_atom + 1)
      IF (bs_env%auto_ri%enabled) maxnp = MAXVAL(bs_env%auto_ri%sizes_opt_RI)

      CALL timeset('d_lP_allocate_field_bounds', handle)

      ALLOCATE (block_field_bound(SIZE(points, 2), blocks%nblocks))

      CALL timestop(handle)

      CALL timeset('d_lP_prepare_field_bounds', handle)

      DO iblock = 1, blocks%nblocks
         block_field_bound(:, iblock) = 0.0_dp
         DO ilocal = blocks%first(iblock), blocks%first(iblock + 1) - 1
            DO ig = 1, SIZE(points, 2)
               block_field_bound(ig, iblock) = MAX(block_field_bound(ig, iblock), &
                                                   ABS(field_squared(ig, blocks%indices(ilocal))))
            END DO
         END DO
      END DO

      CALL timestop(handle)

      ! Bound the additional thread workspaces to 128 MiB per MPI rank.
      ! A single thread retains the original memory bound for large global l grids.
      nthreads = 1
!$    nthreads = omp_get_max_threads()
      nthreads = MIN(nthreads, MAX(1, INT(128.0_dp*1024.0_dp**2/ &
                                          (8.0_dp*(REAL(SIZE(grid_points, 2), dp)*(maxnp + SIZE(points, 2)) + &
                                                   REAL(SIZE(points, 2), dp)*maxnp)))))
      ! Each atom/image owns disjoint P/T columns of Eq. (2).

      !$OMP PARALLEL DEFAULT(NONE) NUM_THREADS(nthreads) &
      !$OMP SHARED(bs_env, grid_points, cells_R_d_lP, points, field_squared, weight, envelopes, &
      !$OMP blocks, threshold, d_lP_R, aux_potentials, cell, particles, cutoff, &
      !$OMP maxnp, block_field_bound, include_lP_R, include_l_blocks_P_R) &
      !$OMP PRIVATE(firstp, handle, iatom, iblock, ig, ikind, ilocal, it, lastp, nactive, nkeep, &
      !$OMP np, nunion, row, center, decay, distance, estimate, basis, &
      !$OMP block_RI, field_work, point_bound, potential, point_needed, union_indices, &
      !$OMP active_l, block_needed, block_active_last, block_active_first) &
      !$OMP REDUCTION(+:omitted, potential_points, block_contractions, screened_blocks, &
      !$OMP coalesced_contractions, contraction_products)

      CALL timeset('d_lP_allocate_thread_buffers', handle)

      ALLOCATE (block_RI(SIZE(grid_points, 2), maxnp), &
                field_work(SIZE(points, 2), SIZE(grid_points, 2)), point_bound(SIZE(points, 2)), &
                potential(SIZE(points, 2), maxnp))

      CALL timestop(handle)

      !$OMP DO COLLAPSE(2) SCHEDULE(DYNAMIC)
      DO it = 1, SIZE(cells_R_d_lP, 2)
         DO iatom = 1, bs_env%n_atom
            ikind = particles(iatom)%atomic_kind%kind_number
            basis => bs_env%basis_set_RI(ikind)%gto_basis_set
            firstp = bs_env%i_RI_start_from_atom(iatom)
            lastp = bs_env%i_RI_end_from_atom(iatom)
            IF (bs_env%auto_ri%enabled) THEN
               firstp = 1 + SUM(bs_env%auto_ri%sizes_opt_RI(:iatom - 1))
               lastp = firstp + bs_env%auto_ri%sizes_opt_RI(iatom) - 1
            END IF
            np = lastp - firstp + 1
            center = pbc(particles(iatom)%r, cell) + MATMUL(bs_env%hmat, REAL(cells_R_d_lP(:, it), dp))
            IF (.NOT. ANY(include_l_blocks_P_R(:, iatom, it))) CYCLE
            CALL timeset('d_lP_screen_atom_image', handle)
            decay = 0.0_dp
            IF (bs_env%ri_metric%potential_type == do_potential_short) THEN
               decay = envelopes(2, iatom)*bs_env%ri_metric%omega**2/ &
                       (envelopes(2, iatom) + 2.0_dp*bs_env%ri_metric%omega**2)
            END IF
            DO ig = 1, SIZE(points, 2)
               point_bound(ig) = envelopes(1, iatom)
               IF (decay > 0.0_dp) point_bound(ig) = point_bound(ig)* &
                                                     EXP(-decay*SUM((points(:, ig) - center)**2))
               IF (cutoff > 0.0_dp) THEN
                  distance = MAX(0.0_dp, NORM2(points(:, ig) - center) - cutoff)
                  point_bound(ig) = point_bound(ig)* &
                                    MIN(1.0_dp, 0.5_dp*envelopes(3, iatom)*cutoff**2* &
                                        EXP(-0.5_dp*envelopes(2, iatom)*distance**2))
               END IF
            END DO
            block_needed = .FALSE.
            point_needed = .FALSE.
            DO iblock = 1, blocks%nblocks
               IF (.NOT. include_l_blocks_P_R(iblock, iatom, it)) CYCLE
               DO ig = 1, SIZE(points, 2)
                  estimate = block_field_bound(ig, iblock)*point_bound(ig)
                  IF (estimate > threshold) THEN
                     block_needed(iblock) = .TRUE.
                     point_needed(ig) = .TRUE.
                  ELSE
                     omitted = omitted + weight*estimate
                  END IF
               END DO
               IF (.NOT. block_needed(iblock)) screened_blocks = screened_blocks + 1_int_8
            END DO
            nunion = 0
            DO ig = 1, SIZE(points, 2)
               IF (point_needed(ig)) THEN
                  nunion = nunion + 1
                  union_indices(nunion) = ig
               END IF
            END DO
            CALL timestop(handle)
            IF (nunion == 0) CYCLE
            potential_points = potential_points + INT(nunion, int_8)
            CALL timeset('d_lP_potential_eval', handle)
            IF (bs_env%auto_ri%enabled) THEN
               CALL evaluate_RI_potential(potential(:, 1:np), &
                                          points(:, union_indices(1:nunion)), &
                                          basis, center, cutoff, aux_potentials(iatom), &
                                          attenuation=MAX(0.0_dp, bs_env%ri_metric%omega))
            ELSE
               CALL evaluate_RI_potential(potential(1:nunion, 1:np), &
                                          points(:, union_indices(1:nunion)), &
                                          basis, center, cutoff, &
                                          attenuation=MAX(0.0_dp, bs_env%ri_metric%omega))
            END IF
            CALL timestop(handle)
            CALL timeset('d_lP_prepare_quadrature', handle)
            nactive = 0
            block_active_first = 0
            block_active_last = -1
            DO iblock = 1, blocks%nblocks
               IF (.NOT. block_needed(iblock)) CYCLE
               block_active_first(iblock) = nactive + 1
               DO ilocal = blocks%first(iblock), blocks%first(iblock + 1) - 1
                  IF (include_lP_R(blocks%indices(ilocal), iatom, it)) THEN
                     nactive = nactive + 1
                     active_l(nactive) = blocks%indices(ilocal)
                  END IF
               END DO
               block_active_last(iblock) = nactive
               IF (.NOT. (block_active_last(iblock) >= block_active_first(iblock))) THEN
                  CPABORT('Empty active integration block')
               END IF
            END DO
            DO iblock = 1, blocks%nblocks
               IF (.NOT. block_needed(iblock)) CYCLE
               DO row = 1, nunion
                  ig = union_indices(row)
                  point_needed(row) = block_field_bound(ig, iblock)*point_bound(ig) > threshold
               END DO
               nkeep = COUNT(point_needed(1:nunion))
               DO ilocal = block_active_first(iblock), block_active_last(iblock)
                  DO row = 1, nunion
                     field_work(row, ilocal) = MERGE( &
                                               field_squared(union_indices(row), active_l(ilocal)), 0.0_dp, point_needed(row))
                  END DO
               END DO
               CPASSERT(nkeep > 0)
               block_contractions = block_contractions + 1_int_8
            END DO
            CALL timestop(handle)
            CALL timeset('d_lP_sum_g_f2_v', handle)
            CALL dgemm('T', 'N', nactive, np, nunion, weight, field_work, SIZE(field_work, 1), &
                       potential, SIZE(potential, 1), 0.0_dp, block_RI, SIZE(block_RI, 1))
            DO ilocal = 1, nactive
               d_lP_R(active_l(ilocal), firstp:lastp, it) = &
                  d_lP_R(active_l(ilocal), firstp:lastp, it) + block_RI(ilocal, 1:np)
            END DO
            CALL timestop(handle)
            coalesced_contractions = coalesced_contractions + 1_int_8
            contraction_products = contraction_products + &
                                   INT(nactive, int_8)*INT(np, int_8)*INT(nunion, int_8)
         END DO
      END DO
      !$OMP END DO NOWAIT

      CALL timeset('d_lP_wait_for_threads', handle)

      !$OMP BARRIER

      CALL timestop(handle)

      CALL timeset('d_lP_release_thread_buffers', handle)

      DEALLOCATE (block_RI, field_work, point_bound, potential)

      CALL timestop(handle)

      !$OMP END PARALLEL

   END SUBROUTINE integrate_potentials

! **************************************************************************************************
!> \brief Squared distance from a point to an axis-aligned spatial l block.
!> \param point ...
!> \param lower ...
!> \param upper ...
!> \return ...
! **************************************************************************************************
   PURE FUNCTION distance_to_box_squared(point, lower, upper) RESULT(distance_squared)
      REAL(dp), INTENT(IN) :: point(3), lower(3), upper(3)
      REAL(dp)             :: distance_squared

      INTEGER              :: idir
      REAL(dp)             :: delta

      distance_squared = 0.0_dp
      DO idir = 1, 3
         delta = MAX(lower(idir) - point(idir), point(idir) - upper(idir), 0.0_dp)
         distance_squared = distance_squared + delta**2
      END DO
   END FUNCTION distance_to_box_squared

! **************************************************************************************************
!> \brief Apply the established l/P/T distance criterion after Eq. (2).
!> \param bs_env ...
!> \param grid_points ...
!> \param cells_R_d_lP Retained integer cell vectors R, indexed (3,R).
!> \param d_lP_R Numerical integrals indexed by local point l, RI function P and cell R.
! **************************************************************************************************
   SUBROUTINE apply_d_lP_distance_cutoff(bs_env, grid_points, cells_R_d_lP, d_lP_R)
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      REAL(dp), INTENT(IN)                       :: grid_points(:, :)
      INTEGER, INTENT(IN)                        :: cells_R_d_lP(:, :)
      REAL(dp), INTENT(INOUT)                    :: d_lP_R(:, :, :)

      INTEGER                                    :: firstp, iatom, il, it, lastp
      REAL(dp)                                   :: center(3), cutoff
      REAL(dp), ALLOCATABLE                      :: cutoffs_lP(:)
      TYPE(cell_type), POINTER                   :: cell
      TYPE(particle_type), POINTER               :: particles(:)

      cell => bs_env%ri_rs%cell
      particles => bs_env%ri_rs%particle_set
      CALL compute_d_lP_cutoffs(bs_env, cutoffs_lP)
      DO it = 1, SIZE(cells_R_d_lP, 2)
         DO iatom = 1, bs_env%n_atom
            center = pbc(particles(iatom)%r, cell) + MATMUL(bs_env%hmat, REAL(cells_R_d_lP(:, it), dp))
            firstp = bs_env%i_RI_start_from_atom(iatom)
            lastp = bs_env%i_RI_end_from_atom(iatom)
            IF (bs_env%auto_ri%enabled) THEN
               firstp = 1 + SUM(bs_env%auto_ri%sizes_opt_RI(:iatom - 1))
               lastp = firstp + bs_env%auto_ri%sizes_opt_RI(iatom) - 1
            END IF
            cutoff = cutoffs_lP(iatom)
            DO il = 1, SIZE(grid_points, 2)
               IF (NORM2(grid_points(:, il) - center) > cutoff) d_lP_R(il, firstp:lastp, it) = 0.0_dp
            END DO
         END DO
      END DO
      DEALLOCATE (cutoffs_lP)

   END SUBROUTINE apply_d_lP_distance_cutoff

! **************************************************************************************************
!> \brief Evaluate v_P(r) for a contracted spherical auxiliary basis, without Gaussian smearing.
!> For a primitive H_l(r) exp(-a r²), H_l is a homogeneous harmonic polynomial.
!> (1) Its potential is H_l(r) (-1/a)^l d^l F(s)/ds^l, s=r².
!> (2) F is the s-Gaussian convolution with theta(rc-|r-r'|)/|r-r'|:
!> F(r²)=pi^(3/2)[2 erf(sqrt(a)r)-erf(sqrt(a)(r-rc))-erf(sqrt(a)(r+rc))]/(2 a^(3/2) r).
!> The existing sphi coefficients supply all contraction and angular normalization factors.
!> A nonpositive rc selects the untruncated Coulomb kernel. No spatial cutoff is applied here.
!> The origin uses a convergent power series; elsewhere Taylor arithmetic differentiates (2).
!> \param values Potentials indexed by integration point and contracted auxiliary function.
!> \param points Cartesian integration points in bohr, indexed (3,point).
!> \param basis Contracted spherical auxiliary basis.
!> \param center Auxiliary atomic center in bohr.
!> \param cutoff Coulomb cutoff in bohr, or nonpositive for the full kernel.
!> \param contracted ...
!> \param attenuation Positive omega (inverse bohr) selects erfc(omega*r)/r.
! **************************************************************************************************
   SUBROUTINE evaluate_RI_potential(values, points, basis, center, cutoff, contracted, attenuation)
      REAL(dp), CONTIGUOUS, INTENT(OUT) :: values(:, :)
      REAL(dp), INTENT(IN)              :: points(:, :)
      TYPE(gto_basis_set_type), POINTER :: basis
      REAL(dp), INTENT(IN)              :: center(3), cutoff
      TYPE(radial_interpolation_type), INTENT(IN), &
         OPTIONAL                                        :: contracted
      REAL(dp), INTENT(IN), OPTIONAL    :: attenuation

      CPASSERT(ASSOCIATED(basis))
      CPASSERT(SIZE(points, 1) == 3)
      IF (PRESENT(contracted)) THEN
         CPASSERT(SIZE(values, 1) >= SIZE(points, 2))
      ELSE
         CPASSERT(SIZE(values, 1) == SIZE(points, 2))
      END IF
      CALL contract_potentials(values, points, basis, center, cutoff, contracted, attenuation)

   END SUBROUTINE evaluate_RI_potential

! **************************************************************************************************
!> \brief Transform spherical coefficients once; bound optional radial tables by radial_table_bytes.
!> \param basis ...
!> \param U_Pp ...
!> \param contracted ...
!> \param cutoff ...
!> \param attenuation ...
!> \param radial_table_bytes ...
!> \param mesh_step ...
!> \param mesh_displacement ...
! **************************************************************************************************
   SUBROUTINE prepare_contracted_potential(basis, U_Pp, contracted, cutoff, attenuation, radial_table_bytes, &
                                           mesh_step, mesh_displacement)
      TYPE(gto_basis_set_type), POINTER            :: basis
      REAL(dp), INTENT(IN), OPTIONAL               :: U_Pp(:, :)
      TYPE(radial_interpolation_type), INTENT(OUT) :: contracted
      REAL(dp), INTENT(IN), OPTIONAL               :: cutoff, attenuation, radial_table_bytes, mesh_step(3, 3), &
                                                      mesh_displacement(3)

      INTEGER                                      :: ico, iopt, ipgf, irow, irow_packed, iset, isgf, ishell, l, ncart, &
                                                      ninterval, np, nrows
      LOGICAL, ALLOCATABLE                         :: angular_used(:)
      REAL(dp)                                     :: bytes_per_node, table_bytes
      REAL(dp), ALLOCATABLE                        :: full_abscissae(:), moments(:, :)

      CPASSERT(ASSOCIATED(basis))
      ALLOCATE (angular_used(0:MAXVAL(basis%lmax)))
      np = basis%nsgf
      IF (PRESENT(U_Pp)) THEN
         CPASSERT(SIZE(U_Pp, 1) == basis%nsgf)
         np = SIZE(U_Pp, 2)
      END IF
      nrows = 0
      DO iset = 1, basis%nset
         angular_used = .FALSE.
         DO ishell = 1, basis%nshell(iset)
            angular_used(basis%l(ishell, iset)) = .TRUE.
         END DO
         DO l = 0, basis%lmax(iset)
            IF (angular_used(l)) nrows = nrows + basis%npgf(iset)*(ncoset(l) - ncoset(l - 1))
         END DO
      END DO
      ! W_g,ia is independent of the contracted shell: sum its coefficients before evaluating W.
      ! Rows enumerate (set, primitive, distinct angular momentum, Cartesian monomial).
      ALLOCATE (contracted%coefficients(nrows, np), source=0.0_dp)
      irow_packed = 0
      DO iset = 1, basis%nset
         ncart = ncoset(basis%lmax(iset))
         angular_used = .FALSE.
         DO ishell = 1, basis%nshell(iset)
            angular_used(basis%l(ishell, iset)) = .TRUE.
         END DO
         DO ipgf = 1, basis%npgf(iset)
            DO l = 0, basis%lmax(iset)
               IF (.NOT. angular_used(l)) CYCLE
               DO ico = ncoset(l - 1) + 1, ncoset(l)
                  irow = (ipgf - 1)*ncart + ico
                  irow_packed = irow_packed + 1
                  DO ishell = 1, basis%nshell(iset)
                     IF (basis%l(ishell, iset) /= l) CYCLE
                     DO isgf = basis%first_sgf(ishell, iset), basis%last_sgf(ishell, iset)
                        IF (PRESENT(U_Pp)) THEN
                           DO iopt = 1, np
                              contracted%coefficients(irow_packed, iopt) = &
                                 contracted%coefficients(irow_packed, iopt) + basis%sphi(irow, isgf)*U_Pp(isgf, iopt)
                           END DO
                        ELSE
                           contracted%coefficients(irow_packed, isgf) = &
                              contracted%coefficients(irow_packed, isgf) + basis%sphi(irow, isgf)
                        END IF
                     END DO
                  END DO
               END DO
            END DO
         END DO
      END DO
      CPASSERT(irow_packed == nrows)
      DEALLOCATE (angular_used)
      IF (PRESENT(U_Pp) .AND. basis%nsgf < nrows) THEN
         CALL prepare_spherical_potential_factorization(basis, U_Pp, contracted)
      END IF
      IF (PRESENT(cutoff) .OR. PRESENT(attenuation)) THEN
         IF (PRESENT(cutoff)) contracted%cutoff = cutoff
         IF (PRESENT(attenuation)) contracted%attenuation = MAX(0.0_dp, attenuation)
         CALL prepare_radial_series(basis, contracted%cutoff, contracted%attenuation, &
                                    contracted%series, contracted%short_series)
         IF (contracted%attenuation > 0.0_dp) THEN
            table_bytes = 16.0_dp*1024.0_dp**2
            IF (PRESENT(radial_table_bytes)) table_bytes = MAX(0.0_dp, radial_table_bytes)
            ! Budget the actual logarithmic nodes, including one primitive preparation buffer.
            bytes_per_node = 8.0_dp*REAL(MAXVAL(basis%lmax) + 7, dp)* &
                             REAL(MAXVAL(basis%npgf)*basis%nset + 1, dp)
            CALL prepare_short_radial_grid(512.0_dp, contracted%short_abscissae)
            table_bytes = MAX(0.0_dp, table_bytes - 8.0_dp*REAL(SIZE(contracted%short_abscissae), dp))
            ninterval = INT(MIN(REAL(SIZE(contracted%short_abscissae), dp), &
                                table_bytes/bytes_per_node)) - 1
            IF (ninterval < 1) THEN
               DEALLOCATE (contracted%short_abscissae)
               RETURN
            END IF
            IF (ninterval < UBOUND(contracted%short_abscissae, 1)) THEN
               CALL MOVE_ALLOC(contracted%short_abscissae, full_abscissae)
               ALLOCATE (contracted%short_abscissae(0:ninterval))
               contracted%short_abscissae(:) = full_abscissae(0:ninterval)
               DEALLOCATE (full_abscissae)
            END IF
            ALLOCATE (contracted%short_moments(0:MAXVAL(basis%lmax) + 6, &
                                               0:ninterval, MAXVAL(basis%npgf), basis%nset))
            DO iset = 1, basis%nset
               DO ipgf = 1, basis%npgf(iset)
                  CALL prepare_short_radial_table(basis%zet(ipgf, iset), contracted%attenuation, &
                                                  MAXVAL(basis%lmax), moments, ninterval, abscissae=contracted%short_abscissae)
                  contracted%short_moments(:, :, ipgf, iset) = moments
                  DEALLOCATE (moments)
               END DO
            END DO
            ! Keep the generic radial lookup at all radii before using remaining storage for mesh nodes.
            table_bytes = MAX(0.0_dp, table_bytes - 8.0_dp*REAL(SIZE(contracted%short_moments, KIND=int_8), dp))
            IF (PRESENT(mesh_step) .AND. PRESENT(mesh_displacement)) THEN
               CALL prepare_mesh_radial_moments(basis, contracted, mesh_step, mesh_displacement, table_bytes)
            END IF
         END IF
      END IF

   END SUBROUTINE prepare_contracted_potential

! **************************************************************************************************
!> \brief Factor sphi into a common angular map and shell/primitive radial scales when exact.
!> This is checked against every spherical coefficient; arbitrary contracted or modified bases
!> retain the original Cartesian path if the factorization does not hold.
!> \param basis ...
!> \param U_Pp ...
!> \param contracted ...
! **************************************************************************************************
   SUBROUTINE prepare_spherical_potential_factorization(basis, U_Pp, contracted)
      TYPE(gto_basis_set_type), POINTER              :: basis
      REAL(dp), INTENT(IN)                           :: U_Pp(:, :)
      TYPE(radial_interpolation_type), INTENT(INOUT) :: contracted

      INTEGER                                        :: c, first, ico, ipgf, iset, ishell, l, lmax, m, ncart, ref_c, ref_m, &
                                                        ref_p, ref_shell, row
      LOGICAL, ALLOCATABLE                           :: used(:)
      REAL(dp)                                       :: actual, expected, ref_abs, ref_value, scale
      REAL(dp), ALLOCATABLE                          :: angular_coeff(:, :, :, :), primitive_scale(:, :, :)

      IF (SIZE(U_Pp, 1) /= basis%nsgf) RETURN
      ALLOCATE (used(basis%nsgf), source=.FALSE.)
      ALLOCATE (angular_coeff(MAXVAL(ncoset(0:MAXVAL(basis%lmax)) - &
                                     ncoset(-1:MAXVAL(basis%lmax) - 1)), &
                              MAXVAL(nso(0:MAXVAL(basis%lmax))), &
                              0:MAXVAL(basis%lmax), basis%nset), source=0.0_dp)
      ALLOCATE (primitive_scale(MAXVAL(basis%npgf), MAXVAL(basis%nshell), basis%nset), &
                source=0.0_dp)
      DO iset = 1, basis%nset
         DO ishell = 1, basis%nshell(iset)
            l = basis%l(ishell, iset)
            first = basis%first_sgf(ishell, iset)
            IF (first < 1 .OR. basis%last_sgf(ishell, iset) /= first + nso(l) - 1) RETURN
            IF (basis%last_sgf(ishell, iset) > basis%nsgf) RETURN
            IF (ANY(used(first:first + nso(l) - 1))) RETURN
            used(first:first + nso(l) - 1) = .TRUE.
         END DO
      END DO
      IF (.NOT. ALL(used)) RETURN
      DO iset = 1, basis%nset
         lmax = basis%lmax(iset)
         DO l = 0, lmax
            ref_abs = 0.0_dp
            ref_shell = 0
            ncart = ncoset(l) - ncoset(l - 1)
            DO ishell = 1, basis%nshell(iset)
               IF (basis%l(ishell, iset) /= l) CYCLE
               first = basis%first_sgf(ishell, iset)
               DO ipgf = 1, basis%npgf(iset)
                  DO m = 1, nso(l)
                     DO c = 1, ncart
                        ico = ncoset(l - 1) + c
                        row = (ipgf - 1)*ncoset(lmax) + ico
                        actual = basis%sphi(row, first + m - 1)
                        IF (ABS(actual) > ref_abs) THEN
                           ref_abs = ABS(actual)
                           ref_value = actual
                           ref_shell = ishell
                           ref_p = ipgf
                           ref_c = c
                           ref_m = m
                        END IF
                     END DO
                  END DO
               END DO
            END DO
            IF (ref_shell == 0) CYCLE
            first = basis%first_sgf(ref_shell, iset)
            DO m = 1, nso(l)
               DO c = 1, ncart
                  ico = ncoset(l - 1) + c
                  row = (ref_p - 1)*ncoset(lmax) + ico
                  angular_coeff(c, m, l, iset) = basis%sphi(row, first + m - 1)
               END DO
            END DO
            DO ishell = 1, basis%nshell(iset)
               IF (basis%l(ishell, iset) /= l) CYCLE
               first = basis%first_sgf(ishell, iset)
               DO ipgf = 1, basis%npgf(iset)
                  row = (ipgf - 1)*ncoset(lmax) + ncoset(l - 1) + ref_c
                  scale = basis%sphi(row, first + ref_m - 1)/ref_value
                  primitive_scale(ipgf, ishell, iset) = scale
                  DO m = 1, nso(l)
                     DO c = 1, ncart
                        ico = ncoset(l - 1) + c
                        row = (ipgf - 1)*ncoset(lmax) + ico
                        actual = basis%sphi(row, first + m - 1)
                        expected = scale*angular_coeff(c, m, l, iset)
                        IF (ABS(actual - expected) > 64.0_dp*EPSILON(1.0_dp)* &
                            MAX(ref_abs, ABS(actual), ABS(expected))) RETURN
                     END DO
                  END DO
               END DO
            END DO
         END DO
      END DO
      CALL MOVE_ALLOC(angular_coeff, contracted%spherical_angular)
      CALL MOVE_ALLOC(primitive_scale, contracted%spherical_primitive)
      ALLOCATE (contracted%spherical_selection(SIZE(U_Pp, 1), SIZE(U_Pp, 2)))
      contracted%spherical_selection(:, :) = U_Pp
      contracted%spherical_factorized = .TRUE.

   END SUBROUTINE prepare_spherical_potential_factorization

! **************************************************************************************************
!> \brief Prepare the origin series in Eq. (2) once for a fixed basis and interaction kernel.
!> The coefficients depend on neither the integration point nor the translated atomic center.
!> \param basis ...
!> \param cutoff ...
!> \param omega ...
!> \param series ...
!> \param short_series ...
! **************************************************************************************************
   SUBROUTINE prepare_radial_series(basis, cutoff, omega, series, short_series)
      TYPE(gto_basis_set_type), POINTER  :: basis
      REAL(dp), INTENT(IN)               :: cutoff, omega
      REAL(dp), ALLOCATABLE, INTENT(OUT) :: series(:, :, :), short_series(:, :, :)

      INTEGER                            :: in, ipgf, iset, lmax
      REAL(dp)                           :: beta_ratio, exponent

      lmax = MAXVAL(basis%lmax)
      ALLOCATE (series(0:lmax + 24, MAXVAL(basis%npgf), basis%nset), &
                short_series(0:lmax + 24, MAXVAL(basis%npgf), basis%nset))
      DO iset = 1, basis%nset
         DO ipgf = 1, basis%npgf(iset)
            exponent = basis%zet(ipgf, iset)
            IF (omega > 0.0_dp) THEN
               CALL origin_series(exponent, 0.0_dp, series(:, ipgf, iset))
               beta_ratio = omega**2/(exponent + omega**2)
               DO in = 0, UBOUND(series, 1)
                  short_series(in, ipgf, iset) = series(in, ipgf, iset)* &
                                                 (1.0_dp - beta_ratio**(REAL(in, dp) + 0.5_dp))
               END DO
            ELSE
               CALL origin_series(exponent, cutoff, series(:, ipgf, iset))
               short_series(:, ipgf, iset) = 0.0_dp
            END IF
         END DO
      END DO

   END SUBROUTINE prepare_radial_series

! **************************************************************************************************
!> \brief Contract primitive potentials from Eq. (1) using the CP2K spherical coefficients.
!> \param values ...
!> \param points ...
!> \param basis ...
!> \param center ...
!> \param cutoff ...
!> \param contracted ...
!> \param attenuation ...
! **************************************************************************************************
   SUBROUTINE contract_potentials(values, points, basis, center, cutoff, contracted, attenuation)
      REAL(dp), CONTIGUOUS, INTENT(OUT) :: values(:, :)
      REAL(dp), INTENT(IN)              :: points(:, :)
      TYPE(gto_basis_set_type), POINTER :: basis
      REAL(dp), INTENT(IN)              :: center(3), cutoff
      TYPE(radial_interpolation_type), INTENT(IN), &
         OPTIONAL                                        :: contracted
      REAL(dp), INTENT(IN), OPTIONAL    :: attenuation

      INTEGER                           :: ico, in, ipgf, ipoint, irow, iset, isgf, ishell, l, lmax, ncart
      LOGICAL                           :: parallel_points, short
      REAL(dp)                          :: beta_ratio, exponent, minimum_radius2, omega, polynomial, radius, relative(3), &
                                           weight
      REAL(dp), ALLOCATABLE             :: long_radial(:), radial(:), series(:, :, :), short_series(:, :, :)
      TYPE(radial_work_type)            :: radial_work

      lmax = MAXVAL(basis%lmax)
      omega = 0.0_dp
      IF (PRESENT(attenuation)) omega = attenuation
      short = omega > 0.0_dp
      IF (PRESENT(contracted)) THEN
         CPASSERT(ALLOCATED(contracted%coefficients))
         CPASSERT(SIZE(contracted%coefficients, 2) == SIZE(values, 2))
         CALL contract_selected_potentials(values, points, basis, center, cutoff, contracted, attenuation)
         RETURN
      END IF
      ALLOCATE (series(0:lmax + 24, MAXVAL(basis%npgf), basis%nset), &
                short_series(0:lmax + 24, MAXVAL(basis%npgf), basis%nset))
      minimum_radius2 = MINVAL(SUM((points - SPREAD(center, 2, SIZE(points, 2)))**2, DIM=1))
      DO iset = 1, basis%nset
         DO ipgf = 1, basis%npgf(iset)
            exponent = basis%zet(ipgf, iset)
            IF (short) THEN
               CALL origin_series(exponent, 0.0_dp, series(:, ipgf, iset))
               beta_ratio = omega**2/(exponent + omega**2)
               DO in = 0, UBOUND(series, 1)
                  short_series(in, ipgf, iset) = series(in, ipgf, iset)* &
                                                 (1.0_dp - beta_ratio**(REAL(in, dp) + 0.5_dp))
               END DO
            ELSE IF (exponent*minimum_radius2 < 0.5_dp) THEN
               CALL origin_series(exponent, cutoff, series(:, ipgf, iset))
            ELSE
               series(:, ipgf, iset) = 0.0_dp
            END IF
         END DO
      END DO
      values = 0.0_dp
      parallel_points = .TRUE.
!$    parallel_points = .NOT. omp_in_parallel()
      !$OMP PARALLEL DEFAULT(NONE) IF(parallel_points) &
      !$OMP SHARED(values, points, basis, center, cutoff, lmax, series, short_series, &
      !$OMP short, omega, ncoset, indco) &
      !$OMP PRIVATE(ipoint, relative, radius, iset, ipgf, exponent, radial, ishell, &
      !$OMP l, isgf, ico, irow, ncart, polynomial, weight, radial_work, long_radial)
      ALLOCATE (radial(0:lmax), long_radial(0:lmax), radial_work%jet(0:lmax), &
                radial_work%gaussian(0:lmax), radial_work%derivative(0:lmax), radial_work%bracket(0:lmax))
      !$OMP DO SCHEDULE(STATIC)
      DO ipoint = 1, SIZE(points, 2)
         relative = points(:, ipoint) - center
         radius = NORM2(relative)
         DO iset = 1, basis%nset
            ncart = ncoset(basis%lmax(iset))
            DO ipgf = 1, basis%npgf(iset)
               exponent = basis%zet(ipgf, iset)
               IF (short) THEN
                  CALL radial_potential_short(radius, exponent, omega, series(:, ipgf, iset), &
                                              short_series(:, ipgf, iset), radial, radial_work, long_radial)
               ELSE
                  CALL radial_potential(radius, exponent, cutoff, series(:, ipgf, iset), radial, radial_work)
               END IF
               DO ishell = 1, basis%nshell(iset)
                  l = basis%l(ishell, iset)
                  DO isgf = basis%first_sgf(ishell, iset), basis%last_sgf(ishell, iset)
                     polynomial = 0.0_dp
                     DO ico = ncoset(l - 1) + 1, ncoset(l)
                        irow = (ipgf - 1)*ncart + ico
                        weight = basis%sphi(irow, isgf)
                        polynomial = polynomial + weight*relative(1)**indco(1, ico)* &
                                     relative(2)**indco(2, ico)*relative(3)**indco(3, ico)
                     END DO
                     values(ipoint, isgf) = values(ipoint, isgf) + polynomial*radial(l)
                  END DO
               END DO
            END DO
         END DO
      END DO
      !$OMP END DO
      DEALLOCATE (radial, long_radial, radial_work%jet, radial_work%gaussian, &
                  radial_work%derivative, radial_work%bracket)
      !$OMP END PARALLEL

   END SUBROUTINE contract_potentials

! **************************************************************************************************
!> \brief Contract selected auxiliary potentials with BLAS after shared radial work.
!> The potential columns are selected AUTO_RI functions throughout this routine.
!> Each thread reuses radial work arrays over all points and Gaussian primitives.
!> \param values ...
!> \param points ...
!> \param basis ...
!> \param center ...
!> \param cutoff ...
!> \param contracted ...
!> \param attenuation ...
! **************************************************************************************************
   SUBROUTINE contract_selected_potentials(values, points, basis, center, cutoff, contracted, attenuation)
      REAL(dp), CONTIGUOUS, INTENT(OUT) :: values(:, :)
      REAL(dp), INTENT(IN)              :: points(:, :)
      TYPE(gto_basis_set_type), POINTER :: basis
      REAL(dp), INTENT(IN)              :: center(3), cutoff
      TYPE(radial_interpolation_type), INTENT(IN), &
         TARGET                                          :: contracted
      REAL(dp), INTENT(IN), OPTIONAL    :: attenuation

      INTEGER                           :: handle, iangular, ico, ig, imesh, in, ipgf, irow, iset, isgf, ishell, l, lmax, m, &
                                           nchannels, ng, np, nrows, nwork
      INTEGER, ALLOCATABLE              :: angular_count(:, :, :), angular_index(:), angular_term(:, :, :, :), &
                                           mesh_channel(:), minimum_l(:)
      LOGICAL                           :: factorized, in_table, parallel_points, prepared, short, tabulated, use_mesh
      LOGICAL, ALLOCATABLE              :: angular_used(:, :)
      REAL(dp)                          :: beta_ratio, exponent, mesh_delta, mesh_position, minimum_radius2, omega, &
                                           radial_sum, radius, radius2, relative(3), spherical_value, x
      REAL(dp), ALLOCATABLE             :: angular_values(:, :, :), coordinate_power(:, :), long_radial(:), monomial(:), &
                                           radial(:), radial_channels(:, :, :), radial_mesh(:), work(:, :)
      REAL(dp), CONTIGUOUS, POINTER     :: coefficient_matrix(:, :)
      REAL(dp), POINTER                 :: series(:, :, :), short_series(:, :, :)
      TYPE(radial_work_type)            :: radial_work

      CALL timeset('d_lP_potential_prepare', handle)

      ng = SIZE(points, 2)
      np = SIZE(values, 2)
      lmax = MAXVAL(basis%lmax)
      nrows = SIZE(contracted%coefficients, 1)
      factorized = contracted%spherical_factorized
      nwork = nrows
      IF (factorized) nwork = basis%nsgf
      ALLOCATE (angular_used(0:lmax, basis%nset), source=.FALSE.)
      ALLOCATE (minimum_l(basis%nset))
      DO iset = 1, basis%nset
         minimum_l(iset) = MINVAL(basis%l(1:basis%nshell(iset), iset))
         DO ishell = 1, basis%nshell(iset)
            angular_used(basis%l(ishell, iset), iset) = .TRUE.
         END DO
      END DO
      omega = 0.0_dp
      IF (PRESENT(attenuation)) omega = attenuation
      short = omega > 0.0_dp
      ALLOCATE (work(nwork, ng))
      prepared = ALLOCATED(contracted%series) .AND. ALLOCATED(contracted%short_series)
      prepared = prepared .AND. ABS(contracted%cutoff - cutoff) <= 0.0_dp .AND. &
                 ABS(contracted%attenuation - omega) <= 0.0_dp
      IF (prepared) THEN
         series => contracted%series
         short_series => contracted%short_series
      ELSE
         ALLOCATE (series(0:lmax + 24, MAXVAL(basis%npgf), basis%nset), &
                   short_series(0:lmax + 24, MAXVAL(basis%npgf), basis%nset))
         minimum_radius2 = MINVAL(SUM((points - SPREAD(center, 2, ng))**2, DIM=1))
         DO iset = 1, basis%nset
            DO ipgf = 1, basis%npgf(iset)
               exponent = basis%zet(ipgf, iset)
               IF (short) THEN
                  CALL origin_series(exponent, 0.0_dp, series(:, ipgf, iset))
                  beta_ratio = omega**2/(exponent + omega**2)
                  DO in = 0, UBOUND(series, 1)
                     short_series(in, ipgf, iset) = series(in, ipgf, iset)* &
                                                    (1.0_dp - beta_ratio**(REAL(in, dp) + 0.5_dp))
                  END DO
               ELSE IF (exponent*minimum_radius2 < 0.5_dp) THEN
                  CALL origin_series(exponent, cutoff, series(:, ipgf, iset))
               ELSE
                  series(:, ipgf, iset) = 0.0_dp
               END IF
            END DO
         END DO
      END IF
      ! Primitive/angular row labels are invariant over the integration points.
      ! Evaluate each tabulated radial channel once, then apply its Cartesian monomial.
      nchannels = 0
      ALLOCATE (mesh_channel(nrows), angular_index(nrows))
      IF (prepared) THEN
         IF (ALLOCATED(contracted%mesh_moments)) THEN
            nchannels = SIZE(contracted%mesh_moments, 2)
            irow = 0
            DO iset = 1, basis%nset
               DO ipgf = 1, basis%npgf(iset)
                  DO l = 0, basis%lmax(iset)
                     IF (.NOT. angular_used(l, iset)) CYCLE
                     DO ico = ncoset(l - 1) + 1, ncoset(l)
                        irow = irow + 1
                        mesh_channel(irow) = contracted%mesh_first(ipgf, iset) + l - minimum_l(iset)
                        angular_index(irow) = ico
                     END DO
                  END DO
               END DO
            END DO
            CPASSERT(irow == nrows)
         END IF
      END IF
      ! Preserve the order of the nonzero polynomial terms; omit only exact zeros.
      IF (factorized) THEN
         ALLOCATE (angular_count(MAXVAL(nso(0:lmax)), 0:lmax, basis%nset), source=0)
         ALLOCATE (angular_term(SIZE(contracted%spherical_angular, 1), MAXVAL(nso(0:lmax)), 0:lmax, basis%nset))
         DO iset = 1, basis%nset
            DO l = 0, basis%lmax(iset)
               IF (.NOT. angular_used(l, iset)) CYCLE
               DO m = 1, nso(l)
                  DO ico = ncoset(l - 1) + 1, ncoset(l)
                     IF (ABS(contracted%spherical_angular(ico - ncoset(l - 1), m, l, iset)) <= 0.0_dp) CYCLE
                     angular_count(m, l, iset) = angular_count(m, l, iset) + 1
                     angular_term(angular_count(m, l, iset), m, l, iset) = ico
                  END DO
               END DO
            END DO
         END DO
      END IF
      tabulated = prepared .AND. ALLOCATED(contracted%short_moments)

      CALL timestop(handle)

      CALL timeset('d_lP_potential_radial', handle)

      parallel_points = .TRUE.
!$    parallel_points = .NOT. omp_in_parallel()
      !$OMP PARALLEL DEFAULT(NONE) IF(parallel_points) &
      !$OMP SHARED(ng, points, center, basis, cutoff, series, short_series, short, omega, &
      !$OMP work, indco, ncoset, lmax, angular_used, minimum_l, contracted, tabulated, prepared, &
      !$OMP nchannels, nrows, nwork, factorized, mesh_channel, angular_index, nso, angular_count, angular_term) &
      !$OMP PRIVATE(ig, ico, relative, radius, radius2, iset, ipgf, radial, monomial, l, irow, radial_work, long_radial, &
      !$OMP x, in_table, coordinate_power, imesh, mesh_delta, mesh_position, use_mesh, radial_mesh, &
      !$OMP radial_channels, angular_values, isgf, m, radial_sum, spherical_value, iangular)
      ALLOCATE (radial(0:lmax), long_radial(0:lmax), radial_work%jet(0:lmax), &
                radial_work%gaussian(0:lmax), radial_work%derivative(0:lmax), radial_work%bracket(0:lmax))
      ALLOCATE (monomial(ncoset(lmax)), coordinate_power(3, 0:lmax), radial_mesh(nchannels))
      IF (factorized) ALLOCATE (radial_channels(0:lmax, MAXVAL(basis%npgf), basis%nset), &
                                angular_values(MAXVAL(nso(0:lmax)), 0:lmax, basis%nset))
      !$OMP DO SCHEDULE(STATIC)
      DO ig = 1, ng
         relative = points(:, ig) - center
         radius2 = SUM(relative**2)
         coordinate_power(:, 0) = 1.0_dp
         DO l = 1, lmax
            coordinate_power(:, l) = coordinate_power(:, l - 1)*relative
         END DO
         DO ico = 1, SIZE(monomial)
            monomial(ico) = coordinate_power(1, indco(1, ico))* &
                            coordinate_power(2, indco(2, ico))*coordinate_power(3, indco(3, ico))
         END DO
         use_mesh = .FALSE.
         mesh_delta = 0.0_dp
         imesh = 0
         IF (prepared) THEN
            IF (ALLOCATED(contracted%mesh_moments)) THEN
               mesh_position = radius2/contracted%mesh_spacing - contracted%mesh_offset
               IF (mesh_position >= -0.5_dp .AND. mesh_position <= REAL(UBOUND(contracted%mesh_moments, 3), dp) + 0.5_dp) THEN
                  imesh = NINT(mesh_position)
                  IF (imesh >= 0 .AND. imesh <= UBOUND(contracted%mesh_moments, 3)) THEN
                     mesh_delta = radius2 - contracted%mesh_spacing*(REAL(imesh, dp) + contracted%mesh_offset)
                     use_mesh = ABS(mesh_delta) <= 64.0_dp*EPSILON(1.0_dp)*MAX(1.0_dp, radius2)
                     use_mesh = use_mesh .AND. contracted%mesh_exponent*ABS(mesh_delta) <= 1.0E-8_dp
                  END IF
               END IF
            END IF
         END IF
         IF (use_mesh) THEN
            IF (factorized .AND. ALLOCATED(contracted%spherical_mesh_first)) THEN
               ! The radial shell contraction was formed once for each squared-radius node.
            ELSE
               radial_mesh(:) = contracted%mesh_moments(1, :, imesh) + &
                                mesh_delta*contracted%mesh_moments(2, :, imesh)
            END IF
            IF (factorized .AND. .NOT. ALLOCATED(contracted%spherical_mesh_first)) THEN
               DO iset = 1, basis%nset
                  DO ipgf = 1, basis%npgf(iset)
                     DO l = 0, basis%lmax(iset)
                        IF (.NOT. angular_used(l, iset)) CYCLE
                        radial_channels(l, ipgf, iset) = &
                           radial_mesh(contracted%mesh_first(ipgf, iset) + l - minimum_l(iset))
                     END DO
                  END DO
               END DO
            ELSE IF (.NOT. factorized) THEN
               DO irow = 1, nrows
                  work(irow, ig) = monomial(angular_index(irow))*radial_mesh(mesh_channel(irow))
               END DO
               CYCLE
            END IF
         ELSE
            radius = SQRT(radius2)
            irow = 0
            DO iset = 1, basis%nset
               DO ipgf = 1, basis%npgf(iset)
                  x = basis%zet(ipgf, iset)*radius2
                  in_table = .FALSE.
                  IF (tabulated) in_table = x <= contracted%short_abscissae(UBOUND(contracted%short_abscissae, 1))
                  IF (in_table) THEN
        CALL evaluate_short_radial_table(x, contracted%short_abscissae, contracted%short_moments(minimum_l(iset):, :, ipgf, iset), &
                                                      radial(minimum_l(iset):basis%lmax(iset)))
                  ELSE IF (short) THEN
                     CALL radial_potential_short(radius, basis%zet(ipgf, iset), omega, &
                                                 series(:, ipgf, iset), short_series(:, ipgf, iset), &
                                                 radial(0:basis%lmax(iset)), radial_work, long_radial)
                  ELSE
                     CALL radial_potential(radius, basis%zet(ipgf, iset), cutoff, &
                                           series(:, ipgf, iset), radial(0:basis%lmax(iset)), radial_work)
                  END IF
                  IF (factorized) THEN
                     DO l = 0, basis%lmax(iset)
                        IF (angular_used(l, iset)) radial_channels(l, ipgf, iset) = radial(l)
                     END DO
                  ELSE
                     DO l = 0, basis%lmax(iset)
                        IF (.NOT. angular_used(l, iset)) CYCLE
                        DO ico = ncoset(l - 1) + 1, ncoset(l)
                           irow = irow + 1
                           work(irow, ig) = monomial(ico)*radial(l)
                        END DO
                     END DO
                  END IF
               END DO
            END DO
         END IF
         IF (factorized) THEN
            work(:, ig) = 0.0_dp
            DO iset = 1, basis%nset
               DO l = 0, basis%lmax(iset)
                  IF (.NOT. angular_used(l, iset)) CYCLE
                  DO m = 1, nso(l)
                     spherical_value = 0.0_dp
                     DO iangular = 1, angular_count(m, l, iset)
                        ico = angular_term(iangular, m, l, iset)
                        spherical_value = spherical_value + monomial(ico)* &
                                          contracted%spherical_angular(ico - ncoset(l - 1), m, l, iset)
                     END DO
                     angular_values(m, l, iset) = spherical_value
                  END DO
               END DO
               DO ishell = 1, basis%nshell(iset)
                  l = basis%l(ishell, iset)
                  IF (use_mesh .AND. ALLOCATED(contracted%spherical_mesh_first)) THEN
                     irow = contracted%spherical_mesh_first(ishell, iset)
                     radial_sum = contracted%mesh_moments(1, irow, imesh) + &
                                  mesh_delta*contracted%mesh_moments(2, irow, imesh)
                  ELSE
                     radial_sum = SUM(radial_channels(l, 1:basis%npgf(iset), iset)* &
                                      contracted%spherical_primitive(1:basis%npgf(iset), ishell, iset))
                  END IF
                  DO m = 1, nso(l)
                     isgf = basis%first_sgf(ishell, iset) + m - 1
                     work(isgf, ig) = radial_sum*angular_values(m, l, iset)
                  END DO
               END DO
            END DO
         END IF
      END DO
      !$OMP END DO
      DEALLOCATE (radial, long_radial, radial_work%jet, radial_work%gaussian, &
                  radial_work%derivative, radial_work%bracket)
      DEALLOCATE (monomial, coordinate_power, radial_mesh)
      IF (factorized) DEALLOCATE (radial_channels, angular_values)
      !$OMP END PARALLEL

      CALL timestop(handle)

      CALL timeset('d_lP_potential_RI_transform', handle)

      ! The caller supplies the full leading dimension; write only the active point rows.
      IF (factorized) THEN
         coefficient_matrix => contracted%spherical_selection
      ELSE
         coefficient_matrix => contracted%coefficients
      END IF
      CALL dgemm('T', 'N', ng, np, nwork, 1.0_dp, work, nwork, &
                 coefficient_matrix, nwork, 0.0_dp, values, SIZE(values, 1))
      IF (factorized) DEALLOCATE (angular_count, angular_term)
      DEALLOCATE (angular_used, minimum_l, mesh_channel, angular_index)
      IF (.NOT. prepared) DEALLOCATE (series, short_series)

      CALL timestop(handle)

   END SUBROUTINE contract_selected_potentials

! **************************************************************************************************
!> \brief Gaussian potential for erfc(omega*r)/r via two full-Coulomb potentials.
!> With beta=a*omega²/(a+omega²), the long-range part equals the beta potential
!> multiplied in angular channel l by (beta/a)^(l+3/2). The origin coefficients
!> combine both terms before evaluation to avoid subtraction at r=0.
!> \param radius ...
!> \param exponent ...
!> \param omega ...
!> \param coefficients ...
!> \param short_coefficients ...
!> \param radial ...
!> \param radial_work ...
!> \param long_radial ...
! **************************************************************************************************
   SUBROUTINE radial_potential_short(radius, exponent, omega, coefficients, short_coefficients, radial, radial_work, long_radial)
      REAL(dp), INTENT(IN)                  :: radius, exponent, omega, coefficients(0:), short_coefficients(0:)
      REAL(dp), INTENT(OUT)                 :: radial(0:)
      TYPE(radial_work_type), INTENT(INOUT) :: radial_work
      REAL(dp), INTENT(INOUT)               :: long_radial(0:)

      INTEGER                               :: il
      REAL(dp)                              :: beta, beta_ratio, factor, lower_exp, ratio_power, root_x, upper_exp, x

      IF (exponent*radius**2 < 0.5_dp) THEN
         CALL radial_potential(radius, exponent, 0.0_dp, short_coefficients, radial, radial_work)
         RETURN
      END IF
      beta_ratio = omega**2/(exponent + omega**2)
      beta = exponent*beta_ratio
      x = exponent*radius**2
      IF (x >= REAL(2*UBOUND(radial, 1) + 8, dp)) THEN
         ! H_l(x,b)=integral_{sqrt(b)}^1 t^(2l) exp(-x t^2) dt; radial_l=(2 pi/a) H_l.
         ! Its upward recurrence is stable here and avoids subtracting two Coulomb derivatives.
         root_x = SQRT(x)
         factor = 2.0_dp*pi/exponent
         radial(0) = factor*SQRT(pi)/(2.0_dp*root_x)* &
                     (ERFC(SQRT(beta_ratio)*root_x) - ERFC(root_x))
         upper_exp = EXP(-x)
         lower_exp = EXP(-beta_ratio*x)
         ratio_power = SQRT(beta_ratio)
         DO il = 1, UBOUND(radial, 1)
            radial(il) = (REAL(2*il - 1, dp)*radial(il - 1) + &
                          factor*(ratio_power*lower_exp - upper_exp))/(2.0_dp*x)
            ratio_power = ratio_power*beta_ratio
         END DO
         RETURN
      END IF
      CALL radial_potential(radius, exponent, 0.0_dp, coefficients, radial, radial_work)
      CALL radial_potential(radius, beta, 0.0_dp, coefficients, long_radial(0:UBOUND(radial, 1)), radial_work)
      DO il = 0, UBOUND(radial, 1)
         radial(il) = radial(il) - beta_ratio**(REAL(il, dp) + 1.5_dp)*long_radial(il)
      END DO

   END SUBROUTINE radial_potential_short

! **************************************************************************************************
!> \brief Expand Eq. (2) in x=a r²: F=(2 pi/a) sum_n c_n x^n near the origin.
!> \param exponent ...
!> \param cutoff ...
!> \param coefficients ...
! **************************************************************************************************
   SUBROUTINE origin_series(exponent, cutoff, coefficients)
      REAL(dp), INTENT(IN)  :: exponent, cutoff
      REAL(dp), INTENT(OUT) :: coefficients(0:)

      INTEGER               :: ij, ik, in
      REAL(dp)              :: factorial, gamma_p, integral, term, weight, y, terms(0:UBOUND(coefficients, 1))

      y = exponent*cutoff**2
      weight = 1.0_dp
      DO ik = 0, UBOUND(terms, 1)
         gamma_p = 1.0_dp
         IF (cutoff > 0.0_dp .AND. y < 200.0_dp) THEN
            ! P(k+1,y) is a positive Poisson tail, avoiding subtraction near zero.
            term = EXP(-y)
            DO ij = 1, ik + 1
               term = term*y/REAL(ij, dp)
            END DO
            integral = term
            DO ij = ik + 2, ik + 1000
               term = term*y/REAL(ij, dp)
               integral = integral + term
               IF (term < EPSILON(1.0_dp)*integral) EXIT
            END DO
            gamma_p = integral
         END IF
         terms(ik) = weight*gamma_p
         weight = weight*2.0_dp/REAL(2*ik + 3, dp)
      END DO
      DO in = 0, UBOUND(coefficients, 1)
         coefficients(in) = 0.0_dp
         factorial = 1.0_dp
         DO ik = in, 0, -1
            coefficients(in) = coefficients(in) + factorial*terms(ik)
            factorial = -factorial/REAL(in - ik + 1, dp)
         END DO
      END DO

   END SUBROUTINE origin_series

! **************************************************************************************************
!> \brief Evaluate the radial derivatives in Eq. (1) by series or Taylor differentiation of (2).
!> \param radius ...
!> \param exponent ...
!> \param cutoff ...
!> \param coefficients ...
!> \param radial ...
!> \param radial_work ...
! **************************************************************************************************
   SUBROUTINE radial_potential(radius, exponent, cutoff, coefficients, radial, radial_work)
      REAL(dp), INTENT(IN)                  :: radius, exponent, cutoff, coefficients(0:)
      REAL(dp), INTENT(OUT)                 :: radial(0:)
      TYPE(radial_work_type), INTENT(INOUT) :: radial_work

      INTEGER                               :: ij, il, in, lmax
      REAL(dp)                              :: displacement, factor, power, prefactor, value, x

      lmax = UBOUND(radial, 1)
      x = exponent*radius**2
      IF (x < 0.5_dp) THEN
         DO il = 0, lmax
            value = 0.0_dp
            power = 1.0_dp
            DO in = il, UBOUND(coefficients, 1)
               factor = 1.0_dp
               DO ij = in - il + 1, in
                  factor = factor*REAL(ij, dp)
               END DO
               value = value + coefficients(in)*factor*power
               power = power*x
            END DO
            radial(il) = 2.0_dp*pi/exponent*(-1.0_dp)**il*value
         END DO
         RETURN
      END IF
      radial_work%bracket(0:lmax) = 0.0_dp
      DO ij = 1, 3
         displacement = radius
         factor = 2.0_dp
         IF (ij > 1) THEN
            IF (cutoff <= 0.0_dp) EXIT
            displacement = radius + REAL(2*ij - 5, dp)*cutoff
            factor = -1.0_dp
         END IF
         radial_work%gaussian(0) = EXP(-exponent*displacement**2)
         IF (lmax > 0) radial_work%gaussian(1) = -2.0_dp*exponent*displacement*radial_work%gaussian(0)
         DO in = 2, lmax
            radial_work%gaussian(in) = -2.0_dp*exponent* &
                                       (displacement*radial_work%gaussian(in - 1) + radial_work%gaussian(in - 2))/REAL(in, dp)
         END DO
         DO in = 1, lmax
            radial_work%bracket(in) = radial_work%bracket(in) + &
                                      factor*2.0_dp*SQRT(exponent/pi)*radial_work%gaussian(in - 1)/REAL(in, dp)
         END DO
      END DO
      IF (cutoff <= 0.0_dp) THEN
         radial_work%bracket(0) = 2.0_dp*ERF(SQRT(exponent)*radius)
      ELSE IF (radius > cutoff) THEN
         radial_work%bracket(0) = ERFC(SQRT(exponent)*(radius - cutoff)) + &
                                  ERFC(SQRT(exponent)*(radius + cutoff)) - 2.0_dp*ERFC(SQRT(exponent)*radius)
      ELSE
         radial_work%bracket(0) = 2.0_dp*ERF(SQRT(exponent)*radius) - &
                                  ERF(SQRT(exponent)*(radius - cutoff)) - ERF(SQRT(exponent)*(radius + cutoff))
      END IF
      prefactor = pi**1.5_dp/(2.0_dp*exponent**1.5_dp)
      radial_work%jet(0) = radial_work%bracket(0)/radius
      DO in = 1, lmax
         radial_work%jet(in) = (radial_work%bracket(in) - radial_work%jet(in - 1))/radius
      END DO
      radial(0) = prefactor*radial_work%jet(0)
      DO il = 1, lmax
         radial_work%derivative(0:lmax) = 0.0_dp
         radial_work%derivative(0) = radial_work%jet(1)/(2.0_dp*radius)
         DO in = 1, lmax - il
            radial_work%derivative(in) = (REAL(in + 1, dp)*radial_work%jet(in + 1)/2.0_dp - radial_work%derivative(in - 1))/radius
         END DO
         radial_work%jet(0:lmax) = radial_work%derivative(0:lmax)
         radial(il) = prefactor*(-1.0_dp/exponent)**il*radial_work%jet(0)
      END DO

   END SUBROUTINE radial_potential

! **************************************************************************************************
!> \brief Place radial nodes at x=0.02*i up to x=2, then at x=2*1.01**(i-100).
!> The final node retains the original table extent. For a nearest-node displacement δ,
!> the sixth-order Taylor remainder of H_j is bounded by
!> |δ|**7/7! * integral_0^1 t**(2*j+14)*exp(-(x-|δ|)*t**2) dt.
!> On the geometric part |δ| <= 0.005*x, so this bound is below 1.1e-17 for j>=0.
!> This is an absolute bound before the Gaussian prefactor 2*pi/a; roundoff is separate.
!> \param xmax ...
!> \param abscissae ...
! **************************************************************************************************
   SUBROUTINE prepare_short_radial_grid(xmax, abscissae)
      REAL(dp), INTENT(IN)               :: xmax
      REAL(dp), ALLOCATABLE, INTENT(OUT) :: abscissae(:)

      INTEGER                            :: ix, last

      CPASSERT(xmax > 0.0_dp)
      IF (xmax <= 2.0_dp) THEN
         last = CEILING(xmax/0.02_dp)
      ELSE
         last = 100 + CEILING(LOG(xmax/2.0_dp)/LOG(1.01_dp))
      END IF
      ALLOCATE (abscissae(0:last))
      DO ix = 0, last - 1
         IF (ix <= 100) THEN
            abscissae(ix) = 0.02_dp*REAL(ix, dp)
         ELSE
            abscissae(ix) = 2.0_dp*EXP(REAL(ix - 100, dp)*LOG(1.01_dp))
         END IF
      END DO
      abscissae(last) = xmax

   END SUBROUTINE prepare_short_radial_grid

! **************************************************************************************************
!> \brief Tabulate H_j(x,b)=integral_sqrt(b)^1 t^(2j)*exp(-x*t²)dt for erfc potentials.
!> A 64-point positive quadrature supplies near-field moments; a stable upward recurrence
!> supplies far-field moments for b<0.9. Narrow intervals retain quadrature to avoid cancellation.
!> Uniform mesh tables select spacing and offset; nonuniform lookup tables supply their abscissae.
!> The caller bounds storage, including one primitive preparation buffer.
!> \param a ...
!> \param omega ...
!> \param lmax ...
!> \param table ...
!> \param ninterval ...
!> \param point_spacing ...
!> \param point_offset ...
!> \param last_moment ...
!> \param abscissae ...
! **************************************************************************************************
   SUBROUTINE prepare_short_radial_table(a, omega, lmax, table, ninterval, point_spacing, point_offset, &
                                         last_moment, abscissae)
      REAL(dp), INTENT(IN)               :: a, omega
      INTEGER, INTENT(IN)                :: lmax
      REAL(dp), ALLOCATABLE, INTENT(OUT) :: table(:, :)
      INTEGER, INTENT(IN), OPTIONAL      :: ninterval
      REAL(dp), INTENT(IN), OPTIONAL     :: point_spacing, point_offset
      INTEGER, INTENT(IN), OPTIONAL      :: last_moment
      REAL(dp), INTENT(IN), OPTIONAL     :: abscissae(0:)

      INTEGER                            :: ig, ij, il, iter, ix, last, order, quadrature_last
      REAL(dp)                           :: b, derivative, dz, factor, lower, lower_exp, offset, p0, p1, p2, ratio_power, &
                                            root_x, spacing, t, t2, term, upper_exp, w, width, x, z

      CPASSERT(a > 0.0_dp .AND. omega > 0.0_dp .AND. lmax >= 0)
      last = 25600
      IF (PRESENT(ninterval)) last = ninterval
      CPASSERT(last > 0)
      spacing = 0.02_dp
      offset = 0.0_dp
      order = lmax + 6
      IF (PRESENT(point_spacing)) spacing = point_spacing
      IF (PRESENT(point_offset)) offset = point_offset
      IF (PRESENT(last_moment)) order = last_moment
      CPASSERT(spacing > 0.0_dp .AND. offset >= 0.0_dp .AND. order >= 0)
      IF (PRESENT(abscissae)) THEN
         CPASSERT(SIZE(abscissae) == last + 1)
         CPASSERT(abscissae(0) >= 0.0_dp)
         CPASSERT(ALL(abscissae(1:) > abscissae(:last - 1)))
      END IF
      ALLOCATE (table(0:order, 0:last), source=0.0_dp)
      b = omega**2/(a + omega**2)
      lower = SQRT(b)
      width = (a/(a + omega**2))/(1.0_dp + lower)
      quadrature_last = last
      IF (b < 0.9_dp) THEN
         IF (PRESENT(abscissae)) THEN
            quadrature_last = COUNT(abscissae <= REAL(2*order + 8, dp)) - 1
         ELSE
            quadrature_last = INT(MIN(REAL(last, dp), MAX(0.0_dp, REAL(2*order + 8, dp)/spacing - offset)))
         END IF
      END IF
      DO ig = 1, 64
         z = COS(pi*(REAL(ig, dp) - 0.25_dp)/64.5_dp)
         DO iter = 1, 30
            p0 = 1.0_dp
            p1 = z
            DO ij = 2, 64
               p2 = (REAL(2*ij - 1, dp)*z*p1 - REAL(ij - 1, dp)*p0)/REAL(ij, dp)
               p0 = p1
               p1 = p2
            END DO
            derivative = 64.0_dp*(z*p1 - p0)/(z*z - 1.0_dp)
            dz = p1/derivative
            z = z - dz
            IF (ABS(dz) < 2.0E-16_dp) EXIT
         END DO
         IF (iter > 30) CPABORT('DLP radial quadrature failed')
         w = 2.0_dp/((1.0_dp - z*z)*derivative**2)
         t = lower + 0.5_dp*width*(1.0_dp + z)
         t2 = t*t
         DO ix = 0, quadrature_last
            x = spacing*(REAL(ix, dp) + offset)
            IF (PRESENT(abscissae)) x = abscissae(ix)
            term = 0.5_dp*width*w*EXP(-x*t2)*2.0_dp*pi/a
            DO il = 0, UBOUND(table, 1)
               table(il, ix) = table(il, ix) + term
               term = term*t2
            END DO
         END DO
      END DO
      factor = 2.0_dp*pi/a
      DO ix = quadrature_last + 1, last
         x = spacing*(REAL(ix, dp) + offset)
         IF (PRESENT(abscissae)) x = abscissae(ix)
         root_x = SQRT(x)
         table(0, ix) = factor*SQRT(pi)/(2.0_dp*root_x)*(ERFC(lower*root_x) - ERFC(root_x))
         lower_exp = EXP(-b*x)
         upper_exp = EXP(-x)
         ratio_power = lower
         DO il = 1, UBOUND(table, 1)
            table(il, ix) = (REAL(2*il - 1, dp)*table(il - 1, ix) + &
                             factor*(ratio_power*lower_exp - upper_exp))/(2.0_dp*x)
            ratio_power = ratio_power*b
         END DO
      END DO

   END SUBROUTINE prepare_short_radial_table

! **************************************************************************************************
!> \brief Evaluate H_l at the nearest table point with its first six derivatives.
!> Since d^k H_l/dx^k=(-1)^k H_(l+k), the stored higher moments provide the derivatives.
!> The nonuniform-node error bound is given in prepare_short_radial_grid.
!> \param x ...
!> \param abscissae ...
!> \param table ...
!> \param radial ...
! **************************************************************************************************
   SUBROUTINE evaluate_short_radial_table(x, abscissae, table, radial)
      REAL(dp), INTENT(IN)  :: x, abscissae(0:), table(0:, 0:)
      REAL(dp), INTENT(OUT) :: radial(0:)

      INTEGER               :: ix, il, ik, ileft, iright, imiddle
      REAL(dp)              :: delta, value
      REAL(dp), PARAMETER   :: inverse(6) = [1.0_dp, 0.5_dp, 1.0_dp/3.0_dp, 0.25_dp, 0.2_dp, 1.0_dp/6.0_dp]
      ileft = 0
      iright = UBOUND(abscissae, 1)
      DO WHILE (iright - ileft > 1)
         imiddle = (ileft + iright)/2
         IF (x < abscissae(imiddle)) THEN
            iright = imiddle
         ELSE
            ileft = imiddle
         END IF
      END DO
      ix = ileft
      IF (abscissae(iright) - x < x - abscissae(ileft)) ix = iright
      delta = abscissae(ix) - x
      DO il = 0, UBOUND(radial, 1)
         value = table(il + 6, ix)
         DO ik = 5, 0, -1
            value = table(il + ik, ix) + delta*value*inverse(ik + 1)
         END DO
         radial(il) = value
      END DO

   END SUBROUTINE evaluate_short_radial_table

! **************************************************************************************************
!> \brief Tabulate H_l(r²) only when squared mesh distances share a one-dimensional lattice.
!> With M=SᵀS and d=r_origin-r_atom, r²=nᵀMn+2nᵀSᵀd+d². Integer coefficients
!> in units q imply r²=q(i+offset). Whole-cell translations preserve this mesh.
!> H_l(s+δ)=H_l(s)-a δ H_(l+1)(s)+O((a δ)²); each lookup checks the residual.
!> Tables cover up to 40 bohr within the storage budget; other points use the generic lookup.
!> Account for all stored channels, their index map and the largest temporary moment table.
!> \param basis ...
!> \param contracted ...
!> \param step ...
!> \param displacement ...
!> \param table_bytes ...
! **************************************************************************************************
   SUBROUTINE prepare_mesh_radial_moments(basis, contracted, step, displacement, table_bytes)
      TYPE(gto_basis_set_type), POINTER              :: basis
      TYPE(radial_interpolation_type), INTENT(INOUT) :: contracted
      REAL(dp), INTENT(IN)                           :: step(3, 3), displacement(3)
      REAL(dp), INTENT(INOUT)                        :: table_bytes

      INTEGER                                        :: idir, ifirst, il, ipgf, iset, ishell, jdir, last, max_l, &
                                                        max_primitive, minimum_l, nchannels, nshell_total, nstored, &
                                                        shell_index
      LOGICAL                                        :: direct_shells
      REAL(dp)                                       :: bytes_per_node, index_bytes, linear(3), metric(3, 3), offset, ratio, &
                                                        required_bytes, spacing
      REAL(dp), ALLOCATABLE                          :: moments(:, :)

      IF (basis%nset < 1) RETURN
      IF (ANY(basis%nshell(1:basis%nset) < 1)) RETURN
      IF (ANY(basis%npgf(1:basis%nset) < 1)) RETURN
      IF (contracted%attenuation <= 0.0_dp .OR. table_bytes <= 0.0_dp) RETURN
      metric = MATMUL(TRANSPOSE(step), step)
      spacing = MIN(metric(1, 1), metric(2, 2), metric(3, 3))
      IF (spacing <= TINY(1.0_dp)) RETURN
      DO idir = 1, 3
         DO jdir = idir, 3
            ratio = metric(idir, jdir)/spacing
            IF (idir /= jdir) ratio = 2.0_dp*ratio
            IF (ABS(ratio - ANINT(ratio)) > 1.0E-10_dp) RETURN
         END DO
      END DO
      linear = 2.0_dp*MATMUL(TRANSPOSE(step), displacement)/spacing
      IF (ANY(ABS(linear - ANINT(linear)) > 1.0E-10_dp)) RETURN
      offset = MODULO(SUM(displacement**2)/spacing, 1.0_dp)
      IF (1600.0_dp/spacing > REAL(HUGE(last) - 1, dp)) RETURN
      last = CEILING(1600.0_dp/spacing)
      IF (last < 1) RETURN
      nchannels = 0
      DO iset = 1, basis%nset
         minimum_l = MINVAL(basis%l(1:basis%nshell(iset), iset))
         nchannels = nchannels + basis%npgf(iset)*(basis%lmax(iset) - minimum_l + 1)
      END DO
      nshell_total = SUM(basis%nshell(1:basis%nset))
      max_l = MAXVAL(basis%lmax(1:basis%nset))
      max_primitive = MAXVAL(basis%npgf(1:basis%nset))
      direct_shells = contracted%spherical_factorized .AND. nshell_total <= nchannels
      nstored = nchannels
      IF (direct_shells) nstored = nshell_total
      bytes_per_node = 8.0_dp*(2.0_dp*REAL(nstored, dp) + REAL(max_l + 2, dp))
      index_bytes = REAL(STORAGE_SIZE(0)/8, dp)*REAL(max_primitive, dp)*REAL(basis%nset, dp)
      IF (direct_shells) THEN
         index_bytes = index_bytes + &
                       REAL(STORAGE_SIZE(0)/8, dp)*REAL(MAXVAL(basis%nshell)*basis%nset, dp)
      END IF
      ! A smaller table still accelerates nearby points; its upper bound selects the generic fallback.
      last = MIN(last, INT(MIN(REAL(HUGE(last), dp), &
                               MAX(0.0_dp, (table_bytes - index_bytes)/bytes_per_node))) - 1)
      IF (last < 1) RETURN
      required_bytes = bytes_per_node*REAL(last + 1, dp) + index_bytes
      IF (required_bytes > table_bytes) RETURN
      ALLOCATE (contracted%mesh_moments(2, nstored, 0:last))
      ALLOCATE (contracted%mesh_first(max_primitive, basis%nset), source=0)
      contracted%mesh_spacing = spacing
      contracted%mesh_offset = offset
      IF (direct_shells) THEN
         ALLOCATE (contracted%spherical_mesh_first(MAXVAL(basis%nshell), basis%nset), source=0)
         shell_index = 0
         DO iset = 1, basis%nset
            DO ishell = 1, basis%nshell(iset)
               shell_index = shell_index + 1
               contracted%spherical_mesh_first(ishell, iset) = shell_index
            END DO
         END DO
         contracted%mesh_moments = 0.0_dp
      END IF
      ifirst = 1
      DO iset = 1, basis%nset
         minimum_l = MINVAL(basis%l(1:basis%nshell(iset), iset))
         DO ipgf = 1, basis%npgf(iset)
            contracted%mesh_exponent = MAX(contracted%mesh_exponent, basis%zet(ipgf, iset))
            contracted%mesh_first(ipgf, iset) = ifirst
            CALL prepare_short_radial_table(basis%zet(ipgf, iset), contracted%attenuation, &
                                            basis%lmax(iset), moments, last, &
                                            basis%zet(ipgf, iset)*spacing, offset, basis%lmax(iset) + 1)
            IF (direct_shells) THEN
               ! Accumulate primitive contributions in their original order for each shell.
               ! The larger primitive-channel table is never materialized.
               DO ishell = 1, basis%nshell(iset)
                  il = basis%l(ishell, iset)
                  shell_index = contracted%spherical_mesh_first(ishell, iset)
                  contracted%mesh_moments(1, shell_index, :) = &
                     contracted%mesh_moments(1, shell_index, :) + &
                     contracted%spherical_primitive(ipgf, ishell, iset)*moments(il, :)
                  contracted%mesh_moments(2, shell_index, :) = &
                     contracted%mesh_moments(2, shell_index, :) + &
                     contracted%spherical_primitive(ipgf, ishell, iset)*(-basis%zet(ipgf, iset)*moments(il + 1, :))
               END DO
               ifirst = ifirst + basis%lmax(iset) - minimum_l + 1
            ELSE
               DO il = minimum_l, basis%lmax(iset)
                  contracted%mesh_moments(1, ifirst, :) = moments(il, :)
                  contracted%mesh_moments(2, ifirst, :) = -basis%zet(ipgf, iset)*moments(il + 1, :)
                  ifirst = ifirst + 1
               END DO
            END IF
            DEALLOCATE (moments)
         END DO
      END DO
      table_bytes = table_bytes - required_bytes

   END SUBROUTINE prepare_mesh_radial_moments

END MODULE ri_rs_kp_integrals
