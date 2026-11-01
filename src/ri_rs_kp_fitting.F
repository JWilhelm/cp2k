!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief AO sampling, D_llp(R), and the periodic RI-RS fit Z_lP(q).
!> \author Jan Wilhelm (with help of ChatGPT, GPT-6 Sol and GPT-6 Astra)
! **************************************************************************************************
MODULE ri_rs_kp_fitting
   USE basis_set_types,                 ONLY: gto_basis_set_type
   USE cell_types,                      ONLY: cell_type,&
                                              pbc
   USE input_constants,                 ONLY: do_potential_short,&
                                              do_potential_truncated
   USE kinds,                           ONLY: dp,&
                                              int_8
   USE machine,                         ONLY: m_walltime
   USE mathconstants,                   ONLY: pi,&
                                              twopi,&
                                              z_one,&
                                              z_zero
   USE message_passing,                 ONLY: mp_comm_split_type_shared,&
                                              mp_comm_type,&
                                              mp_mem_avail_per_rank_GB,&
                                              mp_print_mem_per_rank,&
                                              mp_request_type,&
                                              mp_waitall
   USE particle_types,                  ONLY: particle_type
   USE physcon,                         ONLY: angstrom
   USE post_scf_bandstructure_types,    ONLY: post_scf_bandstructure_type
   USE ri_rs_kp_integrals,              ONLY: compute_d_lP_R,&
                                              compute_d_lP_cutoffs,&
                                              get_local_l_range,&
                                              write_ri_rs_progress
   USE ri_rs_utils,                     ONLY: evaluate_ao_basis_on_points
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'ri_rs_kp_fitting'

   PUBLIC :: compute_Z_lP_q, &
             evaluate_mu_rl_R_values, &
             lattice_phase

CONTAINS

! **************************************************************************************************
!> \brief Build explicit-cell AO values and periodic RI-RS coefficients Z_lP(q).
!> \param bs_env Band-structure environment.
!> \param grid_points Reference-cell RI-RS points r_l.
!> \param ao_values AO values mu(r_l - R), indexed (l,mu,R).
!> \param Z_lP_q Gaussian RI-RS coefficients on the shifted SCF mesh.
!> \param Z_lP_zero ...
! **************************************************************************************************
   SUBROUTINE compute_Z_lP_q(bs_env, grid_points, ao_values, &
                             Z_lP_q, Z_lP_zero)
      TYPE(post_scf_bandstructure_type), POINTER   :: bs_env
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)   :: grid_points
      REAL(dp), ALLOCATABLE, INTENT(INOUT), TARGET :: ao_values(:, :, :)
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)   :: Z_lP_q(:, :, :), Z_lP_zero(:, :, :)

      CHARACTER(LEN=*), PARAMETER                  :: routineN = 'compute_Z_lP_q'

      COMPLEX(KIND=dp), ALLOCATABLE                :: charge_lq(:, :)
      INTEGER                                      :: first_l, handle, last_l, n_D_cells, n_d_lP_cells, n_grid
      INTEGER, ALLOCATABLE                         :: cells_R_D_llp(:, :), cells_R_d_lP(:, :)
      REAL(KIND=dp)                                :: elapsed, start_time
      REAL(KIND=dp), ALLOCATABLE                   :: D_llp_R(:, :, :), d_lP_R(:, :, :), qpoints_head(:, :), qpoints_scf(:, :)
      REAL(KIND=dp), POINTER                       :: phi_lmu_R(:, :, :)

      CALL timeset(routineN, handle)

      start_time = m_walltime()
      IF (bs_env%unit_nr > 0) WRITE (bs_env%unit_nr, '(T2,A)') "RI-RS: Compute Z_lP(k); required: D_ll'^R and d_lP^R"

      IF (bs_env%ri_rs%tikhonov < 0.0_dp) THEN
         CPABORT("RI_RS%TIKHONOV must not be negative")
      END IF

      n_grid = SIZE(grid_points, 2)
      CPASSERT(n_grid > 0)
      CPASSERT(bs_env%nkp_scf_desymm > 0)

      CALL evaluate_mu_rl_R_values(bs_env, grid_points, ao_values)
      phi_lmu_R => ao_values
      CALL build_cells_R_for_d_lP_R(bs_env, grid_points, cells_R_d_lP)
      CALL build_cells_R_for_D_llp(bs_env, phi_lmu_R, cells_R_D_llp)
      n_d_lP_cells = SIZE(cells_R_d_lP, 2)
      n_D_cells = SIZE(cells_R_D_llp, 2)
      CPASSERT(n_d_lP_cells > 0)
      CPASSERT(n_D_cells > 0)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') 'RI-RS d_lP^R cells R:', n_d_lP_cells
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') "RI-RS D_ll'^R cells R:", n_D_cells
         FLUSH (bs_env%unit_nr)
      END IF

      ! Each rank stores a rectangular (l,l') block for every translation A.
      first_l = (n_grid*bs_env%para_env%mepos)/bs_env%para_env%num_pe + 1
      last_l = (n_grid*(bs_env%para_env%mepos + 1))/bs_env%para_env%num_pe
      ALLOCATE (D_llp_R(last_l - first_l + 1, n_grid, n_D_cells))
      CALL compute_D_llp(bs_env, cells_R_D_llp, phi_lmu_R, first_l, last_l, D_llp_R)
      CALL mp_print_mem_per_rank(bs_env%para_env, bs_env%unit_nr, "After distribution of D_ll'^R:")
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A)') 'RI-RS three-center projection d_lP^R: starting'
         FLUSH (bs_env%unit_nr)
      END IF
      ALLOCATE (qpoints_scf(3, bs_env%nkp_scf_desymm))
      qpoints_scf(:, :) = bs_env%kpoints_chi_eps_W%xkp(:, 1:bs_env%nkp_scf_desymm)
      ALLOCATE (qpoints_head(3, 1), SOURCE=0.0_dp)
      CALL compute_d_lP_R(bs_env, grid_points, cells_R_d_lP, d_lP_R, qpoints_head, charge_lq)
      CALL mp_print_mem_per_rank(bs_env%para_env, bs_env%unit_nr, 'After RI-RS three-center projection d_lP^R')

      CALL solve_for_Z_lP_q(bs_env, cells_R_D_llp, cells_R_d_lP, D_llp_R, d_lP_R, first_l, last_l, qpoints_scf, &
                            Z_lP_q, distributed=.TRUE.)
      CALL solve_for_Z_lP_q(bs_env, cells_R_D_llp, cells_R_d_lP, D_llp_R, d_lP_R, first_l, last_l, &
                            qpoints_head, Z_lP_zero, distributed=.FALSE., charge_lq=charge_lq)

      DEALLOCATE (cells_R_d_lP, cells_R_D_llp, D_llp_R, qpoints_scf, d_lP_R)
      DEALLOCATE (qpoints_head)

      elapsed = m_walltime() - start_time
      CALL bs_env%para_env%max(elapsed)
      IF (bs_env%unit_nr > 0) WRITE (bs_env%unit_nr, '(T2,A,T69,F10.1,A)') &
         'Completed computing Z_lP(q), Exec. time:', elapsed, ' s'
      IF (bs_env%unit_nr > 0) FLUSH (bs_env%unit_nr)

      CALL timestop(handle)

   END SUBROUTINE compute_Z_lP_q

! **************************************************************************************************
!> \brief Evaluate mu(r_l - R) for every AO mu, RI-RS point r_l and retained cell R.
!> \param bs_env ...
!> \param grid_points ...
!> \param ao_values AO values mu(r_l - R), indexed (l,mu,R).
! **************************************************************************************************
   SUBROUTINE evaluate_mu_rl_R_values(bs_env, grid_points, ao_values)
      TYPE(post_scf_bandstructure_type), POINTER   :: bs_env
      REAL(KIND=dp), INTENT(IN)                    :: grid_points(:, :)
      REAL(dp), ALLOCATABLE, INTENT(INOUT), TARGET :: ao_values(:, :, :)

      CHARACTER(LEN=*), PARAMETER                  :: routineN = 'evaluate_mu_rl_R_values'

      INTEGER                                      :: first_ao, handle, i_atom, i_cell, i_kind, last_ao, ready
      REAL(dp)                                     :: available_GB, required_bytes
      REAL(KIND=dp)                                :: cutoff_squared, source_position(3)
      TYPE(cell_type), POINTER                     :: cell
      TYPE(gto_basis_set_type), POINTER            :: basis
      TYPE(mp_comm_type)                           :: node_comm
      TYPE(particle_type), DIMENSION(:), POINTER   :: particle_set

      CALL timeset(routineN, handle)

      CPASSERT(.NOT. ALLOCATED(ao_values))
      required_bytes = 8.0_dp*REAL(SIZE(grid_points, 2), dp)*REAL(bs_env%n_ao, dp)* &
                       REAL(SIZE(bs_env%index_to_cell_3c, 2), dp)
      CALL mp_mem_avail_per_rank_GB(bs_env%para_env, available_GB)
      ready = MERGE(1, 0, required_bytes <= 0.5_dp*MAX(0.0_dp, available_GB)*1.0E9_dp)
      CALL bs_env%para_env%min(ready)
      IF (ready == 0) CPABORT('Insufficient memory for per-rank periodic AO grid')
      ALLOCATE (ao_values(SIZE(grid_points, 2), bs_env%n_ao, SIZE(bs_env%index_to_cell_3c, 2)))
      CALL node_comm%from_split_type(bs_env%para_env, mp_comm_split_type_shared, key=bs_env%para_env%mepos)
      IF (node_comm%mepos == 0) THEN
         cell => bs_env%ri_rs%cell
         particle_set => bs_env%ri_rs%particle_set
         ao_values = 0.0_dp

         DO i_cell = 1, SIZE(bs_env%index_to_cell_3c, 2)
            DO i_atom = 1, bs_env%n_atom
               i_kind = particle_set(i_atom)%atomic_kind%kind_number
               basis => bs_env%basis_set_AO(i_kind)%gto_basis_set
               first_ao = bs_env%i_ao_start_from_atom(i_atom)
               last_ao = bs_env%i_ao_end_from_atom(i_atom)
               source_position = pbc(particle_set(i_atom)%r, cell) + &
                                 MATMUL(bs_env%hmat, REAL(bs_env%index_to_cell_3c(:, i_cell), dp))
               cutoff_squared = bs_env%ri_rs%radius_ao_per_atom(i_atom)**2
               IF (bs_env%ri_rs%cutoff_radius_ri_ao > 0.0_dp) THEN
                  cutoff_squared = bs_env%ri_rs%cutoff_radius_ri_ao**2
               END IF
               CALL evaluate_ao_basis_on_points(ao_values(:, first_ao:last_ao, i_cell), &
                                                grid_points, basis, source_position, cell, &
                                                cutoff_squared=cutoff_squared, minimum_image=.FALSE.)
            END DO
         END DO
      END IF
      DO i_cell = 1, SIZE(ao_values, 3)
         CALL node_comm%bcast(ao_values(:, :, i_cell))
      END DO
      CALL node_comm%free()

      CALL timestop(handle)

   END SUBROUTINE evaluate_mu_rl_R_values

! **************************************************************************************************
!> \brief Return exp(i 2 pi k.R).
!> \param kpoint ...
!> \param cell ...
!> \return ...
! **************************************************************************************************
   PURE FUNCTION lattice_phase(kpoint, cell) RESULT(phase)
      REAL(KIND=dp), DIMENSION(3), INTENT(IN) :: kpoint
      INTEGER, DIMENSION(3), INTENT(IN)       :: cell
      COMPLEX(KIND=dp)                        :: phase

      REAL(KIND=dp)                           :: angle

      angle = twopi*SUM(kpoint*REAL(cell, dp))
      phase = CMPLX(COS(angle), SIN(angle), KIND=dp)

   END FUNCTION lattice_phase

! **************************************************************************************************
!> \brief Solve the periodic RI-RS equations at the supplied momentum-transfer points.
!> At zero Tikhonov parameter the Hermitian D matrix is solved in the
!> Moore-Penrose sense, yielding the minimum-norm unregularized fit.
!> \param bs_env ...
!> \param cells_R_D_llp ...
!> \param cells_R_d_lP ...
!> \param D_llp_R ...
!> \param d_lP_R ...
!> \param first_l ...
!> \param last_l ...
!> \param qpoints ...
!> \param Z_lP_q ...
!> \param distributed ...
!> \param charge_lq ...
! **************************************************************************************************
   SUBROUTINE solve_for_Z_lP_q(bs_env, cells_R_D_llp, cells_R_d_lP, D_llp_R, d_lP_R, first_l, last_l, qpoints, &
                               Z_lP_q, distributed, charge_lq)
      TYPE(post_scf_bandstructure_type), POINTER    :: bs_env
      INTEGER, DIMENSION(:, :), INTENT(IN)          :: cells_R_D_llp, cells_R_d_lP
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(IN) :: D_llp_R
      REAL(KIND=dp), CONTIGUOUS, INTENT(IN)         :: d_lP_R(:, :, :)
      INTEGER, INTENT(IN)                           :: first_l, last_l
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)    :: qpoints
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)    :: Z_lP_q(:, :, :)
      LOGICAL, INTENT(IN)                           :: distributed
      COMPLEX(KIND=dp), INTENT(IN), OPTIONAL        :: charge_lq(:, :)

      CHARACTER(LEN=*), PARAMETER                   :: routineN = 'solve_for_Z_lP_q'

      COMPLEX(KIND=dp)                              :: phase
      COMPLEX(KIND=dp), ALLOCATABLE                 :: D_llp_q(:, :), D_llp_q_local(:, :), D_llp_q_packed(:), d_lP_q(:, :), &
                                                       d_lP_q_reference(:, :), rhs(:, :)
      INTEGER                                       :: first_l_rank, handle, i_A, i_batch, i_rank, info, iq, iq_batch, &
                                                       iq_local, iq_store, l, last_l_rank, lp, n_grid, n_q, n_q_local, n_RI, &
                                                       n_RI_reference, n_RI_solve, next_progress
      INTEGER(KIND=int_8)                           :: n_done, n_local
      INTEGER, ALLOCATABLE                          :: displs(:), recvcounts(:)
      LOGICAL, ALLOCATABLE                          :: active_grid_point(:), required_q(:)
      REAL(KIND=dp)                                 :: antihermitian_error, diagonal_imag_error, max_scale, metric_head, &
                                                       progress_start, qpoint(3), volume
      REAL(KIND=dp), ALLOCATABLE                    :: scale(:)

      CALL timeset(routineN, handle)

      n_grid = SIZE(D_llp_R, 2)
      n_RI = bs_env%n_RI
      n_RI_solve = n_RI
      IF (PRESENT(charge_lq)) THEN
         IF (bs_env%ri_metric%potential_type /= do_potential_truncated) THEN
            IF (bs_env%ri_metric%potential_type /= do_potential_short) THEN
               CPABORT('Unsupported RI metric')
            END IF
         END IF
         n_RI_solve = n_RI + 1
      END IF
      n_RI_reference = n_RI
      CPASSERT(SIZE(qpoints, 1) >= 3)
      n_q = SIZE(qpoints, 2)
      IF (PRESENT(charge_lq)) THEN
         CPASSERT(SIZE(charge_lq, 1) == n_grid)
         CPASSERT(SIZE(charge_lq, 2) == n_q)
      END IF
      ! Fit every Gaussian transfer point; the Gamma-correction scalar head is fitted at Gamma.
      ALLOCATE (required_q(n_q), SOURCE=.TRUE.)
      IF (distributed) THEN
         IF (bs_env%para_env%mepos == 0) THEN
            n_q_local = n_q/bs_env%para_env%num_pe
         ELSE IF (bs_env%para_env%mepos <= n_q) THEN
            n_q_local = (n_q - bs_env%para_env%mepos)/bs_env%para_env%num_pe + 1
         ELSE
            n_q_local = 0
         END IF
      ELSE
         n_q_local = n_q
      END IF
      ALLOCATE (Z_lP_q(n_grid, n_RI_solve, n_q_local))
      ALLOCATE (D_llp_q(n_grid, n_grid), D_llp_q_local(last_l - first_l + 1, n_grid), &
                D_llp_q_packed(n_grid*n_grid), recvcounts(bs_env%para_env%num_pe), &
                displs(bs_env%para_env%num_pe), &
                d_lP_q(n_grid, n_RI_solve), d_lP_q_reference(n_grid, n_RI_reference), &
                rhs(n_grid, n_RI_solve))
      DO i_rank = 0, bs_env%para_env%num_pe - 1
         first_l_rank = (n_grid*i_rank)/bs_env%para_env%num_pe + 1
         last_l_rank = (n_grid*(i_rank + 1))/bs_env%para_env%num_pe
         recvcounts(i_rank + 1) = (last_l_rank - first_l_rank + 1)*n_grid
         displs(i_rank + 1) = SUM(recvcounts(1:i_rank))
      END DO
      ALLOCATE (SCALE(n_grid), active_grid_point(n_grid))
      Z_lP_q = z_zero

      progress_start = m_walltime()
      next_progress = 1
      n_done = 0_int_8
      n_local = 0_int_8
      IF (bs_env%para_env%mepos == 0) THEN
         DO iq = MERGE(bs_env%para_env%num_pe, 1, distributed), n_q, bs_env%para_env%num_pe
            IF (required_q(iq)) n_local = n_local + 1_int_8
         END DO
      END IF
      iq_local = 0
      DO i_batch = 0, (n_q - 1)/bs_env%para_env%num_pe
         ! D_ll'(q) = sum_A exp(i q.A) D_ll'^A; gather only on the q rank.
         DO iq_batch = i_batch*bs_env%para_env%num_pe + 1, &
            MIN(n_q, (i_batch + 1)*bs_env%para_env%num_pe)
            IF (.NOT. required_q(iq_batch)) CYCLE
            D_llp_q_local = z_zero
            DO i_A = 1, SIZE(cells_R_D_llp, 2)
               phase = lattice_phase(qpoints(:, iq_batch), cells_R_D_llp(1:3, i_A))
               D_llp_q_local(:, :) = D_llp_q_local(:, :) + phase*D_llp_R(:, :, i_A)
            END DO
            i_rank = MODULO(iq_batch - 1, bs_env%para_env%num_pe)
            IF (distributed) i_rank = MODULO(iq_batch, bs_env%para_env%num_pe)
            CALL bs_env%para_env%gatherv(RESHAPE(D_llp_q_local, [SIZE(D_llp_q_local)]), &
                                         D_llp_q_packed, recvcounts, displs, i_rank)
            IF (bs_env%para_env%mepos == i_rank) THEN
               DO l = 0, bs_env%para_env%num_pe - 1
                  first_l_rank = (n_grid*l)/bs_env%para_env%num_pe + 1
                  last_l_rank = (n_grid*(l + 1))/bs_env%para_env%num_pe
                  D_llp_q(first_l_rank:last_l_rank, :) = RESHAPE( &
                                                         D_llp_q_packed(displs(l + 1) + 1:displs(l + 1) + recvcounts(l + 1)), &
                                                         [last_l_rank - first_l_rank + 1, n_grid])
               END DO
            END IF
         END DO
         CALL fourier_transform_d_lP_R_to_q(bs_env, cells_R_d_lP, d_lP_R, qpoints, i_batch, &
                                            distributed, required_q, d_lP_q_reference, iq)
         IF (iq > n_q) CYCLE
         iq_local = iq_local + 1
         IF (.NOT. required_q(iq)) CYCLE
         IF (distributed) THEN
            iq_store = iq_local
         ELSE
            iq_store = iq
         END IF
         qpoint = qpoints(:, iq)
         IF (PRESENT(charge_lq)) THEN
            ! The Gamma-correction scalar head is separate. Keep the full Gaussian RI.
            CPASSERT(n_RI_reference == n_RI)
            d_lP_q(:, 2:) = d_lP_q_reference
            volume = bs_env%ri_rs%cell%deth
            IF (ALL(bs_env%periodic == [1, 1, 0])) THEN
               ! The separated 2D charge channel is normalized by layer area,
               ! independent of the vacuum-cell height.
               volume = SQRT(SUM([bs_env%hmat(2, 1)*bs_env%hmat(3, 2) - &
                                  bs_env%hmat(3, 1)*bs_env%hmat(2, 2), &
                                  bs_env%hmat(3, 1)*bs_env%hmat(1, 2) - &
                                  bs_env%hmat(1, 1)*bs_env%hmat(3, 2), &
                                  bs_env%hmat(1, 1)*bs_env%hmat(2, 2) - &
                                  bs_env%hmat(2, 1)*bs_env%hmat(1, 2)]**2))
            END IF
            CPASSERT(volume > 0.0_dp)
            IF (bs_env%ri_metric%potential_type == do_potential_short) THEN
               metric_head = pi/bs_env%ri_metric%omega**2
            ELSE
               metric_head = 2.0_dp*pi*bs_env%ri_metric%cutoff_radius**2
            END IF
            d_lP_q(:, 1) = metric_head/SQRT(volume)*charge_lq(:, iq)
         ELSE
            CPASSERT(n_RI_reference == n_RI)
            d_lP_q(:, :) = d_lP_q_reference
         END IF

         antihermitian_error = MAXVAL(ABS(D_llp_q - CONJG(TRANSPOSE(D_llp_q))))
         diagonal_imag_error = 0.0_dp
         DO l = 1, n_grid
            diagonal_imag_error = MAX(diagonal_imag_error, ABS(AIMAG(D_llp_q(l, l))))
         END DO
         IF (antihermitian_error > 100.0_dp*bs_env%eps_filter .OR. &
             diagonal_imag_error > 100.0_dp*bs_env%eps_filter) THEN
            CPABORT("Non-Hermitian periodic RI-RS D(q)")
         END IF
         D_llp_q(:, :) = 0.5_dp*(D_llp_q(:, :) + CONJG(TRANSPOSE(D_llp_q(:, :))))

         DO l = 1, n_grid
            IF (REAL(D_llp_q(l, l), KIND=dp) < -100.0_dp*bs_env%eps_filter) THEN
               CPABORT("Periodic RI-RS D(q) has a negative diagonal element")
            END IF
            SCALE(l) = SQRT(MAX(REAL(D_llp_q(l, l), KIND=dp), TINY(1.0_dp)))
         END DO
         max_scale = MAXVAL(scale)
         active_grid_point(:) = scale > &
                                SQRT(MAX(bs_env%eps_filter, EPSILON(1.0_dp)))*max_scale
         IF (.NOT. ANY(active_grid_point)) THEN
            CPABORT("Periodic RI-RS D(q) has no active grid points")
         END IF

         ! Form D'_ll'(q) = D_ll'(q)/(s_l s_l') + eta delta_ll'.
         DO lp = 1, n_grid
            DO l = 1, n_grid
               IF (active_grid_point(l) .AND. active_grid_point(lp)) THEN
                  D_llp_q(l, lp) = D_llp_q(l, lp)/(SCALE(l)*SCALE(lp))
               ELSE
                  D_llp_q(l, lp) = z_zero
               END IF
            END DO
         END DO
         DO l = 1, n_grid
            D_llp_q(l, l) = D_llp_q(l, l) + bs_env%ri_rs%tikhonov*z_one
            IF (active_grid_point(l)) THEN
               rhs(l, :) = d_lP_q(l, :)/SCALE(l)
            ELSE
               rhs(l, :) = z_zero
               SCALE(l) = 1.0_dp
            END IF
         END DO

         IF (bs_env%ri_rs%tikhonov <= 0.0_dp) THEN
            CALL solve_unregularized_D_llp(D_llp_q, rhs)
         ELSE
            CALL zpotrf('L', n_grid, D_llp_q, n_grid, info)
            IF (info == 0) CALL zpotrs('L', n_grid, n_RI_solve, D_llp_q, n_grid, rhs, n_grid, info)
            IF (info /= 0) CPABORT("Periodic RI-RS regularized solve failed")
         END IF

         DO l = 1, n_grid
            Z_lP_q(l, :, iq_store) = rhs(l, :)/SCALE(l)
         END DO
         IF (bs_env%para_env%mepos == 0 .AND. n_local > 1) THEN
            n_done = n_done + 1_int_8
            CALL write_ri_rs_progress(bs_env%unit_nr, 'Z_lP(q) q-point', &
                                      n_done, n_local, next_progress, progress_start)
         END IF
      END DO
      IF (distributed) THEN
         CPASSERT(iq_local == n_q_local)
      END IF
      IF (.NOT. distributed) CALL bs_env%para_env%sum(Z_lP_q)
      DEALLOCATE (D_llp_q, D_llp_q_local, D_llp_q_packed, recvcounts, displs, &
                  d_lP_q, d_lP_q_reference, rhs, scale, active_grid_point, required_q)

      CALL timestop(handle)

   END SUBROUTINE solve_for_Z_lP_q

! **************************************************************************************************
!> \brief Solve a possibly rank-deficient Hermitian D matrix without Tikhonov regularization.
!> Only eigenvalues indistinguishable from zero at matrix-roundoff precision
!> are excluded from the Moore-Penrose inverse.
!> \param D_llp_q ...
!> \param rhs ...
! **************************************************************************************************
   SUBROUTINE solve_unregularized_D_llp(D_llp_q, rhs)
      COMPLEX(KIND=dp), DIMENSION(:, :), INTENT(INOUT) :: D_llp_q, rhs

      COMPLEX(KIND=dp), ALLOCATABLE                    :: projected_rhs(:, :), work(:)
      INTEGER                                          :: ieigenvalue, info, ngrid
      REAL(KIND=dp)                                    :: rank_tolerance
      REAL(KIND=dp), ALLOCATABLE                       :: eigenvalues(:), rwork(:)

      ngrid = SIZE(D_llp_q, 1)
      CPASSERT(ngrid > 0)
      CPASSERT(SIZE(D_llp_q, 2) == ngrid)
      CPASSERT(SIZE(rhs, 1) == ngrid)
      ALLOCATE (eigenvalues(ngrid), rwork(MAX(1, 3*ngrid - 2)), &
                work(MAX(1, 2*ngrid - 1)), projected_rhs(ngrid, SIZE(rhs, 2)))
      CALL zheev('V', 'L', ngrid, D_llp_q, ngrid, eigenvalues, work, SIZE(work), rwork, info)
      IF (info /= 0) CPABORT('Unregularized periodic RI-RS eigensolver failed')
      rank_tolerance = EPSILON(1.0_dp)*REAL(ngrid, dp)*MAXVAL(ABS(eigenvalues))
      IF (MINVAL(eigenvalues) < -10.0_dp*rank_tolerance) THEN
         CPABORT('Unregularized RI-RS D matrix is not positive semidefinite')
      END IF
      IF (.NOT. ANY(eigenvalues > rank_tolerance)) THEN
         CPABORT('Unregularized RI-RS D matrix has zero numerical rank')
      END IF

      projected_rhs(:, :) = MATMUL(CONJG(TRANSPOSE(D_llp_q)), rhs)
      DO ieigenvalue = 1, ngrid
         IF (eigenvalues(ieigenvalue) > rank_tolerance) THEN
            projected_rhs(ieigenvalue, :) = &
               projected_rhs(ieigenvalue, :)/eigenvalues(ieigenvalue)
         ELSE
            projected_rhs(ieigenvalue, :) = z_zero
         END IF
      END DO
      rhs(:, :) = MATMUL(D_llp_q, projected_rhs)

   END SUBROUTINE solve_unregularized_D_llp

! **************************************************************************************************
!> \brief Enumerate translations T that survive the R_cut^(lP) criterion for at least one (l,P).
!> \param bs_env ...
!> \param grid_points ...
!> \param cells_R_d_lP ...
! **************************************************************************************************
   SUBROUTINE build_cells_R_for_d_lP_R(bs_env, grid_points, cells_R_d_lP)
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN) :: grid_points
      INTEGER, ALLOCATABLE, INTENT(OUT)          :: cells_R_d_lP(:, :)

      INTEGER                                    :: cell_index(3), i_atom, i_l, ix, iy, iz, n_max(3), n_RI, n_work
      INTEGER, ALLOCATABLE                       :: cells_work(:, :)
      LOGICAL                                    :: keep
      REAL(KIND=dp)                              :: center(3), cutoff, max_cutoff
      REAL(KIND=dp), ALLOCATABLE                 :: automatic_cutoffs(:)
      TYPE(particle_type), DIMENSION(:), POINTER :: particle_set

      particle_set => bs_env%ri_rs%particle_set
      CALL compute_d_lP_cutoffs(bs_env, automatic_cutoffs)
      max_cutoff = MAXVAL(automatic_cutoffs)
      DO ix = 1, 3
         IF (bs_env%periodic(ix) == 1) THEN
            ! If |h*T + delta| <= R, then every fractional component obeys
            ! |T_i| <= R*|(h^-1)_i| + |(h^-1 delta)_i|.  Both the atom and the
            ! grid point are mapped into the reference cell, so the last term
            ! is smaller than one.  This reciprocal-vector bound remains safe
            ! for skewed cells, unlike a bound based on direct-vector lengths.
            n_max(ix) = CEILING(max_cutoff*NORM2(bs_env%ri_rs%cell%h_inv(ix, :))) + 1
         ELSE
            n_max(ix) = 0
         END IF
      END DO

      n_work = PRODUCT(2*n_max + 1)
      ALLOCATE (cells_work(3, n_work))
      n_RI = 0
      DO ix = -n_max(1), n_max(1)
         DO iy = -n_max(2), n_max(2)
            DO iz = -n_max(3), n_max(3)
               cell_index = [ix, iy, iz]
               keep = .FALSE.
               DO i_atom = 1, bs_env%n_atom
                  cutoff = automatic_cutoffs(i_atom)
                  center = pbc(particle_set(i_atom)%r, bs_env%ri_rs%cell) + &
                           MATMUL(bs_env%hmat, REAL(cell_index, dp))
                  DO i_l = 1, SIZE(grid_points, 2)
                     IF (NORM2(center - grid_points(:, i_l)) <= cutoff) THEN
                        keep = .TRUE.
                        EXIT
                     END IF
                  END DO
                  IF (keep) EXIT
               END DO
               IF (.NOT. keep) CYCLE
               n_RI = n_RI + 1
               cells_work(:, n_RI) = cell_index
            END DO
         END DO
      END DO

      ALLOCATE (cells_R_d_lP(3, n_RI))
      cells_R_d_lP(:, :) = cells_work(:, 1:n_RI)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T69,F10.1,A)') &
            'Periodic RI-RS automatic R_cut^(lP):', &
            max_cutoff*angstrom, ' Å'
      END IF
      DEALLOCATE (automatic_cutoffs, cells_work)

   END SUBROUTINE build_cells_R_for_d_lP_R

! **************************************************************************************************
!> \brief Enumerate the translations A required by the AO-product D matrix D_ll'^A.
!>
!> Unlike the right-hand side d_lP^T, D_ll'^A is independent of R_cut^(lP). A translation can
!> contribute only if it is the difference of two image cells with nonzero AO values on the
!> reference-cell grid. Keeping a separate cell set prevents Eq. (10) screening from truncating
!> the least-squares metric and avoids enumerating empty three-center image cells.
!> \param bs_env ...
!> \param phi_lmu_R ...
!> \param cells_R_D_llp ...
! **************************************************************************************************
   SUBROUTINE build_cells_R_for_D_llp(bs_env, phi_lmu_R, cells_R_D_llp)
      TYPE(post_scf_bandstructure_type), POINTER    :: bs_env
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(IN) :: phi_lmu_R
      INTEGER, ALLOCATABLE, INTENT(OUT)             :: cells_R_D_llp(:, :)

      INTEGER                                       :: cell_index(3), i, i_R, i_R_prime, n_active, n_D_cells
      INTEGER, ALLOCATABLE                          :: active_cells(:), cells_work(:, :)
      LOGICAL                                       :: found

      ALLOCATE (active_cells(SIZE(phi_lmu_R, 3)))
      n_active = 0
      DO i_R = 1, SIZE(phi_lmu_R, 3)
         IF (.NOT. ANY(phi_lmu_R(:, :, i_R) /= 0.0_dp)) CYCLE
         n_active = n_active + 1
         active_cells(n_active) = i_R
      END DO
      CPASSERT(n_active > 0)

      ALLOCATE (cells_work(3, n_active*n_active))
      n_D_cells = 0
      DO i_R = 1, n_active
         DO i_R_prime = 1, n_active
            cell_index = bs_env%index_to_cell_3c(:, active_cells(i_R)) - &
                         bs_env%index_to_cell_3c(:, active_cells(i_R_prime))
            found = .FALSE.
            DO i = 1, n_D_cells
               IF (ALL(cell_index == cells_work(:, i))) THEN
                  found = .TRUE.
                  EXIT
               END IF
            END DO
            IF (.NOT. found) THEN
               n_D_cells = n_D_cells + 1
               cells_work(:, n_D_cells) = cell_index
            END IF
         END DO
      END DO

      ALLOCATE (cells_R_D_llp(3, n_D_cells))
      cells_R_D_llp(:, :) = cells_work(:, 1:n_D_cells)
      DEALLOCATE (active_cells, cells_work)

   END SUBROUTINE build_cells_R_for_D_llp

! **************************************************************************************************
!> \brief Construct D_ll'^R = [Σ_{S-S'=R,μ} ϕ_μ^S(r_L) ϕ_μ^S'(r_L')]².
!> The progress count enumerates pairs of active cell images (S,S'), not Gaussian pairs.
!> \param bs_env ...
!> \param cells_R_D_llp ...
!> \param phi_lmu_R ...
!> \param first_l ...
!> \param last_l ...
!> \param D_llp_R ...
! **************************************************************************************************
   SUBROUTINE compute_D_llp(bs_env, cells_R_D_llp, phi_lmu_R, first_l, last_l, D_llp_R)
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      INTEGER, DIMENSION(:, :), INTENT(IN)       :: cells_R_D_llp
      REAL(KIND=dp), INTENT(IN)                  :: phi_lmu_R(:, :, :)
      INTEGER, INTENT(IN)                        :: first_l, last_l
      REAL(KIND=dp), CONTIGUOUS, INTENT(OUT)     :: D_llp_R(:, :, :)

      INTEGER                                    :: cell_A(3), handle, i_A, i_R, i_R_prime, next_progress, nrows
      INTEGER(KIND=int_8)                        :: n_done, n_total
      LOGICAL                                    :: found
      LOGICAL, ALLOCATABLE                       :: active_cell(:)
      REAL(KIND=dp)                              :: progress_start
      REAL(KIND=dp), ALLOCATABLE                 :: phi_local(:, :)
      INTERFACE
         SUBROUTINE dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            IMPORT dp
            CHARACTER(LEN=1), INTENT(IN) :: transa, transb
            INTEGER, INTENT(IN)          :: m, n, k, lda, ldb, ldc
            REAL(dp), INTENT(IN)         :: alpha, beta, a(lda, *), b(ldb, *)
            REAL(dp), INTENT(INOUT)      :: c(ldc, *)

         END SUBROUTINE dgemm
      END INTERFACE

      CALL timeset('compute_D_llp', handle)

      ALLOCATE (active_cell(SIZE(phi_lmu_R, 3)), SOURCE=.FALSE.)
      DO i_R = 1, SIZE(phi_lmu_R, 3)
         active_cell(i_R) = ANY(phi_lmu_R(:, :, i_R) /= 0.0_dp)
      END DO
      progress_start = m_walltime()
      next_progress = 1
      n_done = 0_int_8
      n_total = INT(COUNT(active_cell), int_8)**2
      ! Sum the AO images for each owned (l,l') rectangle before squaring.
      D_llp_R = 0.0_dp
      nrows = MAX(0, last_l - first_l + 1)
      ALLOCATE (phi_local(nrows, SIZE(phi_lmu_R, 2)))
      IF (first_l <= last_l) THEN
         DO i_R = 1, SIZE(phi_lmu_R, 3)
            IF (.NOT. active_cell(i_R)) CYCLE
            phi_local(:, :) = phi_lmu_R(first_l:last_l, :, i_R)
            DO i_R_prime = 1, SIZE(phi_lmu_R, 3)
               IF (.NOT. active_cell(i_R_prime)) CYCLE
               IF (n_done > 0) CALL write_ri_rs_progress(bs_env%unit_nr, "D_ll'^R Cell-image pairs", &
                                                         n_done, n_total, next_progress, progress_start)
               n_done = n_done + 1
               cell_A = bs_env%index_to_cell_3c(:, i_R) - &
                        bs_env%index_to_cell_3c(:, i_R_prime)
               CALL find_cell(cell_A, cells_R_D_llp, found, i_A)
               IF (.NOT. found) CYCLE
               CALL dgemm('N', 'T', nrows, SIZE(phi_lmu_R, 1), SIZE(phi_lmu_R, 2), &
                          1.0_dp, phi_local, MAX(1, nrows), phi_lmu_R(:, :, i_R_prime), &
                          MAX(1, SIZE(phi_lmu_R, 1)), 1.0_dp, D_llp_R(:, :, i_A), &
                          MAX(1, nrows))
            END DO
         END DO
      END IF
      IF (first_l <= last_l) CALL write_ri_rs_progress(bs_env%unit_nr, "D_ll'^R Cell-image pairs", &
                                                       n_done, n_total, next_progress, progress_start)
      DO i_A = 1, SIZE(D_llp_R, 3)
         D_llp_R(:, :, i_A) = D_llp_R(:, :, i_A)*D_llp_R(:, :, i_A)
      END DO
      DEALLOCATE (active_cell, phi_local)

      CALL timestop(handle)

   END SUBROUTINE compute_D_llp

! **************************************************************************************************
!> \brief Find a lattice triplet in an explicit image-cell list.
!> \param cell ...
!> \param index_to_cell ...
!> \param found ...
!> \param index ...
! **************************************************************************************************
   SUBROUTINE find_cell(cell, index_to_cell, found, index)
      INTEGER, DIMENSION(3), INTENT(IN)    :: cell
      INTEGER, DIMENSION(:, :), INTENT(IN) :: index_to_cell
      LOGICAL, INTENT(OUT)                 :: found
      INTEGER, INTENT(OUT)                 :: index

      INTEGER                              :: i

      found = .FALSE.
      index = 0
      DO i = 1, SIZE(index_to_cell, 2)
         IF (ALL(cell == index_to_cell(:, i))) THEN
            found = .TRUE.
            index = i
            RETURN
         END IF
      END DO

   END SUBROUTINE find_cell

! **************************************************************************************************
!> \brief Fourier-transform d_lP(R) to d_lP(q) and gather l rows on each q owner.
!> d_lP(q)=Σ_T exp(iqT)d_lP(T); two send buffers overlap transfer with the next q-point transform.
!> Each rank holds all P and T for its l interval; each destination receives one complete q.
!> \param bs_env ...
!> \param cells_R_d_lP ...
!> \param d_lP_R ...
!> \param qpoints ...
!> \param i_batch ...
!> \param distributed ...
!> \param required_q ...
!> \param d_lP_q ...
!> \param iq ...
! **************************************************************************************************
   SUBROUTINE fourier_transform_d_lP_R_to_q(bs_env, cells_R_d_lP, d_lP_R, qpoints, i_batch, distributed, &
                                            required_q, d_lP_q, iq)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: cells_R_d_lP(:, :)
      REAL(dp), CONTIGUOUS, INTENT(IN), TARGET           :: d_lP_R(:, :, :)
      REAL(dp), INTENT(IN)                               :: qpoints(:, :)
      INTEGER, INTENT(IN)                                :: i_batch
      LOGICAL, INTENT(IN)                                :: distributed, required_q(:)
      COMPLEX(dp), ASYNCHRONOUS, CONTIGUOUS, INTENT(OUT) :: d_lP_q(:, :)
      INTEGER, INTENT(OUT)                               :: iq

      COMPLEX(dp)                                        :: phase
      COMPLEX(dp), ALLOCATABLE, ASYNCHRONOUS             :: receive_q(:), send_q(:, :, :)
      INTEGER                                            :: cell_index(3), first_l, handle, i_dest, i_grid, i_l, i_P, i_q, &
                                                            i_rank, i_slot, i_T, last_l, local_first_l, local_last_l, &
                                                            n_active_q, n_grid_P, n_l, n_P, n_ranks, n_recv, position, &
                                                            q_column, q_offset
      INTEGER(KIND=int_8)                                :: work_bytes
      INTEGER, ALLOCATABLE                               :: q_indices(:)
      LOGICAL                                            :: receiving, sending(2), use_blas
      REAL(dp)                                           :: qpoint(3)
      REAL(dp), ALLOCATABLE                              :: phase_imag(:, :), phase_real(:, :), result_imag(:, :), &
                                                            result_real(:, :)
      REAL(dp), CONTIGUOUS, POINTER                      :: RI_matrix(:, :)
      TYPE(mp_request_type)                              :: send_requests(2)
      TYPE(mp_request_type), ALLOCATABLE                 :: recv_requests(:)
      INTERFACE
         SUBROUTINE dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            IMPORT dp
            CHARACTER(LEN=1), INTENT(IN) :: transa, transb
            INTEGER, INTENT(IN)          :: m, n, k, lda, ldb, ldc
            REAL(dp), INTENT(IN)         :: alpha, beta, a(lda, *), b(ldb, *)
            REAL(dp), INTENT(INOUT)      :: c(ldc, *)

         END SUBROUTINE dgemm
      END INTERFACE

      CALL timeset('d_lP_T_to_q', handle)

      n_ranks = bs_env%para_env%num_pe
      i_rank = bs_env%para_env%mepos
      CPASSERT(SIZE(required_q) == SIZE(qpoints, 2))
      q_offset = 0
      IF (distributed) q_offset = 1
      iq = i_batch*n_ranks + MODULO(i_rank - q_offset, n_ranks) + 1
      CALL get_local_l_range(SIZE(d_lP_q, 1), n_ranks, i_rank, local_first_l, local_last_l)
      CPASSERT(SIZE(d_lP_R, 1) == local_last_l - local_first_l + 1)
      CPASSERT(SIZE(d_lP_R, 2) == SIZE(d_lP_q, 2))
      CPASSERT(SIZE(d_lP_q, KIND=int_8) <= INT(HUGE(0), int_8))
      n_P = SIZE(d_lP_q, 2)
      ALLOCATE (recv_requests(n_ranks), send_q(SIZE(d_lP_R, 1), SIZE(d_lP_R, 2), 2))
      d_lP_q = z_zero
      n_recv = 0
      receiving = .FALSE.
      IF (iq <= SIZE(qpoints, 2)) receiving = required_q(iq)
      IF (receiving) THEN
         ! Packed rank blocks: offset=(first_l-1)*n_P, followed by local (l,P) order.
         ! Receive into contiguous storage before assigning the noncontiguous output rows.
         ALLOCATE (receive_q(SIZE(d_lP_q)))
         DO i_dest = 0, n_ranks - 1
            IF (i_dest == i_rank) CYCLE
            CALL get_local_l_range(SIZE(d_lP_q, 1), n_ranks, i_dest, first_l, last_l)
            IF (last_l < first_l .OR. n_P == 0) CYCLE
            n_recv = n_recv + 1
            CALL bs_env%para_env%irecv(receive_q((first_l - 1)*n_P + 1:last_l*n_P), &
                                       i_dest, recv_requests(n_recv), tag=29571)
         END DO
      END IF
      ! Form several Fourier sums from the locally owned l rows using real BLAS.
      ! Bound the additional storage; large cases retain the two-buffer path.
      n_active_q = 0
      ALLOCATE (q_indices(n_ranks))
      DO i_dest = 0, n_ranks - 1
         i_q = i_batch*n_ranks + MODULO(i_dest - q_offset, n_ranks) + 1
         IF (i_q > SIZE(qpoints, 2)) CYCLE
         IF (.NOT. required_q(i_q)) CYCLE
         n_active_q = n_active_q + 1
         q_indices(n_active_q) = i_q
      END DO
      work_bytes = INT(SIZE(d_lP_R, 1), int_8)*INT(SIZE(d_lP_R, 2), int_8)
      use_blas = work_bytes > 0_int_8 .AND. work_bytes <= INT(HUGE(n_grid_P), int_8)
      IF (use_blas) THEN
         n_grid_P = INT(work_bytes)
         work_bytes = 16_int_8*INT(n_active_q, int_8)* &
                      (work_bytes + INT(SIZE(cells_R_d_lP, 2), int_8))
         use_blas = n_active_q >= 4 .AND. SIZE(cells_R_d_lP, 2) >= 32 .AND. &
                    work_bytes <= 64_int_8*1024_int_8*1024_int_8
      END IF
      IF (use_blas) THEN
         ALLOCATE (phase_real(SIZE(cells_R_d_lP, 2), n_active_q), &
                   phase_imag(SIZE(cells_R_d_lP, 2), n_active_q), &
                   result_real(n_grid_P, n_active_q), result_imag(n_grid_P, n_active_q))
         DO q_column = 1, n_active_q
            qpoint(:) = qpoints(1:3, q_indices(q_column))
            DO i_T = 1, SIZE(cells_R_d_lP, 2)
               cell_index(:) = cells_R_d_lP(1:3, i_T)
               phase = lattice_phase(qpoint, cell_index)
               phase_real(i_T, q_column) = REAL(phase, KIND=dp)
               phase_imag(i_T, q_column) = AIMAG(phase)
            END DO
         END DO
         RI_matrix(1:n_grid_P, 1:SIZE(cells_R_d_lP, 2)) => d_lP_R
         CALL dgemm('N', 'N', n_grid_P, n_active_q, SIZE(cells_R_d_lP, 2), &
                    1.0_dp, RI_matrix, n_grid_P, phase_real, SIZE(cells_R_d_lP, 2), &
                    0.0_dp, result_real, n_grid_P)
         CALL dgemm('N', 'N', n_grid_P, n_active_q, SIZE(cells_R_d_lP, 2), &
                    1.0_dp, RI_matrix, n_grid_P, phase_imag, SIZE(cells_R_d_lP, 2), &
                    0.0_dp, result_imag, n_grid_P)
      END IF
      sending = .FALSE.
      q_column = 0
      IF (SIZE(d_lP_R, 1) > 0 .AND. SIZE(d_lP_R, 2) > 0) THEN
         DO i_dest = 0, n_ranks - 1
            i_q = i_batch*n_ranks + MODULO(i_dest - q_offset, n_ranks) + 1
            IF (i_q > SIZE(qpoints, 2)) CYCLE
            IF (.NOT. required_q(i_q)) CYCLE
            q_column = q_column + 1
            i_slot = MODULO(i_dest, 2) + 1
            IF (sending(i_slot)) CALL mp_waitall(send_requests(i_slot:i_slot))
            sending(i_slot) = .FALSE.
            IF (use_blas) THEN
               DO i_P = 1, SIZE(d_lP_R, 2)
                  DO i_grid = 1, SIZE(d_lP_R, 1)
                     i_T = (i_P - 1)*SIZE(d_lP_R, 1) + i_grid
                     send_q(i_grid, i_P, i_slot) = &
                        CMPLX(result_real(i_T, q_column), result_imag(i_T, q_column), KIND=dp)
                  END DO
               END DO
            ELSE
               send_q(:, :, i_slot) = z_zero
               DO i_T = 1, SIZE(cells_R_d_lP, 2)
                  send_q(:, :, i_slot) = send_q(:, :, i_slot) + &
                                         lattice_phase(qpoints(:, i_q), cells_R_d_lP(:, i_T))*d_lP_R(:, :, i_T)
               END DO
            END IF
            IF (i_dest == i_rank) THEN
               d_lP_q(local_first_l:local_last_l, :) = send_q(:, :, i_slot)
            ELSE
               CALL bs_env%para_env%isend(send_q(:, :, i_slot), i_dest, send_requests(i_slot), tag=29571)
               sending(i_slot) = .TRUE.
            END IF
         END DO
      END IF
      DO i_slot = 1, 2
         IF (sending(i_slot)) CALL mp_waitall(send_requests(i_slot:i_slot))
      END DO
      IF (n_recv > 0) CALL mp_waitall(recv_requests(:n_recv))
      IF (receiving) THEN
         DO i_dest = 0, n_ranks - 1
            IF (i_dest == i_rank) CYCLE
            CALL get_local_l_range(SIZE(d_lP_q, 1), n_ranks, i_dest, first_l, last_l)
            n_l = last_l - first_l + 1
            DO i_P = 1, n_P
               position = (first_l - 1)*n_P + (i_P - 1)*n_l
               DO i_l = 1, n_l
                  d_lP_q(first_l + i_l - 1, i_P) = receive_q(position + i_l)
               END DO
            END DO
         END DO
         DEALLOCATE (receive_q)
      END IF
      DEALLOCATE (send_q, recv_requests)

      CALL timestop(handle)

   END SUBROUTINE fourier_transform_d_lP_R_to_q

END MODULE ri_rs_kp_fitting
