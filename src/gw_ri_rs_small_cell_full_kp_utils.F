!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Green functions, screening, self-energy and Gamma correction for periodic RI-RS GW.
!> \author Jan Wilhelm (with help of ChatGPT, GPT-6 Sol and GPT-6 Astra)
! **************************************************************************************************
MODULE gw_ri_rs_small_cell_full_kp_utils
   USE atomic_kind_types,               ONLY: atomic_kind_type,&
                                              get_atomic_kind_set
   USE basis_set_types,                 ONLY: gto_basis_set_type
   USE cell_types,                      ONLY: cell_type,&
                                              get_cell
   USE cp_blacs_env,                    ONLY: cp_blacs_env_create,&
                                              cp_blacs_env_release,&
                                              cp_blacs_env_type
   USE cp_cfm_basic_linalg,             ONLY: cp_cfm_add_on_diag,&
                                              cp_cfm_column_scale,&
                                              cp_cfm_norm,&
                                              cp_cfm_scale_and_add,&
                                              cp_cfm_transpose,&
                                              cp_cfm_uplo_to_full
   USE cp_cfm_cholesky,                 ONLY: cp_cfm_cholesky_decompose,&
                                              cp_cfm_cholesky_invert
   USE cp_cfm_diag,                     ONLY: cp_cfm_heevd,&
                                              cp_cfm_power
   USE cp_cfm_types,                    ONLY: cp_cfm_create,&
                                              cp_cfm_get_info,&
                                              cp_cfm_get_submatrix,&
                                              cp_cfm_release,&
                                              cp_cfm_set_submatrix,&
                                              cp_cfm_type
   USE cp_control_types,                ONLY: dft_control_type
   USE cp_dbcsr_api,                    ONLY: &
        dbcsr_copy, dbcsr_create, dbcsr_distribution_release, dbcsr_distribution_type, &
        dbcsr_get_info, dbcsr_get_readonly_block_p, dbcsr_p_type, dbcsr_release, dbcsr_type, &
        dbcsr_type_antisymmetric, dbcsr_type_no_symmetry
   USE cp_dbcsr_operations,             ONLY: copy_dbcsr_to_fm,&
                                              cp_dbcsr_dist2d_to_dist,&
                                              dbcsr_allocate_matrix_set,&
                                              dbcsr_deallocate_matrix_set
   USE cp_fm_struct,                    ONLY: cp_fm_struct_create,&
                                              cp_fm_struct_release,&
                                              cp_fm_struct_type
   USE cp_fm_types,                     ONLY: cp_fm_create,&
                                              cp_fm_get_info,&
                                              cp_fm_release,&
                                              cp_fm_type
   USE distribution_2d_types,           ONLY: distribution_2d_type
   USE gw_auto_ri_types,                ONLY: auto_ri_type
   USE gw_utils_fm,                     ONLY: local_complex_power
   USE input_constants,                 ONLY: do_potential_coulomb,&
                                              do_potential_short,&
                                              do_potential_truncated,&
                                              evGW0,&
                                              ri_rs_small_cell_full_kp
   USE kinds,                           ONLY: dp,&
                                              int_8
   USE kpoint_coulomb_2c,               ONLY: build_2c_coulomb_matrix_kp_small_cell
   USE kpoint_k_r_trafo_simple,         ONLY: rs_to_kp
   USE kpoint_lattice_fft,              ONLY: cell_to_k_grid_fft,&
                                              k_grid_to_cell_execute,&
                                              k_grid_to_cell_fft,&
                                              k_grid_to_cell_prepare,&
                                              k_grid_to_cell_release,&
                                              k_grid_to_cell_work_type
   USE kpoint_methods,                  ONLY: kp_transform_plan_create,&
                                              kp_transform_plan_type,&
                                              kpoint_init_cell_index
   USE kpoint_types,                    ONLY: get_kpoint_info,&
                                              kpoint_create,&
                                              kpoint_release,&
                                              kpoint_type
   USE libint_2c_3c,                    ONLY: libint_potential_type
   USE machine,                         ONLY: m_flush,&
                                              m_memory,&
                                              m_walltime
   USE mathconstants,                   ONLY: fourpi,&
                                              gaussi,&
                                              pi,&
                                              twopi,&
                                              z_one,&
                                              z_zero
   USE mathlib,                         ONLY: diag_complex,&
                                              gemm_square,&
                                              inv_3x3
   USE message_passing,                 ONLY: mp_comm_split_type_shared,&
                                              mp_comm_type,&
                                              mp_mem_avail_per_rank_GB,&
                                              mp_para_env_release,&
                                              mp_para_env_type
   USE mp2_eri_gpw,                     ONLY: mp2_eri_2c_integrate_gpw
   USE omp_lib,                         ONLY: omp_get_max_threads
   USE orbital_pointers,                ONLY: indco,&
                                              ncoset
   USE parallel_gemm_api,               ONLY: parallel_gemm
   USE particle_methods,                ONLY: get_particle_set
   USE particle_types,                  ONLY: particle_type
   USE post_scf_bandstructure_types,    ONLY: post_scf_bandstructure_type
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type
   USE qs_kind_types,                   ONLY: get_qs_kind,&
                                              qs_kind_type
   USE qs_ks_types,                     ONLY: get_ks_env,&
                                              set_ks_env
   USE qs_moments,                      ONLY: build_local_moment_matrix_rs_img
   USE qs_neighbor_list_types,          ONLY: get_iterator_info,&
                                              neighbor_list_iterate,&
                                              neighbor_list_iterator_create,&
                                              neighbor_list_iterator_p_type,&
                                              neighbor_list_iterator_release,&
                                              neighbor_list_set_p_type,&
                                              release_neighbor_list_sets
   USE qs_tensors,                      ONLY: build_2c_integrals,&
                                              build_2c_neighbor_lists
   USE ri_rs_grid_setup,                ONLY: setup_ri_rs_grid
   USE ri_rs_kp_fitting,                ONLY: compute_Z_lP_q,&
                                              evaluate_mu_rl_R_values,&
                                              lattice_phase
   USE ri_rs_utils,                     ONLY: precompute_ri_rs_radii
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'gw_ri_rs_small_cell_full_kp_utils'

   PUBLIC :: gwkp_type, &
             compute_Sigma_x, &
             compute_g_and_chi, &
             compute_W0, &
             compute_sigma_c_from_w_pq, &
             prepare_ri_rs, prepare_ri_rs_gamma_correction, &
             get_gw_work, release_gw_work

   ! Blocking limits and named slots in the six existing CP2K matrix workspaces.
   INTEGER, PARAMETER :: column_block_cols = 32
   INTEGER, PARAMETER :: kpoint_batch_size = 8, pair_batch_size = 256
   INTEGER, PARAMETER :: w_chi = 1, w_metric = 2, w_coulomb = 3, &
                         w_tmp = 4, w_dielectric = 5, w_output = 6

   TYPE :: pair_transfer_plan_type
      INTEGER :: first_pair = 1, n_grid = 0, n_local = 0, n_q = 0, owner_offset = 0
      INTEGER, ALLOCATABLE :: first_pair_rank(:), n_pair_rank(:)
   END TYPE pair_transfer_plan_type

   ! One complex Fourier coefficient per real-space cell.  The singular head
   ! is kept factored from its two grid-point vectors, avoiding an additional
   ! grid-pair by cell allocation during exchange and correlation.
   TYPE :: gamma_correction_type
      COMPLEX(KIND=dp), ALLOCATABLE :: head_tau(:, :), wing_tau(:, :, :)
      REAL(KIND=dp), ALLOCATABLE :: directions(:, :), radii(:), weights(:)
      REAL(KIND=dp), ALLOCATABLE :: near_q(:, :)
      COMPLEX(KIND=dp), ALLOCATABLE :: slab_head_cell_tau(:, :)
      COMPLEX(KIND=dp), ALLOCATABLE :: slab_wing_cell_frequency(:, :, :)
      COMPLEX(KIND=dp), ALLOCATABLE :: slab_wing_cartesian(:, :, :)
      INTEGER :: n_angular = 0
   END TYPE gamma_correction_type

   TYPE gamma_ao_time_state_type
      INTEGER :: first_row = 1
      COMPLEX(KIND=dp), ALLOCATABLE :: chi_tau(:, :, :), wing_tau(:, :, :), head_tau(:, :, :)
      COMPLEX(KIND=dp), ALLOCATABLE :: metric_rows(:, :)
   END TYPE gamma_ao_time_state_type

   TYPE gamma_atom_projection_type
      INTEGER :: first = 0
      INTEGER, ALLOCATABLE :: positions(:)
      REAL(KIND=dp), ALLOCATABLE :: values(:, :)
   END TYPE gamma_atom_projection_type

   TYPE :: bounded_transfer_type
      INTEGER :: peer_limit = 0
      INTEGER(KIND=int_8) :: offset = 0_int_8, i_block = 0_int_8, n_blocks = 0_int_8
      INTEGER(KIND=int_8), ALLOCATABLE :: send_total(:), recv_total(:)
      INTEGER, ALLOCATABLE :: send_counts(:), recv_counts(:), send_displs(:), recv_displs(:)
      COMPLEX(KIND=dp), ALLOCATABLE :: send_buffer(:), recv_buffer(:)
   END TYPE bounded_transfer_type

   TYPE :: projection_interpolation_type
      INTEGER, ALLOCATABLE :: native_iq(:), offmesh_index(:)
      REAL(KIND=dp), ALLOCATABLE :: weights(:, :)
   END TYPE projection_interpolation_type

   !> values(P,Q,j) stores one q point, with j an imaginary frequency or time.
   !> Each q allocation can be released after its last use independently.
   TYPE ri_q_matrix_block_type
      COMPLEX(KIND=dp), ALLOCATABLE :: values(:, :, :)
   END TYPE ri_q_matrix_block_type

   ! Work data shared by the periodic GW stages; input settings remain in bs_env.
   TYPE :: gwkp_type
      REAL(KIND=dp), ALLOCATABLE                 :: eigenvalues_G(:, :, :)
      COMPLEX(dp), ALLOCATABLE                  :: metric_coulomb_factor(:, :, :), psi_lm_k(:, :, :, :), &
                                                   Z_lP_q(:, :, :), Z_lP_zero(:, :, :)
      REAL(dp), ALLOCATABLE                     :: sigma_c_bands(:, :, :, :, :), sigma_x_bands(:, :, :)
      REAL(dp), ALLOCATABLE                     :: ao_values(:, :, :)
      COMPLEX(dp), ALLOCATABLE                  :: band_orbitals(:, :, :, :)
      TYPE(mp_para_env_type), POINTER           :: group_env => NULL()
      INTEGER                                   :: ngroups = 1, my_group = 0
      TYPE(gamma_correction_type)               :: gamma_correction
      TYPE(projection_interpolation_type)       :: interpolation
      TYPE(ri_q_matrix_block_type), ALLOCATABLE :: chi_q_frequency(:), W_PQ_tau(:)
      ! AO image operators retained only until the Gamma-correction contraction.
      TYPE(dbcsr_p_type), POINTER               :: gamma_corr_d(:, :) => NULL(), &
                                                   gamma_corr_h(:, :) => NULL(), gamma_corr_s(:, :) => NULL()
      TYPE(kp_transform_plan_type)              :: gamma_corr_plan, gamma_corr_scf_plan
      INTEGER, POINTER                          :: gamma_corr_index_to_cell(:, :) => NULL(), &
                                                   gamma_corr_cell_to_index(:, :, :) => NULL()
   END TYPE gwkp_type

CONTAINS

! **************************************************************************************************
!> \brief Preserve distributed AO position, Hamiltonian and overlap images before QS release.
!> Only the sparse operators and their immutable image traversal survive; SCF grids and
!> neighbor lists can be released by the common GW initialization path.
!> \param qs_env Converged Quickstep state.
!> \param bs_env Band-structure settings.
! **************************************************************************************************
   SUBROUTINE prepare_ri_rs_gamma_correction(qs_env, bs_env)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env

      INTEGER                                            :: handle, i_image, i_spin, n_img
      LOGICAL                                            :: created_scf_image_map
      TYPE(dbcsr_p_type), POINTER                        :: matrix_h(:, :), matrix_s(:, :)
      TYPE(gwkp_type), POINTER                           :: gwkp_env
      TYPE(kpoint_type), POINTER                         :: kpoints_all, kpoints_scf
      TYPE(neighbor_list_set_p_type), POINTER            :: sab_all(:), sab_kp(:)

      CALL timeset('prepare_ri_rs_gamma_correction', handle)

      CPASSERT(.NOT. ALLOCATED(bs_env%gwkp_env))
      ALLOCATE (gwkp_type :: bs_env%gwkp_env)
      CALL get_gw_work(bs_env, gwkp_env)
      ALLOCATE (gwkp_env%eigenvalues_G(bs_env%n_ao, bs_env%nkp_scf_desymm, bs_env%n_spin))
      gwkp_env%eigenvalues_G(:, :, :) = bs_env%eigenval_scf(:, 1:bs_env%nkp_scf_desymm, :)

      NULLIFY (kpoints_all)

      CALL get_qs_env(qs_env, kpoints=kpoints_scf, matrix_ks_kp=matrix_h, matrix_s_kp=matrix_s, &
                      sab_all=sab_all, sab_kp=sab_kp)
      CPASSERT(ASSOCIATED(sab_all) .AND. ASSOCIATED(sab_kp))
      CPASSERT(ASSOCIATED(matrix_h) .AND. ASSOCIATED(matrix_s))
      CPASSERT(.NOT. ASSOCIATED(gwkp_env%gamma_corr_d))
      created_scf_image_map = .NOT. ASSOCIATED(kpoints_scf)
      IF (created_scf_image_map) THEN
         CALL kpoint_create(kpoints_scf)
         CALL kpoint_init_cell_index(kpoints_scf, sab_kp, bs_env%para_env, n_img)
      END IF
      CALL kpoint_create(kpoints_all)
      CALL kpoint_init_cell_index(kpoints_all, sab_all, bs_env%para_env, n_img)
      CALL gw_gamma_cell_index_union(kpoints_all, kpoints_scf, &
                                     gwkp_env%gamma_corr_index_to_cell, gwkp_env%gamma_corr_cell_to_index)
      CALL dbcsr_allocate_matrix_set(gwkp_env%gamma_corr_d, 3, SIZE(kpoints_all%index_to_cell, 2))
      CALL build_local_moment_matrix_rs_img(qs_env, gwkp_env%gamma_corr_d)
      CALL kp_transform_plan_create(gwkp_env%gamma_corr_plan, kpoints_all%sab_nl, &
                                    kpoints_all%cell_to_index, SIZE(gwkp_env%gamma_corr_d, 2))
      CALL kp_transform_plan_create(gwkp_env%gamma_corr_scf_plan, kpoints_scf%sab_nl, &
                                    kpoints_scf%cell_to_index, SIZE(matrix_h, 2))
      CALL dbcsr_allocate_matrix_set(gwkp_env%gamma_corr_h, SIZE(matrix_h, 1), SIZE(matrix_h, 2))
      CALL dbcsr_allocate_matrix_set(gwkp_env%gamma_corr_s, SIZE(matrix_s, 1), SIZE(matrix_s, 2))
      DO i_image = 1, SIZE(matrix_h, 2)
         DO i_spin = 1, SIZE(matrix_h, 1)
            ALLOCATE (gwkp_env%gamma_corr_h(i_spin, i_image)%matrix)
            CALL dbcsr_copy(gwkp_env%gamma_corr_h(i_spin, i_image)%matrix, matrix_h(i_spin, i_image)%matrix)
         END DO
      END DO
      DO i_image = 1, SIZE(matrix_s, 2)
         DO i_spin = 1, SIZE(matrix_s, 1)
            ALLOCATE (gwkp_env%gamma_corr_s(i_spin, i_image)%matrix)
            CALL dbcsr_copy(gwkp_env%gamma_corr_s(i_spin, i_image)%matrix, matrix_s(i_spin, i_image)%matrix)
         END DO
      END DO
      CALL kpoint_release(kpoints_all)
      IF (created_scf_image_map) CALL kpoint_release(kpoints_scf)

      CALL timestop(handle)

   END SUBROUTINE prepare_ri_rs_gamma_correction

! **************************************************************************************************
!> \brief Prepare RI-RS radial data, grid values, and Bloch orbitals.
!> \param qs_env Required for periodic AUTO_RI metric optimization via existing two-centre integrals.
!> \param bs_env Band-structure settings and physical results.
!> \param gwkp_env Temporary periodic GW work data.
! **************************************************************************************************
   SUBROUTINE prepare_ri_rs(qs_env, bs_env, gwkp_env)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gwkp_type), INTENT(INOUT)                     :: gwkp_env

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'prepare_ri_rs'

      COMPLEX(dp), ALLOCATABLE                           :: phi_lmu_k(:, :, :)
      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      CALL precompute_ri_rs_radii(bs_env%ri_rs, bs_env%basis_set_AO, bs_env%basis_set_RI, &
                                  bs_env%eps_filter, bs_env%unit_nr)
      IF (bs_env%auto_ri%enabled) THEN
         CALL optimize_periodic_auto_ri_metric(qs_env, bs_env)
      END IF
      CALL setup_ri_rs_grid(bs_env, bs_env%ri_rs%grid_points)
      CALL compute_Z_lP_q(bs_env, bs_env%ri_rs%grid_points, &
                          gwkp_env%ao_values, Z_lP_q=gwkp_env%Z_lP_q, Z_lP_zero=gwkp_env%Z_lP_zero)
      CALL build_all_phi_k(bs_env, gwkp_env%ao_values, phi_lmu_k)
      CALL build_all_psi_k(bs_env, phi_lmu_k, gwkp_env%psi_lm_k)
      DEALLOCATE (phi_lmu_k)
      IF (bs_env%gw_flavour == evGW0) THEN
         CPASSERT(ALLOCATED(gwkp_env%eigenvalues_G))
         CPASSERT(ALLOCATED(bs_env%eigenval_evGW0))
         CPASSERT(SIZE(bs_env%eigenval_scf, 2) >= bs_env%nkp_scf_desymm)
         bs_env%eigenval_evGW0(:, :, :) = bs_env%eigenval_scf
      END IF

      CALL timestop(handle)

   END SUBROUTINE prepare_ri_rs

! **************************************************************************************************
!> \brief Form the grid Green functions and RI polarizability.
!> \param qs_env Quickstep data required by existing integral/lifecycle interfaces.
!> \param bs_env Band-structure settings and physical results.
!> \param gwkp_env Temporary periodic GW work data.
! **************************************************************************************************
   SUBROUTINE compute_G_and_chi(qs_env, bs_env, gwkp_env)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gwkp_type), INTENT(INOUT)                     :: gwkp_env

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'compute_G_and_chi'

      INTEGER                                            :: group_size, handle

      CALL timeset(routineN, handle)

      CALL create_ri_rs_grid_groups(bs_env, SIZE(gwkp_env%ao_values, 1), gwkp_env%group_env, group_size, &
                                    gwkp_env%ngroups, gwkp_env%my_group)
      DEALLOCATE (gwkp_env%ao_values)

      CALL gamma_point_correction(bs_env, qs_env, gwkp_env)
      CPASSERT(.NOT. ALLOCATED(gwkp_env%band_orbitals))
      CALL compute_chi(bs_env, gwkp_env)

      CALL timestop(handle)

   END SUBROUTINE compute_G_and_chi

! **************************************************************************************************
!> \brief Compute ε, Wᶜ and its imaginary-time transform, Eqs. (7)-(8).
!> \param qs_env Required by the existing two-centre integral interfaces.
!> \param bs_env Band-structure settings and physical results.
!> \param gwkp_env Periodic GW work data.
! **************************************************************************************************
   SUBROUTINE compute_W0(qs_env, bs_env, gwkp_env)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gwkp_type), INTENT(INOUT)                     :: gwkp_env

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'compute_W0'

      COMPLEX(KIND=dp), ALLOCATABLE :: chi_k_w(:, :), eps_k_w(:, :), M_inv(:, :, :), &
         MWM_k_w(:, :), V_sqrt(:, :, :), W_aux_local_q(:, :, :), W_k_w(:, :)
      INTEGER :: handle, i_batch, i_group_size, i_t, ikp, ikp_local, j_w, max_condition_index(2), &
         max_metric_chi_index(2), min_dielectric_index(2), my_qgroup, n_batch, n_qgroups, &
         n_required_q, n_RI, nkp_local, own_q, phase_handle, q_ordinal, q_owner, q_root
      INTEGER, ALLOCATABLE                               :: q_roots(:)
      LOGICAL                                            :: grouped_q
      REAL(KIND=dp) :: available_GB, max_tmp, metric_norm_max, metric_tmp, min_tmp, phase_elapsed, &
         phase_start, qnorm, required_bytes, step_start, W_aux_time_norm, Wnorm_tmp
      REAL(KIND=dp), ALLOCATABLE :: dielectric_condition(:, :), dielectric_max_eigenvalue(:, :), &
         dielectric_min_eigenvalue(:, :), frequency_elapsed(:), M_inv_norm(:), &
         metric_chi_norm(:, :), time_weights(:), V_sqrt_norm(:), W_aux_norm(:)
      TYPE(cp_blacs_env_type), POINTER                   :: w_blacs
      TYPE(cp_cfm_type)                                  :: w_matrices(6)
      TYPE(cp_fm_struct_type), POINTER                   :: w_matrix_struct
      TYPE(mp_para_env_type), POINTER                    :: w_group_env

      CALL timeset(routineN, handle)

      CALL compute_gaussian_Minv_and_Vsqrt(bs_env, qs_env, gwkp_env%metric_coulomb_factor, M_inv, V_sqrt, metric_norm_max)
      phase_start = m_walltime()
      n_RI = bs_env%n_RI
      IF (bs_env%nkp_chi_eps_W_orig_plus_extra /= bs_env%nkp_scf_desymm) THEN
         CPABORT("Periodic RI-RS GW requires the complete SCF transfer mesh")
      END IF
      IF (ANY(bs_env%nkp_grid_chi_eps_W_orig /= &
              bs_env%kpoints_scf_desymm%nkp_grid)) THEN
         CPABORT("Periodic RI-RS GW requires the SCF mesh dimensions for W")
      END IF
      nkp_local = COUNT([(MODULO(ikp, bs_env%para_env%num_pe) == &
                          bs_env%para_env%mepos, ikp=1, bs_env%nkp_scf_desymm)])
      CPASSERT(SIZE(gwkp_env%chi_q_frequency) == nkp_local)
      CPASSERT(n_RI == bs_env%n_RI)
      nkp_local = SIZE(M_inv, 3)
      CPASSERT(SIZE(gwkp_env%chi_q_frequency) == nkp_local)
      DO ikp_local = 1, nkp_local
         CPASSERT(ALLOCATED(gwkp_env%chi_q_frequency(ikp_local)%values))
         CPASSERT(SIZE(gwkp_env%chi_q_frequency(ikp_local)%values, 1) == n_RI)
         CPASSERT(SIZE(gwkp_env%chi_q_frequency(ikp_local)%values, 2) == n_RI)
         IF (.NOT. (SIZE(gwkp_env%chi_q_frequency(ikp_local)%values, 3) == bs_env%num_time_freq_points)) THEN
            CPABORT("Internal GW array shape mismatch")
         END IF
      END DO
      n_required_q = bs_env%nkp_scf_desymm
      CPASSERT(n_required_q > 0)
      ! Six distributed solver matrices plus equal workspace reserve; owner buffers remain local.
      ! Select the smallest feasible divisor, independently of the number of q points.
      CALL mp_mem_avail_per_rank_GB(bs_env%para_env, available_GB)
      DO i_group_size = 1, bs_env%para_env%num_pe
         IF (MODULO(bs_env%para_env%num_pe, i_group_size) /= 0) CYCLE
         required_bytes = REAL(n_RI, dp)**2*(32.0_dp + 192.0_dp/REAL(i_group_size, dp)) + &
                          16.0_dp*REAL(n_RI, dp)*32.0_dp
         IF (required_bytes <= 0.5_dp*MAX(0.0_dp, available_GB)*1.0E9_dp) EXIT
      END DO
      IF (i_group_size > bs_env%para_env%num_pe) THEN
         CPABORT('Insufficient node memory for the distributed GW W(q) solver')
      END IF
      n_qgroups = bs_env%para_env%num_pe/i_group_size
      grouped_q = i_group_size > 1
      IF (grouped_q) CALL grouped_w_partition(bs_env%para_env, n_qgroups, my_qgroup, q_roots)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') 'GW W(q) MPI ranks per matrix:', i_group_size
      END IF
      ALLOCATE (gwkp_env%W_PQ_tau(nkp_local))
      ALLOCATE (time_weights(bs_env%num_time_freq_points))
      IF (grouped_q) THEN
         ALLOCATE (W_aux_local_q(0, 0, 0), eps_k_w(0, 0), W_k_w(0, 0))
         IF (nkp_local > 0 .OR. bs_env%para_env%mepos == q_roots(my_qgroup + 1)) THEN
            ALLOCATE (chi_k_w(n_RI, n_RI), MWM_k_w(n_RI, n_RI))
         ELSE
            ALLOCATE (chi_k_w(0, 0), MWM_k_w(0, 0))
         END IF
      ELSE IF (ALL(bs_env%periodic == [1, 1, 1])) THEN
         ALLOCATE (W_aux_local_q(0, 0, 0), eps_k_w(0, 0), W_k_w(0, 0))
         ALLOCATE (chi_k_w(n_RI, n_RI), MWM_k_w(n_RI, n_RI))
      ELSE
         ALLOCATE (W_aux_local_q(n_RI, n_RI, nkp_local), SOURCE=z_zero)
         ALLOCATE (chi_k_w(n_RI, n_RI), eps_k_w(n_RI, n_RI), &
                   MWM_k_w(n_RI, n_RI), W_k_w(n_RI, n_RI))
      END IF
      ALLOCATE (metric_chi_norm(bs_env%nkp_scf_desymm, bs_env%num_time_freq_points), &
                dielectric_min_eigenvalue(bs_env%nkp_scf_desymm, bs_env%num_time_freq_points), &
                dielectric_max_eigenvalue(bs_env%nkp_scf_desymm, bs_env%num_time_freq_points))
      ALLOCATE (M_inv_norm(bs_env%nkp_scf_desymm), &
                V_sqrt_norm(bs_env%nkp_scf_desymm), &
                W_aux_norm(bs_env%nkp_scf_desymm))
      metric_chi_norm = -HUGE(1.0_dp)
      dielectric_min_eigenvalue = HUGE(1.0_dp)
      dielectric_max_eigenvalue = -HUGE(1.0_dp)
      M_inv_norm = 0.0_dp
      V_sqrt_norm = 0.0_dp
      W_aux_norm = 0.0_dp

      IF (grouped_q) THEN
         ALLOCATE (w_group_env)
         CALL w_group_env%from_split(bs_env%para_env, my_qgroup)
         CALL grouped_w_create(w_matrices, w_blacs, w_matrix_struct, w_group_env, n_RI)
         ALLOCATE (frequency_elapsed(bs_env%num_time_freq_points), SOURCE=0.0_dp)
         ! Each group holds one q at a time, reusing its factors for all frequencies.
         ! Ordered world transfers also cover incomplete batches and idle groups.
         DO i_batch = 1, n_required_q, n_qgroups
            n_batch = MIN(n_qgroups, n_required_q - i_batch + 1)
            own_q = 0
            IF (my_qgroup < n_batch) own_q = i_batch + my_qgroup
            DO q_ordinal = 0, n_batch - 1
               ikp = i_batch + q_ordinal
               q_root = q_roots(q_ordinal + 1)
               q_owner = MODULO(ikp, bs_env%para_env%num_pe)
               ikp_local = 1 + (ikp - 1)/bs_env%para_env%num_pe
               DO i_t = 1, 2
                  IF (bs_env%para_env%mepos == q_owner) THEN
                     SELECT CASE (i_t)
                     CASE (1)
                        CALL grouped_w_copy_matrix(bs_env%para_env, n_RI, q_owner, q_root, &
                                                   gwkp_env%metric_coulomb_factor(:, :, ikp_local), chi_k_w)
                     CASE (2)
                        CALL grouped_w_copy_matrix(bs_env%para_env, n_RI, q_owner, q_root, &
                                                   V_sqrt(:, :, ikp_local), chi_k_w)
                     END SELECT
                  ELSE
                     CALL grouped_w_copy_matrix(bs_env%para_env, n_RI, q_owner, q_root, MWM_k_w, chi_k_w)
                  END IF
                  IF (my_qgroup == q_ordinal) CALL grouped_w_set_factor(w_matrices, w_group_env, n_RI, i_t, chi_k_w)
               END DO
               IF (bs_env%para_env%mepos == q_owner) THEN
                  M_inv_norm(ikp) = SQRT(SUM(ABS(M_inv(:, :, ikp_local))**2))
                  V_sqrt_norm(ikp) = SQRT(SUM(ABS(V_sqrt(:, :, ikp_local))**2))
                  ALLOCATE (gwkp_env%W_PQ_tau(ikp_local)%values( &
                            n_RI, n_RI, bs_env%num_time_freq_points), SOURCE=z_zero)
               END IF
            END DO
            DO j_w = 1, bs_env%num_time_freq_points
               step_start = m_walltime()
               CALL timeset('W_screening_on_W_KP', phase_handle)
               time_weights(:) = bs_env%time_frequency_grid%cosine_frequency_to_time(:, j_w)
               DO q_ordinal = 0, n_batch - 1
                  ikp = i_batch + q_ordinal
                  q_owner = MODULO(ikp, bs_env%para_env%num_pe)
                  ikp_local = 1 + (ikp - 1)/bs_env%para_env%num_pe
                  IF (bs_env%para_env%mepos == q_owner) THEN
                     MWM_k_w(:, :) = gwkp_env%chi_q_frequency(ikp_local)%values(:, :, j_w)
                  END IF
                  CALL grouped_w_copy_matrix(bs_env%para_env, n_RI, q_owner, q_roots(q_ordinal + 1), MWM_k_w, chi_k_w)
               END DO
               IF (own_q > 0) THEN
                  CALL grouped_w_solve(w_matrices, w_group_env, n_RI, 0, chi_k_w, &
                                       bs_env%eps_eigval_mat_RI, MWM_k_w, metric_tmp, min_tmp, max_tmp, Wnorm_tmp, &
                                       compute_screened_norm=j_w == 1)
                  IF (w_group_env%mepos == 0) THEN
                     metric_chi_norm(own_q, j_w) = metric_tmp
                     dielectric_min_eigenvalue(own_q, j_w) = min_tmp
                     dielectric_max_eigenvalue(own_q, j_w) = max_tmp
                     IF (j_w == 1) W_aux_norm(own_q) = Wnorm_tmp
                  END IF
               END IF
               DO q_ordinal = 0, n_batch - 1
                  ikp = i_batch + q_ordinal
                  q_owner = MODULO(ikp, bs_env%para_env%num_pe)
                  ikp_local = 1 + (ikp - 1)/bs_env%para_env%num_pe
                  CALL grouped_w_copy_matrix(bs_env%para_env, n_RI, q_roots(q_ordinal + 1), q_owner, MWM_k_w, chi_k_w)
                  IF (bs_env%para_env%mepos == q_owner) THEN
                     CALL accumulate_ri_q_cosine(gwkp_env%W_PQ_tau(ikp_local), chi_k_w, time_weights)
                     IF (j_w == bs_env%num_time_freq_points) DEALLOCATE (gwkp_env%chi_q_frequency(ikp_local)%values)
                  END IF
               END DO
               CALL timestop(phase_handle)
               frequency_elapsed(j_w) = frequency_elapsed(j_w) + m_walltime() - step_start
            END DO
         END DO

         DEALLOCATE (M_inv, V_sqrt)
         DO j_w = 1, bs_env%num_time_freq_points
            CALL bs_env%para_env%max(frequency_elapsed(j_w))
            IF (bs_env%unit_nr > 0) THEN
               WRITE (bs_env%unit_nr, '(T2,A,I3,A,I0,T56,A,T70,F10.1,A)') &
                  'GW W(iω) point ', j_w, ' / ', bs_env%num_time_freq_points, &
                  'Exec. time:', frequency_elapsed(j_w), ' s'
               CALL m_flush(bs_env%unit_nr)
            END IF
         END DO
         DEALLOCATE (frequency_elapsed)
         CALL grouped_w_release(w_matrices, w_matrix_struct, w_blacs)
         CALL mp_para_env_release(w_group_env)
      ELSE IF (ALL(bs_env%periodic == [1, 1, 1])) THEN
         ! One MPI rank per q uses the same filtered dielectric inverse as larger groups.
         ! Reuse its factors across frequencies; different local q counts need no collective.
         ALLOCATE (w_group_env)
         CALL w_group_env%from_split(bs_env%para_env, bs_env%para_env%mepos)
         CALL grouped_w_create(w_matrices, w_blacs, w_matrix_struct, w_group_env, n_RI)
         ALLOCATE (frequency_elapsed(bs_env%num_time_freq_points), SOURCE=0.0_dp)
         ikp_local = 0
         DO ikp = 1, bs_env%nkp_scf_desymm
            IF (MODULO(ikp, bs_env%para_env%num_pe) /= bs_env%para_env%mepos) CYCLE
            ikp_local = ikp_local + 1
            CALL grouped_w_set_factor(w_matrices, w_group_env, n_RI, 1, gwkp_env%metric_coulomb_factor(:, :, ikp_local))
            CALL grouped_w_set_factor(w_matrices, w_group_env, n_RI, 2, V_sqrt(:, :, ikp_local))
            M_inv_norm(ikp) = SQRT(SUM(ABS(M_inv(:, :, ikp_local))**2))
            V_sqrt_norm(ikp) = SQRT(SUM(ABS(V_sqrt(:, :, ikp_local))**2))
            ALLOCATE (gwkp_env%W_PQ_tau(ikp_local)%values(n_RI, n_RI, bs_env%num_time_freq_points), SOURCE=z_zero)
            DO j_w = 1, bs_env%num_time_freq_points
               step_start = m_walltime()
               CALL timeset('W_screening_on_W_KP', phase_handle)
               chi_k_w(:, :) = gwkp_env%chi_q_frequency(ikp_local)%values(:, :, j_w)
               CALL grouped_w_solve(w_matrices, w_group_env, n_RI, 0, chi_k_w, &
                                    bs_env%eps_eigval_mat_RI, MWM_k_w, metric_tmp, min_tmp, max_tmp, Wnorm_tmp, &
                                    compute_screened_norm=j_w == 1)
               metric_chi_norm(ikp, j_w) = metric_tmp
               dielectric_min_eigenvalue(ikp, j_w) = min_tmp
               dielectric_max_eigenvalue(ikp, j_w) = max_tmp
               IF (j_w == 1) W_aux_norm(ikp) = Wnorm_tmp
               CALL timestop(phase_handle)
               CALL timeset('W_response_iomega_2_tau', phase_handle)
               CALL accumulate_ri_q_cosine(gwkp_env%W_PQ_tau(ikp_local), MWM_k_w, &
                                           bs_env%time_frequency_grid%cosine_frequency_to_time(:, j_w))
               CALL timestop(phase_handle)
               frequency_elapsed(j_w) = frequency_elapsed(j_w) + m_walltime() - step_start
            END DO
            DEALLOCATE (gwkp_env%chi_q_frequency(ikp_local)%values)
         END DO
         CPASSERT(ikp_local == nkp_local)
         ! Reduce timings only after unequal local q loops have finished on every rank.
         DO j_w = 1, bs_env%num_time_freq_points
            CALL bs_env%para_env%max(frequency_elapsed(j_w))
            IF (bs_env%unit_nr > 0) THEN
               WRITE (bs_env%unit_nr, '(T2,A,I3,A,I0,T56,A,T70,F10.1,A)') &
                  'GW W(iω) point ', j_w, ' / ', bs_env%num_time_freq_points, &
                  'Exec. time:', frequency_elapsed(j_w), ' s'
               CALL m_flush(bs_env%unit_nr)
            END IF
         END DO
         DEALLOCATE (frequency_elapsed)
         CALL grouped_w_release(w_matrices, w_matrix_struct, w_blacs)
         CALL mp_para_env_release(w_group_env)
      ELSE
         DO j_w = 1, bs_env%num_time_freq_points
            step_start = m_walltime()

            CALL timeset('W_screening_on_W_KP', phase_handle)
            ikp_local = 0
            DO ikp = 1, bs_env%nkp_scf_desymm
               IF (MODULO(ikp, bs_env%para_env%num_pe) /= bs_env%para_env%mepos) CYCLE
               ikp_local = ikp_local + 1
               chi_k_w(:, :) = gwkp_env%chi_q_frequency(ikp_local)%values(:, :, j_w)
               CALL power(chi_k_w, 1.0_dp, bs_env%eps_eigval_mat_RI)
               CALL gemm_square(gwkp_env%metric_coulomb_factor(:, :, ikp_local), 'C', chi_k_w, 'N', &
                                gwkp_env%metric_coulomb_factor(:, :, ikp_local), 'N', eps_k_w)
               metric_chi_norm(ikp, j_w) = SQRT(SUM(ABS(eps_k_w)**2))
               CALL shift_diagonal(eps_k_w, 1.0_dp)
               CALL power(eps_k_w, -1.0_dp, 0.0_dp, &
                          min_ev=dielectric_min_eigenvalue(ikp, j_w), &
                          max_ev=dielectric_max_eigenvalue(ikp, j_w))
               CALL shift_diagonal(eps_k_w, -1.0_dp)
               CALL gemm_square(V_sqrt(:, :, ikp_local), 'N', eps_k_w, 'N', &
                                V_sqrt(:, :, ikp_local), 'C', W_k_w)
               W_aux_local_q(:, :, ikp_local) = W_k_w
               IF (j_w == 1) THEN
                  M_inv_norm(ikp) = SQRT(SUM(ABS(M_inv(:, :, ikp_local))**2))
                  V_sqrt_norm(ikp) = SQRT(SUM(ABS(V_sqrt(:, :, ikp_local))**2))
                  W_aux_norm(ikp) = SQRT(SUM(ABS(W_k_w)**2))
               END IF
            END DO
            CPASSERT(ikp_local == nkp_local)
            CALL timestop(phase_handle)

            CALL timeset('W_response_iomega_2_tau', phase_handle)
            ikp_local = 0
            DO ikp = 1, bs_env%nkp_scf_desymm
               IF (MODULO(ikp, bs_env%para_env%num_pe) /= bs_env%para_env%mepos) CYCLE
               ikp_local = ikp_local + 1
               CALL gemm_square(M_inv(:, :, ikp_local), 'N', &
                                W_aux_local_q(:, :, ikp_local), 'N', &
                                M_inv(:, :, ikp_local), 'N', MWM_k_w)
               IF (j_w == 1) ALLOCATE (gwkp_env%W_PQ_tau(ikp_local)%values( &
                                       n_RI, n_RI, bs_env%num_time_freq_points), SOURCE=z_zero)
               CALL accumulate_ri_q_cosine(gwkp_env%W_PQ_tau(ikp_local), MWM_k_w, &
                                           bs_env%time_frequency_grid%cosine_frequency_to_time(:, j_w))
               IF (j_w == bs_env%num_time_freq_points) DEALLOCATE (gwkp_env%chi_q_frequency(ikp_local)%values)
            END DO
            CPASSERT(ikp_local == nkp_local)
            CALL timestop(phase_handle)
            IF (bs_env%unit_nr > 0) THEN
               WRITE (bs_env%unit_nr, '(T2,A,I3,A,I3,T53,A,T70,F10.1,A)') &
                  'GW W(iω) point ', j_w, ' / ', bs_env%num_time_freq_points, &
                  'Exec. time:', m_walltime() - step_start, ' s'
               CALL m_flush(bs_env%unit_nr)
            END IF
         END DO

      END IF

      ! Release frequency-domain matrices after constructing W(q,tau).
      DEALLOCATE (chi_k_w, eps_k_w, MWM_k_w, W_aux_local_q, W_k_w)
      IF (.NOT. grouped_q) DEALLOCATE (M_inv, V_sqrt)
      W_aux_time_norm = 0.0_dp
      ikp_local = 0
      DO ikp = 1, bs_env%nkp_scf_desymm
         IF (MODULO(ikp, bs_env%para_env%num_pe) /= bs_env%para_env%mepos) CYCLE
         ikp_local = ikp_local + 1
         W_aux_time_norm = W_aux_time_norm + &
                           SUM(ABS(gwkp_env%W_PQ_tau(ikp_local)%values(:, :, 1))**2)
      END DO
      CALL bs_env%para_env%sum(W_aux_time_norm)
      CALL bs_env%para_env%max(metric_chi_norm)
      CALL bs_env%para_env%min(dielectric_min_eigenvalue)
      CALL bs_env%para_env%max(dielectric_max_eigenvalue)
      CALL bs_env%para_env%sum(M_inv_norm)
      CALL bs_env%para_env%sum(V_sqrt_norm)
      CALL bs_env%para_env%sum(W_aux_norm)
      IF (ALL(bs_env%periodic == [1, 1, 0]) .AND. bs_env%unit_nr > 0) THEN
         DO ikp = 1, bs_env%nkp_scf_desymm
            qnorm = SQRT(SUM(MATMUL(twopi*TRANSPOSE(inv_3x3(bs_env%hmat)), &
                                    bs_env%kpoints_chi_eps_W%xkp(:, ikp))**2))
            WRITE (bs_env%unit_nr, '(T2,A,I5,3ES16.7)') &
               'GW2D_DIAG_Q_WC ', ikp, qnorm, W_aux_norm(ikp), V_sqrt_norm(ikp)
         END DO
      END IF
      ALLOCATE (dielectric_condition, MOLD=metric_chi_norm)
      dielectric_condition(:, :) = dielectric_max_eigenvalue/dielectric_min_eigenvalue
      max_metric_chi_index = MAXLOC(metric_chi_norm)
      min_dielectric_index = MINLOC(dielectric_min_eigenvalue)
      max_condition_index = MAXLOC(dielectric_condition)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T73,ES9.1)') &
            'GW response W_PQ(q,iτ_1) norm', &
            SQRT(W_aux_time_norm)
         WRITE (bs_env%unit_nr, '(T2,A,T72,ES9.1)') &
            'GW maximum M_PQ(q) norm:', metric_norm_max
         IF (grouped_q .OR. ALL(bs_env%periodic == [1, 1, 1])) THEN
            WRITE (bs_env%unit_nr, '(T2,A,T74,ES9.1)') &
               'GW lower bound on eigenvalues of ε_PQ(q,iω):', &
               dielectric_min_eigenvalue(min_dielectric_index(1), min_dielectric_index(2))
            WRITE (bs_env%unit_nr, '(T2,A,T74,ES9.1)') &
               'GW upper bound on condition number of ε_PQ(q,iω):', &
               dielectric_condition(max_condition_index(1), max_condition_index(2))
         ELSE
            WRITE (bs_env%unit_nr, '(T2,A,T74,ES9.1)') &
               'GW minimum absolute eigenvalue of ε_PQ(q,iω):', &
               dielectric_min_eigenvalue(min_dielectric_index(1), min_dielectric_index(2))
            WRITE (bs_env%unit_nr, '(T2,A,T74,ES9.1)') &
               'GW maximum condition number of ε_PQ(q,iω):', &
               dielectric_condition(max_condition_index(1), max_condition_index(2))
         END IF
         WRITE (bs_env%unit_nr, '(T2,A,T73,ES9.1)') &
            'GW ||M^-1||_F at maximal transformed χ q:', M_inv_norm(max_metric_chi_index(1))
         WRITE (bs_env%unit_nr, '(T2,A,T73,ES9.1)') &
            'GW ||V^1/2||_F at maximal transformed χ q:', V_sqrt_norm(max_metric_chi_index(1))
         WRITE (bs_env%unit_nr, '(T2,A,T73,ES9.1)') &
            'GW ||W^c||_F at maximal transformed χ q:', W_aux_norm(max_metric_chi_index(1))
      END IF

      CPASSERT(gwkp_env%ngroups == 1 .AND. gwkp_env%my_group == 0)
      CPASSERT(gwkp_env%group_env%num_pe == bs_env%para_env%num_pe)
      CPASSERT(gwkp_env%group_env%mepos == bs_env%para_env%mepos)

      DEALLOCATE (dielectric_condition, dielectric_max_eigenvalue, &
                  dielectric_min_eigenvalue, metric_chi_norm, M_inv_norm, &
                  time_weights, V_sqrt_norm, W_aux_norm)
      phase_elapsed = m_walltime() - phase_start
      CALL bs_env%para_env%max(phase_elapsed)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T69,F12.1)') &
            'GW W(q) wall time [s]:', phase_elapsed
         FLUSH (bs_env%unit_nr)
      END IF
      IF (ALLOCATED(gwkp_env%chi_q_frequency)) DEALLOCATE (gwkp_env%chi_q_frequency)

      CALL timestop(handle)

   END SUBROUTINE compute_W0

! **************************************************************************************************
!> \brief Evaluate exchange and prepare band orbitals for correlation.
!> \param bs_env Band-structure settings and physical results.
!> \param gwkp_env Temporary periodic GW work data.
! **************************************************************************************************
   SUBROUTINE compute_Sigma_x(bs_env, gwkp_env)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gwkp_type), INTENT(INOUT)                     :: gwkp_env

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'compute_Sigma_x'

      COMPLEX(dp), ALLOCATABLE                           :: phi_lmu_k(:, :, :)
      INTEGER                                            :: handle

! AO images are reconstructed only after W has released its large workspace.

      CALL timeset(routineN, handle)

      CALL evaluate_mu_rl_R_values(bs_env, bs_env%ri_rs%grid_points, gwkp_env%ao_values)
      CALL build_all_phi_k(bs_env, gwkp_env%ao_values, phi_lmu_k)
      CALL build_projection_interpolation(bs_env, gwkp_env%interpolation)
      CALL collect_band_bloch_orbitals(bs_env, gwkp_env%ao_values, &
                                       gwkp_env%band_orbitals, phi_lmu_k)
      DEALLOCATE (phi_lmu_k)
      CALL compute_Sigma_x_from_V(bs_env, gwkp_env)
      DEALLOCATE (gwkp_env%ao_values)
      CALL prepare_self_energy_groups(bs_env, gwkp_env)

      CALL timestop(handle)

   END SUBROUTINE compute_Sigma_x

! **************************************************************************************************
!> \brief Improve the conditioning of one common atom-local RI representation on all transfer points.
!> This changes coordinates inside each selected local RI subspace, so its AA fit is unchanged.
!> \param qs_env ...
!> \param bs_env ...
! **************************************************************************************************
   SUBROUTINE optimize_periodic_auto_ri_metric(qs_env, bs_env)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env

      CHARACTER(LEN=*), PARAMETER :: routineN = 'optimize_periodic_auto_ri_metric'

      COMPLEX(KIND=dp), ALLOCATABLE                      :: metric_base(:, :, :), &
                                                            metric_ref(:, :, :), transform(:, :)
      INTEGER                                            :: handle, iatom, ikp, ilocal, irow, istep, &
                                                            itrial, jrow, ncol, nopt, nq, &
                                                            nq_local, nref, nrow
      INTEGER, ALLOCATABLE                               :: metric_cells(:, :)
      REAL(KIND=dp)                                      :: best_worst, elapsed, grad_norm, &
                                                            objective, slope, step, &
                                                            trial_objective, trial_worst, worst
      REAL(KIND=dp), ALLOCATABLE                         :: best(:, :), coefficients(:, :), &
                                                            gradient(:, :), trial(:, :), &
                                                            trial_gradient(:, :)
      TYPE(cp_fm_type), ALLOCATABLE                      :: fm_metric(:)
      TYPE(mp_para_env_type), POINTER                    :: auto_group

      NULLIFY (auto_group)

      CALL timeset(routineN, handle)

      elapsed = m_walltime()
      IF (.NOT. bs_env%do_periodic) THEN
         CPABORT('RI metric optimization requires periodic AUTO_RI')
      END IF
      IF (.NOT. bs_env%auto_ri%enabled .OR. .NOT. bs_env%auto_ri%ready) THEN
         CPABORT('RI metric optimization requires initialized AUTO_RI')
      END IF
      nq = bs_env%nkp_chi_eps_W_orig_plus_extra
      nopt = bs_env%n_RI
      nref = SUM(bs_env%auto_ri%sizes_ref_RI)
      IF (nq < 1 .OR. nopt < 1 .OR. nref < nopt) THEN
         CPABORT('Invalid periodic RI metric optimization dimensions')
      END IF
      nq_local = COUNT([(MODULO(ikp, bs_env%para_env%num_pe) == &
                         bs_env%para_env%mepos, ikp=1, nq)])
      ALLOCATE (metric_ref(nref, nref, nq_local), metric_base(nopt, nopt, nq_local))
      CALL assemble_AA_contraction(bs_env, transform)
      CALL build_ri_2c_lattice(fm_metric, metric_cells, bs_env%ri_metric, &
                               0.0_dp, bs_env, qs_env, .TRUE.)
      CALL ri_2c_lattice_to_owned_q(fm_metric, metric_cells, &
                                    bs_env%kpoints_chi_eps_W%xkp(1:3, 1:nq), metric_ref)
      CALL cp_fm_release(fm_metric)
      DEALLOCATE (metric_cells)
      ilocal = 0
      DO ikp = 1, nq
         IF (MODULO(ikp, bs_env%para_env%num_pe) /= bs_env%para_env%mepos) CYCLE
         ilocal = ilocal + 1
         metric_base(:, :, ilocal) = MATMUL(CONJG(TRANSPOSE(transform)), &
                                            MATMUL(metric_ref(:, :, ilocal), transform))
         metric_base(:, :, ilocal) = 0.5_dp*(metric_base(:, :, ilocal) + &
                                             CONJG(TRANSPOSE(metric_base(:, :, ilocal))))
      END DO
      DEALLOCATE (metric_ref)

      CALL prepare_auto_ri_q_group(bs_env, metric_base, auto_group)

      ALLOCATE (coefficients(nopt, nopt), best(nopt, nopt), &
                trial(nopt, nopt), gradient(nopt, nopt), trial_gradient(nopt, nopt))
      coefficients(:, :) = 0.0_dp
      DO irow = 1, nopt
         coefficients(irow, irow) = 1.0_dp
      END DO
      CALL evaluate_auto_ri_condition(bs_env, metric_base, coefficients, &
                                      objective, worst, gradient, auto_group)
      best(:, :) = coefficients
      best_worst = worst
      DO istep = 1, 30
         grad_norm = SQRT(SUM(gradient*gradient))
         IF (grad_norm < 1.0E-8_dp) EXIT
         step = 1.0_dp/MAX(1.0_dp, MAXVAL(ABS(gradient)))
         slope = SUM(gradient*gradient)
         DO itrial = 1, 12
            trial(:, :) = coefficients - step*gradient
            CALL evaluate_auto_ri_condition(bs_env, metric_base, trial, &
                                            trial_objective, trial_worst, trial_gradient, auto_group)
            IF (trial_objective <= objective - 1.0E-4_dp*step*slope) EXIT
            step = 0.5_dp*step
         END DO
         IF (itrial > 12) EXIT
         coefficients(:, :) = trial
         gradient(:, :) = trial_gradient
         objective = trial_objective
         worst = trial_worst
         IF (worst < best_worst) THEN
            best_worst = worst
            best(:, :) = coefficients
         END IF
      END DO
      IF (bs_env%unit_nr > 0) WRITE (bs_env%unit_nr, '(T2,A,T72,ES9.1E2)') &
         'Completed Auto RI, max. condition number of M_PQ metric matrix:', best_worst

      irow = 1
      DO iatom = 1, bs_env%n_atom
         nrow = bs_env%auto_ri%sizes_ref_RI(iatom)
         ncol = bs_env%auto_ri%sizes_opt_RI(iatom)
         jrow = bs_env%auto_ri%U_Pp_atom_offset(iatom)
         bs_env%auto_ri%U_Pp_atom(jrow:jrow + nrow*ncol - 1) = &
            RESHAPE(MATMUL(RESHAPE(bs_env%auto_ri%U_Pp_atom( &
                                   jrow:jrow + nrow*ncol - 1), [nrow, ncol]), &
                           best(irow:irow + ncol - 1, irow:irow + ncol - 1)), [nrow*ncol])
         irow = irow + ncol
      END DO
      elapsed = m_walltime() - elapsed
      CALL bs_env%para_env%max(elapsed)
      IF (bs_env%unit_nr > 0) WRITE (bs_env%unit_nr, '(T2,A,T69,F10.1,A)') &
         'Auto RI metric optimization, exec. time:', elapsed, ' s'
      IF (ASSOCIATED(auto_group)) CALL mp_para_env_release(auto_group)

      CALL timestop(handle)

   END SUBROUTINE optimize_periodic_auto_ri_metric

! **************************************************************************************************
!> \brief Assembles the q-independent, atom-diagonal AA contraction from stored coefficients.
!> \param bs_env ...
!> \param U_Pp Atom-diagonal reference-to-optimized AA contraction.
! **************************************************************************************************
   SUBROUTINE assemble_AA_contraction(bs_env, U_Pp)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: U_Pp(:, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'assemble_AA_contraction'

      INTEGER                                            :: first, first_opt, first_ref, handle, &
                                                            iatom, last, nopt, nref

      CALL timeset(routineN, handle)

      IF (.NOT. bs_env%auto_ri%ready .OR. &
          .NOT. ALLOCATED(bs_env%auto_ri%sizes_ref_RI)) THEN
         CPABORT('Periodic AUTO_RI contraction is not initialized')
      END IF
      ALLOCATE (U_Pp(SUM(bs_env%auto_ri%sizes_ref_RI), &
                     SUM(bs_env%auto_ri%sizes_opt_RI)), source=z_zero)
      first_ref = 1
      first_opt = 1
      DO iatom = 1, bs_env%n_atom
         nref = bs_env%auto_ri%sizes_ref_RI(iatom)
         nopt = bs_env%auto_ri%sizes_opt_RI(iatom)
         first = bs_env%auto_ri%U_Pp_atom_offset(iatom)
         last = first + nref*nopt - 1
         U_Pp(first_ref:first_ref + nref - 1, first_opt:first_opt + nopt - 1) = &
            RESHAPE(bs_env%auto_ri%U_Pp_atom(first:last), [nref, nopt])
         first_ref = first_ref + nref
         first_opt = first_opt + nopt
      END DO
      IF (ANY(SUM(ABS(U_Pp)**2, DIM=1) <= TINY(1.0_dp))) THEN
         CPABORT('Periodic AUTO_RI contraction has an empty optimized direction')
      END IF

      CALL timestop(handle)

   END SUBROUTINE assemble_AA_contraction

! **************************************************************************************************
!> \brief Evaluate the smooth scale-invariant metric objective and its local-block gradient.
!> \param bs_env ...
!> \param metric_base ...
!> \param coefficients ...
!> \param objective ...
!> \param worst ...
!> \param gradient ...
!> \param auto_group ...
! **************************************************************************************************
   SUBROUTINE evaluate_auto_ri_condition(bs_env, metric_base, coefficients, &
                                         objective, worst, gradient, auto_group)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: metric_base(:, :, :)
      REAL(KIND=dp), INTENT(IN)                          :: coefficients(:, :)
      REAL(KIND=dp), INTENT(OUT)                         :: objective, worst
      REAL(KIND=dp), CONTIGUOUS, INTENT(OUT)             :: gradient(:, :)
      TYPE(mp_para_env_type), POINTER                    :: auto_group

      COMPLEX(KIND=dp), ALLOCATABLE                      :: inverse(:, :), projected(:, :), &
                                                            response(:, :)
      INTEGER                                            :: handle, iatom, ilocal, jrow, n, &
                                                            n_singular, ncol, nlocal, nq, &
                                                            phase_handle
      LOGICAL                                            :: singular
      REAL(KIND=dp)                                      :: largest, smallest, trace_h, trace_inverse
      REAL(KIND=dp), ALLOCATABLE                         :: eigval(:)

      CALL timeset('auto_ri_condition_evaluate', handle)

      n = SIZE(coefficients, 1)
      nq = bs_env%nkp_chi_eps_W_orig_plus_extra
      nlocal = SIZE(metric_base, 3)
      objective = 0.0_dp
      worst = 0.0_dp
      gradient(:, :) = 0.0_dp
      n_singular = 0
      CALL timeset("auto_ri_local_q_work", phase_handle)
      IF (ASSOCIATED(auto_group)) THEN
         CALL evaluate_auto_ri_group(bs_env, auto_group, metric_base, coefficients, &
                                     objective, worst, gradient, n_singular)
      ELSE
         !$OMP PARALLEL DEFAULT(NONE) IF(nlocal > 1) &
         !$OMP SHARED(bs_env, coefficients, metric_base, n, nlocal, gradient, n_singular, objective, worst) &
         !$OMP PRIVATE(eigval, ilocal, inverse, jrow, largest, &
         !$OMP         projected, response, smallest, trace_h, trace_inverse, singular)
         ALLOCATE (projected(n, n), inverse(n, n), response(n, n), eigval(n))
         !$OMP DO SCHEDULE(STATIC, 1) ORDERED
         DO ilocal = 1, nlocal
            CALL auto_ri_project_atom_blocks(bs_env%auto_ri%sizes_opt_RI, coefficients, &
                                             metric_base(:, :, ilocal), projected, response)
            projected(:, :) = 0.5_dp*(projected + CONJG(TRANSPOSE(projected)))
            CALL auto_ri_metric_inverse(projected, eigval, inverse)
            largest = MAXVAL(eigval)
            smallest = MINVAL(eigval)
            singular = smallest <= 1.0E-12_dp*largest .OR. largest <= 0.0_dp
            IF (.NOT. singular) THEN
               trace_h = SUM(eigval)
               trace_inverse = SUM(1.0_dp/eigval)

               CALL auto_ri_block_gradient(bs_env, coefficients, projected, inverse, &
                                           trace_h, trace_inverse, response)
            END IF
            ! Keep the q sum independent of thread timing; ill-conditioned metrics
            ! amplify otherwise tiny reduction-order differences during optimization.
            !$OMP ORDERED
            IF (singular) THEN
               n_singular = n_singular + 1
            ELSE
               worst = MAX(worst, largest/smallest)
               objective = objective + LOG(trace_h*trace_inverse/REAL(n*n, dp))
               gradient(:, :) = gradient + 2.0_dp*REAL(response, dp)
            END IF
            !$OMP END ORDERED
         END DO
         !$OMP END DO
         DEALLOCATE (projected, inverse, response, eigval)
         !$OMP END PARALLEL
      END IF
      CALL timestop(phase_handle)
      IF (n_singular > 0) THEN
         objective = HUGE(1.0_dp)
         worst = HUGE(1.0_dp)
         gradient(:, :) = 0.0_dp
      END IF
      CALL timeset("auto_ri_global_reduction", phase_handle)
      CALL bs_env%para_env%sum(objective)
      CALL bs_env%para_env%max(worst)
      CALL bs_env%para_env%sum(gradient)
      CALL timestop(phase_handle)
      objective = objective/REAL(nq, dp)
      gradient(:, :) = gradient/REAL(nq, dp)
      jrow = 1
      DO iatom = 1, bs_env%n_atom
         ncol = bs_env%auto_ri%sizes_opt_RI(iatom)
         gradient(jrow:jrow + ncol - 1, :jrow - 1) = 0.0_dp
         gradient(jrow:jrow + ncol - 1, jrow + ncol:) = 0.0_dp
         jrow = jrow + ncol
      END DO

      CALL timestop(handle)

   END SUBROUTINE evaluate_auto_ri_condition

! **************************************************************************************************
!> \brief Assign all ranks to balanced q groups only when there are fewer q than ranks.
!> Move each existing M(q) to its group root once; do not replicate it on helpers.
!> \param bs_env ...
!> \param metric_base ...
!> \param group_env ...
! **************************************************************************************************
   SUBROUTINE prepare_auto_ri_q_group(bs_env, metric_base, group_env)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(dp), ALLOCATABLE, INTENT(INOUT)            :: metric_base(:, :, :)
      TYPE(mp_para_env_type), POINTER                    :: group_env

      COMPLEX(dp), ALLOCATABLE                           :: empty(:, :), owned(:, :, :)
      INTEGER                                            :: ilocal, iq, mygroup, n, nq, owner
      INTEGER, ALLOCATABLE                               :: roots(:)

      nq = bs_env%nkp_chi_eps_W_orig_plus_extra
      NULLIFY (group_env)
      IF (nq >= bs_env%para_env%num_pe) RETURN
      n = SIZE(metric_base, 1)
      CALL grouped_w_partition(bs_env%para_env, nq, mygroup, roots)
      ALLOCATE (group_env)
      CALL group_env%from_split(bs_env%para_env, mygroup)
      ALLOCATE (owned(n, n, MERGE(1, 0, group_env%mepos == 0)), empty(0, 0))
      DO iq = 1, nq
         owner = MODULO(iq, bs_env%para_env%num_pe)
         ilocal = (iq - 1)/bs_env%para_env%num_pe + 1
         IF (bs_env%para_env%mepos == owner) THEN
            IF (group_env%mepos == 0) THEN
               CALL grouped_w_copy_matrix(bs_env%para_env, n, owner, roots(iq), &
                                          metric_base(:, :, ilocal), owned(:, :, 1))
            ELSE
               CALL grouped_w_copy_matrix(bs_env%para_env, n, owner, roots(iq), &
                                          metric_base(:, :, ilocal), empty)
            END IF
         ELSE IF (bs_env%para_env%mepos == roots(iq)) THEN
            CALL grouped_w_copy_matrix(bs_env%para_env, n, owner, roots(iq), empty, owned(:, :, 1))
         END IF
      END DO
      CALL MOVE_ALLOC(owned, metric_base)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') 'AUTO_RI distributed q groups', nq
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') 'AUTO_RI maximum MPI ranks per q group', &
            (bs_env%para_env%num_pe + nq - 1)/nq
         FLUSH (bs_env%unit_nr)
      END IF
   END SUBROUTINE prepare_auto_ri_q_group

! **************************************************************************************************
!> \brief Evaluate one group's q contribution with distributed spectrum and inverse.
!> The root forms H=C^T M C and its atom-block gradient; helper ranks participate
!> in the distributed solves but contribute no duplicate objective or gradient.
!> \param bs_env ...
!> \param group_env ...
!> \param metric_base ...
!> \param coefficients ...
!> \param objective ...
!> \param worst ...
!> \param gradient ...
!> \param n_singular ...
! **************************************************************************************************
   SUBROUTINE evaluate_auto_ri_group(bs_env, group_env, metric_base, coefficients, &
                                     objective, worst, gradient, n_singular)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(mp_para_env_type), POINTER                    :: group_env
      COMPLEX(dp), INTENT(IN)                            :: metric_base(:, :, :)
      REAL(dp), INTENT(IN)                               :: coefficients(:, :)
      REAL(dp), INTENT(OUT)                              :: objective, worst, gradient(:, :)
      INTEGER, INTENT(OUT)                               :: n_singular

      COMPLEX(dp), ALLOCATABLE                           :: inverse(:, :), projected(:, :), &
                                                            response(:, :)
      INTEGER                                            :: n, nowner
      REAL(dp)                                           :: largest, smallest, trace_h, trace_inverse
      REAL(dp), ALLOCATABLE                              :: eigenvalues(:)

      n = SIZE(coefficients, 1)
      nowner = MERGE(n, 0, group_env%mepos == 0)
      ALLOCATE (projected(nowner, nowner), inverse(nowner, nowner), &
                response(nowner, nowner), eigenvalues(n))
      objective = 0.0_dp
      worst = 0.0_dp
      gradient(:, :) = 0.0_dp
      n_singular = 0
      IF (group_env%mepos == 0) THEN
         CALL auto_ri_project_atom_blocks(bs_env%auto_ri%sizes_opt_RI, coefficients, &
                                          metric_base(:, :, 1), projected, response)
         projected(:, :) = 0.5_dp*(projected + CONJG(TRANSPOSE(projected)))
      END IF
      CALL auto_ri_grouped_inverse(group_env, projected, eigenvalues, inverse)
      IF (group_env%mepos /= 0) RETURN
      largest = MAXVAL(eigenvalues)
      smallest = MINVAL(eigenvalues)
      IF (smallest <= 1.0E-12_dp*largest .OR. largest <= 0.0_dp) THEN
         n_singular = 1
         RETURN
      END IF
      trace_h = SUM(eigenvalues)
      trace_inverse = SUM(1.0_dp/eigenvalues)
      CALL auto_ri_block_gradient(bs_env, coefficients, projected, inverse, &
                                  trace_h, trace_inverse, response)
      worst = largest/smallest
      objective = LOG(trace_h*trace_inverse/REAL(n*n, dp))
      gradient(:, :) = 2.0_dp*REAL(response, dp)
   END SUBROUTINE evaluate_auto_ri_group

! **************************************************************************************************
!> \brief Form H_AB=C_A^T M_AB C_B using the allowed atom-diagonal coefficient blocks.
!> C_A transforms the RI functions on atom A. The two block products cost
!> O(N_RI^2 max_A n_A), instead of multiplying the zero off-atom blocks of C.
!> The caller supplies one existing complex workspace; only one n_A^2 block is added.
!> \param sizes ...
!> \param coefficients ...
!> \param metric ...
!> \param projected ...
!> \param work ...
! **************************************************************************************************
   SUBROUTINE auto_ri_project_atom_blocks(sizes, coefficients, metric, projected, work)
      INTEGER, INTENT(IN)                                :: sizes(:)
      REAL(dp), INTENT(IN)                               :: coefficients(:, :)
      COMPLEX(dp), INTENT(IN) :: metric(SIZE(coefficients, 1), SIZE(coefficients, 1))
      COMPLEX(dp), INTENT(OUT) :: projected(SIZE(coefficients, 1), SIZE(coefficients, 1)), &
         work(SIZE(coefficients, 1), SIZE(coefficients, 1))

      COMPLEX(dp), ALLOCATABLE                           :: atom_coefficients(:, :)
      INTEGER                                            :: handle, iatom, ifirst, ilast, n, nblock

      CALL timeset('auto_ri_project_atom_blocks', handle)

      n = SIZE(coefficients, 1)
      CPASSERT(n > 0 .AND. SIZE(coefficients, 2) == n)
      CPASSERT(SIZE(sizes) > 0)
      CPASSERT(ALL(sizes >= 0) .AND. SUM(sizes) == n)
      CPASSERT(SIZE(metric, 1) == n .AND. SIZE(metric, 2) == n)
      CPASSERT(SIZE(projected, 1) == n .AND. SIZE(projected, 2) == n)
      CPASSERT(SIZE(work, 1) == n .AND. SIZE(work, 2) == n)
      ALLOCATE (atom_coefficients(MAXVAL(sizes), MAXVAL(sizes)))
      ifirst = 1
      DO iatom = 1, SIZE(sizes)
         nblock = sizes(iatom)
         IF (nblock == 0) CYCLE
         ilast = ifirst + nblock - 1
         atom_coefficients(:nblock, :nblock) = CMPLX(coefficients(ifirst:ilast, ifirst:ilast), 0.0_dp, dp)
         CALL zgemm('N', 'N', n, nblock, nblock, z_one, metric(1, ifirst), n, &
                    atom_coefficients, SIZE(atom_coefficients, 1), z_zero, work(1, ifirst), n)
         ifirst = ilast + 1
      END DO
      ifirst = 1
      DO iatom = 1, SIZE(sizes)
         nblock = sizes(iatom)
         IF (nblock == 0) CYCLE
         ilast = ifirst + nblock - 1
         atom_coefficients(:nblock, :nblock) = CMPLX(coefficients(ifirst:ilast, ifirst:ilast), 0.0_dp, dp)
         CALL zgemm('T', 'N', nblock, n, nblock, z_one, &
                    atom_coefficients, SIZE(atom_coefficients, 1), work(ifirst, 1), n, &
                    z_zero, projected(ifirst, 1), n)
         ifirst = ilast + 1
      END DO
      DEALLOCATE (atom_coefficients)

      CALL timestop(handle)

   END SUBROUTINE auto_ri_project_atom_blocks

! **************************************************************************************************
!> \brief Compute the spectrum and inverse of H=CᵀMC without constructing eigenvectors.
!> Keep the existing λ_min > 10^-12 λ_max positivity test. Cholesky computes H^-1
!> only for accepted matrices; rejected optimizer trials retain the original handling.
!> \param matrix ...
!> \param eigenvalues ...
!> \param inverse ...
! **************************************************************************************************
   SUBROUTINE auto_ri_metric_inverse(matrix, eigenvalues, inverse)
      COMPLEX(dp), INTENT(IN)                            :: matrix(:, :)
      REAL(dp), INTENT(OUT)                              :: eigenvalues(:)
      COMPLEX(dp), INTENT(OUT)                           :: inverse(:, :)

      COMPLEX(dp), ALLOCATABLE                           :: work(:)
      INTEGER                                            :: handle, icol, info, irow, liwork, &
                                                            lrwork, lwork, n
      INTEGER, ALLOCATABLE                               :: iwork(:)
      REAL(dp), ALLOCATABLE                              :: rwork(:)

      n = SIZE(matrix, 1)
      CPASSERT(n > 0 .AND. SIZE(matrix, 2) == n)
      CPASSERT(SIZE(eigenvalues) == n)
      CPASSERT(SIZE(inverse, 1) == n .AND. SIZE(inverse, 2) == n)

      CALL timeset('auto_ri_metric_spectrum', handle)

      ALLOCATE (work(1), rwork(1), iwork(1))
      inverse = matrix
      CALL zheevd('N', 'U', n, inverse, n, eigenvalues, work, -1, rwork, -1, iwork, -1, info)
      IF (info /= 0) CPABORT('AUTO_RI eigenvalue workspace query failed')
      lwork = CEILING(REAL(work(1), dp))
      lrwork = CEILING(rwork(1))
      liwork = iwork(1)
      DEALLOCATE (work, rwork, iwork)
      ALLOCATE (work(lwork), rwork(lrwork), iwork(liwork))
      inverse = matrix
      CALL zheevd('N', 'U', n, inverse, n, eigenvalues, work, lwork, rwork, lrwork, iwork, liwork, info)
      IF (info /= 0) CPABORT('AUTO_RI metric eigenvalue calculation failed')
      DEALLOCATE (work, rwork, iwork)

      CALL timestop(handle)

      inverse = z_zero
      IF (MAXVAL(eigenvalues) <= 0.0_dp) RETURN
      IF (MINVAL(eigenvalues) <= 1.0E-12_dp*MAXVAL(eigenvalues)) RETURN
      inverse = matrix

      CALL timeset('auto_ri_metric_cholesky', handle)

      CALL zpotrf('U', n, inverse, n, info)
      IF (info /= 0) CPABORT('AUTO_RI positive metric factorization failed')
      CALL zpotri('U', n, inverse, n, info)
      IF (info /= 0) CPABORT('AUTO_RI metric inversion failed')
      DO icol = 1, n
         DO irow = icol + 1, n
            inverse(irow, icol) = CONJG(inverse(icol, irow))
         END DO
      END DO

      CALL timestop(handle)

   END SUBROUTINE auto_ri_metric_inverse

! **************************************************************************************************
!> \brief Differentiate log[tr(H)tr(H^-1)/n²] for the allowed atomwise blocks of C.
!> With H=CᵀMC, G=2C^-ᵀ Re[H/tr(H)-H^-1/tr(H^-1)]. Solve each block of
!> CᵀG=2Re[H/tr(H)-H^-1/tr(H^-1)], avoiding H^-2 and cancellation in MC H^-2.
!> \param bs_env ...
!> \param coefficients ...
!> \param matrix ...
!> \param inverse ...
!> \param trace_h ...
!> \param trace_inverse ...
!> \param response ...
! **************************************************************************************************
   SUBROUTINE auto_ri_block_gradient(bs_env, coefficients, matrix, inverse, trace_h, trace_inverse, response)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(dp), INTENT(IN)                               :: coefficients(:, :)
      COMPLEX(dp), INTENT(IN)                            :: matrix(:, :), inverse(:, :)
      REAL(dp), INTENT(IN)                               :: trace_h, trace_inverse
      COMPLEX(dp), INTENT(OUT)                           :: response(:, :)

      INTEGER                                            :: first, handle, iatom, info, last, n
      INTEGER, ALLOCATABLE                               :: pivots(:)
      REAL(dp), ALLOCATABLE                              :: block(:, :), gradient(:, :)

      CALL timeset('auto_ri_atom_gradient', handle)

      response = z_zero
      first = 1
      DO iatom = 1, bs_env%n_atom
         n = bs_env%auto_ri%sizes_opt_RI(iatom)
         last = first + n - 1
         CPASSERT(n >= 0 .AND. last <= SIZE(coefficients, 1))
         IF (n == 0) CYCLE
         ALLOCATE (block(n, n), gradient(n, n), pivots(n))
         block(:, :) = TRANSPOSE(coefficients(first:last, first:last))
         gradient(:, :) = 2.0_dp*REAL(matrix(first:last, first:last)/trace_h - &
                                      inverse(first:last, first:last)/trace_inverse, dp)
         CALL dgesv(n, n, block, n, pivots, gradient, n, info)
         IF (info /= 0) CPABORT('AUTO_RI accepted coefficient block is singular')
         ! The caller retains its factor of two in accumulating the q gradient.
         response(first:last, first:last) = CMPLX(0.5_dp*gradient, 0.0_dp, dp)
         DEALLOCATE (block, gradient, pivots)
         first = last + 1
      END DO
      CPASSERT(first == SIZE(coefficients, 1) + 1)

      CALL timestop(handle)

   END SUBROUTINE auto_ri_block_gradient

! **************************************************************************************************
!> \brief Use all MPI ranks as one memory-minimal RI-RS grid group.
!>
!> The routine estimates the larger of the chi, W, exchange, and correlation-self-energy peaks.
!> Chi and W initially use the full global communicator.  After W is built,
!> prepare_self_energy_groups chooses smaller time groups from the live memory budget.
!> W(i omega) is transformed to W_PQ(q,i tau) while q remains distributed over all ranks:
!>
!>   H_chi(g) = H_distributed/g + H_replicated + n_owned H_time,
!>   H_W(P)   = H_auxiliary-time/P + H_auxiliary-work/P + H_stream,
!>   H_Sigma(g) = H_grid/g + H_auxiliary-time/g + H_replicated + n_owned H_time.
!>
!> where H_stream is a bounded auxiliary-basis image batch.  Only the current imaginary-time
!> W_ll'(R,i tau) is materialized; it is consumed immediately by Sigma_c.
!>
!> All ranks form one group and exactly one distributed copy of every k/q-dependent input is
!> retained.  This is the minimum-total-memory layout and also maximizes the ranks over which grid
!> rows are distributed.  The reserve covers MPI and library workspaces; allocations already live
!> when this routine is called are included in the measured unavailable memory.
!> \param bs_env ...
!> \param n_grid ...
!> \param group_env ...
!> \param group_size ...
!> \param ngroups ...
!> \param my_group ...
! **************************************************************************************************
   SUBROUTINE create_ri_rs_grid_groups(bs_env, n_grid, group_env, group_size, ngroups, my_group)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: n_grid
      TYPE(mp_para_env_type), POINTER                    :: group_env
      INTEGER, INTENT(OUT)                               :: group_size, ngroups, my_group

      CHARACTER(len=*), PARAMETER :: memory_error = &
         'Periodic RI-RS memory estimate exceeds available memory after the workspace reserve', &
         routineN = 'create_ri_rs_grid_groups'
      REAL(KIND=dp), PARAMETER                           :: max_memory_reserve_GB = 2.0_dp, &
                                                            min_memory_reserve_GB = 0.5_dp

      INTEGER                                            :: handle, nowned, nthreads
      INTEGER(int_8)                                     :: used_bytes
      REAL(KIND=dp) :: bytes_per_rank, chi_distributed_bytes, chi_stream_bytes, chi_time_bytes, &
         estimated_peak_bytes, estimated_total_bytes, mem_avail_GB, mem_used_GB, &
         memory_reserve_GB, n_aux, n_full, sigma_c_distributed_bytes, sigma_c_owned_bytes, &
         sigma_c_replicated_bytes, sigma_x_replicated_bytes, W_aux_group_bytes, W_aux_total_bytes, &
         W_aux_world_bytes, W_owned_bytes, W_replicated_bytes

! Account separately for arrays distributed by grid row, arrays replicated on every
! rank, and results retained for each imaginary-time point owned by a rank.

      CALL timeset(routineN, handle)

      n_aux = REAL(bs_env%n_RI + 1, dp)
      n_full = REAL(PRODUCT(bs_env%kpoints_scf_desymm%nkp_grid), dp)
      chi_time_bytes = 16.0_dp*n_aux**2* &
                       REAL(bs_env%nkp_scf_desymm*bs_env%num_time_freq_points, dp)
      chi_distributed_bytes = REAL(n_grid, dp)**2* &
                              (32.0_dp*REAL(bs_env%nkp_scf_desymm, dp) + &
                               16.0_dp*n_full + &
                               48.0_dp*REAL(bs_env%nkp_scf_desymm, dp)) + &
                              16.0_dp*REAL(n_grid, dp)*n_aux + &
                              48.0_dp*n_aux**2* &
                              REAL(bs_env%nkp_scf_desymm, dp) + chi_time_bytes
      ! Temporary RI projection matrices or thread-private weighted shared orbitals.
      nthreads = 1
!$    nthreads = omp_get_max_threads()
      chi_stream_bytes = MAX(16.0_dp*n_aux**2, &
                             16.0_dp*REAL(n_grid, dp)*REAL(bs_env%n_ao*nthreads, dp))
      ! Exchange grid rows use the world communicator, not a tau group.  The three reciprocal
      ! complex grids and two complete-cell complex grids are therefore divided over all ranks.
      sigma_x_replicated_bytes = REAL(n_grid, dp)**2/REAL(bs_env%para_env%num_pe, dp)* &
                                 (48.0_dp*REAL(bs_env%nkp_scf_desymm, dp) + &
                                  32.0_dp*n_full) + &
                                 16.0_dp*REAL(n_grid, dp)*REAL(bs_env%n_ao, dp) + &
                                 16.0_dp*REAL(bs_env%n_ao, dp)**2 + &
                                 8.0_dp*REAL(bs_env%n_ao, dp)**2* &
                                 REAL(bs_env%nimages_scf_desymm, dp) + &
                                 16.0_dp*REAL(n_grid, dp)* &
                                 (n_aux + REAL(bs_env%n_ao, dp))/ &
                                 REAL(bs_env%para_env%num_pe, dp) + &
                                 48.0_dp*n_aux**2
      sigma_c_distributed_bytes = REAL(n_grid, dp)**2* &
                                  (16.0_dp*(3.0_dp*REAL(bs_env%nkp_scf_desymm, dp) + &
                                            n_full + 1.0_dp) + &
                                   40.0_dp*n_full + &
                                   48.0_dp*REAL(bs_env%nkp_scf_desymm, dp)) + &
                                  16.0_dp*REAL(n_grid, dp)*n_aux
      sigma_c_replicated_bytes = 8.0_dp*REAL(bs_env%n_ao, dp)**2* &
                                 REAL(bs_env%nimages_scf_desymm, dp)
      sigma_c_owned_bytes = 16.0_dp*REAL(bs_env%n_ao, dp)**2* &
                            REAL(bs_env%nimages_scf_desymm*bs_env%n_spin, dp)
      ! W_PQ(q,i tau) uses the global q distribution until the bounded transpose to tau groups.
      ! Projection to W_ll'(q/R,i tau) occurs inside Sigma_c for one time point at a time.
      W_aux_world_bytes = 16.0_dp*n_aux**2* &
                          REAL(bs_env%nkp_scf_desymm*bs_env%num_time_freq_points, dp)/ &
                          REAL(bs_env%para_env%num_pe, dp)
      W_owned_bytes = W_aux_world_bytes + n_aux**2* &
                      (64.0_dp*REAL((bs_env%nkp_chi_eps_W_orig_plus_extra + &
                                     bs_env%para_env%num_pe - 1)/bs_env%para_env%num_pe, dp) + &
                       32.0_dp*REAL((bs_env%nkp_scf_desymm + bs_env%para_env%num_pe - 1)/ &
                                    bs_env%para_env%num_pe, dp)) + &
                      chi_time_bytes/REAL(bs_env%para_env%num_pe, dp)
      ! Streaming W_PQ(R) in fixed image batches bounds the only auxiliary real-space buffer;
      ! no rank holds a complete W_PQ(R) or W_ll'(q/R) object.
      W_replicated_bytes = n_aux**2* &
                           (8.0_dp*REAL(MIN(32, PRODUCT(bs_env%kpoints_scf_desymm%nkp_grid)), dp) + &
                            96.0_dp)
      CALL m_memory(used_bytes)
      mem_used_GB = REAL(used_bytes, dp)*1.0E-9_dp
      CALL bs_env%para_env%max(mem_used_GB)
      CALL mp_mem_avail_per_rank_GB(bs_env%para_env, mem_avail_GB)
      IF (mem_avail_GB > 0.0_dp) THEN
         ! Reserve at least half of the remaining memory for MPI and library workspaces,
         ! but do not demand 2 GB per rank when the node has less memory available.
         memory_reserve_GB = MIN(max_memory_reserve_GB, &
                                 MAX(min_memory_reserve_GB, 0.5_dp*mem_avail_GB))
         bytes_per_rank = MAX(0.0_dp, mem_avail_GB - memory_reserve_GB)*1.0E9_dp
      ELSE
         memory_reserve_GB = 0.0_dp
         bytes_per_rank = 8.0_dp*1024.0_dp**3
      END IF
      group_size = bs_env%para_env%num_pe
      ngroups = 1
      my_group = 0
      nowned = (bs_env%num_time_freq_points + group_size - 1)/group_size
      W_aux_group_bytes = 16.0_dp*n_aux**2* &
                          REAL(bs_env%nkp_scf_desymm*nowned, dp)/REAL(group_size, dp)
      estimated_peak_bytes = MAX( &
                             chi_distributed_bytes/REAL(group_size, dp) + &
                             chi_stream_bytes, &
                             W_replicated_bytes + W_owned_bytes, &
                             sigma_x_replicated_bytes, &
                             sigma_c_distributed_bytes/REAL(group_size, dp) + &
                             sigma_c_replicated_bytes + &
                             REAL(nowned, dp)*sigma_c_owned_bytes + W_aux_group_bytes)
      IF (estimated_peak_bytes > bytes_per_rank) THEN
         IF (bs_env%unit_nr > 0) THEN
            WRITE (bs_env%unit_nr, '(T2,A,T72,F9.2)') &
               'Periodic RI-RS available memory per rank [GB]', mem_avail_GB
            WRITE (bs_env%unit_nr, '(T2,A,T72,F9.2)') &
               'Periodic RI-RS workspace reserve per rank [GB]', memory_reserve_GB
            WRITE (bs_env%unit_nr, '(T2,A,T72,F9.2)') &
               'Periodic RI-RS usable memory after workspace reserve [GB]', &
               bytes_per_rank*1.0E-9_dp
            WRITE (bs_env%unit_nr, '(T2,A,T72,F9.2)') &
               'Periodic RI-RS minimum additional peak per rank [GB]', &
               estimated_peak_bytes*1.0E-9_dp
         END IF
         CPWARN(memory_error)
      END IF
      W_aux_total_bytes = 16.0_dp*n_aux**2* &
                          REAL(bs_env%nkp_scf_desymm*bs_env%num_time_freq_points, dp)
      estimated_total_bytes = mem_used_GB*1.0E9_dp*REAL(bs_env%para_env%num_pe, dp) + &
                              MAX(chi_distributed_bytes + &
                                  8.0_dp*n_aux**2* &
                                  REAL(bs_env%para_env%num_pe + 1, dp), &
                                  REAL(bs_env%para_env%num_pe, dp)* &
                                  (W_replicated_bytes + W_owned_bytes), &
                                  REAL(bs_env%para_env%num_pe, dp)*sigma_x_replicated_bytes, &
                                  sigma_c_distributed_bytes + &
                                  REAL(bs_env%para_env%num_pe, dp)*sigma_c_replicated_bytes + &
                                  REAL(bs_env%num_time_freq_points, dp)*sigma_c_owned_bytes + &
                                  W_aux_total_bytes)
      ALLOCATE (group_env)
      CALL group_env%from_split(bs_env%para_env, my_group)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T69,I9,A)') &
            'Estimated total peak memory (all MPI ranks):', &
            NINT(estimated_total_bytes*1.0E-9_dp), ' GB'
      END IF

      CALL timestop(handle)

   END SUBROUTINE create_ri_rs_grid_groups

! **************************************************************************************************
!> \brief Retain G orbitals, Z and fixed W in the smallest memory-feasible time groups.
!>
!> Each group owns τ = group + 1, group + 1 + ngroups, ... . Within a group,
!> orbitals use owner MOD(k-1,g), whereas Z and W use owner MOD(q,g).
!> W remains in the auxiliary basis. A single q block is broadcast during this
!> one-time redistribution; each auxiliary W(q,τ) goes only to its owning rank.
!> \param bs_env Band-structure settings.
!> \param gwkp_env Work arrays and their MPI distribution.
! **************************************************************************************************
   SUBROUTINE prepare_self_energy_groups(bs_env, gwkp_env)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gwkp_type), INTENT(INOUT)                     :: gwkp_env

      CHARACTER(LEN=*), PARAMETER :: routineN = 'prepare_self_energy_groups'

      COMPLEX(KIND=dp), ALLOCATABLE                      :: psi_block(:, :, :), &
                                                            psi_group(:, :, :, :), Z_block(:, :), &
                                                            Z_group(:, :, :)
      INTEGER :: handle, i_dest, i_group_size, i_root, i_source, i_tag, ik, ik_local, iq, &
         iq_local, it, it_local, n_ao, n_aux, n_grid, n_k, n_local_k, n_local_q, n_local_time, &
         n_spin, n_stored_q, n_threads, n_time, n_world, phase_handle
      REAL(KIND=dp) :: available_GB, block_bytes, budget_bytes, grid_bytes, input_bytes, n_pairs, &
         peak_bytes, persistent_bytes, result_bytes
      TYPE(mp_comm_type)                                 :: column_env
      TYPE(ri_q_matrix_block_type), ALLOCATABLE          :: W_group(:)

      CALL timeset(routineN, handle)

      n_threads = 1
!$    n_threads = omp_get_max_threads()
      n_world = bs_env%para_env%num_pe
      n_grid = SIZE(gwkp_env%psi_lm_k, 1)
      n_ao = SIZE(gwkp_env%psi_lm_k, 2)
      n_spin = SIZE(gwkp_env%psi_lm_k, 4)
      n_aux = SIZE(gwkp_env%Z_lP_q, 2)
      n_k = bs_env%nkp_scf_desymm
      n_time = bs_env%num_time_freq_points
      CALL mp_mem_avail_per_rank_GB(bs_env%para_env, available_GB)
      CALL bs_env%para_env%min(available_GB)
      input_bytes = 16.0_dp*(REAL(SIZE(gwkp_env%psi_lm_k, KIND=int_8), dp) + &
                             REAL(SIZE(gwkp_env%Z_lP_q, KIND=int_8), dp) + REAL(ri_q_storage_elements(gwkp_env%W_PQ_tau), dp))
      CALL bs_env%para_env%min(input_bytes)
      ! World inputs remain alive during redistribution. Count the complete new
      ! group storage in addition to measured usage; retain a library reserve.
      budget_bytes = MAX(0.0_dp, available_GB - MAX(1.0_dp, 0.1_dp*available_GB))*1.0E9_dp
      block_bytes = 16.0_dp*REAL(n_grid, dp)*REAL(n_ao*n_spin + n_aux, dp)
      result_bytes = 32.0_dp*REAL(n_ao, dp)*REAL(SIZE(bs_env%eigenval_scf, 2), dp)* &
                     REAL(n_time*n_spin, dp)
      DO i_group_size = MIN(2, MAX(1, n_world/2)), n_world
         IF (MODULO(n_world, i_group_size) /= 0) CYCLE
         gwkp_env%ngroups = n_world/i_group_size
         IF (gwkp_env%ngroups > n_time) CYCLE
         n_local_k = (n_k + i_group_size - 1)/i_group_size
         n_local_time = (n_time + gwkp_env%ngroups - 1)/gwkp_env%ngroups
         n_stored_q = (n_k + i_group_size - 1)/i_group_size
         persistent_bytes = 16.0_dp*(REAL(n_local_k*n_grid, dp)*REAL(n_ao*n_spin + n_aux, dp) + &
                                     REAL(n_aux, dp)**2*REAL(n_stored_q*n_local_time, dp))
         n_pairs = REAL(CEILING(REAL(n_grid, dp)**2/REAL(i_group_size, dp), KIND=int_8), dp)
         ! G(k), gwkp_env%W_PQ_tau(R), and Σ(R) coexist: 16 + 8 + 8 bytes per pair and mesh point.
         ! q matrices, band projection, and Green blocks belong to separate stages.
         grid_bytes = n_pairs*32.0_dp*REAL(n_k, dp) + &
                      MAX(16.0_dp*REAL(n_grid, dp)**2*REAL(n_local_k, dp) + &
                          16.0_dp*REAL(n_aux, dp)**2*REAL(n_local_k, dp) + &
                          16.0_dp*REAL(n_threads*n_grid*(n_ao + n_aux), dp), &
                          projection_workspace_bytes(n_grid, n_pairs, n_k, n_ao, n_threads)) + &
                      32.0_dp*REAL(n_threads*n_k, dp) + 256.0_dp*1024.0_dp**2
         peak_bytes = MAX(persistent_bytes + block_bytes, &
                          persistent_bytes - input_bytes + grid_bytes + result_bytes)
         IF (gwkp_env%ngroups == 1) peak_bytes = grid_bytes + result_bytes
         IF (peak_bytes <= budget_bytes) EXIT
      END DO
      IF (i_group_size > n_world) THEN
         CPABORT('Insufficient node memory for GW self-energy time groups')
      END IF
      gwkp_env%ngroups = n_world/i_group_size
      gwkp_env%my_group = bs_env%para_env%mepos/i_group_size
      CALL mp_para_env_release(gwkp_env%group_env)
      ALLOCATE (gwkp_env%group_env)
      CALL gwkp_env%group_env%from_split(bs_env%para_env, gwkp_env%my_group)
      CPASSERT(gwkp_env%group_env%num_pe == i_group_size)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,F9.3)') &
            'GW self-energy workspace budget per rank [GB]', budget_bytes*1.0E-9_dp
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') 'GW self-energy MPI ranks per time group', i_group_size
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') 'GW self-energy number of time groups', gwkp_env%ngroups
         WRITE (bs_env%unit_nr, '(T2,A,T72,F9.3)') &
            'GW self-energy additional peak per rank [GB]', peak_bytes*1.0E-9_dp
         FLUSH (bs_env%unit_nr)
      END IF
      IF (gwkp_env%ngroups == 1) THEN
         CALL timestop(handle)
         RETURN
      END IF
      n_local_k = COUNT([(MODULO(ik - 1, i_group_size) == gwkp_env%group_env%mepos, ik=1, n_k)])
      n_local_q = COUNT([(MODULO(iq, i_group_size) == gwkp_env%group_env%mepos, iq=1, n_k)])
      n_local_time = (n_time - gwkp_env%my_group + gwkp_env%ngroups - 1)/gwkp_env%ngroups
      ALLOCATE (psi_group(n_grid, n_ao, n_local_k, n_spin), &
                Z_group(n_grid, n_aux, n_local_q), &
                psi_block(n_grid, n_ao, n_spin), Z_block(n_grid, n_aux))
      ALLOCATE (W_group(n_local_q))
      ! Ranks with the same within-group index need the same k/q slices.
      ! Communicate only along this column, independently of the other columns.
      CALL column_env%from_split(bs_env%para_env, gwkp_env%group_env%mepos)
      CPASSERT(column_env%mepos == gwkp_env%my_group)
      CPASSERT(column_env%num_pe == gwkp_env%ngroups)
      ik_local = 0
      iq_local = 0
      CALL timeset('Sigma_groups_redistribute_psi', phase_handle)
      DO ik = gwkp_env%group_env%mepos + 1, n_k, i_group_size
         i_root = MODULO(ik - 1, n_world)/i_group_size
         IF (column_env%mepos == i_root) psi_block(:, :, :) = gwkp_env%psi_lm_k(:, :, (ik - 1)/n_world + 1, :)
         CALL column_env%bcast(psi_block, i_root)
         ik_local = ik_local + 1
         psi_group(:, :, ik_local, :) = psi_block
      END DO
      CALL timestop(phase_handle)
      CALL timeset('Sigma_groups_redistribute_ZW', phase_handle)
      DO iq = MODULO(gwkp_env%group_env%mepos - 1, i_group_size) + 1, n_k, i_group_size
         i_root = MODULO(iq, n_world)/i_group_size
         IF (column_env%mepos == i_root) Z_block(:, :) = gwkp_env%Z_lP_q(:, :, (iq - 1)/n_world + 1)
         CALL column_env%bcast(Z_block, i_root)
         iq_local = iq_local + 1
         Z_group(:, :, iq_local) = Z_block
         ALLOCATE (W_group(iq_local)%values(n_aux, n_aux, n_local_time))
         DO it = 1, n_time
            i_dest = MODULO(it - 1, gwkp_env%ngroups)
            IF (column_env%mepos == i_root) THEN
               IF (i_dest == i_root) THEN
                  it_local = (it - 1)/gwkp_env%ngroups + 1
                  W_group(iq_local)%values(:, :, it_local) = gwkp_env%W_PQ_tau((iq - 1)/n_world + 1)%values(:, :, it)
               ELSE
                  CALL column_env%send(gwkp_env%W_PQ_tau((iq - 1)/n_world + 1)%values(:, :, it), i_dest, 0)
               END IF
            ELSE IF (column_env%mepos == i_dest) THEN
               i_source = i_root
               i_tag = 0
               it_local = (it - 1)/gwkp_env%ngroups + 1
               CALL column_env%recv(W_group(iq_local)%values(:, :, it_local), i_source, i_tag)
            END IF
         END DO
         IF (column_env%mepos == i_root) DEALLOCATE (gwkp_env%W_PQ_tau((iq - 1)/n_world + 1)%values)
      END DO
      CALL timestop(phase_handle)
      CALL column_env%free()
      CALL MOVE_ALLOC(psi_group, gwkp_env%psi_lm_k)
      CALL MOVE_ALLOC(Z_group, gwkp_env%Z_lP_q)
      CALL MOVE_ALLOC(W_group, gwkp_env%W_PQ_tau)
      DEALLOCATE (psi_block, Z_block)

      CALL timestop(handle)

   END SUBROUTINE prepare_self_energy_groups

! **************************************************************************************************
!> \brief Compute chi_PQ^R(i tau) through the periodic real-space grid representation.
!> \param bs_env Band-structure settings and physical results.
!> \param gwkp_env Periodic GW work data.
! **************************************************************************************************
   SUBROUTINE compute_chi(bs_env, gwkp_env)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gwkp_type), INTENT(INOUT), TARGET             :: gwkp_env

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'compute_chi'

      COMPLEX(KIND=dp), ALLOCATABLE :: chi_grid_local_q(:, :, :), chi_grid_q(:, :, :), &
         chi_q(:, :, :), G_grid_k(:, :, :), work(:, :), Z_block(:, :)
      COMPLEX(KIND=dp), ALLOCATABLE, TARGET              :: psi_time(:, :, :, :), Z_group(:, :, :)
      COMPLEX(KIND=dp), POINTER                          :: psi_values(:, :, :, :), Z_time(:, :, :)
      INTEGER :: first_row, handle, i_cell, i_group, i_q_world, i_rank, i_root, i_size, i_t, &
         i_time_group, i_time_local, iq, iq_local, ispin, last_row, n_aux, n_cells, n_grid, &
         n_groups, n_k_local, n_local, n_q_local, n_threads, n_time_local, n_world, nrows_local, &
         phase_handle
      INTEGER, ALLOCATABLE                               :: full_cells(:, :)
      REAL(KIND=dp) :: atom_block_count, available_GB, budget, input_bytes, live_bytes, n_q_max, &
         n_row_max, n_stored_max, peak, q_bytes, rows_bytes, step_elapsed, step_start, tau, &
         time_bytes
      REAL(KIND=dp), ALLOCATABLE                         :: chi_grid_R(:, :, :), G_occ_R(:, :, :), &
                                                            green_norm_sq(:), qpoints(:, :)
      TYPE(mp_comm_type)                                 :: column_env, node_env
      TYPE(mp_para_env_type), POINTER                    :: chi_env
      TYPE(ri_q_matrix_block_type), ALLOCATABLE          :: chi_partial(:)

      CALL timeset(routineN, handle)

      CPASSERT(gwkp_env%ngroups == 1 .AND. gwkp_env%my_group == 0)
      CALL build_full_cell_grid(bs_env, full_cells)
      CPASSERT(.NOT. ALLOCATED(gwkp_env%band_orbitals))
      n_grid = SIZE(gwkp_env%psi_lm_k, 1)
      n_aux = SIZE(gwkp_env%Z_lP_q, 2)
      n_world = gwkp_env%group_env%num_pe
      ALLOCATE (qpoints(3, bs_env%nkp_scf_desymm))
      DO iq = 1, bs_env%nkp_scf_desymm
         qpoints(:, iq) = momentum_transfer_point(bs_env, iq)
      END DO
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') 'GW stored response q points:', &
            SIZE(qpoints, 2)
      END IF
      CALL mp_mem_avail_per_rank_GB(gwkp_env%group_env, available_GB)
      budget = MAX(0.0_dp, available_GB - MAX(1.0_dp, 0.1_dp*available_GB))*1.0E9_dp
      n_cells = SIZE(full_cells, 2)
      n_threads = omp_get_max_threads()
      CALL node_env%from_split_type(gwkp_env%group_env, mp_comm_split_type_shared, key=gwkp_env%group_env%mepos)
      DO i_size = MIN(2, MAX(1, n_world/2)), n_world
         IF (MODULO(n_world, i_size) /= 0) CYCLE
         n_groups = n_world/i_size
         IF (n_groups > bs_env%num_time_freq_points) CYCLE
         i_rank = MODULO(gwkp_env%group_env%mepos, i_size)
         i_group = gwkp_env%group_env%mepos/i_size
         n_time_local = (bs_env%num_time_freq_points + n_groups - 1 - i_group)/n_groups
         n_row_max = REAL(n_grid*(i_rank + 1)/i_size - n_grid*i_rank/i_size, dp)
         n_q_max = REAL(COUNT([(MODULO(iq, i_size) == i_rank, &
                                iq=1, bs_env%nkp_scf_desymm)]), dp)
         n_stored_max = n_q_max
         input_bytes = 0.0_dp
         IF (n_groups > 1) THEN
            input_bytes = 16.0_dp*REAL(n_grid*n_aux, dp)*n_q_max
            input_bytes = input_bytes + &
                          16.0_dp*REAL(n_grid*bs_env%n_ao*bs_env%n_spin, dp)* &
                          REAL((bs_env%nkp_scf_desymm + i_size - 1 - i_rank)/i_size, dp)
         END IF
         time_bytes = 16.0_dp*REAL(n_aux, dp)**2*n_stored_max*REAL(n_time_local, dp)
         ! Keep only this group's time slices, not a full frequency set per group.
         ! Lattice FFT work holds two mesh vectors per thread, not another grid matrix.
         rows_bytes = n_row_max*REAL(n_grid, dp)* &
                      (16.0_dp*REAL(bs_env%nkp_scf_desymm, dp) + 16.0_dp*REAL(n_cells, dp))
         q_bytes = 16.0_dp*REAL(n_grid, dp)**2*n_q_max
         live_bytes = MAX(rows_bytes + &
                          16.0_dp*REAL(n_threads*n_grid*bs_env%n_ao, dp), &
                          16.0_dp*n_row_max*REAL(n_grid*bs_env%nkp_scf_desymm, dp) + q_bytes, &
                          q_bytes + 16.0_dp*REAL(n_grid*n_aux, dp)*MAX(1.0_dp, MIN(REAL(n_threads, dp), n_q_max)))
         peak = input_bytes + time_bytes + 16.0_dp*REAL(n_aux, dp)**2*n_q_max + &
                MAX(live_bytes, 16.0_dp*REAL(n_aux, dp)**2*REAL(bs_env%num_time_freq_points, dp)) + &
                32.0_dp*REAL(n_threads*bs_env%nkp_scf_desymm, dp) + 256.0_dp*1024.0_dp**2
         ! Cyclic rank placement can balance unequal q ownership on the same node.
         CALL node_env%sum(peak)
         peak = peak/REAL(node_env%num_pe, dp)
         CALL gwkp_env%group_env%max(peak)
         IF (peak <= budget) EXIT
      END DO
      CALL node_env%free()
      IF (i_size > n_world) THEN
         CPABORT("Insufficient node memory for GW time groups")
      END IF
      n_groups = n_world/i_size
      i_group = gwkp_env%group_env%mepos/i_size
      ALLOCATE (chi_env)
      CALL chi_env%from_split(gwkp_env%group_env, i_group)
      CALL column_env%from_split(gwkp_env%group_env, chi_env%mepos)
      Z_time => gwkp_env%Z_lP_q
      IF (n_groups > 1) THEN
         n_local = COUNT([(MODULO(iq, i_size) == chi_env%mepos, iq=1, bs_env%nkp_scf_desymm)])
         ALLOCATE (Z_group(n_grid, n_aux, n_local), Z_block(n_grid, n_aux))
         iq_local = 0
         DO iq = MODULO(chi_env%mepos - 1, i_size) + 1, bs_env%nkp_scf_desymm, i_size
            i_root = MODULO(iq, n_world)/i_size
            IF (column_env%mepos == i_root) Z_block(:, :) = gwkp_env%Z_lP_q(:, :, (iq - 1)/n_world + 1)
            CALL column_env%bcast(Z_block, i_root)
            iq_local = iq_local + 1
            Z_group(:, :, iq_local) = Z_block
         END DO
         DEALLOCATE (Z_block)
         Z_time => Z_group
      END IF
      IF (n_groups > 1) THEN
         n_k_local = (bs_env%nkp_scf_desymm + i_size - 1 - chi_env%mepos)/i_size
         ALLOCATE (psi_time(n_grid, bs_env%n_ao, n_k_local, bs_env%n_spin))
         ALLOCATE (Z_block(n_grid, bs_env%n_ao))
         DO ispin = 1, bs_env%n_spin
            iq_local = 0
            DO iq = chi_env%mepos + 1, bs_env%nkp_scf_desymm, i_size
               i_root = MODULO(iq - 1, n_world)/i_size
               IF (column_env%mepos == i_root) Z_block(:, :) = gwkp_env%psi_lm_k(:, :, (iq - 1)/n_world + 1, ispin)
               CALL column_env%bcast(Z_block, i_root)
               iq_local = iq_local + 1
               psi_time(:, :, iq_local, ispin) = Z_block
            END DO
         END DO
         DEALLOCATE (Z_block)
         psi_values => psi_time
      ELSE
         psi_values => gwkp_env%psi_lm_k
      END IF
      n_time_local = (bs_env%num_time_freq_points + n_groups - 1 - i_group)/n_groups
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') 'GW χ MPI ranks per time group', i_size
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') 'GW χ number of time groups', n_groups
         CALL m_flush(bs_env%unit_nr)
      END IF
      CALL local_grid_row_range(n_grid, chi_env, first_row, last_row, nrows_local)
      n_cells = SIZE(full_cells, 2)
      CPASSERT(bs_env%n_atom > 0)
      atom_block_count = REAL(bs_env%n_atom, dp)**2
      ALLOCATE (green_norm_sq(2*n_cells))
      n_q_local = 0
      DO iq = 1, bs_env%nkp_scf_desymm
         IF (MODULO(iq, chi_env%num_pe) == chi_env%mepos) n_q_local = n_q_local + 1
      END DO
      ALLOCATE (chi_q(n_aux, n_aux, n_q_local))
      ALLOCATE (chi_partial(n_q_local))
      iq_local = 0
      DO iq = MODULO(chi_env%mepos - 1, i_size) + 1, SIZE(qpoints, 2), i_size
         iq_local = iq_local + 1
         ALLOCATE (chi_partial(iq_local)%values(n_aux, n_aux, n_time_local), SOURCE=z_zero)
      END DO

      IF (bs_env%unit_nr > 0) WRITE (bs_env%unit_nr, '(A)') ' '
      DO i_t = i_group + 1, bs_env%num_time_freq_points, n_groups
         step_start = m_walltime()
         tau = bs_env%time_frequency_grid%imaginary_time(i_t)
         chi_q = z_zero
         DO ispin = 1, bs_env%n_spin
            CALL timeset('chi_grid_build', phase_handle)
            ALLOCATE (G_grid_k(nrows_local, n_grid, bs_env%nkp_scf_desymm))
            ALLOCATE (G_occ_R(nrows_local, n_grid, SIZE(full_cells, 2)))
            ALLOCATE (chi_grid_R, MOLD=G_occ_R)
            CALL build_green_grid_k_rows(bs_env, psi_values(:, :, :, ispin), &
                                         chi_env, tau, ispin, &
                                         .TRUE., first_row, last_row, G_grid_k)
            CALL gw_k_grid_to_real_cells(G_grid_k, bs_env%kpoints_scf_desymm%xkp, &
                                         bs_env%kpoints_scf_desymm%nkp_grid, full_cells, &
                                         G_occ_R)
            CALL build_green_grid_k_rows(bs_env, psi_values(:, :, :, ispin), &
                                         chi_env, tau, ispin, &
                                         .FALSE., first_row, last_row, G_grid_k)
            CALL gw_k_grid_to_real_cells(G_grid_k, bs_env%kpoints_scf_desymm%xkp, &
                                         bs_env%kpoints_scf_desymm%nkp_grid, full_cells, &
                                         chi_grid_R)
            ! Each rank owns complete columns but only its local AO-grid rows.
            ! One reduction over the time group yields the Frobenius norms of
            ! the entire occupied and virtual G_ll'(R, tau) matrices.
            DO i_cell = 1, n_cells
               green_norm_sq(i_cell) = SUM(G_occ_R(:, :, i_cell)**2)
               green_norm_sq(n_cells + i_cell) = SUM(chi_grid_R(:, :, i_cell)**2)
            END DO
            CALL chi_env%sum(green_norm_sq)
            DO i_cell = 1, n_cells
               IF (SQRT(green_norm_sq(i_cell))/atom_block_count < bs_env%eps_filter) THEN
                  G_occ_R(:, :, i_cell) = 0.0_dp
               END IF
               IF (SQRT(green_norm_sq(n_cells + i_cell))/atom_block_count < bs_env%eps_filter) THEN
                  chi_grid_R(:, :, i_cell) = 0.0_dp
               END IF
            END DO
            DEALLOCATE (G_grid_k)
            chi_grid_R(:, :, :) = bs_env%spin_degeneracy*G_occ_R*chi_grid_R
            DEALLOCATE (G_occ_R)
            ALLOCATE (chi_grid_q(nrows_local, n_grid, bs_env%nkp_scf_desymm))
            CALL product_to_k_pair_centered(bs_env, full_cells, qpoints, chi_grid_R, &
                                            chi_grid_q, first_row, 0)
            DEALLOCATE (chi_grid_R)
            CALL timestop(phase_handle)

            CALL timeset('chi_projection', phase_handle)
            CALL fft_rows_to_q_matrices(chi_grid_q, n_grid, chi_env, chi_grid_local_q)
            DEALLOCATE (chi_grid_q)
            ! χ_PQ(q) = Σ_ll′ Z_lP(q)* χ_ll′(q) Z_l′Q(q).
            ! Parallelize distinct q matrices; leave singleton GEMM threading to BLAS.
            !$OMP PARALLEL DEFAULT(NONE) IF(n_q_local > 1) &
            !$OMP SHARED(n_grid, n_aux, n_q_local, chi_grid_local_q, Z_time, chi_q) &
            !$OMP PRIVATE(work, iq_local)
            ALLOCATE (work(n_grid, n_aux))
            !$OMP DO SCHEDULE(STATIC)
            DO iq_local = 1, n_q_local
               CALL zgemm('N', 'N', n_grid, n_aux, n_grid, z_one, &
                          chi_grid_local_q(:, :, iq_local), n_grid, &
                          Z_time(:, :, iq_local), n_grid, z_zero, work, n_grid)
               CALL zgemm('C', 'N', n_aux, n_aux, n_grid, z_one, &
                          Z_time(:, :, iq_local), n_grid, work, n_grid, z_one, &
                          chi_q(:, :, iq_local), n_aux)
            END DO
            !$OMP END DO
            DEALLOCATE (work)
            !$OMP END PARALLEL
            DEALLOCATE (chi_grid_local_q)
            CALL timestop(phase_handle)
         END DO
         ! The legacy inverse FFT retained only Re[χ(R)].  On a complete
         ! Gamma-centred mesh this is [χ(q) + χ(-q)*]/2 element by element.
         CALL timeset('chi_q_reality', phase_handle)
         DO iq_local = 1, n_q_local
            i_time_local = (i_t - 1)/n_groups + 1
            chi_partial(iq_local)%values(:, :, i_time_local) = chi_q(:, :, iq_local)
         END DO
         CALL timestop(phase_handle)
         step_elapsed = m_walltime() - step_start
         IF (bs_env%unit_nr > 0) THEN
            WRITE (bs_env%unit_nr, '(T2,A,I3,A,I0,T57,A,T71,F10.1,A)') &
               'GW χ(iτ) point ', i_t, ' / ', bs_env%num_time_freq_points, &
               'Exec. time:', step_elapsed, ' s'
            CALL m_flush(bs_env%unit_nr)
         END IF
      END DO
      DEALLOCATE (green_norm_sq)

      IF (ALLOCATED(psi_time)) DEALLOCATE (psi_time)
      CALL timeset('chi_time_groups_sum', phase_handle)
      n_local = COUNT([(MODULO(iq, n_world) == gwkp_env%group_env%mepos, iq=1, bs_env%nkp_scf_desymm)])
      ALLOCATE (gwkp_env%chi_q_frequency(n_local))
      iq_local = 0
      DO iq = MODULO(chi_env%mepos - 1, i_size) + 1, bs_env%nkp_scf_desymm, i_size
         iq_local = iq_local + 1
         i_root = MODULO(iq, n_world)/i_size
         i_q_world = (iq - 1)/n_world + 1
         IF (column_env%mepos == i_root) THEN
            ALLOCATE (gwkp_env%chi_q_frequency(i_q_world)%values(n_aux, n_aux, &
                                                                 bs_env%num_time_freq_points), SOURCE=z_zero)
         END IF
         ! Transfer each computed time slice once. Only its world q owner forms
         ! the complete frequency result; partial frequency sets are never replicated.
         DO i_t = 1, bs_env%num_time_freq_points
            i_time_group = MODULO(i_t - 1, n_groups)
            i_time_local = (i_t - 1)/n_groups + 1
            IF (column_env%mepos == i_time_group) THEN
               chi_q(:, :, iq_local) = chi_partial(iq_local)%values(:, :, i_time_local)
            END IF
            IF (n_groups > 1) CALL column_env%bcast(chi_q(:, :, iq_local), i_time_group)
            IF (column_env%mepos == i_root) THEN
               CALL accumulate_ri_q_cosine(gwkp_env%chi_q_frequency(i_q_world), chi_q(:, :, iq_local), &
                                           bs_env%time_frequency_grid%cosine_time_to_frequency(:, i_t))
            END IF
         END DO
         DEALLOCATE (chi_partial(iq_local)%values)
      END DO
      DEALLOCATE (chi_partial)
      NULLIFY (Z_time)
      IF (ALLOCATED(Z_group)) DEALLOCATE (Z_group)
      CALL column_env%free()
      CALL mp_para_env_release(chi_env)
      CALL timestop(phase_handle)
      DEALLOCATE (chi_q, full_cells, qpoints)

      CALL timestop(handle)

   END SUBROUTINE compute_chi

! **************************************************************************************************
!> \brief Gamma-correction head and wing coefficients for the near-Gamma cells.
!> Construct the q->0 dielectric limit along face-cubature and mesh-sample directions
!> from SCF dipole transitions and the directly fitted Gamma Gaussian body.
!> \param bs_env ...
!> \param qs_env ...
!> \param gwkp_env RI-RS work data and Gamma correction.
! **************************************************************************************************
   SUBROUTINE gamma_point_correction(bs_env, qs_env, gwkp_env)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(gwkp_type), INTENT(INOUT)                     :: gwkp_env

      CHARACTER(LEN=*), PARAMETER :: routineN = 'gamma_point_correction'

      COMPLEX(KIND=dp)                                   :: eps_00, eps_inv_00, pi_00, pi_q(3), &
                                                            q_direction_complex(3), &
                                                            schur_contribution, schur_denom
      COMPLEX(KIND=dp), ALLOCATABLE :: dipole(:, :, :, :, :), eps_P_cartesian(:, :), &
         F_external(:, :), F_response_cartesian(:, :), F_zero(:, :), head_freq(:, :), &
         metric_zero(:, :), pi_cartesian(:, :), response_cartesian(:, :), schur_cartesian(:, :), &
         U_Pp_zero(:, :), V_sqrt_zero(:, :), wing_cartesian(:, :), wing_freq(:, :, :), &
         wing_local(:)
      INTEGER :: a, detail_handle, handle, i_cartesian, i_direction, i_t, ikp, j_cartesian, j_w, &
         n_angular, n_directions, n_near, n_w_kpoints
      INTEGER, ALLOCATABLE                               :: metric_cells(:, :), nmo_spin(:)
      LOGICAL                                            :: is_slab, valid_layout
      REAL(KIND=dp) :: frequency, near_q(3, 8), phase_elapsed, phase_start, q_cart(3), &
         q_direction(3), qnorm, reciprocal(3, 3), scaled(3), tau, volume, weight
      REAL(KIND=dp), ALLOCATABLE                         :: angular_directions(:, :), &
                                                            angular_radii(:), angular_weights(:), &
                                                            V_zero_real(:, :)
      TYPE(cp_fm_type), ALLOCATABLE                      :: fm_metric(:)
      TYPE(gamma_ao_time_state_type)                     :: gamma_time_state

      CALL timeset(routineN, handle)

      phase_start = m_walltime()
      is_slab = ALL(bs_env%periodic == [1, 1, 0])
      valid_layout = ALL(SHAPE(gwkp_env%Z_lP_zero) == [SIZE(gwkp_env%psi_lm_k, 1), bs_env%n_RI + 1, 1])
      CPASSERT(valid_layout)

      volume = bs_env%ri_rs%cell%deth
      CPASSERT(volume > 0.0_dp)
      n_w_kpoints = PRODUCT(bs_env%nkp_grid_chi_eps_W_orig)
      CPASSERT(n_w_kpoints > 0)
      reciprocal = twopi*TRANSPOSE(inv_3x3(bs_env%hmat))
      IF (is_slab) THEN
         CALL slab_reciprocal_vectors(bs_env, reciprocal(:, 1:2), volume)
         CALL slab_cell_face_directions(bs_env, angular_directions, angular_weights, angular_radii)
      ELSE
         CALL gamma_cell_face_directions(bs_env, angular_directions, angular_weights, angular_radii)
      END IF
      n_angular = SIZE(angular_weights)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,I10)') &
            'Number of angular points for the Γ-point correction of W:', n_angular
      END IF
      n_near = PRODUCT(MIN(2, bs_env%nkp_grid_chi_eps_W_orig))
      n_directions = n_angular + n_near
      a = 0
      DO ikp = 1, n_w_kpoints
         q_direction = bs_env%kpoints_chi_eps_W%xkp(1:3, ikp) - &
                       ANINT(bs_env%kpoints_chi_eps_W%xkp(1:3, ikp))
         scaled = ABS(q_direction)*REAL(bs_env%nkp_grid_chi_eps_W_orig, dp)
         IF (is_slab) THEN
            IF (MAXVAL(ABS(scaled(1:2) - 0.5_dp)) >= 1.0E-9_dp) CYCLE
         ELSE
            IF (MAXVAL(ABS(scaled - 0.5_dp)) >= 1.0E-9_dp) CYCLE
         END IF
         a = a + 1
         IF (a > n_near) CPABORT('Too many Gamma-neighbor transfer points')
         near_q(:, a) = q_direction
      END DO
      IF (a /= n_near) CPABORT('Inconsistent Gamma-neighbor transfer points')
      ALLOCATE (gwkp_env%gamma_correction%directions(3, n_directions), gwkp_env%gamma_correction%weights(n_directions), &
                gwkp_env%gamma_correction%radii(n_directions), gwkp_env%gamma_correction%near_q(3, n_near))
      gwkp_env%gamma_correction%n_angular = n_angular
      gwkp_env%gamma_correction%near_q(:, :) = near_q(:, 1:n_near)
      gwkp_env%gamma_correction%directions(:, 1:n_angular) = angular_directions
      gwkp_env%gamma_correction%weights(1:n_angular) = angular_weights
      gwkp_env%gamma_correction%radii(1:n_angular) = angular_radii
      DEALLOCATE (angular_directions, angular_weights, angular_radii)
      DO i_direction = 1, n_near
         q_cart = MATMUL(reciprocal, near_q(:, i_direction))
         qnorm = SQRT(SUM(q_cart*q_cart))
         gwkp_env%gamma_correction%directions(:, n_angular + i_direction) = q_cart/qnorm
         gwkp_env%gamma_correction%weights(n_angular + i_direction) = 0.0_dp
         gwkp_env%gamma_correction%radii(n_angular + i_direction) = qnorm
      END DO
      ALLOCATE (head_freq(n_directions, bs_env%num_time_freq_points), &
                wing_freq(SIZE(gwkp_env%Z_lP_zero, 1), n_directions, bs_env%num_time_freq_points))
      ! The finite RI metric at q=0 retains its complete Gaussian channel.
      ! The singular full-Coulomb Gamma-correction head is integrated separately.
      CALL timeset('gamma_ri_metric_build', detail_handle)
      ALLOCATE (metric_zero(bs_env%n_RI, bs_env%n_RI), &
                V_sqrt_zero(bs_env%n_RI, bs_env%n_RI), F_zero(bs_env%n_RI, bs_env%n_RI))
      CALL build_ri_2c_lattice(fm_metric, metric_cells, bs_env%ri_metric, &
                               0.0_dp, bs_env, qs_env, reference_basis=bs_env%auto_ri%enabled)
      IF (bs_env%auto_ri%enabled) THEN
         CALL gamma_metric_contract_atoms(fm_metric, bs_env%auto_ri, metric_zero, bs_env%unit_nr)
      ELSE
         CALL ri_2c_lattice_to_kp(fm_metric, metric_cells, [0.0_dp, 0.0_dp, 0.0_dp], metric_zero)
      END IF
      CALL cp_fm_release(fm_metric)
      DEALLOCATE (metric_cells)
      CALL timestop(detail_handle)
      IF (is_slab) THEN
         CALL slab_gamma_factors(qs_env, bs_env, metric_zero, F_zero, F_external)
         IF (ALLOCATED(U_Pp_zero)) DEALLOCATE (U_Pp_zero)
      ELSE
         CALL timeset('gamma_ri_coulomb_zero', detail_handle)
         CALL gamma_ri_coulomb_g0(qs_env, bs_env%auto_ri, V_zero_real)
         CPASSERT(SIZE(V_zero_real, 1) == bs_env%n_RI)
         V_sqrt_zero(:, :) = CMPLX(V_zero_real, 0.0_dp, dp)
         DEALLOCATE (V_zero_real)
         CALL timestop(detail_handle)
         CALL timeset('gamma_ri_metric_factor', detail_handle)
         CALL gamma_metric_factor_distributed(bs_env%para_env, bs_env%regularization_RI, &
                                              metric_zero, V_sqrt_zero, F_zero)
         CALL timestop(detail_handle)
      END IF
      DEALLOCATE (metric_zero, V_sqrt_zero)

      ! Gamma-correction matrix elements in the actual SCF MO gauge.  q_x cancels analytically against
      ! the 1/q and 1/q^2 factors, so q_x=1 is used after the limit has been taken.
      CALL gw_gamma_transition_moments(bs_env, gwkp_env, bs_env%kpoints_scf_desymm, &
                                       bs_env%cfm_mo_coeff_kp(1:bs_env%nkp_scf_desymm, :), gwkp_env%eigenvalues_G, &
                                       bs_env%n_occ(1:bs_env%n_spin), dipole, nmo_spin)
      CPASSERT(SIZE(dipole, 2) == SIZE(gwkp_env%psi_lm_k, 3))
      CPASSERT(ALL(nmo_spin >= bs_env%n_ao))

      ALLOCATE (eps_P_cartesian(bs_env%n_RI, 3), pi_cartesian(3, 3), &
                response_cartesian(bs_env%n_RI, 3), schur_cartesian(3, 3), &
                wing_cartesian(SIZE(gwkp_env%Z_lP_zero, 1), 3), &
                F_response_cartesian(bs_env%n_RI, 3), wing_local(bs_env%n_RI))
      IF (is_slab) ALLOCATE (gwkp_env%gamma_correction%slab_wing_cartesian(SIZE(gwkp_env%Z_lP_zero, 1), 3, &
                                                                           bs_env%num_time_freq_points), SOURCE=z_zero)
      head_freq = z_zero
      wing_freq = z_zero
      CALL gamma_ao_time_prepare(bs_env, gwkp_env%psi_lm_k, dipole, volume, gamma_time_state)
      DO j_w = 1, bs_env%num_time_freq_points
         frequency = bs_env%time_frequency_grid%frequency(j_w)
         CPASSERT(bs_env%eps_eigval_mat_RI == 0.0_dp)
         CALL gamma_ao_time_response(bs_env, gamma_time_state, gwkp_env%Z_lP_zero(:, 2:, 1), F_zero, frequency, &
                                     eps_P_cartesian, pi_cartesian, response_cartesian)
         CALL timeset('gamma_headwing_apply', detail_handle)
         CALL zgemm('C', 'N', 3, 3, bs_env%n_RI, CMPLX(fourpi, 0.0_dp, dp), &
                    eps_P_cartesian, bs_env%n_RI, response_cartesian, bs_env%n_RI, &
                    z_zero, schur_cartesian, 3)
         IF (is_slab) THEN
            CALL zgemm('N', 'N', bs_env%n_RI, 3, bs_env%n_RI, z_one, &
                       F_external, bs_env%n_RI, response_cartesian, bs_env%n_RI, &
                       z_zero, F_response_cartesian, bs_env%n_RI)
            CALL zgemm('N', 'N', SIZE(gwkp_env%Z_lP_zero, 1), 3, bs_env%n_RI, &
                       CMPLX(twopi/ri_metric_head_zero(bs_env), 0.0_dp, dp), &
                       gwkp_env%Z_lP_zero(:, 2:, 1), SIZE(gwkp_env%Z_lP_zero, 1), F_response_cartesian, bs_env%n_RI, &
                       z_zero, wing_cartesian, SIZE(gwkp_env%Z_lP_zero, 1))
            gwkp_env%gamma_correction%slab_wing_cartesian(:, :, j_w) = -wing_cartesian
         ELSE
            CALL zgemm('N', 'N', bs_env%n_RI, 3, bs_env%n_RI, z_one, &
                       F_zero, bs_env%n_RI, response_cartesian, bs_env%n_RI, &
                       z_zero, F_response_cartesian, bs_env%n_RI)
            CALL zgemm('N', 'N', SIZE(gwkp_env%Z_lP_zero, 1), 3, bs_env%n_RI, &
                       CMPLX(fourpi/ri_metric_head_zero(bs_env), 0.0_dp, dp), &
                       gwkp_env%Z_lP_zero(:, 2:, 1), SIZE(gwkp_env%Z_lP_zero, 1), F_response_cartesian, bs_env%n_RI, &
                       z_zero, wing_cartesian, SIZE(gwkp_env%Z_lP_zero, 1))
         END IF
         DO i_direction = 1, n_directions
            q_direction_complex(:) = CMPLX(gwkp_env%gamma_correction%directions(:, i_direction), 0.0_dp, dp)
            pi_q(:) = MATMUL(pi_cartesian, q_direction_complex)
            pi_00 = DOT_PRODUCT(q_direction_complex, pi_q)
            eps_00 = z_one + fourpi*pi_00
            schur_contribution = z_zero
            DO j_cartesian = 1, 3
               DO i_cartesian = 1, 3
                  schur_contribution = schur_contribution + &
                                       CONJG(q_direction_complex(i_cartesian))* &
                                       schur_cartesian(i_cartesian, j_cartesian)*q_direction_complex(j_cartesian)
               END DO
            END DO
            IF (is_slab) THEN
               ! A(d,ω) = d†[α - E† B⁻¹ E]d is the screened 2D polarizability.
               head_freq(i_direction, j_w) = pi_00 - schur_contribution/fourpi
               IF (REAL(head_freq(i_direction, j_w), dp) < -1.0E-10_dp .OR. &
                   ABS(AIMAG(head_freq(i_direction, j_w))) > 1.0E-8_dp) THEN
                  CPABORT('2D Gamma polarizability is not positive real')
               END IF
               head_freq(i_direction, j_w) = &
                  CMPLX(MAX(0.0_dp, REAL(head_freq(i_direction, j_w), dp)), 0.0_dp, dp)
               CYCLE
            END IF
            schur_denom = eps_00 - schur_contribution
            IF (REAL(schur_denom, dp) <= 0.0_dp .OR. &
                ABS(AIMAG(schur_denom)) > 1.0E-8_dp*MAX(ABS(schur_denom), 1.0_dp)) THEN
               CPABORT("Gamma dielectric Schur complement is not positive real")
            END IF
            eps_inv_00 = z_one/schur_denom
            head_freq(i_direction, j_w) = eps_inv_00 - z_one
            wing_freq(:, i_direction, j_w) = -eps_inv_00* &
                                             MATMUL(wing_cartesian, q_direction_complex)
         END DO
         CALL timestop(detail_handle)

      END DO

      IF (is_slab) THEN
         CALL slab_gamma_corr_cell_integrals(bs_env, gwkp_env%gamma_correction, head_freq)
      END IF
      CALL timeset('gamma_time_transform', detail_handle)
      ALLOCATE (gwkp_env%gamma_correction%head_tau(n_directions, bs_env%num_time_freq_points), &
                gwkp_env%gamma_correction%wing_tau(SIZE(gwkp_env%Z_lP_zero, 1), n_directions, bs_env%num_time_freq_points))
      gwkp_env%gamma_correction%head_tau = z_zero
      gwkp_env%gamma_correction%wing_tau = z_zero
      DO i_t = 1, bs_env%num_time_freq_points
         tau = bs_env%time_frequency_grid%imaginary_time(i_t)
         DO j_w = 1, bs_env%num_time_freq_points
            frequency = bs_env%time_frequency_grid%frequency(j_w)
            weight = bs_env%time_frequency_grid%cosine_frequency_to_time(i_t, j_w)
            gwkp_env%gamma_correction%head_tau(:, i_t) = gwkp_env%gamma_correction%head_tau(:, i_t) + weight*head_freq(:, j_w)
            gwkp_env%gamma_correction%wing_tau(:, :, i_t) = &
               gwkp_env%gamma_correction%wing_tau(:, :, i_t) + weight*wing_freq(:, :, j_w)
         END DO
      END DO
      CALL timestop(detail_handle)
      DEALLOCATE (head_freq, wing_freq)
      CALL gamma_ao_time_release(gamma_time_state)
      IF (ALLOCATED(F_external)) DEALLOCATE (F_external)

      DEALLOCATE (dipole, &
                  eps_P_cartesian, F_zero, nmo_spin, &
                  pi_cartesian, response_cartesian, schur_cartesian, &
                  F_response_cartesian, wing_cartesian, wing_local)
      phase_elapsed = m_walltime() - phase_start
      CALL bs_env%para_env%max(phase_elapsed)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T69,F12.1)') &
            'GW Γ-point correction wall time [s]:', phase_elapsed
         FLUSH (bs_env%unit_nr)
      END IF

      CALL timestop(handle)

   END SUBROUTINE gamma_point_correction

! **************************************************************************************************
!> \brief Finite Gamma Coulomb body from the unchanged GPW two-centre integrator.
!> The periodic Poisson solve omits the singular G=0 Fourier coefficient. Each MPI group
!> integrates distinct reference auxiliary columns; atom-local AUTO_RI coefficients then
!> project both indices without allocating a full reference-space Coulomb matrix.
!> \param qs_env ...
!> \param auto_ri ...
!> \param V_zero ...
! **************************************************************************************************
   SUBROUTINE gamma_ri_coulomb_g0(qs_env, auto_ri, V_zero)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(auto_ri_type), INTENT(IN)                     :: auto_ri
      REAL(KIND=dp), ALLOCATABLE, INTENT(OUT)            :: V_zero(:, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'gamma_ri_coulomb_g0'

      INTEGER :: handle, i_col, i_group, iatom, ikind, j_first, j_last, n_groups, n_local, n_opt, &
         n_ref, nimages_old, nsgf, opt_first, ref_first, u_first
      INTEGER, ALLOCATABLE                               :: kind_of(:)
      LOGICAL                                            :: created_gpw_kpoints
      REAL(KIND=dp)                                      :: progression
      REAL(KIND=dp), ALLOCATABLE                         :: projected(:, :), reference(:, :), &
                                                            U_atom(:, :)
      TYPE(atomic_kind_type), POINTER                    :: atomic_kinds(:)
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(kpoint_type), POINTER                         :: gpw_kpoints, previous_kpoints
      TYPE(libint_potential_type)                        :: coulomb
      TYPE(mp_para_env_type), POINTER                    :: column_env, para_env
      TYPE(neighbor_list_set_p_type), POINTER            :: sab_orb(:)
      TYPE(particle_type), POINTER                       :: particles(:)
      TYPE(qs_kind_type), POINTER                        :: qs_kinds(:)

      CALL timeset(routineN, handle)

      NULLIFY (atomic_kinds, dft_control, para_env, particles, qs_kinds, sab_orb)
      CALL get_qs_env(qs_env, para_env=para_env, sab_orb=sab_orb, dft_control=dft_control, &
                      atomic_kind_set=atomic_kinds, particle_set=particles, qs_kind_set=qs_kinds)
      CALL get_atomic_kind_set(atomic_kinds, kind_of=kind_of)
      n_ref = 0
      DO iatom = 1, SIZE(particles)
         ikind = kind_of(iatom)
         CALL get_qs_kind(qs_kinds(ikind), nsgf=nsgf, basis_type='RI_AUX')
         n_ref = n_ref + nsgf
      END DO
      n_opt = n_ref
      IF (auto_ri%enabled) THEN
         CPASSERT(SIZE(auto_ri%sizes_ref_RI) == SIZE(particles))
         CPASSERT(SIZE(auto_ri%sizes_opt_RI) == SIZE(particles))
         CPASSERT(SIZE(auto_ri%U_Pp_atom_offset) == SIZE(particles))
         CPASSERT(SUM(auto_ri%sizes_ref_RI) == n_ref)
         n_opt = SUM(auto_ri%sizes_opt_RI)
      END IF
      CPASSERT(n_ref > 0 .AND. n_opt > 0)
      ALLOCATE (V_zero(n_opt, n_opt), SOURCE=0.0_dp)

      ! prepare_gpw uses half the WFC_GPW cutoffs. Match the unchanged DFT grid exactly;
      ! reject a non-geometric grid rather than silently changing this calculation.
      progression = dft_control%qs_control%progression_factor
      CPASSERT(progression > 0.0_dp)
      IF (ABS(dft_control%qs_control%e_cutoff(1) - dft_control%qs_control%cutoff) > &
          1.0E-10_dp*dft_control%qs_control%cutoff) THEN
         CPABORT('Gamma GPW Coulomb requires matching first DFT multigrid cutoff')
      END IF
      DO i_col = 2, SIZE(dft_control%qs_control%e_cutoff)
         IF (ABS(dft_control%qs_control%e_cutoff(i_col) - &
                 dft_control%qs_control%e_cutoff(i_col - 1)/progression) > &
             1.0E-10_dp*dft_control%qs_control%e_cutoff(i_col)) THEN
            CPABORT('Gamma GPW Coulomb requires geometric DFT multigrid cutoffs')
         END IF
      END DO
      coulomb%potential_type = do_potential_coulomb
      coulomb%cutoff_radius = -1.0_dp
      coulomb%omega = 0.0_dp
      coulomb%filename = ''

      n_groups = MIN(n_ref, para_env%num_pe)
      i_group = MODULO(para_env%mepos, n_groups)
      j_first = 1 + n_ref*i_group/n_groups
      j_last = n_ref*(i_group + 1)/n_groups
      n_local = j_last - j_first + 1
      ALLOCATE (column_env)
      CALL column_env%from_split(para_env, i_group)
      ALLOCATE (reference(n_ref, n_local))
      ! GPW tasks use AO neighbor-list images, which can extend beyond the GW cell grid.
      ! Rebuild that map after QS partial release using the same neighbor list as GPW.
      NULLIFY (previous_kpoints, gpw_kpoints)
      CALL get_ks_env(qs_env%ks_env, kpoints=previous_kpoints)
      created_gpw_kpoints = .NOT. ASSOCIATED(previous_kpoints)
      nimages_old = dft_control%nimages
      IF (created_gpw_kpoints) THEN
         CALL kpoint_create(gpw_kpoints)
         CALL kpoint_init_cell_index(gpw_kpoints, sab_orb, para_env, dft_control%nimages)
         CALL set_ks_env(qs_env%ks_env, kpoints=gpw_kpoints)
      END IF
      CALL mp2_eri_2c_integrate_gpw(qs_env, column_env, j_first, j_last, &
                                    [dft_control%qs_control%cutoff, dft_control%qs_control%relative_cutoff], &
                                    coulomb, sab_orb, reference, kind_of)
      IF (created_gpw_kpoints) THEN
         CALL set_ks_env(qs_env%ks_env, kpoints=previous_kpoints)
         CALL kpoint_release(gpw_kpoints)
      END IF
      dft_control%nimages = nimages_old

      IF (auto_ri%enabled) THEN
         ALLOCATE (projected(n_opt, n_local))
         ref_first = 1
         opt_first = 1
         DO iatom = 1, SIZE(particles)
            n_ref = auto_ri%sizes_ref_RI(iatom)
            nsgf = auto_ri%sizes_opt_RI(iatom)
            u_first = auto_ri%U_Pp_atom_offset(iatom)
            CPASSERT(n_ref > 0 .AND. nsgf > 0)
            IF (u_first < 1 .OR. &
                u_first - 1 > SIZE(auto_ri%U_Pp_atom) - n_ref*nsgf) THEN
               CPABORT('Invalid AUTO_RI atom transform offset')
            END IF
            ALLOCATE (U_atom(n_ref, nsgf))
            U_atom(:, :) = RESHAPE(auto_ri%U_Pp_atom(u_first:u_first + n_ref*nsgf - 1), &
                                   [n_ref, nsgf])
            projected(opt_first:opt_first + nsgf - 1, :) = &
               MATMUL(TRANSPOSE(U_atom), reference(ref_first:ref_first + n_ref - 1, :))
            DEALLOCATE (U_atom)
            ref_first = ref_first + n_ref
            opt_first = opt_first + nsgf
         END DO
         ref_first = 1
         opt_first = 1
         DO iatom = 1, SIZE(particles)
            n_ref = auto_ri%sizes_ref_RI(iatom)
            nsgf = auto_ri%sizes_opt_RI(iatom)
            i_col = MAX(j_first, ref_first)
            IF (i_col <= MIN(j_last, ref_first + n_ref - 1)) THEN
               u_first = auto_ri%U_Pp_atom_offset(iatom)
               ALLOCATE (U_atom(n_ref, nsgf))
               U_atom(:, :) = RESHAPE(auto_ri%U_Pp_atom(u_first:u_first + n_ref*nsgf - 1), &
                                      [n_ref, nsgf])
               V_zero(:, opt_first:opt_first + nsgf - 1) = &
                  V_zero(:, opt_first:opt_first + nsgf - 1) + &
                  MATMUL(projected(:, i_col - j_first + 1:MIN(j_last, ref_first + n_ref - 1) - j_first + 1), &
                         U_atom(i_col - ref_first + 1:MIN(j_last, ref_first + n_ref - 1) - ref_first + 1, :))
               DEALLOCATE (U_atom)
            END IF
            ref_first = ref_first + n_ref
            opt_first = opt_first + nsgf
         END DO
         DEALLOCATE (projected)
      ELSE
         V_zero(:, j_first:j_last) = reference(:, :)
      END IF
      DEALLOCATE (reference, kind_of)
      IF (column_env%mepos /= 0) V_zero = 0.0_dp
      CALL mp_para_env_release(column_env)
      DO i_col = 1, n_opt
         CALL para_env%sum(V_zero(:, i_col))
      END DO
      V_zero(:, :) = 0.5_dp*(V_zero + TRANSPOSE(V_zero))

      CALL timestop(handle)

   END SUBROUTINE gamma_ri_coulomb_g0

! **************************************************************************************************
!> \brief Build the full Gaussian RI metric and bare Coulomb operators.
!> The Gamma-cell head is corrected separately from these operators.
!> \param bs_env ...
!> \param qs_env ...
!> \param M_inv_V_sqrt ...
!> \param M_inv ...
!> \param V_sqrt ...
!> \param metric_norm_max ...
! **************************************************************************************************
   SUBROUTINE compute_gaussian_Minv_and_Vsqrt(bs_env, qs_env, M_inv_V_sqrt, M_inv, V_sqrt, metric_norm_max)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(qs_environment_type), POINTER                 :: qs_env
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: M_inv_V_sqrt(:, :, :), M_inv(:, :, :), &
                                                            V_sqrt(:, :, :)
      REAL(KIND=dp), INTENT(OUT)                         :: metric_norm_max

      CHARACTER(LEN=*), PARAMETER :: routineN = 'compute_gaussian_Minv_and_Vsqrt'

      COMPLEX(KIND=dp), ALLOCATABLE                      :: M_ref(:, :, :), U_Pp_q(:, :), &
                                                            V_ref(:, :, :)
      INTEGER                                            :: handle, ikp, ilocal, n_ref, n_RI, nkp, &
                                                            nkp_local
      INTEGER, ALLOCATABLE                               :: metric_cells(:, :)
      TYPE(cp_fm_type), ALLOCATABLE                      :: fm_metric(:)

      CALL timeset(routineN, handle)

      CPASSERT(bs_env%gw_implementation == ri_rs_small_cell_full_kp)
      n_RI = bs_env%n_RI
      n_ref = n_RI
      IF (bs_env%auto_ri%enabled) n_ref = SUM(bs_env%auto_ri%sizes_ref_RI)
      nkp = bs_env%nkp_scf_desymm
      CPASSERT(bs_env%nkp_chi_eps_W_orig == nkp)
      nkp_local = COUNT([(MODULO(ikp, bs_env%para_env%num_pe) == &
                          bs_env%para_env%mepos, ikp=1, nkp)])
      ALLOCATE (M_ref(n_ref, n_ref, nkp_local), V_ref(n_ref, n_ref, nkp_local))
      ALLOCATE (M_inv_V_sqrt(n_RI, n_RI, nkp_local), M_inv(n_RI, n_RI, nkp_local), &
                V_sqrt(n_RI, n_RI, nkp_local))
      M_ref = z_zero
      V_ref = z_zero
      ! The full Ewald Coulomb matrix is finite on the shifted 3D q mesh. No
      ! G=0 auxiliary channel is removed at each transfer point.
      bs_env%kpoints_chi_eps_W%nkp_grid = bs_env%nkp_grid_chi_eps_W_orig
      CALL build_2c_coulomb_matrix_kp_small_cell(V_ref, qs_env, bs_env%kpoints_chi_eps_W, &
                                                 bs_env%size_lattice_sum_V, basis_type='RI_AUX', &
                                                 ikp_start=1, ikp_end=nkp, use_ewald=ALL(bs_env%periodic == [1, 1, 1]))
      CALL build_ri_2c_lattice(fm_metric, metric_cells, bs_env%ri_metric, &
                               0.0_dp, bs_env, qs_env, reference_basis=bs_env%auto_ri%enabled)
      CALL ri_2c_lattice_to_owned_q(fm_metric, metric_cells, &
                                    bs_env%kpoints_chi_eps_W%xkp(1:3, 1:nkp), M_ref)
      CALL cp_fm_release(fm_metric)
      DEALLOCATE (metric_cells)
      metric_norm_max = 0.0_dp
      ilocal = 0
      DO ikp = 1, nkp
         IF (MODULO(ikp, bs_env%para_env%num_pe) /= bs_env%para_env%mepos) CYCLE
         ilocal = ilocal + 1
         IF (bs_env%auto_ri%enabled) THEN
            CALL assemble_AA_contraction(bs_env, U_Pp_q)
            M_inv(:, :, ilocal) = MATMUL(CONJG(TRANSPOSE(U_Pp_q)), &
                                         MATMUL(M_ref(:, :, ilocal), U_Pp_q))
            V_sqrt(:, :, ilocal) = MATMUL(CONJG(TRANSPOSE(U_Pp_q)), &
                                          MATMUL(V_ref(:, :, ilocal), U_Pp_q))
            DEALLOCATE (U_Pp_q)
         ELSE
            M_inv(:, :, ilocal) = M_ref(:, :, ilocal)
            V_sqrt(:, :, ilocal) = V_ref(:, :, ilocal)
         END IF
         metric_norm_max = MAX(metric_norm_max, SQRT(SUM(ABS(M_inv(:, :, ilocal))**2)))
         CALL power(M_inv(:, :, ilocal), -1.0_dp, 0.0_dp, &
                    relative_eps=bs_env%regularization_RI)
         CALL power(V_sqrt(:, :, ilocal), 0.5_dp, 0.0_dp)
         M_inv_V_sqrt(:, :, ilocal) = MATMUL(M_inv(:, :, ilocal), V_sqrt(:, :, ilocal))
      END DO
      CALL bs_env%para_env%max(metric_norm_max)
      DEALLOCATE (M_ref, V_ref)

      CALL timestop(handle)

   END SUBROUTINE compute_gaussian_Minv_and_Vsqrt

! **************************************************************************************************
!> \brief Shift every diagonal element of a complex square matrix.
!> \param matrix ...
!> \param shift ...
! **************************************************************************************************
   SUBROUTINE shift_diagonal(matrix, shift)
      COMPLEX(KIND=dp), DIMENSION(:, :), INTENT(INOUT)   :: matrix
      REAL(KIND=dp), INTENT(IN)                          :: shift

      INTEGER                                            :: i

      CPASSERT(SIZE(matrix, 1) == SIZE(matrix, 2))
      DO i = 1, SIZE(matrix, 1)
         matrix(i, i) = matrix(i, i) + shift
      END DO

   END SUBROUTINE shift_diagonal

! **************************************************************************************************
!> \brief Gauss-Legendre nodes and weights on [-1,1].
!> \param n ...
!> \param nodes ...
!> \param weights ...
! **************************************************************************************************
   SUBROUTINE gauss_legendre_cell_rule(n, nodes, weights)
      INTEGER, INTENT(IN)                                :: n
      REAL(KIND=dp), ALLOCATABLE, INTENT(OUT)            :: nodes(:), weights(:)

      INTEGER                                            :: i, iter, j
      REAL(KIND=dp)                                      :: derivative, p0, p1, p2, z, z_next

      ALLOCATE (nodes(n), weights(n))
      DO i = 1, (n + 1)/2
         z = COS(pi*(REAL(i, dp) - 0.25_dp)/(REAL(n, dp) + 0.5_dp))
         DO iter = 1, 50
            p0 = 1.0_dp
            p1 = 0.0_dp
            DO j = 1, n
               p2 = p1
               p1 = p0
               p0 = ((2*j - 1)*z*p1 - (j - 1)*p2)/REAL(j, dp)
            END DO
            derivative = REAL(n, dp)*(z*p0 - p1)/(z*z - 1.0_dp)
            z_next = z - p0/derivative
            IF (ABS(z_next - z) < 1.0E-15_dp) EXIT
            z = z_next
         END DO
         IF (iter > 50) CPABORT('Gauss-Legendre face quadrature did not converge')
         nodes(i) = -z_next
         nodes(n + 1 - i) = z_next
         weights(i) = 2.0_dp/((1.0_dp - z_next*z_next)*derivative**2)
         weights(n + 1 - i) = weights(i)
      END DO
   END SUBROUTINE gauss_legendre_cell_rule

! **************************************************************************************************
!> \brief Angular cubature over the cells adjacent to Gamma.
!> With mesh dimensions n_i, the fractional half-widths are h_i=min(1/n_i,1/2).
!> The face Jacobian is Omega_BZ*PRODUCT(h_i)/r**3.  The region contains
!> PRODUCT(min(2,n_i)) distinct mesh cells; a one-point direction covers its
!> entire Brillouin-zone interval exactly once, not two periodic copies.
!> \param bs_env ...
!> \param directions ...
!> \param weights ...
!> \param radii ...
! **************************************************************************************************
   SUBROUTINE gamma_cell_face_directions(bs_env, directions, weights, radii)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(KIND=dp), ALLOCATABLE, INTENT(OUT)            :: directions(:, :), weights(:), radii(:)

      INTEGER, PARAMETER                                 :: n = 6

      INTEGER                                            :: face, i, ia, ib, j, k, n_angular, nk, &
                                                            sign_index
      REAL(KIND=dp)                                      :: bmat(3, 3), jacobian_scale, q(3), &
                                                            qfrac(3), region_volume, vol_check, &
                                                            volume
      REAL(KIND=dp), ALLOCATABLE                         :: nodes(:), rule_weights(:)

      ! Six faces with a 6-by-6 Gauss-Legendre rule give 216 angular points.
      n_angular = 6*n*n
      nk = PRODUCT(bs_env%nkp_grid_chi_eps_W_orig)
      jacobian_scale = REAL(PRODUCT(MIN(2, bs_env%nkp_grid_chi_eps_W_orig)), dp)/8.0_dp
      region_volume = 8.0_dp*jacobian_scale/REAL(nk, dp)
      volume = bs_env%ri_rs%cell%deth
      bmat = twopi*TRANSPOSE(inv_3x3(bs_env%hmat))
      CALL gauss_legendre_cell_rule(n, nodes, rule_weights)
      ALLOCATE (directions(3, n_angular), weights(n_angular), radii(n_angular))
      face = 0
      DO i = 1, 3
         j = MODULO(i, 3) + 1
         k = MODULO(i + 1, 3) + 1
         DO sign_index = -1, 1, 2
            DO ia = 1, n
               DO ib = 1, n
                  face = face + 1
                  qfrac(i) = REAL(sign_index, dp)/REAL(MAX(2, bs_env%nkp_grid_chi_eps_W_orig(i)), dp)
                  qfrac(j) = nodes(ia)/REAL(MAX(2, bs_env%nkp_grid_chi_eps_W_orig(j)), dp)
                  qfrac(k) = nodes(ib)/REAL(MAX(2, bs_env%nkp_grid_chi_eps_W_orig(k)), dp)
                  q = MATMUL(bmat, qfrac)
                  radii(face) = SQRT(SUM(q*q))
                  directions(:, face) = q/radii(face)
                  weights(face) = (twopi**3/volume)/(fourpi*REAL(nk, dp)*radii(face)**3)* &
                                  rule_weights(ia)*rule_weights(ib)*jacobian_scale
               END DO
            END DO
         END DO
      END DO
      CPASSERT(face == n_angular)
      vol_check = volume*fourpi/(3.0_dp*twopi**3)*SUM(weights*radii**3)
      IF (ABS(vol_check - region_volume) > 1.0E-11_dp/REAL(nk, dp)) THEN
         CPABORT('Gamma-cell angular quadrature failed its exact volume check')
      END IF
      DEALLOCATE (nodes, rule_weights)
   END SUBROUTINE gamma_cell_face_directions

! **************************************************************************************************
!> \brief Integrate 1/q**2 over the distinct shifted-mesh cells around Gamma.
!> The regular mesh sum already contains the near-Gamma model samples, which are
!> subtracted here.  The returned coefficient multiplies the factored head
!> density 4*pi*Z_l0(0)*conjg(Z_l'0(0))/M_0(0)**2.
!> \param bs_env ...
!> \param cells ...
!> \param factors ...
! **************************************************************************************************
   SUBROUTINE singular_head_cell_factors(bs_env, cells, factors)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: cells(:, :)
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: factors(:)

      COMPLEX(KIND=dp)                                   :: f0, phase
      INTEGER                                            :: icell, idir, iq, n_near, nk
      REAL(KIND=dp)                                      :: a, L, q(3), qcart(3), qnorm, rcart(3), &
                                                            scaled(3), volume
      REAL(KIND=dp), ALLOCATABLE                         :: directions(:, :), radii(:), weights(:)

      IF (ALL(bs_env%periodic == [1, 1, 0])) THEN
         CALL singular_head_cell_factors_2d(bs_env, cells, factors)
         RETURN
      END IF
      nk = PRODUCT(bs_env%nkp_grid_chi_eps_W_orig)
      CALL gamma_cell_face_directions(bs_env, directions, weights, radii)
      volume = bs_env%ri_rs%cell%deth
      ALLOCATE (factors(SIZE(cells, 2)))
      n_near = 0
      DO iq = 1, nk
         q = bs_env%kpoints_chi_eps_W%xkp(1:3, iq) - &
             ANINT(bs_env%kpoints_chi_eps_W%xkp(1:3, iq))
         scaled = ABS(q)*REAL(bs_env%nkp_grid_chi_eps_W_orig, dp)
         IF (MAXVAL(ABS(scaled - 0.5_dp)) < 1.0E-9_dp) n_near = n_near + 1
      END DO
      IF (n_near /= PRODUCT(MIN(2, bs_env%nkp_grid_chi_eps_W_orig))) THEN
         CPABORT('Inconsistent number of distinct Gamma-neighbor cells')
      END IF
      DO icell = 1, SIZE(cells, 2)
         rcart = MATMUL(bs_env%hmat, REAL(cells(:, icell), dp))
         factors(icell) = z_zero
         DO idir = 1, SIZE(weights)
            L = radii(idir)
            a = DOT_PRODUCT(directions(:, idir), rcart)
            IF (ABS(a*L) < 1.0E-4_dp) THEN
               f0 = CMPLX(L - a*a*L**3/6.0_dp + a**4*L**5/120.0_dp, &
                          -a*L*L/2.0_dp + a**3*L**4/24.0_dp, dp)
            ELSE
               phase = EXP(-gaussi*a*L)
               f0 = (z_one - phase)/(gaussi*a)
            END IF
            factors(icell) = factors(icell) + weights(idir)*f0
         END DO
         factors(icell) = factors(icell)*volume*fourpi/(twopi**3)
         DO iq = 1, nk
            q = bs_env%kpoints_chi_eps_W%xkp(1:3, iq) - &
                ANINT(bs_env%kpoints_chi_eps_W%xkp(1:3, iq))
            scaled = ABS(q)*REAL(bs_env%nkp_grid_chi_eps_W_orig, dp)
            IF (MAXVAL(ABS(scaled - 0.5_dp)) >= 1.0E-9_dp) CYCLE
            qcart = MATMUL(twopi*TRANSPOSE(inv_3x3(bs_env%hmat)), q)
            qnorm = SQRT(SUM(qcart*qcart))
            phase = EXP(-gaussi*DOT_PRODUCT(qcart, rcart))
            factors(icell) = factors(icell) - phase/(REAL(nk, dp)*qnorm*qnorm)
         END DO
      END DO
      DEALLOCATE (directions, weights, radii)
   END SUBROUTINE singular_head_cell_factors

! **************************************************************************************************
!> \brief Replace the four shifted XY transfer samples by their 2D cell integral
!> of 1/|q|. The face-to-polar Jacobian resolves the integrable singularity
!> without sampling q=0. The result has units of inverse in-plane momentum.
!> \param bs_env ...
!> \param cells ...
!> \param factors ...
! **************************************************************************************************
   SUBROUTINE singular_head_cell_factors_2d(bs_env, cells, factors)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: cells(:, :)
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: factors(:)

      INTEGER, PARAMETER                                 :: n_face = 24

      COMPLEX(KIND=dp)                                   :: phase, radial
      INTEGER                                            :: axis, icell, iq, it, j, n_near, nk, &
                                                            sign_index
      REAL(KIND=dp) :: a1(3), a2(3), area, area_bz, b(3, 2), cross_product(3), dq(3), jacobian, &
         length, normal(3), qcart(3), qfrac(3), rcart(3), scaled(2), t
      REAL(KIND=dp), ALLOCATABLE                         :: nodes(:), rule_weights(:)

      IF (.NOT. ALL(bs_env%periodic == [1, 1, 0])) THEN
         CPABORT('2D Coulomb cell correction requires PERIODIC XY')
      END IF
      nk = PRODUCT(bs_env%nkp_grid_chi_eps_W_orig)
      a1 = bs_env%hmat(:, 1)
      a2 = bs_env%hmat(:, 2)
      normal = [a1(2)*a2(3) - a1(3)*a2(2), &
                a1(3)*a2(1) - a1(1)*a2(3), &
                a1(1)*a2(2) - a1(2)*a2(1)]
      area = SQRT(SUM(normal**2))
      CPASSERT(area > 0.0_dp)
      normal = normal/area
      b(:, 1) = twopi*[a2(2)*normal(3) - a2(3)*normal(2), &
                       a2(3)*normal(1) - a2(1)*normal(3), &
                       a2(1)*normal(2) - a2(2)*normal(1)]/area
      b(:, 2) = twopi*[normal(2)*a1(3) - normal(3)*a1(2), &
                       normal(3)*a1(1) - normal(1)*a1(3), &
                       normal(1)*a1(2) - normal(2)*a1(1)]/area
      area_bz = twopi**2/area
      CALL gauss_legendre_cell_rule(n_face, nodes, rule_weights)
      ALLOCATE (factors(SIZE(cells, 2)))
      n_near = 0
      DO iq = 1, nk
         qfrac = bs_env%kpoints_chi_eps_W%xkp(:, iq) - &
                 ANINT(bs_env%kpoints_chi_eps_W%xkp(:, iq))
         scaled = ABS(qfrac(1:2))*REAL(bs_env%nkp_grid_chi_eps_W_orig(1:2), dp)
         IF (MAXVAL(ABS(scaled - 0.5_dp)) < 1.0E-9_dp) n_near = n_near + 1
      END DO
      IF (n_near /= PRODUCT(MIN(2, bs_env%nkp_grid_chi_eps_W_orig(1:2)))) THEN
         CPABORT('Inconsistent number of 2D Gamma-neighbor cells')
      END IF
      DO icell = 1, SIZE(cells, 2)
         rcart = MATMUL(bs_env%hmat, REAL(cells(:, icell), dp))
         factors(icell) = z_zero
         DO axis = 1, 2
            j = 3 - axis
            dq = b(:, j)/REAL(MAX(2, bs_env%nkp_grid_chi_eps_W_orig(j)), dp)
            DO sign_index = -1, 1, 2
               DO it = 1, n_face
                  t = nodes(it)
                  qfrac = 0.0_dp
                  qfrac(axis) = REAL(sign_index, dp)/ &
                                REAL(MAX(2, bs_env%nkp_grid_chi_eps_W_orig(axis)), dp)
                  qfrac(j) = t/REAL(MAX(2, bs_env%nkp_grid_chi_eps_W_orig(j)), dp)
                  qcart = b(:, 1)*qfrac(1) + b(:, 2)*qfrac(2)
                  length = SQRT(SUM(qcart*qcart))
                  cross_product = [qcart(2)*dq(3) - qcart(3)*dq(2), &
                                   qcart(3)*dq(1) - qcart(1)*dq(3), &
                                   qcart(1)*dq(2) - qcart(2)*dq(1)]
                  jacobian = ABS(DOT_PRODUCT(normal, cross_product))/length**2
                  t = DOT_PRODUCT(qcart, rcart)/length
                  IF (ABS(t*length) < 1.0E-4_dp) THEN
                     radial = CMPLX(length - t*t*length**3/6.0_dp, &
                                    -t*length**2/2.0_dp + t**3*length**4/24.0_dp, dp)
                  ELSE
                     radial = (z_one - EXP(-gaussi*t*length))/(gaussi*t)
                  END IF
                  factors(icell) = factors(icell) + &
                                   rule_weights(it)*jacobian*radial/area_bz
               END DO
            END DO
         END DO
         DO iq = 1, nk
            qfrac = bs_env%kpoints_chi_eps_W%xkp(:, iq) - &
                    ANINT(bs_env%kpoints_chi_eps_W%xkp(:, iq))
            scaled = ABS(qfrac(1:2))*REAL(bs_env%nkp_grid_chi_eps_W_orig(1:2), dp)
            IF (MAXVAL(ABS(scaled - 0.5_dp)) >= 1.0E-9_dp) CYCLE
            qcart = b(:, 1)*qfrac(1) + b(:, 2)*qfrac(2)
            length = SQRT(SUM(qcart*qcart))
            phase = EXP(-gaussi*DOT_PRODUCT(qcart, rcart))
            factors(icell) = factors(icell) - phase/(REAL(nk, dp)*length)
         END DO
      END DO
      DEALLOCATE (nodes, rule_weights)
   END SUBROUTINE singular_head_cell_factors_2d

! **************************************************************************************************
!> \brief Separate the 2D charge mode and factor the neutral Γ Coulomb body.
!> For charge vector f, P=I-ff†/(f†f) and V_b=P V_reg(0) P. The Gamma correction
!> density map is M⁻¹ P V_b¹ᐟ²; the external-potential map is M⁻¹ V_reg P V_b⁻¹ᐟ².
!> Keeping the second map distinct retains the finite charge/body couplings.
!> \param qs_env ...
!> \param bs_env ...
!> \param metric ...
!> \param F_density ...
!> \param F_external ...
! **************************************************************************************************
   SUBROUTINE slab_gamma_factors(qs_env, bs_env, metric, F_density, F_external)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(INOUT)                    :: metric(:, :)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: F_density(:, :)
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: F_external(:, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'slab_gamma_factors'

      COMPLEX(KIND=dp), ALLOCATABLE :: body_inverse(:, :), body_sqrt(:, :), charge(:), &
         charge_contracted(:), projector(:, :), U(:, :), V_contracted(:, :), V_owned(:, :, :), &
         V_regular(:, :)
      INTEGER                                            :: handle, i_col, i_row, n_local, n_ref, &
                                                            n_ri
      TYPE(kpoint_type), POINTER                         :: gamma

      CALL timeset(routineN, handle)

      n_ri = bs_env%n_RI
      n_ref = n_ri
      IF (bs_env%auto_ri%enabled) n_ref = SUM(bs_env%auto_ri%sizes_ref_RI)
      n_local = 0
      IF (MODULO(1, bs_env%para_env%num_pe) == bs_env%para_env%mepos) n_local = 1
      ALLOCATE (gamma, V_owned(n_ref, n_ref, n_local), V_regular(n_ref, n_ref), charge(n_ref))
      ALLOCATE (gamma%xkp(3, 1))
      gamma%xkp = 0.0_dp
      gamma%nkp = 1
      gamma%nkp_grid = bs_env%nkp_grid_chi_eps_W_orig
      V_owned = z_zero
      CALL build_2c_coulomb_matrix_kp_small_cell(V_owned, qs_env, gamma, &
                                                 bs_env%size_lattice_sum_V, 'RI_AUX', 1, 1)
      V_regular(:, :) = z_zero
      CALL gamma_auxiliary_charges(qs_env, charge)
      IF (n_local == 1) THEN
         V_regular(:, :) = V_owned(:, :, 1)
      END IF
      CALL bs_env%para_env%sum(V_regular)
      DEALLOCATE (gamma%xkp)
      DEALLOCATE (gamma, V_owned)
      IF (bs_env%auto_ri%enabled) THEN
         CALL assemble_AA_contraction(bs_env, U)
         ALLOCATE (V_contracted(n_ri, n_ri), charge_contracted(n_ri))
         V_contracted(:, :) = MATMUL(CONJG(TRANSPOSE(U)), MATMUL(V_regular, U))
         charge_contracted(:) = MATMUL(CONJG(TRANSPOSE(U)), charge)
         CALL MOVE_ALLOC(V_contracted, V_regular)
         CALL MOVE_ALLOC(charge_contracted, charge)
         DEALLOCATE (U)
      END IF
      CPASSERT(REAL(DOT_PRODUCT(charge, charge), dp) > 0.0_dp)
      charge(:) = charge/SQRT(REAL(DOT_PRODUCT(charge, charge), dp))
      ALLOCATE (projector(n_ri, n_ri), body_sqrt(n_ri, n_ri), body_inverse(n_ri, n_ri))
      DO i_col = 1, n_ri
         DO i_row = 1, n_ri
            projector(i_row, i_col) = -charge(i_row)*CONJG(charge(i_col))
         END DO
         projector(i_col, i_col) = projector(i_col, i_col) + z_one
      END DO
      body_sqrt(:, :) = MATMUL(projector, MATMUL(V_regular, projector))
      body_inverse(:, :) = body_sqrt
      CALL power(body_sqrt, 0.5_dp, 0.0_dp, relative_eps=bs_env%regularization_RI)
      CALL power(body_inverse, -0.5_dp, 0.0_dp, relative_eps=bs_env%regularization_RI)
      CALL power(metric, -1.0_dp, 0.0_dp, relative_eps=bs_env%regularization_RI)
      F_density = MATMUL(metric, MATMUL(projector, body_sqrt))
      ALLOCATE (F_external(n_ri, n_ri))
      F_external(:, :) = MATMUL(metric, MATMUL(V_regular, MATMUL(projector, body_inverse)))
      DEALLOCATE (V_regular, charge, projector, body_sqrt, body_inverse)

      CALL timestop(handle)

   END SUBROUTINE slab_gamma_factors

! **************************************************************************************************
!> \brief In-plane reciprocal vectors and area for an arbitrary XY cell.
!> \param bs_env ...
!> \param b ...
!> \param area ...
! **************************************************************************************************
   SUBROUTINE slab_reciprocal_vectors(bs_env, b, area)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(KIND=dp), INTENT(OUT)                         :: b(3, 2), area

      REAL(KIND=dp)                                      :: a1(3), a2(3), normal(3)

      a1 = bs_env%hmat(:, 1)
      a2 = bs_env%hmat(:, 2)
      normal = [a1(2)*a2(3) - a1(3)*a2(2), a1(3)*a2(1) - a1(1)*a2(3), &
                a1(1)*a2(2) - a1(2)*a2(1)]
      area = SQRT(SUM(normal**2))
      CPASSERT(area > 0.0_dp)
      normal = normal/area
      b(:, 1) = twopi*[a2(2)*normal(3) - a2(3)*normal(2), &
                       a2(3)*normal(1) - a2(1)*normal(3), &
                       a2(1)*normal(2) - a2(2)*normal(1)]/area
      b(:, 2) = twopi*[normal(2)*a1(3) - normal(3)*a1(2), &
                       normal(3)*a1(1) - normal(1)*a1(3), &
                       normal(1)*a1(2) - normal(2)*a1(1)]/area
   END SUBROUTINE slab_reciprocal_vectors

! **************************************************************************************************
!> \brief Polar boundary quadrature of the four shifted Γ-neighbor cells.
!> Directions d, radii L and weights approximate dθ on the bounding parallelogram.
!> \param bs_env ...
!> \param directions ...
!> \param weights ...
!> \param radii ...
! **************************************************************************************************
   SUBROUTINE slab_cell_face_directions(bs_env, directions, weights, radii)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(KIND=dp), ALLOCATABLE, INTENT(OUT)            :: directions(:, :), weights(:), radii(:)

      INTEGER, PARAMETER                                 :: n_face = 24

      INTEGER                                            :: i_axis, i_direction, i_node, i_sign, &
                                                            j_axis
      REAL(KIND=dp)                                      :: area, b(3, 2), cross(3), dq(3), q(3), &
                                                            volume_check
      REAL(KIND=dp), ALLOCATABLE                         :: nodes(:), rule(:)

      CALL slab_reciprocal_vectors(bs_env, b, area)
      CALL gauss_legendre_cell_rule(n_face, nodes, rule)
      ALLOCATE (directions(3, 4*n_face), weights(4*n_face), radii(4*n_face))
      i_direction = 0
      DO i_axis = 1, 2
         j_axis = 3 - i_axis
         dq = b(:, j_axis)/REAL(bs_env%nkp_grid_chi_eps_W_orig(j_axis), dp)
         DO i_sign = -1, 1, 2
            DO i_node = 1, n_face
               i_direction = i_direction + 1
               q = REAL(i_sign, dp)*b(:, i_axis)/REAL(bs_env%nkp_grid_chi_eps_W_orig(i_axis), dp) &
                   + nodes(i_node)*dq
               radii(i_direction) = SQRT(SUM(q*q))
               directions(:, i_direction) = q/radii(i_direction)
               cross = [q(2)*dq(3) - q(3)*dq(2), q(3)*dq(1) - q(1)*dq(3), &
                        q(1)*dq(2) - q(2)*dq(1)]
               weights(i_direction) = rule(i_node)*SQRT(SUM(cross**2))/SUM(q*q)
            END DO
         END DO
      END DO
      volume_check = SUM(weights*radii**2)/2.0_dp
      IF (ABS(volume_check/(4.0_dp*twopi**2/area/ &
                            REAL(PRODUCT(bs_env%nkp_grid_chi_eps_W_orig), dp)) - 1.0_dp) > 1.0E-8_dp) THEN
         CPABORT('2D Gamma cell quadrature failed its area check')
      END IF
      DEALLOCATE (nodes, rule)
   END SUBROUTINE slab_cell_face_directions

! **************************************************************************************************
!> \brief Integrate the finite 2D Gamma-correction head and wings before the time transform.
!> With A(d,ω)=d†[α-E†B⁻¹E]d and v=2π/q, Wc_00=-4π²A/(1+2πqA).
!> Wc_l0=-2π t_l(d,ω)/(1+2πqA); t is linear in d. Area-normalized
!> cell integrals minus the four mesh samples retain exp(-iqR). Smooth body
!> terms, including the O(q) Gamma-correction body update, retain the regular mesh rule.
!> Only scalar/Cartesian weights are stored per cell, never a grid-pair tensor.
!> \param bs_env ...
!> \param model ...
!> \param alpha ...
! **************************************************************************************************
   SUBROUTINE slab_gamma_corr_cell_integrals(bs_env, model, alpha)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gamma_correction_type), INTENT(INOUT)         :: model
      COMPLEX(KIND=dp), INTENT(IN)                       :: alpha(:, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'slab_gamma_corr_cell_integrals'
      INTEGER, PARAMETER                                 :: n_radial = 32

      COMPLEX(KIND=dp)                                   :: integral, phase
      COMPLEX(KIND=dp), ALLOCATABLE                      :: head_frequency(:, :)
      INTEGER                                            :: handle, i_cell, i_direction, &
                                                            i_frequency, i_node, i_t, n_cells, &
                                                            n_frequency, nk
      INTEGER, ALLOCATABLE                               :: cells(:, :)
      REAL(KIND=dp)                                      :: a, area, area_bz, b(3, 2), length, &
                                                            projection, q, qcart(3), &
                                                            radial_weight, rcart(3), start_time
      REAL(KIND=dp), ALLOCATABLE                         :: nodes(:), rule(:)

      CALL timeset(routineN, handle)

      start_time = m_walltime()
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A)') 'GW 2D: integrate finite Gamma-correction head and wings'
         CALL m_flush(bs_env%unit_nr)
      END IF
      CALL build_full_cell_grid(bs_env, cells)
      CALL slab_reciprocal_vectors(bs_env, b, area)
      area_bz = twopi**2/area
      nk = PRODUCT(bs_env%nkp_grid_chi_eps_W_orig)
      n_cells = SIZE(cells, 2)
      n_frequency = bs_env%num_time_freq_points
      ALLOCATE (head_frequency(n_cells, n_frequency), SOURCE=z_zero)
      ALLOCATE (model%slab_wing_cell_frequency(3, n_frequency, n_cells), SOURCE=z_zero)
      CALL gauss_legendre_cell_rule(n_radial, nodes, rule)
      DO i_cell = bs_env%para_env%mepos + 1, n_cells, bs_env%para_env%num_pe
         rcart = MATMUL(bs_env%hmat, REAL(cells(:, i_cell), dp))
         DO i_direction = 1, SIZE(model%radii)
            length = model%radii(i_direction)
            projection = DOT_PRODUCT(model%directions(:, i_direction), rcart)
            DO i_frequency = 1, n_frequency
               a = REAL(alpha(i_direction, i_frequency), dp)
               integral = z_zero
               IF (i_direction <= model%n_angular) THEN
                  DO i_node = 1, n_radial
                     q = 0.5_dp*length*(nodes(i_node) + 1.0_dp)
                     radial_weight = 0.5_dp*length*rule(i_node)*q/(1.0_dp + twopi*q*a)
                     integral = integral + radial_weight*EXP(-gaussi*q*projection)
                  END DO
                  integral = integral*model%weights(i_direction)/area_bz
               ELSE
                  qcart = MATMUL(b, model%near_q(1:2, i_direction - model%n_angular))
                  phase = EXP(-gaussi*DOT_PRODUCT(qcart, rcart))
                  integral = -phase/(REAL(nk, dp)*(1.0_dp + twopi*length*a))
               END IF
               head_frequency(i_cell, i_frequency) = head_frequency(i_cell, i_frequency) &
                                                     - twopi**2*a*integral/ri_metric_head_zero(bs_env)**2
               model%slab_wing_cell_frequency(:, i_frequency, i_cell) = &
                  model%slab_wing_cell_frequency(:, i_frequency, i_cell) &
                  + integral*model%directions(:, i_direction)
            END DO
         END DO
      END DO
      CALL bs_env%para_env%sum(head_frequency)
      CALL bs_env%para_env%sum(model%slab_wing_cell_frequency)
      ALLOCATE (model%slab_head_cell_tau(n_cells, n_frequency), SOURCE=z_zero)
      DO i_t = 1, n_frequency
         DO i_frequency = 1, n_frequency
            model%slab_head_cell_tau(:, i_t) = model%slab_head_cell_tau(:, i_t) &
                                               + bs_env%time_frequency_grid%cosine_frequency_to_time(i_t, i_frequency) &
                                               *head_frequency(:, i_frequency)
         END DO
      END DO
      DEALLOCATE (cells, nodes, rule, head_frequency)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T69,F12.1)') &
            'GW 2D Gamma-correction cell integrals wall time [s]:', m_walltime() - start_time
         CALL m_flush(bs_env%unit_nr)
      END IF

      CALL timestop(handle)

   END SUBROUTINE slab_gamma_corr_cell_integrals

! **************************************************************************************************
!> \brief Apply the integrated 2D head and wings to local grid rows or pairs.
!> \param bs_env ...
!> \param cells ...
!> \param model ...
!> \param i_t ...
!> \param Z_zero_q ...
!> \param first_row ...
!> \param first_pair ...
!> \param W_grid_R ...
! **************************************************************************************************
   SUBROUTINE add_slab_gamma_corr_cell_correction(bs_env, cells, model, i_t, Z_zero_q, &
                                                  first_row, first_pair, W_grid_R)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: cells(:, :)
      TYPE(gamma_correction_type), INTENT(IN)            :: model
      INTEGER, INTENT(IN)                                :: i_t
      COMPLEX(KIND=dp), INTENT(IN)                       :: Z_zero_q(:, :, :)
      INTEGER, INTENT(IN)                                :: first_row, first_pair
      REAL(KIND=dp), INTENT(INOUT)                       :: W_grid_R(:, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'add_slab_gamma_corr_cell_correction'

      COMPLEX(KIND=dp)                                   :: coefficient(3)
      COMPLEX(KIND=dp), ALLOCATABLE                      :: wing_left(:), wing_right(:)
      INTEGER                                            :: handle, i_cell, i_frequency, n_grid

      CALL timeset(routineN, handle)

      n_grid = SIZE(Z_zero_q, 1)
      CPASSERT(SIZE(cells, 2) == SIZE(model%slab_head_cell_tau, 1))
      ALLOCATE (wing_left(n_grid), wing_right(n_grid))
      DO i_cell = 1, SIZE(cells, 2)
         wing_left(:) = z_zero
         wing_right(:) = z_zero
         DO i_frequency = 1, bs_env%num_time_freq_points
            coefficient = model%slab_wing_cell_frequency(:, i_frequency, i_cell) &
                          *bs_env%time_frequency_grid%cosine_frequency_to_time(i_t, i_frequency)
            wing_left(:) = wing_left + MATMUL(model%slab_wing_cartesian(:, :, i_frequency), coefficient)
            wing_right(:) = wing_right &
                            + MATMUL(CONJG(model%slab_wing_cartesian(:, :, i_frequency)), coefficient)
         END DO
         CALL add_gamma_corr_pair_cell(bs_env, cells, i_cell, model%slab_head_cell_tau(i_cell, i_t), &
                                       Z_zero_q(:, 1, 1), first_row, first_pair, W_grid_R, &
                                       wing_left, wing_right)
      END DO
      DEALLOCATE (wing_left, wing_right)

      CALL timestop(handle)

   END SUBROUTINE add_slab_gamma_corr_cell_correction

! **************************************************************************************************
!> \brief Replace the distinct near-Gamma model samples by their cell integrals.
!> The directional Gamma-correction response is kept in low-rank form: one head scalar
!> and two wing vectors per real-space cell, never a grid-pair angular tensor.
!> \param bs_env ...
!> \param cells ...
!> \param model ...
!> \param i_t ...
!> \param Z_zero_q ...
!> \param first_row ...
!> \param first_pair ...
!> \param W_grid_R ...
! **************************************************************************************************
   SUBROUTINE add_gamma_corr_cell_correction(bs_env, cells, model, i_t, Z_zero_q, &
                                             first_row, first_pair, W_grid_R)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: cells(:, :)
      TYPE(gamma_correction_type), INTENT(IN)            :: model
      INTEGER, INTENT(IN)                                :: i_t
      COMPLEX(KIND=dp), INTENT(IN)                       :: Z_zero_q(:, :, :)
      INTEGER, INTENT(IN)                                :: first_row, first_pair
      REAL(KIND=dp), INTENT(INOUT)                       :: W_grid_R(:, :, :)

      COMPLEX(KIND=dp)                                   :: f0, f1, head_scalar, kernel_h, kernel_w, &
                                                            phase
      COMPLEX(KIND=dp), ALLOCATABLE                      :: wing_left(:), wing_right(:)
      INTEGER                                            :: icell, idir, n_grid, nk
      REAL(KIND=dp)                                      :: a, L, qcart(3), qnorm, rcart(3), volume

      n_grid = SIZE(Z_zero_q, 1)
      nk = PRODUCT(bs_env%nkp_grid_chi_eps_W_orig)
      volume = bs_env%ri_rs%cell%deth
      ALLOCATE (wing_left(n_grid), wing_right(n_grid))
      DO icell = 1, SIZE(cells, 2)
         rcart = MATMUL(bs_env%hmat, REAL(cells(:, icell), dp))
         head_scalar = z_zero
         wing_left(:) = z_zero
         wing_right(:) = z_zero
         DO idir = 1, SIZE(model%weights)
            IF (idir <= model%n_angular) THEN
               L = model%radii(idir)
               a = DOT_PRODUCT(model%directions(:, idir), rcart)
               IF (ABS(a*L) < 1.0E-4_dp) THEN
                  f0 = CMPLX(L - a*a*L**3/6.0_dp + a**4*L**5/120.0_dp, &
                             -a*L*L/2.0_dp + a**3*L**4/24.0_dp, dp)
                  f1 = CMPLX(L*L/2.0_dp - a*a*L**4/8.0_dp, &
                             -a*L**3/3.0_dp + a**3*L**5/30.0_dp, dp)
               ELSE
                  phase = EXP(-gaussi*a*L)
                  f0 = (z_one - phase)/(gaussi*a)
                  f1 = (phase*(z_one + gaussi*a*L) - z_one)/(a*a)
               END IF
               kernel_h = volume*fourpi/(twopi**3)*model%weights(idir)*f0
               kernel_w = volume*fourpi/(twopi**3)*model%weights(idir)*f1
            ELSE
               qcart = MATMUL(twopi*TRANSPOSE(inv_3x3(bs_env%hmat)), &
                              model%near_q(:, idir - model%n_angular))
               qnorm = SQRT(SUM(qcart*qcart))
               phase = EXP(-gaussi*DOT_PRODUCT(qcart, rcart))
               kernel_h = -phase/(REAL(nk, dp)*qnorm*qnorm)
               kernel_w = -phase/(REAL(nk, dp)*qnorm)
            END IF
            head_scalar = head_scalar + kernel_h*model%head_tau(idir, i_t)
            wing_left(:) = wing_left(:) + kernel_w*model%wing_tau(:, idir, i_t)
            wing_right(:) = wing_right(:) + kernel_w*CONJG(model%wing_tau(:, idir, i_t))
         END DO
         head_scalar = head_scalar*fourpi/ &
                       ri_metric_head_zero(bs_env)**2
         CALL add_gamma_corr_pair_cell(bs_env, cells, icell, head_scalar, Z_zero_q(:, 1, 1), &
                                       first_row, first_pair, W_grid_R, wing_left, wing_right)
      END DO
      DEALLOCATE (wing_left, wing_right)
   END SUBROUTINE add_gamma_corr_cell_correction

! **************************************************************************************************
!> \brief Contract the occupied density with the bare RI Coulomb operator.
!> \param bs_env Band-structure settings and physical results.
!> \param gwkp_env Periodic GW work data.
! **************************************************************************************************
   SUBROUTINE compute_Sigma_x_from_V(bs_env, gwkp_env)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gwkp_type), INTENT(INOUT), TARGET             :: gwkp_env

      CHARACTER(LEN=*), PARAMETER :: routineN = 'compute_Sigma_x_from_V'

      COMPLEX(KIND=dp), ALLOCATABLE :: D_grid_k(:, :, :), D_grid_R_complex(:, :, :), &
         head_cell_factor(:), interaction_q(:, :, :), Sigma_grid_k(:, :, :), V_grid_q(:, :, :), &
         V_grid_R_complex(:, :, :)
      INTEGER                                            :: first_row, handle, i_cell, i_col, i_row, &
                                                            iq, iq_local, ispin, last_row, n_grid, &
                                                            n_q_local, nrows_local, phase_handle
      INTEGER, ALLOCATABLE                               :: full_cells(:, :)
      REAL(KIND=dp)                                      :: head_metric_zero
      REAL(KIND=dp), ALLOCATABLE                         :: qpoints(:, :), Sigma_grid_R(:, :, :), &
                                                            V_grid_R(:, :, :)

      CALL timeset(routineN, handle)

      ALLOCATE (gwkp_env%sigma_x_bands(bs_env%n_ao, bs_env%nkp_bs_and_DOS, bs_env%n_spin))
      n_grid = SIZE(gwkp_env%psi_lm_k, 1)
      CALL local_grid_row_range(n_grid, bs_env%para_env, first_row, last_row, nrows_local)
      ALLOCATE (V_grid_q(nrows_local, n_grid, bs_env%nkp_scf_desymm))
      n_q_local = SIZE(gwkp_env%Z_lP_q, 3)
      IF (SIZE(gwkp_env%Z_lP_zero, 2) /= bs_env%n_RI + 1 .OR. SIZE(gwkp_env%Z_lP_zero, 3) /= 1) THEN
         CPABORT("Periodic RI-RS GW requires the direct Gamma auxiliary fit")
      END IF
      ALLOCATE (qpoints(3, bs_env%nkp_scf_desymm))
      DO iq = 1, bs_env%nkp_scf_desymm
         qpoints(:, iq) = momentum_transfer_point(bs_env, iq)
      END DO
      CPASSERT(SIZE(gwkp_env%Z_lP_q, 2) == bs_env%n_RI)
      CPASSERT(ALLOCATED(gwkp_env%metric_coulomb_factor))
      ALLOCATE (interaction_q(SIZE(gwkp_env%Z_lP_q, 2), SIZE(gwkp_env%Z_lP_q, 2), n_q_local))
      DO iq_local = 1, n_q_local
         interaction_q(:, :, iq_local) = MATMUL(gwkp_env%metric_coulomb_factor(:, :, iq_local), &
                                                CONJG(TRANSPOSE(gwkp_env%metric_coulomb_factor(:, :, iq_local))))
      END DO
      DEALLOCATE (gwkp_env%metric_coulomb_factor)
      CALL timeset('Sigma_x_project_V_grid', phase_handle)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A)') 'GW exchange: projecting bare Coulomb onto RI-RS grid'
         CALL m_flush(bs_env%unit_nr)
      END IF
      CALL project_auxiliary_interaction_q_to_fft_rows(gwkp_env%Z_lP_q, interaction_q, &
                                                       bs_env%nkp_scf_desymm, bs_env%para_env, &
                                                       V_grid_q)
      CALL timestop(phase_handle)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A)') 'GW exchange: bare-Coulomb grid projection complete'
         CALL m_flush(bs_env%unit_nr)
      END IF
      DEALLOCATE (interaction_q)
      CALL build_full_cell_grid(bs_env, full_cells)
      ALLOCATE (V_grid_R_complex(nrows_local, n_grid, SIZE(full_cells, 2)))

      CALL k_grid_to_cell_fft(V_grid_q, qpoints, bs_env%kpoints_scf_desymm%nkp_grid, &
                              full_cells, V_grid_R_complex)
      DEALLOCATE (V_grid_q)
      CALL singular_head_cell_factors(bs_env, full_cells, head_cell_factor)
      head_metric_zero = ri_metric_head_zero(bs_env)
      ALLOCATE (V_grid_R(nrows_local, n_grid, SIZE(full_cells, 2)))
      V_grid_R(:, :, :) = REAL(V_grid_R_complex, KIND=dp)
      DEALLOCATE (V_grid_R_complex)
      DO iq = 1, SIZE(full_cells, 2)
         IF (ALL(bs_env%periodic == [1, 1, 0])) THEN
            CALL add_gamma_corr_pair_cell(bs_env, full_cells, iq, &
                                          twopi/head_metric_zero**2*head_cell_factor(iq), &
                                          gwkp_env%Z_lP_zero(:, 1, 1), first_row, 0, V_grid_R)
         ELSE
            CALL add_gamma_corr_pair_cell(bs_env, full_cells, iq, &
                                          fourpi/head_metric_zero**2*head_cell_factor(iq), &
                                          gwkp_env%Z_lP_zero(:, 1, 1), first_row, 0, V_grid_R)
         END IF
      END DO
      DEALLOCATE (head_cell_factor)

      DO ispin = 1, bs_env%n_spin
         CALL timeset('Sigma_x_density_grid', phase_handle)
         ALLOCATE (D_grid_k(nrows_local, n_grid, bs_env%nkp_scf_desymm))
         CALL build_green_grid_k_rows(bs_env, gwkp_env%psi_lm_k(:, :, :, ispin), &
                                      bs_env%para_env, 0.0_dp, ispin, .TRUE., &
                                      first_row, last_row, D_grid_k)
         ALLOCATE (D_grid_R_complex(nrows_local, n_grid, SIZE(full_cells, 2)))
         CALL k_grid_to_cell_fft(D_grid_k, bs_env%kpoints_scf_desymm%xkp, &
                                 bs_env%kpoints_scf_desymm%nkp_grid, full_cells, D_grid_R_complex)
         DEALLOCATE (D_grid_k)
         ALLOCATE (Sigma_grid_R(nrows_local, n_grid, SIZE(full_cells, 2)))
         !$OMP PARALLEL DO COLLAPSE(2) DEFAULT(NONE) SCHEDULE(STATIC) &
         !$OMP SHARED(D_grid_R_complex, V_grid_R, Sigma_grid_R, nrows_local, n_grid, full_cells) &
         !$OMP PRIVATE(i_cell, i_col, i_row)
         DO i_cell = 1, SIZE(full_cells, 2)
            DO i_col = 1, n_grid
               DO i_row = 1, nrows_local
                  Sigma_grid_R(i_row, i_col, i_cell) = &
                     -REAL(D_grid_R_complex(i_row, i_col, i_cell), KIND=dp)*V_grid_R(i_row, i_col, i_cell)
               END DO
            END DO
         END DO
         !$OMP END PARALLEL DO
         DEALLOCATE (D_grid_R_complex)
         IF (ispin == bs_env%n_spin) DEALLOCATE (V_grid_R)
         ALLOCATE (Sigma_grid_k(nrows_local, n_grid, bs_env%nkp_scf_desymm))
         CALL product_to_k_pair_centered(bs_env, full_cells, bs_env%kpoints_scf_desymm%xkp, &
                                         Sigma_grid_R, Sigma_grid_k, first_row, 0)
         DEALLOCATE (Sigma_grid_R)
         CALL timestop(phase_handle)

         CALL self_energy_RS_to_Bloch(bs_env, Sigma_grid_k, first_row, last_row, ispin, &
                                      gwkp_env%sigma_x_bands(:, :, ispin), gwkp_env%interpolation, &
                                      gwkp_env%band_orbitals)
         DEALLOCATE (Sigma_grid_k)
      END DO

      DEALLOCATE (full_cells, qpoints)

      CALL timestop(handle)

   END SUBROUTINE compute_Sigma_x_from_V

! **************************************************************************************************
!> \brief Form Σc with one owner per ordered (l,l') pair through both lattice FFTs.
!> \param bs_env Band-structure settings and physical results.
!> \param gwkp_env Periodic GW work data.
! **************************************************************************************************
   SUBROUTINE compute_Sigma_c_from_W_PQ(bs_env, gwkp_env)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gwkp_type), INTENT(INOUT), TARGET             :: gwkp_env

      CHARACTER(LEN=*), PARAMETER :: routineN = 'compute_Sigma_c_from_W_PQ'

      COMPLEX(dp), ALLOCATABLE, TARGET                   :: scf_orbitals(:, :, :, :)
      COMPLEX(KIND=dp), ALLOCATABLE                      :: G_grid_k(:, :, :)
      COMPLEX(KIND=dp), POINTER                          :: scf_values(:, :, :, :)
      INTEGER                                            :: handle, i_group, i_t, iq, ispin, n_grid, &
                                                            n_group_times, n_pairs_local, &
                                                            phase_handle, shared_ranks
      INTEGER, ALLOCATABLE                               :: full_cells(:, :)
      LOGICAL                                            :: valid_layout
      REAL(KIND=dp)                                      :: step_elapsed, step_start, tau
      REAL(KIND=dp), ALLOCATABLE :: qpoints(:, :), sigma_band(:, :), Sigma_grid_R(:, :, :), &
         sigma_points(:, :), sigma_result_buffer(:), step_times(:), W_grid_R(:, :, :)
      TYPE(pair_transfer_plan_type)                      :: green_plan, W_plan

      CALL timeset(routineN, handle)

      CPASSERT(ALLOCATED(gwkp_env%band_orbitals))
      IF (ALLOCATED(gwkp_env%sigma_c_bands)) DEALLOCATE (gwkp_env%sigma_c_bands)
      CALL build_full_cell_grid(bs_env, full_cells)
      n_grid = SIZE(gwkp_env%psi_lm_k, 1)
      CALL initialize_pair_transfer(W_plan, n_grid, bs_env%nkp_scf_desymm, 0, gwkp_env%group_env)
      CALL initialize_pair_transfer(green_plan, n_grid, bs_env%nkp_scf_desymm, -1, gwkp_env%group_env)
      n_pairs_local = W_plan%n_local
      CPASSERT(green_plan%first_pair == W_plan%first_pair)
      CPASSERT(green_plan%n_local == n_pairs_local)
      n_group_times = (bs_env%num_time_freq_points - gwkp_env%my_group + gwkp_env%ngroups - 1)/gwkp_env%ngroups
      DO iq = 1, SIZE(gwkp_env%W_PQ_tau)
         CPASSERT(ALLOCATED(gwkp_env%W_PQ_tau(iq)%values))
         IF (.NOT. (SIZE(gwkp_env%W_PQ_tau(iq)%values, 1) == SIZE(gwkp_env%Z_lP_q, 2))) THEN
            CPABORT("Internal GW array shape mismatch")
         END IF
         valid_layout = SIZE(gwkp_env%W_PQ_tau(iq)%values, 3) == n_group_times
         CPASSERT(valid_layout)
      END DO
      ALLOCATE (G_grid_k(n_pairs_local, 1, bs_env%nkp_scf_desymm))
      ALLOCATE (W_grid_R(n_pairs_local, 1, SIZE(full_cells, 2)))
      ALLOCATE (Sigma_grid_R, MOLD=W_grid_R)
      ALLOCATE (qpoints(3, bs_env%nkp_scf_desymm))
      ALLOCATE (sigma_points(3, bs_env%nkp_scf_desymm))
      DO iq = 1, bs_env%nkp_scf_desymm
         qpoints(:, iq) = momentum_transfer_point(bs_env, iq)
         sigma_points(:, iq) = bs_env%kpoints_scf_desymm%xkp(:, iq)
      END DO
      ALLOCATE (sigma_band(bs_env%n_ao, bs_env%nkp_bs_and_DOS))
      ALLOCATE (gwkp_env%sigma_c_bands(bs_env%n_ao, bs_env%nkp_bs_and_DOS, &
                                       bs_env%num_time_freq_points, 2, bs_env%n_spin))
      ALLOCATE (step_times(bs_env%num_time_freq_points), SOURCE=0.0_dp)
      gwkp_env%sigma_c_bands = 0.0_dp
      NULLIFY (scf_values)
      IF (orbitals_cover_scf(bs_env, gwkp_env%band_orbitals)) THEN
         scf_values => gwkp_env%band_orbitals
      ELSE
         CALL collect_scf_grid_orbitals(bs_env, gwkp_env%psi_lm_k, gwkp_env%group_env, scf_orbitals)
         IF (ALLOCATED(scf_orbitals)) scf_values => scf_orbitals
      END IF
      shared_ranks = MERGE(1, 0, ASSOCIATED(scf_values))
      CALL bs_env%para_env%sum(shared_ranks)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') &
            'GW Sigma ranks avoiding Green redistribution:', shared_ranks
         CALL m_flush(bs_env%unit_nr)
      END IF

      DO i_t = gwkp_env%my_group + 1, bs_env%num_time_freq_points, gwkp_env%ngroups
         step_start = m_walltime()
         i_group = (i_t - gwkp_env%my_group - 1)/gwkp_env%ngroups + 1
         CALL timeset('W_RI_2_grid_pairs', phase_handle)
         CALL project_auxiliary_time_q_to_fft_pairs(gwkp_env%Z_lP_q, &
                                                    gwkp_env%W_PQ_tau, i_group, &
                                                    gwkp_env%group_env, W_plan, G_grid_k)
         CALL timestop(phase_handle)

         CALL timeset('W_grid_pairs_2_RS', phase_handle)
         CALL gw_k_grid_to_real_cells(G_grid_k, qpoints, &
                                      bs_env%kpoints_scf_desymm%nkp_grid, full_cells, &
                                      W_grid_R)
         CALL timestop(phase_handle)
         IF (ALL(bs_env%periodic == [1, 1, 0])) THEN
            CALL add_slab_gamma_corr_cell_correction(bs_env, full_cells, gwkp_env%gamma_correction, i_t, &
                                                     gwkp_env%Z_lP_zero, 0, W_plan%first_pair, W_grid_R)
         ELSE
            CALL add_gamma_corr_cell_correction(bs_env, full_cells, gwkp_env%gamma_correction, i_t, &
                                                gwkp_env%Z_lP_zero, 0, W_plan%first_pair, W_grid_R)
         END IF
         CALL truncate_grid_W_by_pair_distance_pairs(bs_env, full_cells, W_plan, W_grid_R)
         tau = bs_env%time_frequency_grid%imaginary_time(i_t)
         DO ispin = 1, bs_env%n_spin
            CALL timeset('Sigma_c_green_pairs_to_k', phase_handle)
            CALL build_green_grid_k_pairs(bs_env, gwkp_env%psi_lm_k(:, :, :, ispin), &
                                          gwkp_env%group_env, green_plan, tau, ispin, .TRUE., G_grid_k, scf_values)
            CALL gw_k_grid_to_real_cells(G_grid_k, bs_env%kpoints_scf_desymm%xkp, &
                                         bs_env%kpoints_scf_desymm%nkp_grid, full_cells, Sigma_grid_R)
            CALL filter_full_green_cells(bs_env, gwkp_env%group_env, Sigma_grid_R)
            Sigma_grid_R(:, :, :) = -Sigma_grid_R*W_grid_R
            CALL product_to_k_pair_centered(bs_env, full_cells, sigma_points, Sigma_grid_R, &
                                            G_grid_k, 0, W_plan%first_pair)
            CALL timestop(phase_handle)
            CALL self_energy_pairs_to_Bloch(bs_env, G_grid_k, W_plan%first_pair, ispin, &
                                            sigma_band, gwkp_env%group_env, gwkp_env%interpolation, gwkp_env%band_orbitals)
            IF (gwkp_env%group_env%mepos == 0) gwkp_env%sigma_c_bands(:, :, i_t, 2, ispin) = sigma_band

            CALL timeset('Sigma_c_green_pairs_to_k', phase_handle)
            CALL build_green_grid_k_pairs(bs_env, gwkp_env%psi_lm_k(:, :, :, ispin), &
                                          gwkp_env%group_env, green_plan, tau, ispin, .FALSE., G_grid_k, scf_values)
            CALL gw_k_grid_to_real_cells(G_grid_k, bs_env%kpoints_scf_desymm%xkp, &
                                         bs_env%kpoints_scf_desymm%nkp_grid, full_cells, Sigma_grid_R)
            CALL filter_full_green_cells(bs_env, gwkp_env%group_env, Sigma_grid_R)
            Sigma_grid_R(:, :, :) = Sigma_grid_R*W_grid_R
            CALL product_to_k_pair_centered(bs_env, full_cells, sigma_points, Sigma_grid_R, &
                                            G_grid_k, 0, W_plan%first_pair)
            CALL timestop(phase_handle)
            CALL self_energy_pairs_to_Bloch(bs_env, G_grid_k, W_plan%first_pair, ispin, &
                                            sigma_band, gwkp_env%group_env, gwkp_env%interpolation, gwkp_env%band_orbitals)
            IF (gwkp_env%group_env%mepos == 0) gwkp_env%sigma_c_bands(:, :, i_t, 1, ispin) = sigma_band
         END DO
         step_elapsed = m_walltime() - step_start
         IF (gwkp_env%group_env%mepos == 0) step_times(i_t) = step_elapsed
      END DO

      CALL bs_env%para_env%sum(step_times)
      IF (bs_env%unit_nr > 0) THEN
         DO i_t = 1, bs_env%num_time_freq_points
            WRITE (bs_env%unit_nr, '(T2,A,I3,A,I0,T57,A,T71,F10.1,A)') &
               'GW Σ_c(iτ) point ', i_t, ' / ', bs_env%num_time_freq_points, &
               'Exec. time:', step_times(i_t), ' s'
         END DO
         CALL m_flush(bs_env%unit_nr)
      END IF

      CALL timeset('Sigma_c_result_redistribution', phase_handle)
      ALLOCATE (sigma_result_buffer(SIZE(gwkp_env%sigma_c_bands)))
      sigma_result_buffer(:) = RESHAPE(gwkp_env%sigma_c_bands, [SIZE(gwkp_env%sigma_c_bands)])
      CALL bs_env%para_env%sum(sigma_result_buffer)
      gwkp_env%sigma_c_bands(:, :, :, :, :) = RESHAPE(sigma_result_buffer, SHAPE(gwkp_env%sigma_c_bands))
      CALL timestop(phase_handle)

      DEALLOCATE (full_cells, G_grid_k, qpoints, &
                  sigma_band, sigma_points, sigma_result_buffer, Sigma_grid_R, &
                  step_times, W_grid_R)
      IF (ALLOCATED(scf_orbitals)) DEALLOCATE (scf_orbitals)

      CALL timestop(handle)

   END SUBROUTINE compute_Sigma_c_from_W_PQ

! **************************************************************************************************
!> \brief Drop an entire real-space Green matrix only when its full Frobenius norm,
!>        divided by the number of atom pairs, is below EPS_FILTER.
!> \param bs_env ...
!> \param para_env ...
!> \param G_R ...
! **************************************************************************************************
   SUBROUTINE filter_full_green_cells(bs_env, para_env, G_R)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(mp_para_env_type), POINTER                    :: para_env
      REAL(KIND=dp), INTENT(INOUT)                       :: G_R(:, :, :)

      INTEGER                                            :: i_cell, n_cells
      REAL(KIND=dp)                                      :: atom_block_count
      REAL(KIND=dp), ALLOCATABLE                         :: norm_squared(:)

      CPASSERT(bs_env%n_atom > 0)
      n_cells = SIZE(G_R, 3)
      atom_block_count = REAL(bs_env%n_atom, dp)**2
      ALLOCATE (norm_squared(n_cells))
      DO i_cell = 1, n_cells
         norm_squared(i_cell) = SUM(G_R(:, :, i_cell)**2)
      END DO
      CALL para_env%sum(norm_squared)
      DO i_cell = 1, n_cells
         IF (SQRT(norm_squared(i_cell))/atom_block_count < bs_env%eps_filter) THEN
            G_R(:, :, i_cell) = 0.0_dp
         END IF
      END DO
      DEALLOCATE (norm_squared)
   END SUBROUTINE filter_full_green_cells

! **************************************************************************************************
!> \brief Apply the same real-space cutoff to ordered pairs owned by this rank.
!> \param bs_env ...
!> \param cells ...
!> \param plan ...
!> \param W_grid_R ...
! **************************************************************************************************
   SUBROUTINE truncate_grid_W_by_pair_distance_pairs(bs_env, cells, plan, W_grid_R)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: cells
      TYPE(pair_transfer_plan_type), INTENT(IN)          :: plan
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(INOUT)   :: W_grid_R

      INTEGER                                            :: i_l, i_local, i_lp, i_pair, i_R
      REAL(KIND=dp)                                      :: cell_index(3), cutoff_squared, &
                                                            displacement(3), lattice_vector(3)

      IF (bs_env%ri_rs%cutoff_radius_v_w <= 0.0_dp) RETURN
      CPASSERT(ALLOCATED(bs_env%ri_rs%grid_points))
      CPASSERT(SIZE(W_grid_R, 1) == plan%n_local)
      CPASSERT(SIZE(W_grid_R, 2) == 1)
      CPASSERT(SIZE(cells, 2) == SIZE(W_grid_R, 3))
      cutoff_squared = bs_env%ri_rs%cutoff_radius_v_w**2
      !$OMP PARALLEL DO DEFAULT(NONE) SCHEDULE(STATIC) &
      !$OMP SHARED(bs_env, cells, cutoff_squared, plan, W_grid_R) &
      !$OMP PRIVATE(cell_index, displacement, i_l, i_local, i_lp, i_pair, i_R, lattice_vector)
      DO i_R = 1, SIZE(cells, 2)
         cell_index(:) = REAL(cells(:, i_R), dp)
         lattice_vector(:) = MATMUL(bs_env%hmat, cell_index)
         DO i_local = 1, plan%n_local
            i_pair = plan%first_pair + i_local - 1
            i_l = MODULO(i_pair - 1, plan%n_grid) + 1
            i_lp = (i_pair - 1)/plan%n_grid + 1
            displacement = bs_env%ri_rs%grid_points(:, i_l) - &
                           bs_env%ri_rs%grid_points(:, i_lp) - lattice_vector
            IF (SUM(displacement**2) > cutoff_squared) W_grid_R(i_local, 1, i_R) = 0.0_dp
         END DO
      END DO
      !$OMP END PARALLEL DO

   END SUBROUTINE truncate_grid_W_by_pair_distance_pairs

! **************************************************************************************************
!> \brief Collect SCF grid orbitals ψ_lm(k) on each rank for pair-local Green functions.
!> G_ll′(k,τ) = Σ_m ψ_lm(k) ψ_l′m(k)* exp(-|τ(ε_mk-ε_F)|), restricted to occupied or virtual m.
!> Copy the original cyclic k-owned orbitals without interpolation or band truncation.
!> Pair owners evaluate their local G elements without transferring a full matrix at every τ.
!> Keep the existing distributed path when the additional shared orbitals and thread work do not fit.
!> \param bs_env ...
!> \param psi ...
!> \param para_env ...
!> \param orbitals ...
! **************************************************************************************************
   SUBROUTINE collect_scf_grid_orbitals(bs_env, psi, para_env, orbitals)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: psi(:, :, :, :)
      TYPE(mp_para_env_type), POINTER                    :: para_env
      COMPLEX(dp), ALLOCATABLE, INTENT(INOUT)            :: orbitals(:, :, :, :)

      REAL(KIND=dp), PARAMETER :: fractions(9) = [0.001_dp, 0.003_dp, 0.01_dp, 0.03_dp, 0.1_dp, &
         0.3_dp, 0.6_dp, 0.9_dp, 1.0_dp]

      COMPLEX(KIND=dp), ALLOCATABLE                      :: plane(:, :)
      INTEGER                                            :: handle, i_local, i_progress, i_root, &
                                                            ikp, ispin, n_done, n_grid, n_owned, &
                                                            n_total, ready
      REAL(KIND=dp)                                      :: available_GB, required_bytes, started

      CALL timeset('collect_scf_grid_orbitals', handle)

      n_grid = SIZE(psi, 1)
      CPASSERT(n_grid > 0 .AND. SIZE(psi, 2) == bs_env%n_ao)
      CPASSERT(SIZE(psi, 4) == bs_env%n_spin)
      n_owned = (bs_env%nkp_scf_desymm + para_env%num_pe - 1 - para_env%mepos)/para_env%num_pe
      CPASSERT(SIZE(psi, 3) == n_owned)
      required_bytes = 16.0_dp*REAL(n_grid, dp)*REAL(bs_env%n_ao, dp)* &
                       (REAL(bs_env%nkp_scf_desymm, dp)*REAL(bs_env%n_spin, dp) + &
                        REAL(1 + omp_get_max_threads(), dp))
      CALL mp_mem_avail_per_rank_GB(para_env, available_GB)
      ready = MERGE(1, 0, required_bytes <= 0.5_dp*MAX(0.0_dp, available_GB)*1.0E9_dp)
      CALL para_env%min(ready)
      IF (ready == 0) THEN
         CALL timestop(handle)
         RETURN
      END IF
      ALLOCATE (orbitals(n_grid, bs_env%n_ao, bs_env%nkp_scf_desymm, bs_env%n_spin))
      ALLOCATE (plane(n_grid, bs_env%n_ao))
      started = m_walltime()
      i_progress = 1
      n_total = bs_env%n_spin*bs_env%nkp_scf_desymm
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A)') 'GW: sharing existing SCF orbitals between ranks on each node'
         CALL m_flush(bs_env%unit_nr)
      END IF
      DO ispin = 1, bs_env%n_spin
         DO ikp = 1, bs_env%nkp_scf_desymm
            i_root = MODULO(ikp - 1, para_env%num_pe)
            i_local = 1 + (ikp - 1)/para_env%num_pe
            IF (para_env%mepos == i_root) plane(:, :) = psi(:, :, i_local, ispin)
            CALL para_env%bcast(plane, i_root)
            orbitals(:, :, ikp, ispin) = plane
            n_done = (ispin - 1)*bs_env%nkp_scf_desymm + ikp
            IF (i_progress <= SIZE(fractions)) THEN
               IF (REAL(n_done, dp) >= fractions(i_progress)*REAL(n_total, dp)) THEN
               IF (bs_env%unit_nr > 0) THEN
                  WRITE (bs_env%unit_nr, '(T2,A,T26,I5,A,I0,T40,F5.1,A,T55,A,T69,F10.1,A)') &
                     'GW SCF k-points ', n_done, ' / ', n_total, &
                     100.0_dp*REAL(n_done, dp)/REAL(n_total, dp), ' %', &
                     'Total time:', m_walltime() - started, ' s'
                  CALL m_flush(bs_env%unit_nr)
               END IF
               DO WHILE (i_progress <= SIZE(fractions))
                  IF (REAL(n_done, dp) < fractions(i_progress)*REAL(n_total, dp)) EXIT
                  i_progress = i_progress + 1
               END DO
               END IF
            END IF
         END DO
      END DO
      DEALLOCATE (plane)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T69,F12.1)') 'GW SCF orbitals: GB per rank', &
            16.0_dp*REAL(SIZE(orbitals, KIND=int_8), dp)*1.0E-9_dp
         CALL m_flush(bs_env%unit_nr)
      END IF

      CALL timestop(handle)

   END SUBROUTINE collect_scf_grid_orbitals

! **************************************************************************************************
!> \brief ...
!> \param bs_env ...
!> \param orbitals ...
!> \return ...
! **************************************************************************************************
   LOGICAL FUNCTION orbitals_cover_scf(bs_env, orbitals) RESULT(available)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(dp), ALLOCATABLE, INTENT(IN)               :: orbitals(:, :, :, :)

      INTEGER                                            :: ikp
      REAL(KIND=dp)                                      :: delta_k(3)

      available = .FALSE.
      IF (.NOT. ALLOCATED(orbitals)) RETURN
      IF (SIZE(orbitals, 2) /= bs_env%n_ao) RETURN
      IF (SIZE(orbitals, 3) < bs_env%nkp_scf_desymm) RETURN
      IF (SIZE(orbitals, 4) /= bs_env%n_spin) RETURN
      IF (SIZE(bs_env%kpoints_DOS%xkp, 2) < bs_env%nkp_scf_desymm) RETURN
      DO ikp = 1, bs_env%nkp_scf_desymm
         delta_k = bs_env%kpoints_DOS%xkp(:, ikp) - bs_env%kpoints_scf_desymm%xkp(:, ikp)
         IF (MAXVAL(ABS(delta_k - ANINT(delta_k))) > 64.0_dp*EPSILON(1.0_dp)) RETURN
      END DO
      available = .TRUE.
   END FUNCTION orbitals_cover_scf

! **************************************************************************************************
!> \brief Form occupied or virtual Green matrices at k owners and redistribute FFT rows.
!> \param bs_env ...
!> \param psi_lm_k ...
!> \param group_env ...
!> \param tau ...
!> \param ispin ...
!> \param occupied ...
!> \param first_row ...
!> \param last_row ...
!> \param G_grid_k ...
! **************************************************************************************************
   SUBROUTINE build_green_grid_k_rows(bs_env, psi_lm_k, group_env, tau, ispin, occupied, first_row, &
                                      last_row, G_grid_k)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), DIMENSION(:, :, :), INTENT(IN)   :: psi_lm_k
      TYPE(mp_para_env_type), POINTER                    :: group_env
      REAL(KIND=dp), INTENT(IN)                          :: tau
      INTEGER, INTENT(IN)                                :: ispin
      LOGICAL, INTENT(IN)                                :: occupied
      INTEGER, INTENT(IN)                                :: first_row, last_row
      COMPLEX(KIND=dp), DIMENSION(:, :, :), INTENT(OUT)  :: G_grid_k

      CHARACTER(LEN=*), PARAMETER :: routineN = 'build_green_grid_k_rows'

      COMPLEX(KIND=dp), ALLOCATABLE                      :: scaled_coeff(:, :)
      INTEGER                                            :: band, first_band, handle, i_band, &
                                                            i_dest, ikp, ikp_local, max_rows, &
                                                            n_bands, phase_handle
      REAL(KIND=dp)                                      :: tau_E
      TYPE(bounded_transfer_type)                        :: transfer
      TYPE(gwkp_type), POINTER                           :: gwkp_env

      CALL timeset(routineN, handle)

      CALL get_gw_work(bs_env, gwkp_env)

      first_band = 1
      n_bands = bs_env%n_occ(ispin)
      IF (.NOT. occupied) THEN
         first_band = n_bands + 1
         n_bands = bs_env%n_ao - n_bands
      END IF
      CPASSERT(n_bands >= 0 .AND. n_bands <= bs_env%n_ao)
      ! The occupied and virtual Green functions contain disjoint band sums.
      ! Omit the identically zero columns of the other band sector.

      ALLOCATE (scaled_coeff(SIZE(psi_lm_k, 1), MAX(1, n_bands)))
      scaled_coeff(:, :) = z_zero
      CALL initialize_product_row_transfer(transfer, group_env, SIZE(psi_lm_k, 1), &
                                           bs_env%nkp_scf_desymm, -1, n_bands, max_rows)
      DO WHILE (next_transfer_block(transfer))
         CALL timeset('chi_green_local_products', phase_handle)
         ikp_local = 0
         DO ikp = 1, bs_env%nkp_scf_desymm
            IF (MODULO(ikp - 1, group_env%num_pe) /= group_env%mepos) CYCLE
            ikp_local = ikp_local + 1
            IF (n_bands == 0) THEN
               transfer%send_buffer = z_zero
               CYCLE
            END IF
            scaled_coeff(:, 1:n_bands) = psi_lm_k(:, first_band:first_band + n_bands - 1, ikp_local)
            DO i_band = 1, n_bands
               band = first_band + i_band - 1
               tau_E = ABS(0.5_dp*tau*(gwkp_env%eigenvalues_G(band, ikp, ispin) - &
                                       bs_env%e_fermi(ispin)))
               IF (tau_E < bs_env%stabilize_exp) THEN
                  scaled_coeff(:, i_band) = scaled_coeff(:, i_band)*EXP(-tau_E)
               ELSE
                  scaled_coeff(:, i_band) = z_zero
               END IF
            END DO
            DO i_dest = 0, group_env%num_pe - 1
               CALL pack_product_row_block(transfer, i_dest, scaled_coeff(:, 1:n_bands), &
                                           scaled_coeff(:, 1:n_bands), SIZE(scaled_coeff, 1), n_bands, ikp_local, max_rows)
            END DO
         END DO
         CPASSERT(ikp_local <= SIZE(psi_lm_k, 3))
         CPASSERT(SIZE(G_grid_k, 1) == MAX(0, last_row - first_row + 1))
         CALL timestop(phase_handle)
         CALL timeset('chi_green_redistribute', phase_handle)
         CALL exchange_transfer_block(transfer, group_env)
         CALL unpack_product_row_block(transfer, group_env, bs_env%nkp_scf_desymm, -1, G_grid_k)
         CALL timestop(phase_handle)
      END DO
      DEALLOCATE (scaled_coeff)

      CALL timestop(handle)

   END SUBROUTINE build_green_grid_k_rows

! **************************************************************************************************
!> \brief Evaluate G_ll′(k,τ) directly for locally owned ordered grid pairs.
!> Pair i=l+(l′-1)*n_grid follows Fortran column order; all native k points are local.
!> \param bs_env ...
!> \param orbitals ...
!> \param first_pair ...
!> \param tau ...
!> \param ispin ...
!> \param occupied ...
!> \param G_pairs ...
! **************************************************************************************************
   SUBROUTINE build_green_shared_pairs(bs_env, orbitals, first_pair, tau, ispin, occupied, G_pairs)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: orbitals(:, :, :)
      INTEGER, INTENT(IN)                                :: first_pair
      REAL(KIND=dp), INTENT(IN)                          :: tau
      INTEGER, INTENT(IN)                                :: ispin
      LOGICAL, INTENT(IN)                                :: occupied
      COMPLEX(KIND=dp), INTENT(OUT)                      :: G_pairs(:, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'build_green_shared_pairs'

      COMPLEX(KIND=dp), ALLOCATABLE                      :: columns(:, :), scaled_coeff(:, :)
      INTEGER                                            :: first_band, first_col, handle, i_band, &
                                                            i_col, i_pair, i_row, ikp, last_col, &
                                                            n_bands, n_columns, n_grid, n_local
      LOGICAL                                            :: valid_layout
      REAL(KIND=dp)                                      :: tau_E
      TYPE(gwkp_type), POINTER                           :: gwkp_env

      CALL timeset(routineN, handle)

      CALL get_gw_work(bs_env, gwkp_env)

      n_grid = SIZE(orbitals, 1)
      n_local = SIZE(G_pairs, 1)
      CPASSERT(n_grid > 0 .AND. SIZE(orbitals, 2) == bs_env%n_ao)
      CPASSERT(SIZE(orbitals, 3) >= bs_env%nkp_scf_desymm)
      IF (.NOT. (SIZE(G_pairs, 2) == 1 .AND. SIZE(G_pairs, 3) == bs_env%nkp_scf_desymm)) THEN
         CPABORT("Internal GW array shape mismatch")
      END IF
      valid_layout = first_pair >= 1 .AND. first_pair + n_local - 1 <= n_grid*n_grid
      CPASSERT(valid_layout)
      first_band = 1
      n_bands = bs_env%n_occ(ispin)
      IF (.NOT. occupied) THEN
         first_band = n_bands + 1
         n_bands = bs_env%n_ao - n_bands
      END IF
      CPASSERT(n_bands >= 0 .AND. n_bands <= bs_env%n_ao)
      IF (n_local == 0 .OR. n_bands == 0) THEN
         G_pairs = z_zero
         CALL timestop(handle)
         RETURN
      END IF
      first_col = (first_pair - 1)/n_grid + 1
      last_col = (first_pair + n_local - 2)/n_grid + 1
      n_columns = last_col - first_col + 1
      ! G_ll′(k,τ) = Σ_n [ψ_ln(k) exp(-|τ(ε_nk-ε_F)|/2)]
      !                         [ψ_l′n(k) exp(-|τ(ε_nk-ε_F)|/2)]*.
      !$OMP PARALLEL DEFAULT(NONE) &
      !$OMP SHARED(bs_env, orbitals, first_pair, first_band, first_col, n_bands, n_columns, &
      !$OMP        n_grid, n_local, tau, ispin, G_pairs, gwkp_env) &
      !$OMP PRIVATE(scaled_coeff, columns, i_band, i_col, i_pair, i_row, ikp, tau_E)
      ALLOCATE (scaled_coeff(n_grid, n_bands), columns(n_grid, n_columns))
      !$OMP DO SCHEDULE(STATIC)
      DO ikp = 1, bs_env%nkp_scf_desymm
         scaled_coeff(:, :) = orbitals(:, first_band:first_band + n_bands - 1, ikp)
         DO i_band = 1, n_bands
            tau_E = ABS(0.5_dp*tau*(gwkp_env%eigenvalues_G(first_band + i_band - 1, ikp, ispin) - &
                                    bs_env%e_fermi(ispin)))
            IF (tau_E < bs_env%stabilize_exp) THEN
               scaled_coeff(:, i_band) = scaled_coeff(:, i_band)*EXP(-tau_E)
            ELSE
               scaled_coeff(:, i_band) = z_zero
            END IF
         END DO
         CALL zgemm('N', 'C', n_grid, n_columns, n_bands, z_one, scaled_coeff, n_grid, &
                    scaled_coeff(first_col, 1), n_grid, z_zero, columns, n_grid)
         DO i_pair = 1, n_local
            i_row = MODULO(first_pair + i_pair - 2, n_grid) + 1
            i_col = (first_pair + i_pair - 2)/n_grid + 1 - first_col + 1
            G_pairs(i_pair, 1, ikp) = columns(i_row, i_col)
         END DO
      END DO
      !$OMP END DO
      DEALLOCATE (scaled_coeff, columns)
      !$OMP END PARALLEL

      CALL timestop(handle)

   END SUBROUTINE build_green_shared_pairs

! **************************************************************************************************
!> \brief Form local Green pairs from shared orbitals, retaining the distributed fallback.
!> \param bs_env ...
!> \param psi_lm_k ...
!> \param group_env ...
!> \param plan ...
!> \param tau ...
!> \param ispin ...
!> \param occupied ...
!> \param G_grid_k ...
!> \param scf_values ...
! **************************************************************************************************
   SUBROUTINE build_green_grid_k_pairs(bs_env, psi_lm_k, group_env, plan, tau, ispin, &
                                       occupied, G_grid_k, scf_values)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), DIMENSION(:, :, :), INTENT(IN)   :: psi_lm_k
      TYPE(mp_para_env_type), POINTER                    :: group_env
      TYPE(pair_transfer_plan_type), INTENT(IN)          :: plan
      REAL(KIND=dp), INTENT(IN)                          :: tau
      INTEGER, INTENT(IN)                                :: ispin
      LOGICAL, INTENT(IN)                                :: occupied
      COMPLEX(KIND=dp), DIMENSION(:, :, :), INTENT(OUT)  :: G_grid_k
      COMPLEX(KIND=dp), INTENT(IN), POINTER              :: scf_values(:, :, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'build_green_grid_k_pairs'

      COMPLEX(KIND=dp), ALLOCATABLE                      :: G_k_local(:, :, :), scaled_coeff(:, :)
      INTEGER                                            :: band, first_band, handle, i_band, ikp, &
                                                            ikp_local, n_bands, nowned_k
      REAL(KIND=dp)                                      :: tau_E
      TYPE(gwkp_type), POINTER                           :: gwkp_env

      IF (ASSOCIATED(scf_values)) THEN
         CALL build_green_shared_pairs(bs_env, scf_values(:, :, :, ispin), &
                                       plan%first_pair, tau, ispin, occupied, G_grid_k)
         RETURN
      END IF

      CALL timeset(routineN, handle)

      CALL get_gw_work(bs_env, gwkp_env)

      first_band = 1
      n_bands = bs_env%n_occ(ispin)
      IF (.NOT. occupied) THEN
         first_band = n_bands + 1
         n_bands = bs_env%n_ao - n_bands
      END IF
      CPASSERT(n_bands >= 0 .AND. n_bands <= bs_env%n_ao)
      ! The occupied and virtual Green functions contain disjoint band sums.
      ! Omit the identically zero columns of the other band sector.
      CPASSERT(plan%owner_offset == -1)
      ALLOCATE (G_k_local(SIZE(psi_lm_k, 1), SIZE(psi_lm_k, 1), SIZE(psi_lm_k, 3)))
      nowned_k = COUNT([(MODULO(ikp - 1, group_env%num_pe) == &
                         group_env%mepos, ikp=1, bs_env%nkp_scf_desymm)])
      CPASSERT(SIZE(psi_lm_k, 3) == nowned_k)
      !$OMP PARALLEL DEFAULT(NONE) &
      !$OMP SHARED(bs_env, G_k_local, group_env, ispin, first_band, n_bands, psi_lm_k, &
      !$OMP        gwkp_env, tau) &
      !$OMP PRIVATE(band, i_band, ikp, ikp_local, scaled_coeff, tau_E)
      ALLOCATE (scaled_coeff(SIZE(psi_lm_k, 1), MAX(1, n_bands)))
      !$OMP DO SCHEDULE(STATIC)
      DO ikp_local = 1, SIZE(psi_lm_k, 3)
         ikp = group_env%mepos + 1 + (ikp_local - 1)*group_env%num_pe
         IF (n_bands == 0) THEN
            G_k_local(:, :, ikp_local) = z_zero
            CYCLE
         END IF
         scaled_coeff(:, 1:n_bands) = psi_lm_k(:, first_band:first_band + n_bands - 1, ikp_local)
         DO i_band = 1, n_bands
            band = first_band + i_band - 1
            tau_E = ABS(0.5_dp*tau*(gwkp_env%eigenvalues_G(band, ikp, ispin) - &
                                    bs_env%e_fermi(ispin)))
            IF (tau_E < bs_env%stabilize_exp) THEN
               scaled_coeff(:, i_band) = scaled_coeff(:, i_band)*EXP(-tau_E)
            ELSE
               scaled_coeff(:, i_band) = z_zero
            END IF
         END DO
         CALL zgemm('N', 'C', SIZE(scaled_coeff, 1), SIZE(scaled_coeff, 1), &
                    n_bands, z_one, scaled_coeff, SIZE(scaled_coeff, 1), &
                    scaled_coeff, SIZE(scaled_coeff, 1), z_zero, G_k_local(:, :, ikp_local), &
                    SIZE(G_k_local, 1))
      END DO
      !$OMP END DO
      DEALLOCATE (scaled_coeff)
      !$OMP END PARALLEL
      CALL q_matrices_to_fft_pairs(G_k_local, group_env, plan, G_grid_k)
      DEALLOCATE (G_k_local)

      CALL timestop(handle)

   END SUBROUTINE build_green_grid_k_pairs

! **************************************************************************************************
!> \brief Cache grid values of every Bloch molecular orbital without replicating all MO matrices.
!> \param bs_env ...
!> \param phi_lmu_k ...
!> \param psi_lm_k ...
! **************************************************************************************************
   SUBROUTINE build_all_psi_k(bs_env, phi_lmu_k, psi_lm_k)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), DIMENSION(:, :, :), INTENT(IN)   :: phi_lmu_k
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: psi_lm_k(:, :, :, :)

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'build_all_psi_k'
      REAL(KIND=dp), PARAMETER :: coefficient_batch_bytes = 128.0E6_dp

      COMPLEX(KIND=dp), ALLOCATABLE                      :: mo_coeff_batch(:, :)
      INTEGER                                            :: batch_size, first_owned, handle, &
                                                            i_batch, ikp, ikp_local, ispin, &
                                                            n_batch, n_owned, n_owned_max, &
                                                            root_rank
      TYPE(gwkp_type), POINTER                           :: gwkp_env

      CALL timeset(routineN, handle)

      CALL get_gw_work(bs_env, gwkp_env)

      n_owned_max = (bs_env%nkp_scf_desymm + bs_env%para_env%num_pe - 1)/ &
                    bs_env%para_env%num_pe
      ALLOCATE (psi_lm_k(SIZE(phi_lmu_k, 1), bs_env%n_ao, SIZE(phi_lmu_k, 3), &
                         bs_env%n_spin), SOURCE=z_zero)
      batch_size = MAX(1, INT(coefficient_batch_bytes/ &
                              (16.0_dp*REAL(bs_env%n_ao, dp)**2)))
      batch_size = MIN(batch_size, n_owned_max)
      ALLOCATE (mo_coeff_batch(bs_env%n_ao, bs_env%n_ao*batch_size))
      DO ispin = 1, bs_env%n_spin
         DO root_rank = 0, MIN(bs_env%para_env%num_pe, bs_env%nkp_scf_desymm) - 1
            n_owned = (bs_env%nkp_scf_desymm - root_rank - 1)/bs_env%para_env%num_pe + 1
            DO first_owned = 1, n_owned, batch_size
               n_batch = MIN(batch_size, n_owned - first_owned + 1)
               mo_coeff_batch(:, 1:bs_env%n_ao*n_batch) = z_zero
               DO i_batch = 1, n_batch
                  ikp = root_rank + 1 + &
                        (first_owned + i_batch - 2)*bs_env%para_env%num_pe
                  CALL copy_local_cfm(bs_env%cfm_mo_coeff_kp(ikp, ispin), &
                                      mo_coeff_batch(:, (i_batch - 1)*bs_env%n_ao + 1: &
                                                     i_batch*bs_env%n_ao))
               END DO
               CALL bs_env%para_env%sum(mo_coeff_batch(:, 1:bs_env%n_ao*n_batch), root_rank)
               IF (root_rank == bs_env%para_env%mepos) THEN
                  DO i_batch = 1, n_batch
                     ikp = root_rank + 1 + &
                           (first_owned + i_batch - 2)*bs_env%para_env%num_pe
                     ikp_local = (ikp - 1)/bs_env%para_env%num_pe + 1
                     psi_lm_k(:, :, ikp_local, ispin) = MATMUL( &
                                                        phi_lmu_k(:, :, ikp_local), &
                                                        mo_coeff_batch(:, (i_batch - 1)*bs_env%n_ao + 1:i_batch*bs_env%n_ao))
                  END DO
               END IF
            END DO
         END DO
      END DO
      DEALLOCATE (mo_coeff_batch)

      CALL timestop(handle)

   END SUBROUTINE build_all_psi_k

! **************************************************************************************************
!> \brief Cache Bloch sums of the RI-RS grid values for the k-points owned by this rank.
!> \param bs_env ...
!> \param phi_lmu_R ...
!> \param phi_lmu_k ...
! **************************************************************************************************
   SUBROUTINE build_all_phi_k(bs_env, phi_lmu_R, phi_lmu_k)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(IN)      :: phi_lmu_R
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: phi_lmu_k(:, :, :)

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'build_all_phi_k'

      INTEGER                                            :: handle, ikp, ikp_local, n_k_local

      CALL timeset(routineN, handle)

      n_k_local = 0
      DO ikp = 1, bs_env%nkp_scf_desymm
         IF (MODULO(ikp - 1, bs_env%para_env%num_pe) == bs_env%para_env%mepos) THEN
            n_k_local = n_k_local + 1
         END IF
      END DO
      ALLOCATE (phi_lmu_k(SIZE(phi_lmu_R, 1), SIZE(phi_lmu_R, 2), n_k_local))
      phi_lmu_k = z_zero
      ikp_local = 0
      DO ikp = 1, bs_env%nkp_scf_desymm
         IF (MODULO(ikp - 1, bs_env%para_env%num_pe) /= bs_env%para_env%mepos) CYCLE
         ikp_local = ikp_local + 1
         CALL build_phi_k(phi_lmu_R, bs_env%index_to_cell_3c, &
                          bs_env%kpoints_scf_desymm%xkp(:, ikp), phi_lmu_k(:, :, ikp_local))
      END DO
      CPASSERT(ikp_local == n_k_local)

      CALL timestop(handle)

   END SUBROUTINE build_all_phi_k

! **************************************************************************************************
!> \brief Assemble the Bloch sum of explicitly translated AO values.
!> \param phi_lmu_R ...
!> \param index_to_cell ...
!> \param kpoint ...
!> \param phi_k ...
! **************************************************************************************************
   SUBROUTINE build_phi_k(phi_lmu_R, index_to_cell, kpoint, phi_k)
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(IN)      :: phi_lmu_R
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: index_to_cell
      REAL(KIND=dp), DIMENSION(3), INTENT(IN)            :: kpoint
      COMPLEX(KIND=dp), DIMENSION(:, :), INTENT(OUT)     :: phi_k

      INTEGER                                            :: i_R

      phi_k = z_zero
      DO i_R = 1, SIZE(phi_lmu_R, 3)
         phi_k = phi_k + lattice_phase(kpoint, index_to_cell(:, i_R))*phi_lmu_R(:, :, i_R)
      END DO

   END SUBROUTINE build_phi_k

! **************************************************************************************************
!> \brief Copy the locally owned elements of a distributed complex full matrix.
!> \param cfm ...
!> \param array ...
! **************************************************************************************************
   SUBROUTINE copy_local_cfm(cfm, array)
      TYPE(cp_cfm_type)                                  :: cfm
      COMPLEX(KIND=dp), DIMENSION(:, :), INTENT(OUT)     :: array

      INTEGER                                            :: col, i_col, i_row, ncol_local, &
                                                            nrow_local, row
      INTEGER, DIMENSION(:), POINTER                     :: col_indices, row_indices

      CALL cp_cfm_get_info(cfm, nrow_local=nrow_local, ncol_local=ncol_local, &
                           row_indices=row_indices, col_indices=col_indices)
      array = z_zero
      DO i_col = 1, ncol_local
         col = col_indices(i_col)
         DO i_row = 1, nrow_local
            row = row_indices(i_row)
            array(row, col) = cfm%local_data(i_row, i_col)
         END DO
      END DO
   END SUBROUTINE copy_local_cfm

! **************************************************************************************************
!> \brief Construct a complete cell grid dual to the uniform SCF k-point mesh.
!> \param bs_env ...
!> \param index_to_cell ...
! **************************************************************************************************
   SUBROUTINE build_full_cell_grid(bs_env, index_to_cell)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, ALLOCATABLE, INTENT(OUT)                  :: index_to_cell(:, :)

      INTEGER                                            :: cell(3), i1, i2, i3, i_cell, &
                                                            i_direction, lower(3), upper(3)

      DO i_direction = 1, 3
         lower(i_direction) = -bs_env%kpoints_scf_desymm%nkp_grid(i_direction)/2
         upper(i_direction) = lower(i_direction) + &
                              bs_env%kpoints_scf_desymm%nkp_grid(i_direction) - 1
      END DO
      ALLOCATE (index_to_cell(3, PRODUCT(bs_env%kpoints_scf_desymm%nkp_grid)))
      i_cell = 0
      DO i1 = lower(1), upper(1)
         DO i2 = lower(2), upper(2)
            DO i3 = lower(3), upper(3)
               i_cell = i_cell + 1
               cell = [i1, i2, i3]
               index_to_cell(:, i_cell) = cell
            END DO
         END DO
      END DO

   END SUBROUTINE build_full_cell_grid

! **************************************************************************************************
!> \brief Return the shifted transfer quadrature in SCF-point ordering.
!> \param bs_env ...
!> \param iq ...
!> \return ...
! **************************************************************************************************
   PURE FUNCTION momentum_transfer_point(bs_env, iq) RESULT(qpoint)
      TYPE(post_scf_bandstructure_type), INTENT(IN), &
         POINTER                                         :: bs_env
      INTEGER, INTENT(IN)                                :: iq
      REAL(KIND=dp)                                      :: qpoint(3)

      qpoint = bs_env%kpoints_chi_eps_W%xkp(:, iq)

   END FUNCTION momentum_transfer_point

! **************************************************************************************************
!> \brief Zero-wave-vector Fourier value of the selected RI metric kernel.
!> \param bs_env ...
!> \return ...
! **************************************************************************************************
   FUNCTION ri_metric_head_zero(bs_env) RESULT(head)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(KIND=dp)                                      :: head

      IF (bs_env%ri_metric%potential_type == do_potential_short) THEN
         IF (bs_env%ri_metric%omega <= 0.0_dp) THEN
            CPABORT('RI attenuation must be positive')
         END IF
         head = pi/bs_env%ri_metric%omega**2
      ELSE
         IF (bs_env%ri_metric%potential_type /= do_potential_truncated) THEN
            CPABORT('Unsupported RI metric')
         END IF
         head = twopi*bs_env%ri_metric%cutoff_radius**2
      END IF
   END FUNCTION ri_metric_head_zero

! **************************************************************************************************
!> \brief Form distributed I_PQ(R) = (φ_P,0 | v | φ_Q,R) over every screened neighbor cell.
!>
!> The finite-range potential v and the Gaussian screening radii determine the cell set.
!> No cell is dropped because it lies outside the SCF Born-von Karman cell. Each real matrix
!> remains distributed over the existing RI matrix communicator; callers release fm_R.
!> The reciprocal interaction is I_PQ(q) = sum_R exp(i q.R) I_PQ(R).
!> \param fm_R ...
!> \param cells ...
!> \param potential ...
!> \param regularization ...
!> \param bs_env ...
!> \param qs_env ...
!> \param reference_basis ...
! **************************************************************************************************
   SUBROUTINE build_ri_2c_lattice(fm_R, cells, potential, regularization, bs_env, qs_env, reference_basis)
      TYPE(cp_fm_type), ALLOCATABLE, INTENT(OUT)         :: fm_R(:)
      INTEGER, ALLOCATABLE, INTENT(OUT)                  :: cells(:, :)
      TYPE(libint_potential_type), INTENT(IN)            :: potential
      REAL(KIND=dp), INTENT(IN)                          :: regularization
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(qs_environment_type), POINTER                 :: qs_env
      LOGICAL, INTENT(IN), OPTIONAL                      :: reference_basis

      CHARACTER(LEN=*), PARAMETER :: routineN = 'build_ri_2c_lattice'

      INTEGER                                            :: bounds(3), cell_index(3), handle, icell, &
                                                            ix, iy, iz, ncells
      INTEGER, ALLOCATABLE                               :: occupied(:, :, :), occupied_flat(:), &
                                                            sizes(:)
      INTEGER, POINTER                                   :: col_sizes(:), row_sizes(:)
      LOGICAL                                            :: use_reference_basis
      TYPE(cp_blacs_env_type), POINTER                   :: blacs
      TYPE(cp_fm_struct_type), POINTER                   :: reference_matrix_struct
      TYPE(dbcsr_distribution_type)                      :: distribution
      TYPE(dbcsr_type), ALLOCATABLE                      :: matrices(:)
      TYPE(distribution_2d_type), POINTER                :: dist_2d
      TYPE(kpoint_type), POINTER                         :: kpoints_ptr
      TYPE(kpoint_type), TARGET                          :: kpoints
      TYPE(mp_para_env_type), POINTER                    :: para
      TYPE(neighbor_list_iterator_p_type), POINTER       :: iterator(:)
      TYPE(neighbor_list_set_p_type), POINTER            :: neighbors(:)
      TYPE(particle_type), POINTER                       :: particles(:)
      TYPE(qs_kind_type), POINTER                        :: qs_kinds(:)

      CALL timeset(routineN, handle)

      use_reference_basis = .FALSE.
      IF (PRESENT(reference_basis)) use_reference_basis = reference_basis
      NULLIFY (blacs, para, reference_matrix_struct)
      CALL get_qs_env(qs_env, distribution_2d=dist_2d, particle_set=particles, qs_kind_set=qs_kinds, &
                      blacs_env=blacs, para_env=para)
      ALLOCATE (sizes(bs_env%n_atom))
      CALL get_particle_set(particles, qs_kinds, nsgf=sizes, basis=bs_env%basis_set_RI)
      NULLIFY (neighbors, iterator)
      CALL build_2c_neighbor_lists(neighbors, bs_env%basis_set_RI, bs_env%basis_set_RI, &
                                   potential, "full_2c_RI", qs_env, sym_ij=.FALSE., dist_2d=dist_2d)
      bounds = 0
      CALL neighbor_list_iterator_create(iterator, neighbors)
      DO WHILE (neighbor_list_iterate(iterator) == 0)
         CALL get_iterator_info(iterator, cell=cell_index)
         bounds = MAX(bounds, ABS(cell_index))
      END DO
      CALL neighbor_list_iterator_release(iterator)
      CALL bs_env%para_env%max(bounds)
      ALLOCATE (occupied(-bounds(1):bounds(1), -bounds(2):bounds(2), -bounds(3):bounds(3)))
      occupied = 0
      CALL neighbor_list_iterator_create(iterator, neighbors)
      DO WHILE (neighbor_list_iterate(iterator) == 0)
         CALL get_iterator_info(iterator, cell=cell_index)
         occupied(cell_index(1), cell_index(2), cell_index(3)) = 1
      END DO
      CALL neighbor_list_iterator_release(iterator)
      ALLOCATE (occupied_flat(SIZE(occupied)))
      occupied_flat(:) = RESHAPE(occupied, [SIZE(occupied)])
      CALL bs_env%para_env%max(occupied_flat)
      occupied(:, :, :) = RESHAPE(occupied_flat, SHAPE(occupied))
      DEALLOCATE (occupied_flat)
      ncells = SUM(occupied)
      CPASSERT(ncells > 0)
      ALLOCATE (cells(3, ncells), fm_R(ncells))
      ALLOCATE (kpoints%cell_to_index(-bounds(1):bounds(1), -bounds(2):bounds(2), -bounds(3):bounds(3)))
      kpoints%cell_to_index = 0
      icell = 0
      DO ix = -bounds(1), bounds(1)
         DO iy = -bounds(2), bounds(2)
            DO iz = -bounds(3), bounds(3)
               IF (occupied(ix, iy, iz) == 0) CYCLE
               icell = icell + 1
               cells(:, icell) = [ix, iy, iz]
               kpoints%cell_to_index(ix, iy, iz) = icell
            END DO
         END DO
      END DO
      DEALLOCATE (occupied)
      kpoints_ptr => kpoints
      ALLOCATE (matrices(ncells))
      CALL cp_dbcsr_dist2d_to_dist(dist_2d, distribution)
      ALLOCATE (row_sizes(SIZE(sizes)), col_sizes(SIZE(sizes)))
      row_sizes = sizes
      col_sizes = sizes
      CALL dbcsr_create(matrices(1), "full RI two-center", distribution, dbcsr_type_no_symmetry, row_sizes, col_sizes)
      DEALLOCATE (row_sizes, col_sizes)
      DO icell = 2, ncells
         CALL dbcsr_create(matrices(icell), template=matrices(1))
      END DO
      CALL build_2c_integrals(matrices, 0.0_dp, qs_env, neighbors, bs_env%basis_set_RI, &
                              bs_env%basis_set_RI, potential, do_kpoints=.TRUE., &
                              ext_kpoints=kpoints_ptr, regularization_RI=regularization)
      DO icell = 1, ncells
         IF (use_reference_basis) THEN
            IF (.NOT. bs_env%auto_ri%enabled) THEN
               CPABORT('Reference RI lattice requires AUTO_RI')
            END IF
            IF (.NOT. ASSOCIATED(reference_matrix_struct)) THEN
               CALL cp_fm_struct_create(reference_matrix_struct, context=blacs, nrow_global=SUM(sizes), &
                                        ncol_global=SUM(sizes), para_env=para)
            END IF
            CALL cp_fm_create(fm_R(icell), reference_matrix_struct)
         ELSE
            CALL cp_fm_create(fm_R(icell), bs_env%fm_RI_RI%matrix_struct)
         END IF
         CALL copy_dbcsr_to_fm(matrices(icell), fm_R(icell))
         CALL dbcsr_release(matrices(icell))
      END DO
      CALL dbcsr_distribution_release(distribution)
      CALL release_neighbor_list_sets(neighbors)
      IF (ASSOCIATED(reference_matrix_struct)) CALL cp_fm_struct_release(reference_matrix_struct)
      DEALLOCATE (matrices, sizes, kpoints%cell_to_index)

      CALL timestop(handle)

   END SUBROUTINE build_ri_2c_lattice

! **************************************************************************************************
!> \brief Evaluate I_PQ(q) = sum_R exp(i q.R) I_PQ(R) collectively into an existing dense matrix.
!>
!> q is in reciprocal fractional coordinates, so q.R includes 2π. The real-space matrices
!> stay distributed; no global RI-by-RI-by-cell array is replicated on the ranks.
!> With owner present, only that rank receives matrix_q; other ranks pass an empty matrix.
!> \param fm_R ...
!> \param cells ...
!> \param qpoint ...
!> \param matrix_q ...
!> \param owner ...
! **************************************************************************************************
   SUBROUTINE ri_2c_lattice_to_kp(fm_R, cells, qpoint, matrix_q, owner)
      TYPE(cp_fm_type), INTENT(IN)                       :: fm_R(:)
      INTEGER, INTENT(IN)                                :: cells(:, :)
      REAL(KIND=dp), INTENT(IN)                          :: qpoint(3)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: matrix_q(:, :)
      INTEGER, INTENT(IN), OPTIONAL                      :: owner

      CHARACTER(LEN=*), PARAMETER :: routineN = 'ri_2c_lattice_to_kp'

      COMPLEX(KIND=dp)                                   :: phase
      COMPLEX(KIND=dp), ALLOCATABLE                      :: block_q(:, :), local_q(:, :)
      INTEGER                                            :: block_size(2), handle, icell, icol, &
                                                            irank, irow, max_size(2), my_rank, &
                                                            ncols, nrows
      INTEGER, ALLOCATABLE                               :: block_cols(:), block_rows(:)
      INTEGER, POINTER                                   :: cols(:), rows(:)
      LOGICAL                                            :: valid_layout

      CALL timeset(routineN, handle)

      CPASSERT(SIZE(fm_R) > 0)
      CPASSERT(SIZE(cells, 1) == 3)
      CPASSERT(SIZE(cells, 2) == SIZE(fm_R))
      CALL cp_fm_get_info(fm_R(1), nrow_local=nrows, ncol_local=ncols, row_indices=rows, col_indices=cols)
      matrix_q = z_zero
      IF (PRESENT(owner)) THEN
         valid_layout = owner >= 0 .AND. owner < fm_R(1)%matrix_struct%para_env%num_pe
         CPASSERT(valid_layout)
         ! Only the owner receives a global matrix; other ranks retain one distributed block.
         ALLOCATE (local_q(nrows, ncols))
         local_q = z_zero
         DO icell = 1, SIZE(fm_R)
            phase = EXP(CMPLX(0.0_dp, twopi*DOT_PRODUCT(qpoint, REAL(cells(:, icell), dp)), dp))
            local_q(:, :) = local_q + phase*fm_R(icell)%local_data(1:nrows, 1:ncols)
         END DO
         max_size = [nrows, ncols]
         CALL fm_R(1)%matrix_struct%para_env%max(max_size)
         ALLOCATE (block_rows(max_size(1)), block_cols(max_size(2)), block_q(max_size(1), max_size(2)))
         my_rank = fm_R(1)%matrix_struct%para_env%mepos
         DO irank = 0, fm_R(1)%matrix_struct%para_env%num_pe - 1
            block_size = [nrows, ncols]
            IF (my_rank == irank) THEN
               block_rows(1:nrows) = rows(1:nrows)
               block_cols(1:ncols) = cols(1:ncols)
               block_q(1:nrows, 1:ncols) = local_q
            END IF
            CALL fm_R(1)%matrix_struct%para_env%bcast(block_size, irank)
            IF (ANY(block_size == 0)) CYCLE
            CALL fm_R(1)%matrix_struct%para_env%bcast(block_rows(1:block_size(1)), irank)
            CALL fm_R(1)%matrix_struct%para_env%bcast(block_cols(1:block_size(2)), irank)
            CALL fm_R(1)%matrix_struct%para_env%bcast(block_q(1:block_size(1), 1:block_size(2)), irank)
            IF (my_rank /= owner) CYCLE
            DO icol = 1, block_size(2)
               DO irow = 1, block_size(1)
                  matrix_q(block_rows(irow), block_cols(icol)) = block_q(irow, icol)
               END DO
            END DO
         END DO
         DEALLOCATE (local_q, block_q, block_rows, block_cols)
      ELSE
         DO icell = 1, SIZE(fm_R)
            phase = EXP(CMPLX(0.0_dp, twopi*DOT_PRODUCT(qpoint, REAL(cells(:, icell), dp)), dp))
            DO icol = 1, ncols
               DO irow = 1, nrows
                  matrix_q(rows(irow), cols(icol)) = matrix_q(rows(irow), cols(icol)) + &
                                                     phase*fm_R(icell)%local_data(irow, icol)
               END DO
            END DO
         END DO
         CALL fm_R(1)%matrix_struct%para_env%sum(matrix_q)
      END IF

      CALL timestop(handle)

   END SUBROUTINE ri_2c_lattice_to_kp

! **************************************************************************************************
!> \brief Fourier transform distributed real RI lattice blocks to cyclic q owners in one exchange.
!>
!> I_PQ(q) = Σ_R exp(2πiq·R) I_PQ(R).  Each sender transforms its own matrix block
!> for every q and sends the resulting blocks to their q owners, iq mod P.
!> \param fm_R ...
!> \param cells ...
!> \param qpoints ...
!> \param matrices_q ...
!> \param required_q ...
! **************************************************************************************************
   SUBROUTINE ri_2c_lattice_to_owned_q(fm_R, cells, qpoints, matrices_q, required_q)
      TYPE(cp_fm_type), INTENT(IN)                       :: fm_R(:)
      INTEGER, INTENT(IN)                                :: cells(:, :)
      REAL(KIND=dp), INTENT(IN)                          :: qpoints(:, :)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: matrices_q(:, :, :)
      LOGICAL, INTENT(IN), OPTIONAL                      :: required_q(:)

      COMPLEX(KIND=dp)                                   :: phase
      COMPLEX(KIND=dp), ALLOCATABLE                      :: block_q(:, :), phase_x(:), phase_y(:), &
                                                            phase_z(:), recv_buffer(:), &
                                                            send_buffer(:)
      INTEGER :: dest, handle, i_axis, icell, icol, iq, iq_local, irow, max_cell(3), min_cell(3), &
         n_q, n_q_active_local, n_q_local, n_RI, ncols, nprocs, nrecv, nrows, nsend, pos, source
      INTEGER, ALLOCATABLE :: block_metadata(:), recv_counts(:), recv_displs(:), send_counts(:), &
         send_displs(:), send_positions(:), source_cols(:, :), source_rows(:, :), &
         source_sizes(:, :)
      INTEGER, POINTER                                   :: cols(:), rows(:)
      LOGICAL, ALLOCATABLE                               :: active_q(:)
      TYPE(mp_para_env_type), POINTER                    :: para_env

      CALL timeset('ri_2c_lattice_to_owned_q', handle)

      CPASSERT(SIZE(fm_R) > 0)
      CPASSERT(SIZE(cells, 1) == 3 .AND. SIZE(cells, 2) == SIZE(fm_R))
      CPASSERT(SIZE(qpoints, 1) == 3)
      min_cell = MINVAL(cells, DIM=2)
      max_cell = MAXVAL(cells, DIM=2)
      ALLOCATE (phase_x(min_cell(1):max_cell(1)), phase_y(min_cell(2):max_cell(2)), &
                phase_z(min_cell(3):max_cell(3)))
      n_q = SIZE(qpoints, 2)
      ALLOCATE (active_q(n_q), SOURCE=.TRUE.)
      IF (PRESENT(required_q)) THEN
         CPASSERT(SIZE(required_q) == n_q)
         active_q(:) = required_q(:)
      END IF
      n_RI = SIZE(matrices_q, 1)
      CPASSERT(SIZE(matrices_q, 2) == n_RI)
      para_env => fm_R(1)%matrix_struct%para_env
      nprocs = para_env%num_pe
      n_q_local = COUNT([(MODULO(iq, nprocs) == para_env%mepos, iq=1, n_q)])
      n_q_active_local = COUNT([(MODULO(iq, nprocs) == para_env%mepos .AND. &
                                 active_q(iq), iq=1, n_q)])
      CPASSERT(SIZE(matrices_q, 3) == n_q_local)
      CALL cp_fm_get_info(fm_R(1), nrow_local=nrows, ncol_local=ncols, &
                          row_indices=rows, col_indices=cols)

      ALLOCATE (block_metadata(2 + 2*n_RI), source_sizes(2, 0:nprocs - 1), &
                source_rows(n_RI, 0:nprocs - 1), source_cols(n_RI, 0:nprocs - 1))
      DO source = 0, nprocs - 1
         block_metadata = 0
         IF (para_env%mepos == source) THEN
            block_metadata(1:2) = [nrows, ncols]
            block_metadata(3:2 + nrows) = rows(1:nrows)
            block_metadata(3 + n_RI:2 + n_RI + ncols) = cols(1:ncols)
         END IF
         CALL para_env%bcast(block_metadata, source)
         source_sizes(:, source) = block_metadata(1:2)
         source_rows(:, source) = block_metadata(3:2 + n_RI)
         source_cols(:, source) = block_metadata(3 + n_RI:2 + 2*n_RI)
      END DO

      ALLOCATE (send_counts(0:nprocs - 1), recv_counts(0:nprocs - 1), &
                send_displs(0:nprocs - 1), recv_displs(0:nprocs - 1), &
                send_positions(0:nprocs - 1))
      DO dest = 0, nprocs - 1
         send_counts(dest) = nrows*ncols* &
                             COUNT([(MODULO(iq, nprocs) == dest .AND. active_q(iq), iq=1, n_q)])
      END DO
      send_displs(0) = 0
      DO dest = 1, nprocs - 1
         send_displs(dest) = send_displs(dest - 1) + send_counts(dest - 1)
      END DO
      nsend = send_displs(nprocs - 1) + send_counts(nprocs - 1)
      CALL para_env%alltoall(send_counts, recv_counts, 1)
      recv_displs(0) = 0
      DO source = 1, nprocs - 1
         recv_displs(source) = recv_displs(source - 1) + recv_counts(source - 1)
      END DO
      nrecv = recv_displs(nprocs - 1) + recv_counts(nprocs - 1)
      ALLOCATE (send_buffer(MAX(1, nsend)), recv_buffer(MAX(1, nrecv)), &
                block_q(nrows, ncols))
      send_positions(:) = send_displs(:)
      DO iq = 1, n_q
         IF (.NOT. active_q(iq)) CYCLE
         dest = MODULO(iq, nprocs)
         IF (nrows*ncols == 0) CYCLE
         ! On a regular lattice the phase factors into three one-dimensional factors.
         ! Reuse these small tables for every local matrix element and cell instead of
         ! evaluating one complex exponential for every cell on every MPI rank.
         DO i_axis = min_cell(1), max_cell(1)
            phase_x(i_axis) = EXP(CMPLX(0.0_dp, twopi*qpoints(1, iq)*REAL(i_axis, dp), dp))
         END DO
         DO i_axis = min_cell(2), max_cell(2)
            phase_y(i_axis) = EXP(CMPLX(0.0_dp, twopi*qpoints(2, iq)*REAL(i_axis, dp), dp))
         END DO
         DO i_axis = min_cell(3), max_cell(3)
            phase_z(i_axis) = EXP(CMPLX(0.0_dp, twopi*qpoints(3, iq)*REAL(i_axis, dp), dp))
         END DO
         block_q = z_zero
         DO icell = 1, SIZE(fm_R)
            phase = phase_x(cells(1, icell))*phase_y(cells(2, icell))* &
                    phase_z(cells(3, icell))
            block_q(:, :) = block_q + phase*fm_R(icell)%local_data(1:nrows, 1:ncols)
         END DO
         pos = send_positions(dest)
         send_buffer(pos + 1:pos + nrows*ncols) = RESHAPE(block_q, [nrows*ncols])
         send_positions(dest) = pos + nrows*ncols
      END DO
      CALL para_env%alltoall(send_buffer, send_counts, send_displs, &
                             recv_buffer, recv_counts, recv_displs)
      matrices_q = z_zero
      DO source = 0, nprocs - 1
         nrows = source_sizes(1, source)
         ncols = source_sizes(2, source)
         CPASSERT(recv_counts(source) == nrows*ncols*n_q_active_local)
         pos = recv_displs(source)
         iq_local = 0
         DO iq = 1, n_q
            IF (MODULO(iq, nprocs) /= para_env%mepos) CYCLE
            iq_local = iq_local + 1
            IF (.NOT. active_q(iq)) CYCLE
            DO icol = 1, ncols
               DO irow = 1, nrows
                  matrices_q(source_rows(irow, source), source_cols(icol, source), iq_local) = &
                     recv_buffer(pos + (icol - 1)*nrows + irow)
               END DO
            END DO
            pos = pos + nrows*ncols
         END DO
      END DO

      DEALLOCATE (active_q, phase_x, phase_y, phase_z)

      CALL timestop(handle)

   END SUBROUTINE ri_2c_lattice_to_owned_q

! **************************************************************************************************
!> \brief Compute the spectrum and inverse of H=C^T M(q) C on one MPI q group.
!> H is Hermitian. The root supplies H in bounded column blocks; eigensolver,
!> Cholesky factorization and inversion use block-cyclic distributed matrices.
!> Preserve the local path's rejection rule lambda_min <= 1e-12 lambda_max.
!> Only the root receives H^-1 for the existing atom-block objective gradient.
!> \param group_env ...
!> \param matrix ...
!> \param eigenvalues ...
!> \param inverse ...
! **************************************************************************************************
   SUBROUTINE auto_ri_grouped_inverse(group_env, matrix, eigenvalues, inverse)
      TYPE(mp_para_env_type), POINTER                    :: group_env
      COMPLEX(dp), INTENT(IN)                            :: matrix(:, :)
      REAL(dp), INTENT(OUT)                              :: eigenvalues(:)
      COMPLEX(dp), INTENT(OUT)                           :: inverse(:, :)

      INTEGER                                            :: handle, info, n, phase_handle
      TYPE(cp_blacs_env_type), POINTER                   :: blacs
      TYPE(cp_cfm_type)                                  :: distributed, scratch
      TYPE(cp_fm_struct_type), POINTER                   :: structure

      NULLIFY (blacs, structure)

      CALL timeset('auto_ri_grouped_inverse', handle)

      n = SIZE(eigenvalues)
      CALL cp_blacs_env_create(blacs, group_env)
      CALL cp_fm_struct_create(structure, context=blacs, nrow_global=n, ncol_global=n)
      CALL cp_cfm_create(distributed, structure)
      CALL cp_cfm_create(scratch, structure)
      CALL set_owner_matrix(distributed, matrix, group_env, 0, n)
      CALL timeset('auto_ri_grouped_spectrum', phase_handle)
      CALL cp_cfm_heevd(distributed, scratch, eigenvalues)
      CALL timestop(phase_handle)
      IF (MAXVAL(eigenvalues) > 0.0_dp .AND. &
          MINVAL(eigenvalues) > 1.0E-12_dp*MAXVAL(eigenvalues)) THEN
         CALL set_owner_matrix(distributed, matrix, group_env, 0, n)
         CALL timeset('auto_ri_grouped_cholesky', phase_handle)
         CALL cp_cfm_cholesky_decompose(distributed, info_out=info)
         IF (info /= 0) CPABORT('Distributed AUTO_RI metric factorization failed')
         CALL cp_cfm_cholesky_invert(distributed, info_out=info)
         IF (info /= 0) CPABORT('Distributed AUTO_RI metric inversion failed')
         CALL cp_cfm_uplo_to_full(distributed, scratch, uplo='U')
         CALL timestop(phase_handle)
         CALL get_owner_matrix(distributed, inverse, group_env, 0, n)
      ELSE
         inverse = CMPLX(0.0_dp, 0.0_dp, dp)
      END IF
      CALL cp_cfm_release(scratch)
      CALL cp_cfm_release(distributed)
      CALL cp_fm_struct_release(structure)
      CALL cp_blacs_env_release(blacs)

      CALL timestop(handle)

   END SUBROUTINE auto_ri_grouped_inverse

! **************************************************************************************************
!> \brief Cache Gamma-correction response in imaginary time, distributed by RI-RS grid rows.
!>
!> Let psi_li=sum_mu phi_mu(r_l) C_mu,i(k), and let m be the midpoint
!> between the highest occupied and lowest virtual eigenvalue at each k/spin.
!> Gocc_ll'(tau)=sum_i psi_li conjg(psi_l'i) exp[(epsilon_i-m)tau].
!> Gvir_ll'(tau)=sum_a psi_la conjg(psi_l'a) exp[-(epsilon_a-m)tau].
!> These are the projected AO Green matrices, evaluated associatively.
!> With f_k=spin_degeneracy*w_k and d_ia,c=-i*dipole_ia,c/sqrt(volume):
!> B_ll'(tau)=sum_k f_k conjg(Gocc_ll') Gvir_ll'.
!> E_lc(tau)=sum_kia f_k conjg(psi_li) psi_la d_ia,c exp[-Delta_ia*tau].
!> P_cd(tau)=sum_kia f_k conjg(d_ia,c) d_ia,d exp[-Delta_ia*tau].
!> Each rank owns disjoint rows of B and E; only the small P is replicated.
!> A k-owner broadcasts its orbitals once, then all ranks process owned rows.
!> No RI-pair by transition tensor or replicated grid-pair cache is formed.
!> The cache is created once and reused for all frequencies. The existing
!> minimax cosine transform approximates 2*Delta/(Delta**2+omega**2);
!> its relative kernel error is checked independently of the iterative solve.
!> \param bs_env ...
!> \param psi ...
!> \param dipole ...
!> \param volume ...
!> \param state ...
! **************************************************************************************************
   SUBROUTINE gamma_ao_time_prepare(bs_env, psi, dipole, volume, state)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: psi(:, :, :, :), dipole(:, :, :, :, :)
      REAL(KIND=dp), INTENT(IN)                          :: volume
      TYPE(gamma_ao_time_state_type), INTENT(INOUT)      :: state

      INTEGER                                            :: handle

      CALL timeset('gamma_ao_time_prepare', handle)

      CALL gamma_ao_time_release(state)
      CALL initialize_time_state(bs_env, SIZE(psi, 1), state)
      CALL validate_time_transform(bs_env)
      CALL accumulate_time_state(bs_env, psi, dipole, volume, state)

      CALL timestop(handle)

   END SUBROUTINE gamma_ao_time_prepare

! **************************************************************************************************
!> \brief Transform cached response and solve (I+F^H Z^H B Z F) X = F^H Z^H E.
!> Z_lP and F are existing metric maps. All three right sides use the same
!> row-distributed frequency matrix. Head and wings use the same time transform.
!> \param bs_env ...
!> \param state ...
!> \param Z ...
!> \param F ...
!> \param frequency ...
!> \param E ...
!> \param P ...
!> \param X ...
! **************************************************************************************************
   SUBROUTINE gamma_ao_time_response(bs_env, state, Z, F, frequency, E, P, X)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gamma_ao_time_state_type), INTENT(INOUT)      :: state
      COMPLEX(KIND=dp), INTENT(IN)                       :: Z(:, :), F(:, :)
      REAL(KIND=dp), INTENT(IN)                          :: frequency
      COMPLEX(KIND=dp), INTENT(OUT)                      :: E(:, :), P(:, :), X(:, :)

      COMPLEX(KIND=dp)                                   :: head_frequency(3, 3)
      COMPLEX(KIND=dp), ALLOCATABLE                      :: chi_frequency(:, :), wing_frequency(:, :)
      INTEGER                                            :: handle

      CALL timeset('gamma_ao_time_response', handle)

      CALL transform_time_state(bs_env, state, frequency, chi_frequency, wing_frequency, head_frequency)
      CALL initialize_metric_grid_rows(state, Z, F)
      CALL project_gamma_corr_wing(bs_env, wing_frequency, head_frequency, state%metric_rows, E, P)
      CALL solve_gamma_corr_responses(bs_env, state%metric_rows, chi_frequency, state%first_row, E, X)

      CALL timestop(handle)

   END SUBROUTINE gamma_ao_time_response

! **************************************************************************************************
!> \brief Release the cache owned by the enclosing Gamma-correction call.
!> \param state ...
! **************************************************************************************************
   SUBROUTINE gamma_ao_time_release(state)
      TYPE(gamma_ao_time_state_type), INTENT(INOUT)      :: state

      IF (ALLOCATED(state%chi_tau)) DEALLOCATE (state%chi_tau)
      IF (ALLOCATED(state%wing_tau)) DEALLOCATE (state%wing_tau)
      IF (ALLOCATED(state%head_tau)) DEALLOCATE (state%head_tau)
      IF (ALLOCATED(state%metric_rows)) DEALLOCATE (state%metric_rows)
      state%first_row = 1
   END SUBROUTINE gamma_ao_time_release

! **************************************************************************************************
!> \brief Partition grid rows and allocate only the local part of the time cache.
!> \param bs_env ...
!> \param n_grid ...
!> \param state ...
! **************************************************************************************************
   SUBROUTINE initialize_time_state(bs_env, n_grid, state)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: n_grid
      TYPE(gamma_ao_time_state_type), INTENT(INOUT)      :: state

      INTEGER                                            :: n_rank, n_rows, n_time, rank
      REAL(KIND=dp)                                      :: cache_gib

      CPASSERT(n_grid > 0)
      n_time = bs_env%num_time_freq_points
      CPASSERT(n_time > 0)
      rank = bs_env%para_env%mepos
      n_rank = bs_env%para_env%num_pe
      state%first_row = INT(INT(rank, int_8)*n_grid/n_rank) + 1
      n_rows = INT(INT(rank + 1, int_8)*n_grid/n_rank) - state%first_row + 1
      ALLOCATE (state%chi_tau(n_rows, n_grid, n_time), SOURCE=z_zero)
      ALLOCATE (state%wing_tau(n_rows, 3, n_time), SOURCE=z_zero)
      ALLOCATE (state%head_tau(3, 3, n_time), SOURCE=z_zero)
      cache_gib = 16.0_dp*(REAL(n_rows, dp)*n_grid*n_time + REAL(n_rows, dp)*3*n_time + 9*n_time)/1024.0_dp**3
      CALL bs_env%para_env%max(cache_gib)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T65,F12.3,A4)') &
            'Gamma time cache largest rank allocation:', cache_gib, ' GiB'
         CALL m_flush(bs_env%unit_nr)
      END IF
   END SUBROUTINE initialize_time_state

! **************************************************************************************************
!> \brief Bound sampled transform errors over the physical transition interval.
!> \param bs_env ...
! **************************************************************************************************
   SUBROUTINE validate_time_transform(bs_env)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env

      REAL(KIND=dp), PARAMETER :: eps_gamma_time_transform = 1.0E-5_dp

      CHARACTER(LEN=256)                                 :: error_message
      INTEGER                                            :: i, k, no, nv, spin, t, w
      REAL(KIND=dp)                                      :: approximate, delta, exact, hi, lo, &
                                                            omega, worst
      TYPE(gwkp_type), POINTER                           :: gwkp_env

      CALL get_gw_work(bs_env, gwkp_env)

      worst = 0.0_dp
      IF (bs_env%para_env%mepos == 0) THEN
         DO spin = 1, bs_env%n_spin
            no = bs_env%n_occ(spin)
            nv = bs_env%n_vir(spin)
            IF (no == 0 .OR. nv == 0) CYCLE
            DO k = 1, bs_env%nkp_scf_desymm
               lo = gwkp_env%eigenvalues_G(no + 1, k, spin) - gwkp_env%eigenvalues_G(no, k, spin)
               hi = gwkp_env%eigenvalues_G(no + nv, k, spin) - gwkp_env%eigenvalues_G(1, k, spin)
               IF (lo <= 0.0_dp) THEN
                  CPABORT('Gamma response requires a positive transition gap')
               END IF
               DO i = 1, 32
                  delta = lo*(hi/lo)**(REAL(i - 1, dp)/31.0_dp)
                  DO w = 1, SIZE(bs_env%time_frequency_grid%frequency)
                     omega = bs_env%time_frequency_grid%frequency(w)
                     exact = 2.0_dp*delta/(delta*delta + omega*omega)
                     approximate = 0.0_dp
                     DO t = 1, bs_env%num_time_freq_points
                        approximate = approximate + bs_env%time_frequency_grid%cosine_time_to_frequency(w, t)* &
                                      EXP(-delta*bs_env%time_frequency_grid%imaginary_time(t))
                     END DO
                     worst = MAX(worst, ABS(approximate - exact)/exact)
                  END DO
               END DO
            END DO
         END DO
      END IF
      CALL bs_env%para_env%max(worst)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T61,ES20.12)') &
            'Gamma time transform sampled relative error:', worst
         CALL m_flush(bs_env%unit_nr)
      END IF
      IF (worst > eps_gamma_time_transform) THEN
         WRITE (error_message, '(A,ES12.4,A,ES12.4,A)') &
            'Gamma time transform relative error ', worst, &
            ' exceeds tolerance ', eps_gamma_time_transform, &
            '. Increase NUM_TIME_FREQ_POINTS and check convergence.'
         CPABORT(TRIM(error_message))
      END IF
   END SUBROUTINE validate_time_transform

! **************************************************************************************************
!> \brief Form the local time-cache rows for every k point, using one orbital broadcast.
!> \param bs_env ...
!> \param psi ...
!> \param dipole ...
!> \param volume ...
!> \param state ...
! **************************************************************************************************
   SUBROUTINE accumulate_time_state(bs_env, psi, dipole, volume, state)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: psi(:, :, :, :), dipole(:, :, :, :, :)
      REAL(KIND=dp), INTENT(IN)                          :: volume
      TYPE(gamma_ao_time_state_type), INTENT(INOUT)      :: state

      COMPLEX(KIND=dp), ALLOCATABLE                      :: gamma_corr(:, :, :), orbitals(:, :)
      INTEGER                                            :: c, k, local_k, no, nv, owner, spin

      CPASSERT(volume > 0.0_dp)
      DO spin = 1, bs_env%n_spin
         no = bs_env%n_occ(spin)
         nv = bs_env%n_vir(spin)
         IF (no == 0 .OR. nv == 0) CYCLE
         ALLOCATE (orbitals(SIZE(psi, 1), no + nv), gamma_corr(no, nv, 3))
         DO k = 1, bs_env%nkp_scf_desymm
            owner = MODULO(k - 1, bs_env%para_env%num_pe)
            IF (bs_env%para_env%mepos == owner) THEN
               local_k = (k - 1)/bs_env%para_env%num_pe + 1
               CPASSERT(local_k <= SIZE(psi, 3))
               orbitals(:, :) = psi(:, 1:no + nv, local_k, spin)
               DO c = 1, 3
                  gamma_corr(:, :, c) = -gaussi*dipole(spin, local_k, c, 1:no, 1:nv)/SQRT(volume)
               END DO
            END IF
            CALL bs_env%para_env%bcast(orbitals, owner)
            CALL bs_env%para_env%bcast(gamma_corr, owner)
            CALL accumulate_k_time_rows(bs_env, state, orbitals, gamma_corr, k, spin, owner)
         END DO
         DEALLOCATE (orbitals, gamma_corr)
      END DO
      CALL bs_env%para_env%sum(state%head_tau)
   END SUBROUTINE accumulate_time_state

! **************************************************************************************************
!> \brief Weight only owned orbital rows in Green and Gamma-correction products; accumulate the tiny head once.
!> Gocc_ll'=sum_i [psi_li exp((epsilon_i-m)tau)] conjg(psi_l'i), and analogously for Gvir.
!> The wing uses both fully weighted local factors: conjg(psi_li) psi_la d_ia exp(-Delta_ia*tau).
!> Global orbitals remain unweighted, avoiding a replicated time-weighted orbital array.
!> \param bs_env ...
!> \param state ...
!> \param orbitals ...
!> \param gamma_corr Transition moments entering the Gamma correction.
!> \param k ...
!> \param spin ...
!> \param owner ...
! **************************************************************************************************
   SUBROUTINE accumulate_k_time_rows(bs_env, state, orbitals, gamma_corr, k, spin, owner)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gamma_ao_time_state_type), INTENT(INOUT)      :: state
      COMPLEX(KIND=dp), INTENT(IN)                       :: orbitals(:, :), gamma_corr(:, :, :)
      INTEGER, INTENT(IN)                                :: k, spin, owner

      COMPLEX(KIND=dp), ALLOCATABLE                      :: block(:, :), go(:, :), gv(:, :), &
                                                            scaled(:, :)
      INTEGER                                            :: a, c, d, first, i, last, ng, no, nr, nv, &
                                                            t
      REAL(KIND=dp)                                      :: head_weight, mid, tau, weight
      REAL(KIND=dp), ALLOCATABLE                         :: occ_factor(:), vir_factor(:)
      TYPE(gwkp_type), POINTER                           :: gwkp_env

      CALL get_gw_work(bs_env, gwkp_env)

      no = bs_env%n_occ(spin)
      nv = bs_env%n_vir(spin)
      ng = SIZE(orbitals, 1)
      nr = SIZE(state%chi_tau, 1)
      first = state%first_row
      last = first + nr - 1
      mid = 0.5_dp*(gwkp_env%eigenvalues_G(no, k, spin) + gwkp_env%eigenvalues_G(no + 1, k, spin))
      weight = bs_env%spin_degeneracy*bs_env%kpoints_scf_desymm%wkp(k)
      ALLOCATE (occ_factor(no), vir_factor(nv), scaled(nr, no + nv))
      ALLOCATE (go(nr, ng), gv(nr, ng), block(nr, no))
      DO t = 1, bs_env%num_time_freq_points
         tau = bs_env%time_frequency_grid%imaginary_time(t)
         DO i = 1, no
            occ_factor(i) = EXP(0.5_dp*(gwkp_env%eigenvalues_G(i, k, spin) - mid)*tau)
            scaled(:, i) = orbitals(first:last, i)*occ_factor(i)**2
         END DO
         DO a = 1, nv
            vir_factor(a) = EXP(-0.5_dp*(gwkp_env%eigenvalues_G(no + a, k, spin) - mid)*tau)
            scaled(:, no + a) = orbitals(first:last, no + a)*vir_factor(a)**2
         END DO
         IF (nr > 0) THEN
            CALL zgemm('N', 'C', nr, ng, no, z_one, scaled, nr, &
                       orbitals, ng, z_zero, go, nr)
            CALL zgemm('N', 'C', nr, ng, nv, z_one, scaled(1, no + 1), nr, &
                       orbitals(:, no + 1:), ng, z_zero, gv, nr)
            state%chi_tau(:, :, t) = state%chi_tau(:, :, t) + weight*CONJG(go)*gv
            DO c = 1, 3
               CALL zgemm('N', 'T', nr, no, nv, z_one, scaled(1, no + 1), nr, &
                          gamma_corr(:, :, c), no, z_zero, block, nr)
               state%wing_tau(:, c, t) = state%wing_tau(:, c, t) + &
                                         weight*SUM(CONJG(scaled(:, 1:no))*block, DIM=2)
            END DO
         END IF
         IF (bs_env%para_env%mepos == owner) THEN
            DO a = 1, nv
               DO i = 1, no
                  head_weight = weight*(occ_factor(i)*vir_factor(a))**2
                  DO d = 1, 3
                     DO c = 1, 3
                        state%head_tau(c, d, t) = state%head_tau(c, d, t) + &
                                                  head_weight*CONJG(gamma_corr(i, a, c))*gamma_corr(i, a, d)
                     END DO
                  END DO
               END DO
            END DO
         END IF
      END DO
      DEALLOCATE (occ_factor, vir_factor, scaled, go, gv, block)
   END SUBROUTINE accumulate_k_time_rows

! **************************************************************************************************
!> \brief Apply the existing cosine coefficients to the local cache, without rebuilding Green matrices.
!> \param bs_env ...
!> \param state ...
!> \param frequency ...
!> \param chi_frequency ...
!> \param wing_frequency ...
!> \param head_frequency ...
! **************************************************************************************************
   SUBROUTINE transform_time_state(bs_env, state, frequency, chi_frequency, wing_frequency, head_frequency)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gamma_ao_time_state_type), INTENT(IN)         :: state
      REAL(KIND=dp), INTENT(IN)                          :: frequency
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: chi_frequency(:, :), wing_frequency(:, :)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: head_frequency(3, 3)

      INTEGER                                            :: t, w
      REAL(KIND=dp)                                      :: coefficient

      CPASSERT(ALLOCATED(state%chi_tau))
      w = MINLOC(ABS(bs_env%time_frequency_grid%frequency - frequency), DIM=1)
      coefficient = ABS(bs_env%time_frequency_grid%frequency(w) - frequency)
      CPASSERT(coefficient < 1.0E-12_dp)
      ALLOCATE (chi_frequency(SIZE(state%chi_tau, 1), SIZE(state%chi_tau, 2)), SOURCE=z_zero)
      ALLOCATE (wing_frequency(SIZE(state%wing_tau, 1), 3), SOURCE=z_zero)
      head_frequency = z_zero
      DO t = 1, bs_env%num_time_freq_points
         coefficient = bs_env%time_frequency_grid%cosine_time_to_frequency(w, t)
         chi_frequency(:, :) = chi_frequency + coefficient*state%chi_tau(:, :, t)
         wing_frequency(:, :) = wing_frequency + coefficient*state%wing_tau(:, :, t)
         head_frequency = head_frequency + coefficient*state%head_tau(:, :, t)
      END DO
   END SUBROUTINE transform_time_state

! **************************************************************************************************
!> \brief Form T_lQ = sum_P Z_lP F_PQ once on the owned grid rows.
!> The same T is reused for all frequencies; no replicated grid-by-RI map is formed.
!> \param state ...
!> \param Z ...
!> \param F ...
! **************************************************************************************************
   SUBROUTINE initialize_metric_grid_rows(state, Z, F)
      TYPE(gamma_ao_time_state_type), INTENT(INOUT)      :: state
      COMPLEX(KIND=dp), INTENT(IN)                       :: Z(:, :), F(:, :)

      COMPLEX(KIND=dp), ALLOCATABLE                      :: Z_rows(:, :)
      INTEGER                                            :: handle, i_first, i_last, nr

      IF (ALLOCATED(state%metric_rows)) RETURN

      CALL timeset('gamma_metric_grid_rows', handle)

      nr = SIZE(state%chi_tau, 1)
      i_first = state%first_row
      i_last = i_first + nr - 1
      ALLOCATE (state%metric_rows(nr, SIZE(F, 2)))
      IF (nr > 0) THEN
         ALLOCATE (Z_rows, SOURCE=Z(i_first:i_last, :))
         CALL zgemm('N', 'N', nr, SIZE(F, 2), SIZE(F, 1), z_one, &
                    Z_rows, nr, F, SIZE(F, 1), z_zero, state%metric_rows, nr)
         DEALLOCATE (Z_rows)
      END IF

      CALL timestop(handle)

   END SUBROUTINE initialize_metric_grid_rows

! **************************************************************************************************
!> \brief Project the wing as E_Qc = sum_l conjg(T_lQ) E_lc over disjoint rows.
!> \param bs_env ...
!> \param wing_frequency ...
!> \param head_frequency ...
!> \param T_rows ...
!> \param E ...
!> \param P ...
! **************************************************************************************************
   SUBROUTINE project_gamma_corr_wing(bs_env, wing_frequency, head_frequency, T_rows, E, P)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: wing_frequency(:, :), &
                                                            head_frequency(3, 3), T_rows(:, :)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: E(:, :), P(:, :)

      INTEGER                                            :: nr

      nr = SIZE(wing_frequency, 1)
      E = z_zero
      IF (nr > 0) CALL zgemm('C', 'N', SIZE(T_rows, 2), 3, nr, z_one, T_rows, nr, &
                             wing_frequency, nr, z_zero, E, SIZE(E, 1))
      CALL bs_env%para_env%sum(E)
      P = head_frequency
   END SUBROUTINE project_gamma_corr_wing

! **************************************************************************************************
!> \brief Apply T^H chi T with distributed rows and reductions of three-column vectors.
!> The grid vector is assembled from disjoint row contributions, never from replicated T.
!> \param bs_env ...
!> \param T_rows ...
!> \param chi ...
!> \param first_row ...
!> \param x ...
!> \param y ...
!> \param aux_y ...
!> \param grid_x ...
!> \param grid_y ...
! **************************************************************************************************
   SUBROUTINE apply_gamma_corr_body(bs_env, T_rows, chi, first_row, x, y, aux_y, grid_x, grid_y)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: T_rows(:, :), chi(:, :)
      INTEGER, INTENT(IN)                                :: first_row
      COMPLEX(KIND=dp), INTENT(IN)                       :: x(:, :)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: y(:, :)
      COMPLEX(KIND=dp), INTENT(INOUT)                    :: aux_y(:, :), grid_x(:, :), grid_y(:, :)

      INTEGER                                            :: i_last, nc, nr

      nr = SIZE(chi, 1)
      nc = SIZE(x, 2)
      i_last = first_row + nr - 1
      grid_x = z_zero
      IF (nr > 0) THEN
         CALL zgemm('N', 'N', nr, nc, SIZE(T_rows, 2), z_one, T_rows, nr, &
                    x, SIZE(x, 1), z_zero, grid_y, nr)
         grid_x(first_row:i_last, :) = grid_y
      END IF
      CALL bs_env%para_env%sum(grid_x)
      aux_y = z_zero
      IF (nr > 0) THEN
         CALL zgemm('N', 'N', nr, nc, SIZE(chi, 2), z_one, chi, nr, &
                    grid_x, SIZE(grid_x, 1), z_zero, grid_y, nr)
         CALL zgemm('C', 'N', SIZE(T_rows, 2), nc, nr, z_one, T_rows, nr, &
                    grid_y, nr, z_zero, aux_y, SIZE(aux_y, 1))
      END IF
      CALL bs_env%para_env%sum(aux_y)
      y = aux_y
   END SUBROUTINE apply_gamma_corr_body

! **************************************************************************************************
!> \brief Solve and verify all three Gamma-correction right sides.
!> \param bs_env ...
!> \param T_rows ...
!> \param chi ...
!> \param first_row ...
!> \param E ...
!> \param X ...
! **************************************************************************************************
   SUBROUTINE solve_gamma_corr_responses(bs_env, T_rows, chi, first_row, E, X)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: T_rows(:, :), chi(:, :)
      INTEGER, INTENT(IN)                                :: first_row
      COMPLEX(KIND=dp), INTENT(IN)                       :: E(:, :)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: X(:, :)

      INTEGER                                            :: worst_iterations
      REAL(KIND=dp)                                      :: worst_residual

      CALL solve_gamma_corr_body(bs_env, T_rows, chi, first_row, E, X, worst_iterations, worst_residual)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T69,I12)') 'Gamma time response maximum iterations:', worst_iterations
         WRITE (bs_env%unit_nr, '(T2,A,T61,ES20.12)') 'Gamma time response true relative residual:', worst_residual
      END IF
   END SUBROUTINE solve_gamma_corr_responses

! **************************************************************************************************
!> \brief Independent conjugate gradients for three directions, sharing matrix actions and reductions.
!> Each column c retains its own alpha_c, beta_c and convergence test; columns are not coupled.
!> Before accepting a column, recompute b_c-(I+F^H Z^H chi Z F)x_c independently.
!> \param bs_env ...
!> \param T_rows ...
!> \param chi ...
!> \param first_row ...
!> \param b ...
!> \param x ...
!> \param iterations ...
!> \param residual ...
! **************************************************************************************************
   SUBROUTINE solve_gamma_corr_body(bs_env, T_rows, chi, first_row, b, x, iterations, residual)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: T_rows(:, :), chi(:, :)
      INTEGER, INTENT(IN)                                :: first_row
      COMPLEX(KIND=dp), INTENT(IN)                       :: b(:, :)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: x(:, :)
      INTEGER, INTENT(OUT)                               :: iterations
      REAL(KIND=dp), INTENT(OUT)                         :: residual

      REAL(KIND=dp), PARAMETER                           :: tolerance = 1.0E-12_dp

      COMPLEX(KIND=dp)                                   :: denominator
      COMPLEX(KIND=dp), ALLOCATABLE                      :: aux_y(:, :), direction(:, :), &
                                                            grid_x(:, :), grid_y(:, :), &
                                                            PRODUCT(:, :), r(:, :), true_r(:, :)
      INTEGER                                            :: handle, i_c, i_iter, nc
      LOGICAL                                            :: active(SIZE(b, 2)), &
                                                            check_column(SIZE(b, 2))
      REAL(KIND=dp)                                      :: alpha, norm_b(SIZE(b, 2)), relative_r, &
                                                            rr(SIZE(b, 2)), rr_new(SIZE(b, 2))

      CALL timeset('gamma_body_iterative_solve', handle)

      nc = SIZE(b, 2)
      x = z_zero
      iterations = 0
      residual = 0.0_dp
      DO i_c = 1, nc
         rr(i_c) = REAL(DOT_PRODUCT(b(:, i_c), b(:, i_c)), dp)
      END DO
      norm_b = SQRT(rr)
      active = norm_b > 0.0_dp
      ALLOCATE (grid_x(SIZE(chi, 2), nc), grid_y(SIZE(chi, 1), nc), aux_y(SIZE(T_rows, 2), nc))
      ALLOCATE (r, SOURCE=b)
      ALLOCATE (direction, SOURCE=b)
      ALLOCATE (PRODUCT(SIZE(b, 1), nc), true_r(SIZE(b, 1), nc))
      DO i_iter = 1, 4*SIZE(b, 1)
         IF (.NOT. ANY(active)) EXIT
         CALL apply_gamma_corr_body(bs_env, T_rows, chi, first_row, direction, product, aux_y, grid_x, grid_y)
         PRODUCT(:, :) = product + direction
         check_column = .FALSE.
         DO i_c = 1, nc
            IF (.NOT. active(i_c)) CYCLE
            denominator = DOT_PRODUCT(direction(:, i_c), PRODUCT(:, i_c))
            IF (REAL(denominator, dp) <= 0.0_dp .OR. &
                ABS(AIMAG(denominator)) > 1.0E-10_dp*ABS(denominator)) THEN
               CPABORT('Gamma body action is not positive Hermitian')
            END IF
            alpha = rr(i_c)/REAL(denominator, dp)
            x(:, i_c) = x(:, i_c) + alpha*direction(:, i_c)
            r(:, i_c) = r(:, i_c) - alpha*PRODUCT(:, i_c)
            rr_new(i_c) = REAL(DOT_PRODUCT(r(:, i_c), r(:, i_c)), dp)
            check_column(i_c) = SQRT(rr_new(i_c)) <= tolerance*norm_b(i_c)
         END DO
         iterations = i_iter
         IF (ANY(check_column)) THEN
            CALL apply_gamma_corr_body(bs_env, T_rows, chi, first_row, x, true_r, aux_y, grid_x, grid_y)
            true_r(:, :) = b - x - true_r
         END IF
         DO i_c = 1, nc
            IF (.NOT. active(i_c)) CYCLE
            IF (check_column(i_c)) THEN
               relative_r = SQRT(REAL(DOT_PRODUCT(true_r(:, i_c), true_r(:, i_c)), dp))/norm_b(i_c)
               IF (relative_r <= 5.0_dp*tolerance) THEN
                  residual = MAX(residual, relative_r)
                  active(i_c) = .FALSE.
                  direction(:, i_c) = z_zero
               ELSE
                  r(:, i_c) = true_r(:, i_c)
                  direction(:, i_c) = r(:, i_c)
                  rr(i_c) = REAL(DOT_PRODUCT(r(:, i_c), r(:, i_c)), dp)
               END IF
            ELSE
               direction(:, i_c) = r(:, i_c) + (rr_new(i_c)/rr(i_c))*direction(:, i_c)
               rr(i_c) = rr_new(i_c)
            END IF
         END DO
      END DO
      IF (ANY(active)) CPABORT('Matrix-free Gamma body solve did not converge')

      CALL timestop(handle)

   END SUBROUTINE solve_gamma_corr_body

! **************************************************************************************************
!> \brief Contract M_pq=Σ_ABPQ U^A_Pp [Σ_R S_AB,PQ(R)] U^B_Qq using atom-local U (1).
!> P,Q enumerate reference functions on atoms A,B; p,q enumerate their optimized
!> functions. Each distributed S element contributes once. No atom pair, image,
!> or coefficient is screened. Sum the resulting optimized columns over MPI (2).
!> Only atom-local transformation rows and one atom-pair intermediate are stored;
!> the final optimized M remains replicated. This is exactly the real q=0 metric.
!> \param images ...
!> \param auto_ri ...
!> \param metric ...
!> \param unit_nr ...
! **************************************************************************************************
   SUBROUTINE gamma_metric_contract_atoms(images, auto_ri, metric, unit_nr)
      TYPE(cp_fm_type), INTENT(IN)                       :: images(:)
      TYPE(auto_ri_type), INTENT(IN)                     :: auto_ri
      COMPLEX(KIND=dp), INTENT(OUT)                      :: metric(:, :)
      INTEGER, INTENT(IN)                                :: unit_nr

      INTEGER                                            :: handle
      TYPE(gamma_atom_projection_type), ALLOCATABLE      :: left(:), right(:)

      CALL timeset('gamma_metric_contract_atoms', handle)

      CALL prepare_gamma_atom_projections(images, auto_ri, left, right)
      CALL accumulate_gamma_atom_metric(images, left, right, metric, unit_nr)
      DEALLOCATE (left, right)

      CALL timestop(handle)

   END SUBROUTINE gamma_metric_contract_atoms

! **************************************************************************************************
!> \brief Extract the distributed reference rows of each atom-local U in equation (1).
!> \param images ...
!> \param auto_ri ...
!> \param left ...
!> \param right ...
! **************************************************************************************************
   SUBROUTINE prepare_gamma_atom_projections(images, auto_ri, left, right)
      TYPE(cp_fm_type), INTENT(IN)                       :: images(:)
      TYPE(auto_ri_type), INTENT(IN)                     :: auto_ri
      TYPE(gamma_atom_projection_type), ALLOCATABLE, &
         INTENT(OUT)                                     :: left(:), right(:)

      INTEGER                                            :: nc, ngc, ngr, nr
      INTEGER, POINTER                                   :: cols(:), rows(:)
      LOGICAL                                            :: valid_layout

      CPASSERT(SIZE(images) > 0)
      CPASSERT(auto_ri%ready)
      CPASSERT(ALLOCATED(auto_ri%sizes_ref_RI))
      CPASSERT(ALLOCATED(auto_ri%sizes_opt_RI))
      CPASSERT(ALLOCATED(auto_ri%U_Pp_atom))
      CPASSERT(ALLOCATED(auto_ri%U_Pp_atom_offset))
      CPASSERT(SIZE(auto_ri%sizes_ref_RI) > 0)
      CPASSERT(SIZE(auto_ri%sizes_ref_RI) == SIZE(auto_ri%sizes_opt_RI))
      CPASSERT(SIZE(auto_ri%U_Pp_atom_offset) == SIZE(auto_ri%sizes_ref_RI))
      valid_layout = ALL(auto_ri%sizes_ref_RI > 0) .AND. ALL(auto_ri%sizes_opt_RI > 0)
      CPASSERT(valid_layout)
      CALL cp_fm_get_info(images(1), nrow_local=nr, ncol_local=nc, &
                          nrow_global=ngr, ncol_global=ngc, row_indices=rows, col_indices=cols)
      CPASSERT(ngr == SUM(auto_ri%sizes_ref_RI) .AND. ngc == ngr)
      CALL extract_gamma_atom_rows(auto_ri, rows(1:nr), left)
      CALL extract_gamma_atom_rows(auto_ri, cols(1:nc), right)
   END SUBROUTINE prepare_gamma_atom_projections

! **************************************************************************************************
!> \brief Map local reference indices onto their atom blocks in equation (1).
!> \param auto_ri ...
!> \param indices ...
!> \param blocks ...
! **************************************************************************************************
   SUBROUTINE extract_gamma_atom_rows(auto_ri, indices, blocks)
      TYPE(auto_ri_type), INTENT(IN)                     :: auto_ri
      INTEGER, INTENT(IN)                                :: indices(:)
      TYPE(gamma_atom_projection_type), ALLOCATABLE, &
         INTENT(OUT)                                     :: blocks(:)

      INTEGER                                            :: first_opt, first_ref, iatom, icol, &
                                                            ilocal, ipos, irow, nopt, nref, offset

      ALLOCATE (blocks(SIZE(auto_ri%sizes_ref_RI)))
      first_ref = 1
      first_opt = 1
      DO iatom = 1, SIZE(blocks)
         nref = auto_ri%sizes_ref_RI(iatom)
         nopt = auto_ri%sizes_opt_RI(iatom)
         offset = auto_ri%U_Pp_atom_offset(iatom)
         CPASSERT(offset > 0 .AND. nref <= HUGE(0)/nopt)
         CPASSERT(offset - 1 <= SIZE(auto_ri%U_Pp_atom) - nref*nopt)
         ALLOCATE (blocks(iatom)%positions(COUNT(indices >= first_ref .AND. indices < first_ref + nref)))
         ALLOCATE (blocks(iatom)%values(SIZE(blocks(iatom)%positions), nopt))
         blocks(iatom)%first = first_opt
         ipos = 0
         DO ilocal = 1, SIZE(indices)
            irow = indices(ilocal) - first_ref + 1
            IF (irow < 1 .OR. irow > nref) CYCLE
            ipos = ipos + 1
            blocks(iatom)%positions(ipos) = ilocal
            DO icol = 1, nopt
               blocks(iatom)%values(ipos, icol) = auto_ri%U_Pp_atom(offset + irow - 1 + (icol - 1)*nref)
            END DO
         END DO
         first_ref = first_ref + nref
         first_opt = first_opt + nopt
      END DO
   END SUBROUTINE extract_gamma_atom_rows

! **************************************************************************************************
!> \brief Evaluate atom-pair products (1) locally and reduce optimized columns (2).
!> \param images ...
!> \param left ...
!> \param right ...
!> \param metric ...
!> \param unit_nr ...
! **************************************************************************************************
   SUBROUTINE accumulate_gamma_atom_metric(images, left, right, metric, unit_nr)
      TYPE(cp_fm_type), INTENT(IN)                       :: images(:)
      TYPE(gamma_atom_projection_type), INTENT(IN)       :: left(:), right(:)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: metric(:, :)
      INTEGER, INTENT(IN)                                :: unit_nr

      REAL(KIND=dp), PARAMETER :: fractions(9) = [0.001_dp, 0.003_dp, 0.01_dp, 0.03_dp, 0.1_dp, &
         0.3_dp, 0.6_dp, 0.9_dp, 1.0_dp]

      INTEGER                                            :: ia, ib, icell, iprogress, n_opt_total, &
                                                            nc, noleft, nopt, noright, nr, nref, &
                                                            pfirst, qfirst
      REAL(KIND=dp)                                      :: local_time, reduce_time, start_time, &
                                                            stats(3), tick
      REAL(KIND=dp), ALLOCATABLE                         :: block(:, :), PRODUCT(:, :), value(:, :)

      nopt = MAXVAL([(SIZE(left(ia)%values, 2), ia=1, SIZE(left))])
      nref = MAX(1, MAXVAL([(SIZE(left(ia)%positions), ia=1, SIZE(left))]), &
                 MAXVAL([(SIZE(right(ib)%positions), ib=1, SIZE(right))]))
      n_opt_total = SUM([(SIZE(left(ia)%values, 2), ia=1, SIZE(left))])
      CPASSERT(ALL(SHAPE(metric) == n_opt_total))
      ALLOCATE (block(nref, nref), PRODUCT(nref, nopt), value(nopt, nopt))
      metric = z_zero
      start_time = m_walltime()
      local_time = 0.0_dp
      reduce_time = 0.0_dp
      iprogress = 1
      IF (unit_nr > 0) THEN
         WRITE (unit_nr, '(T2,A)') 'Γ RI metric: atom-block contraction started'
         CALL m_flush(unit_nr)
      END IF
      DO ib = 1, SIZE(right)
         tick = m_walltime()
         nc = SIZE(right(ib)%positions)
         noright = SIZE(right(ib)%values, 2)
         qfirst = right(ib)%first
         DO ia = 1, SIZE(left)
            nr = SIZE(left(ia)%positions)
            IF (nr == 0 .OR. nc == 0) CYCLE
            noleft = SIZE(left(ia)%values, 2)
            pfirst = left(ia)%first
            block(1:nr, 1:nc) = 0.0_dp
            DO icell = 1, SIZE(images)
               block(1:nr, 1:nc) = block(1:nr, 1:nc) + &
                                   images(icell)%local_data(left(ia)%positions, right(ib)%positions)
            END DO
            CALL dgemm('N', 'N', nr, noright, nc, 1.0_dp, block, nref, &
                       right(ib)%values, nc, 0.0_dp, product, nref)
            CALL dgemm('T', 'N', noleft, noright, nr, 1.0_dp, left(ia)%values, nr, &
                       product, nref, 0.0_dp, value, nopt)
            metric(pfirst:pfirst + noleft - 1, qfirst:qfirst + noright - 1) = &
               CMPLX(value(1:noleft, 1:noright), 0.0_dp, dp)
         END DO
         local_time = local_time + m_walltime() - tick
         tick = m_walltime()
         CPASSERT(noright <= HUGE(0)/SIZE(metric, 1))
         CALL images(1)%matrix_struct%para_env%sum(metric(:, qfirst:qfirst + noright - 1))
         reduce_time = reduce_time + m_walltime() - tick
         IF (iprogress <= SIZE(fractions)) THEN
            IF (REAL(ib, dp) >= fractions(iprogress)*SIZE(right)) THEN
               IF (unit_nr > 0) THEN
                  WRITE (unit_nr, '(T2,A,I10,A,I10,T64,F13.3,A4)') &
                     'Γ metric atoms ', ib, ' / ', SIZE(right), m_walltime() - start_time, ' s  '
                  CALL m_flush(unit_nr)
               END IF
               DO WHILE (iprogress <= SIZE(fractions))
                  IF (REAL(ib, dp) < fractions(iprogress)*SIZE(right)) EXIT
                  iprogress = iprogress + 1
               END DO
            END IF
         END IF
      END DO
      stats = local_time
      CALL images(1)%matrix_struct%para_env%min(stats(1))
      CALL images(1)%matrix_struct%para_env%sum(stats(2))
      CALL images(1)%matrix_struct%para_env%max(stats(3))
      stats(2) = stats(2)/images(1)%matrix_struct%para_env%num_pe
      CALL images(1)%matrix_struct%para_env%max(reduce_time)
      IF (unit_nr > 0) THEN
         WRITE (unit_nr, '(T2,A,T48,3ES11.3)') 'Γ atom work seconds min/mean/max:', stats
         WRITE (unit_nr, '(T2,A,T68,ES13.5)') 'Γ atom reduction/wait maximum seconds:', reduce_time
      END IF
      DEALLOCATE (block, product, value)
      IF (unit_nr > 0) THEN
         WRITE (unit_nr, '(T2,A)') 'Γ RI metric: atom-block contraction completed'
         CALL m_flush(unit_nr)
      END IF
   END SUBROUTINE accumulate_gamma_atom_metric

! **************************************************************************************************
!> \brief Construct F=M^-1 V^1/2 with distributed Hermitian eigensystems.
!> For each input A, ifirst set A=(A+Aᴴ)/2 (1), then diagonalize A=U diag(λ) Uᴴ.
!> Define A^a=U diag(f(λ)) Uᴴ (2), where f(λ)=λ^a only for λ>max(0,η λ_max),
!> and zero otherwise. M uses a=-1 and the existing relative cutoff η; V uses
!> a=1/2 and η=0. Finally F=M^-1 (V^1/2)ᴴ (3), matching the dense Γ path.
!> Inputs and the final F remain replicated; eigenvectors and matrix products
!> are block-cyclic distributed. No spectral threshold or physical input changes.
!> \param para_env ...
!> \param relative_cutoff ...
!> \param metric ...
!> \param coulomb ...
!> \param factor ...
! **************************************************************************************************
   SUBROUTINE gamma_metric_factor_distributed(para_env, relative_cutoff, metric, coulomb, factor)
      TYPE(mp_para_env_type), INTENT(INOUT), TARGET      :: para_env
      REAL(KIND=dp), INTENT(IN)                          :: relative_cutoff
      COMPLEX(KIND=dp), INTENT(IN)                       :: metric(:, :), coulomb(:, :)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: factor(:, :)

      INTEGER                                            :: handle, n
      TYPE(cp_blacs_env_type), POINTER                   :: blacs
      TYPE(cp_cfm_type)                                  :: coulomb_d, factor_d, metric_d, work_d
      TYPE(cp_fm_struct_type), POINTER                   :: structure

      CALL timeset('gamma_metric_factor_distributed', handle)

      n = SIZE(metric, 1)
      CPASSERT(n > 0 .AND. relative_cutoff >= 0.0_dp)
      CPASSERT(ALL(SHAPE(metric) == [n, n]))
      CPASSERT(ALL(SHAPE(coulomb) == [n, n]))
      CPASSERT(ALL(SHAPE(factor) == [n, n]))
      CALL cp_blacs_env_create(blacs, para_env)
      CALL cp_fm_struct_create(structure, context=blacs, nrow_global=n, ncol_global=n)
      CALL cp_cfm_create(metric_d, structure)
      CALL cp_cfm_create(coulomb_d, structure)
      CALL cp_cfm_create(work_d, structure)
      CALL cp_cfm_create(factor_d, structure)
      CALL cp_cfm_set_submatrix(metric_d, metric)
      CALL cp_cfm_set_submatrix(coulomb_d, coulomb)
      CALL retained_hermitian_power(metric_d, work_d, factor_d, n, -1.0_dp, relative_cutoff)
      CALL retained_hermitian_power(coulomb_d, work_d, factor_d, n, 0.5_dp, 0.0_dp)
      CALL parallel_gemm('N', 'C', n, n, n, z_one, metric_d, coulomb_d, z_zero, factor_d)
      CALL collect_factor_columns(factor_d, factor)
      CALL cp_cfm_release(metric_d)
      CALL cp_cfm_release(coulomb_d)
      CALL cp_cfm_release(work_d)
      CALL cp_cfm_release(factor_d)
      CALL cp_fm_struct_release(structure)
      CALL cp_blacs_env_release(blacs)

      CALL timestop(handle)

   END SUBROUTINE gamma_metric_factor_distributed

! **************************************************************************************************
!> \brief Apply Hermitization (1) and the strictly retained spectral power (2).
!> \param matrix ...
!> \param eigenvectors ...
!> \param work ...
!> \param n ...
!> \param exponent ...
!> \param relative_cutoff ...
! **************************************************************************************************
   SUBROUTINE retained_hermitian_power(matrix, eigenvectors, work, n, exponent, relative_cutoff)
      TYPE(cp_cfm_type), INTENT(INOUT)                   :: matrix, eigenvectors, work
      INTEGER, INTENT(IN)                                :: n
      REAL(KIND=dp), INTENT(IN)                          :: exponent, relative_cutoff

      COMPLEX(KIND=dp), ALLOCATABLE                      :: scales(:)
      INTEGER                                            :: i_mode
      REAL(KIND=dp)                                      :: cutoff
      REAL(KIND=dp), ALLOCATABLE                         :: eigenvalues(:)

      CALL cp_cfm_transpose(matrix, 'C', work)
      CALL cp_cfm_scale_and_add(CMPLX(0.5_dp, 0.0_dp, dp), matrix, CMPLX(0.5_dp, 0.0_dp, dp), work)
      ALLOCATE (eigenvalues(n), scales(n))
      CALL cp_cfm_heevd(matrix, eigenvectors, eigenvalues)
      cutoff = MAX(0.0_dp, relative_cutoff*MAXVAL(eigenvalues))
      scales = z_zero
      DO i_mode = 1, n
         IF (eigenvalues(i_mode) > cutoff) scales(i_mode) = CMPLX(eigenvalues(i_mode)**(0.5_dp*exponent), 0.0_dp, dp)
      END DO
      CALL cp_cfm_column_scale(eigenvectors, scales)
      CALL parallel_gemm('N', 'C', n, n, n, z_one, eigenvectors, eigenvectors, z_zero, matrix)
      DEALLOCATE (eigenvalues, scales)
   END SUBROUTINE retained_hermitian_power

! **************************************************************************************************
!> \brief Collect factor (3) by BLACS column blocks with bounded MPI element counts.
!> \param matrix ...
!> \param factor ...
! **************************************************************************************************
   SUBROUTINE collect_factor_columns(matrix, factor)
      TYPE(cp_cfm_type), INTENT(IN)                      :: matrix
      COMPLEX(KIND=dp), INTENT(OUT)                      :: factor(:, :)

      INTEGER                                            :: i_column, i_last, n_columns, n_rows

      n_rows = SIZE(factor, 1)
      CPASSERT(n_rows > 0)
      n_columns = MIN(matrix%matrix_struct%ncol_block, HUGE(0)/n_rows)
      CPASSERT(n_columns > 0)
      i_column = 1
      DO
         i_last = i_column + (MIN(n_columns, SIZE(factor, 2) - i_column + 1) - 1)
         CALL cp_cfm_get_submatrix(matrix, factor(:, i_column:i_last), start_col=i_column)
         IF (i_last == SIZE(factor, 2)) EXIT
         i_column = i_last + 1
      END DO
   END SUBROUTINE collect_factor_columns

! **************************************************************************************************
!> \brief Form Gamma-correction d_ia^α(k) for cyclically owned SCF k points.
!> Reuses the existing AO integral builders and Bloch transformation.
!> \brief Interband Gamma-correction moments in the same SCF MO gauge as the GW body.
!> The SCF Hamiltonian contains the nonlocal pseudopotential; its k derivative
!> therefore includes the nonlocal contribution without a separate commutator.
!> \param bs_env ...
!> \param gwkp_env Gamma-correction AO operators, released after their contraction.
!> \param kpoints ...
!> \param mo_coeff ...
!> \param eigenvalues ...
!> \param nocc_spin ...
!> \param dipole ...
!> \param nmo_spin ...
! **************************************************************************************************
   SUBROUTINE gw_gamma_transition_moments(bs_env, gwkp_env, kpoints, mo_coeff, eigenvalues, nocc_spin, &
                                          dipole, nmo_spin)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gwkp_type), INTENT(INOUT)                     :: gwkp_env
      TYPE(kpoint_type), INTENT(IN), POINTER             :: kpoints
      TYPE(cp_cfm_type), DIMENSION(:, :), INTENT(IN)     :: mo_coeff
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(IN)      :: eigenvalues
      INTEGER, DIMENSION(:), INTENT(IN)                  :: nocc_spin
      COMPLEX(KIND=dp), ALLOCATABLE, &
         DIMENSION(:, :, :, :, :), INTENT(OUT)           :: dipole
      INTEGER, ALLOCATABLE, DIMENSION(:), INTENT(OUT)    :: nmo_spin

      REAL(KIND=dp), PARAMETER                           :: eps_degenerate = 1.0E-10_dp

      COMPLEX(KIND=dp), ALLOCATABLE :: coeff_dense(:, :), d_k(:, :), dh_k(:, :), ds_k(:, :), &
         projected_d(:, :), projected_h(:, :), projected_s(:, :)
      INTEGER :: handle, i_col, i_dir, i_k, i_k_local, i_mo_a, i_mo_i, i_row, i_spin, n_ao, &
         n_col_local, n_img, n_k, n_k_local, n_mo, n_mo_matrix, n_row_local, n_row_matrix, n_spin
      INTEGER, ALLOCATABLE                               :: local_kpoints(:)
      INTEGER, DIMENSION(3)                              :: periodic
      INTEGER, DIMENSION(:), POINTER                     :: col_indices, row_indices
      INTEGER, DIMENSION(:, :), POINTER                  :: index_to_cell
      INTEGER, DIMENSION(:, :, :), POINTER               :: cell_to_index
      LOGICAL                                            :: valid_layout
      REAL(KIND=dp)                                      :: energy_diff, hmat(3, 3)
      REAL(KIND=dp), ALLOCATABLE                         :: d_rs(:, :, :, :), h_rs(:, :, :, :), &
                                                            s_rs(:, :, :, :)
      TYPE(cell_type), POINTER                           :: cell
      TYPE(mp_para_env_type), POINTER                    :: para_env

      CALL timeset('gw_gamma_transition_moments', handle)

      cell => bs_env%ri_rs%cell
      para_env => bs_env%para_env
      index_to_cell => gwkp_env%gamma_corr_index_to_cell
      cell_to_index => gwkp_env%gamma_corr_cell_to_index
      CPASSERT(ASSOCIATED(index_to_cell) .AND. ASSOCIATED(cell_to_index))
      CALL get_cell(cell=cell, h=hmat, periodic=periodic)
      CALL get_kpoint_info(kpoints, nkp=n_k)
      n_spin = bs_env%n_spin
      CPASSERT(SIZE(mo_coeff, 1) >= n_k .AND. SIZE(mo_coeff, 2) == n_spin)
      valid_layout = SIZE(eigenvalues, 2) >= n_k .AND. SIZE(eigenvalues, 3) == n_spin
      CPASSERT(valid_layout)
      CPASSERT(SIZE(nocc_spin) == n_spin)
      ALLOCATE (nmo_spin(n_spin))
      DO i_spin = 1, n_spin
         CALL cp_cfm_get_info(mo_coeff(1, i_spin), nrow_global=n_row_matrix, ncol_global=n_mo_matrix)
         IF (i_spin == 1) n_ao = n_row_matrix
         valid_layout = n_row_matrix == n_ao .AND. n_mo_matrix == SIZE(eigenvalues, 1)
         CPASSERT(valid_layout)
         nmo_spin(i_spin) = n_mo_matrix
      END DO
      CPASSERT(ALL(nocc_spin >= 0 .AND. nocc_spin <= nmo_spin))

      n_img = SIZE(index_to_cell, 2)
      ALLOCATE (s_rs(1, n_ao, n_ao, n_img), h_rs(n_spin, n_ao, n_ao, n_img), &
                d_rs(3, n_ao, n_ao, n_img), source=0.0_dp)
      CALL gamma_corr_ao_images(gwkp_env%gamma_corr_s, gwkp_env%gamma_corr_scf_plan, s_rs, cell_to_index)
      CALL gamma_corr_ao_images(gwkp_env%gamma_corr_h, gwkp_env%gamma_corr_scf_plan, h_rs, cell_to_index)
      CALL gamma_corr_ao_images(gwkp_env%gamma_corr_d, gwkp_env%gamma_corr_plan, d_rs, cell_to_index)
      CALL dbcsr_deallocate_matrix_set(gwkp_env%gamma_corr_s)
      CALL dbcsr_deallocate_matrix_set(gwkp_env%gamma_corr_h)
      CALL dbcsr_deallocate_matrix_set(gwkp_env%gamma_corr_d)
      gwkp_env%gamma_corr_plan = kp_transform_plan_type()
      gwkp_env%gamma_corr_scf_plan = kp_transform_plan_type()

      n_k_local = (n_k + para_env%num_pe - 1 - para_env%mepos)/para_env%num_pe
      ALLOCATE (local_kpoints(n_k_local))
      DO i_k_local = 1, n_k_local
         local_kpoints(i_k_local) = para_env%mepos + 1 + (i_k_local - 1)*para_env%num_pe
      END DO
      ALLOCATE (dipole(n_spin, n_k_local, 3, MAXVAL(nocc_spin), MAXVAL(nmo_spin - nocc_spin)), source=z_zero)
      ALLOCATE (dh_k(n_ao, n_ao), ds_k(n_ao, n_ao), d_k(n_ao, n_ao))
      DO i_k = 1, n_k
         DO i_spin = 1, n_spin
            n_mo = nmo_spin(i_spin)
            ALLOCATE (coeff_dense(n_ao, n_mo), source=z_zero)
            CALL cp_cfm_get_info(mo_coeff(i_k, i_spin), nrow_local=n_row_local, &
                                 ncol_local=n_col_local, row_indices=row_indices, col_indices=col_indices)
            DO i_col = 1, n_col_local
               DO i_row = 1, n_row_local
                  coeff_dense(row_indices(i_row), col_indices(i_col)) = &
                     mo_coeff(i_k, i_spin)%local_data(i_row, i_col)
               END DO
            END DO
            CALL para_env%sum(coeff_dense, root=MODULO(i_k - 1, para_env%num_pe))
            IF (MODULO(i_k - 1, para_env%num_pe) == para_env%mepos) THEN
               i_k_local = (i_k - 1)/para_env%num_pe + 1
               CPASSERT(local_kpoints(i_k_local) == i_k)
               IF (nocc_spin(i_spin) > 0 .AND. n_mo > nocc_spin(i_spin)) THEN
                  ALLOCATE (projected_h(n_mo - nocc_spin(i_spin), nocc_spin(i_spin)), &
                            projected_s(n_mo - nocc_spin(i_spin), nocc_spin(i_spin)), &
                            projected_d(n_mo - nocc_spin(i_spin), nocc_spin(i_spin)))
                  DO i_dir = 1, 3
                     IF (periodic(i_dir) == 0) CYCLE
                     CALL rs_to_kp(h_rs(i_spin, :, :, :), dh_k, index_to_cell, &
                                   kpoints%xkp(:, i_k), i_dir, hmat)
                     CALL rs_to_kp(s_rs(1, :, :, :), ds_k, index_to_cell, &
                                   kpoints%xkp(:, i_k), i_dir, hmat)
                     CALL rs_to_kp(d_rs(i_dir, :, :, :), d_k, index_to_cell, kpoints%xkp(:, i_k))
                     projected_h(:, :) = MATMUL(CONJG(TRANSPOSE(coeff_dense(:, nocc_spin(i_spin) + 1:n_mo))), &
                                                MATMUL(dh_k, coeff_dense(:, 1:nocc_spin(i_spin))))
                     projected_s(:, :) = MATMUL(CONJG(TRANSPOSE(coeff_dense(:, nocc_spin(i_spin) + 1:n_mo))), &
                                                MATMUL(ds_k, coeff_dense(:, 1:nocc_spin(i_spin))))
                     projected_d(:, :) = MATMUL(CONJG(TRANSPOSE(coeff_dense(:, nocc_spin(i_spin) + 1:n_mo))), &
                                                MATMUL(d_k, coeff_dense(:, 1:nocc_spin(i_spin))))
                     DO i_mo_i = 1, nocc_spin(i_spin)
                        DO i_mo_a = 1, n_mo - nocc_spin(i_spin)
                           energy_diff = eigenvalues(nocc_spin(i_spin) + i_mo_a, i_k, i_spin) - &
                                         eigenvalues(i_mo_i, i_k, i_spin)
                           IF (ABS(energy_diff) <= eps_degenerate) CYCLE
                           dipole(i_spin, i_k_local, i_dir, i_mo_i, i_mo_a) = &
                              (-gaussi*projected_h(i_mo_a, i_mo_i) + &
                               gaussi*eigenvalues(nocc_spin(i_spin) + i_mo_a, i_k, i_spin)* &
                               projected_s(i_mo_a, i_mo_i))/energy_diff + projected_d(i_mo_a, i_mo_i)
                        END DO
                     END DO
                  END DO
                  DEALLOCATE (projected_h, projected_s, projected_d)
               END IF
            END IF
            DEALLOCATE (coeff_dense)
         END DO
      END DO
      DEALLOCATE (dh_k, ds_k, d_k, h_rs, s_rs, d_rs, local_kpoints)
      DEALLOCATE (gwkp_env%gamma_corr_index_to_cell, gwkp_env%gamma_corr_cell_to_index)
      NULLIFY (gwkp_env%gamma_corr_index_to_cell, gwkp_env%gamma_corr_cell_to_index)

      CALL timestop(handle)

   END SUBROUTINE gw_gamma_transition_moments

! **************************************************************************************************
!> \brief Scatter one low-rank Gamma correction into the centred grid-pair images.
!> For s=nearest_integer[H^{-1}(r_l-r_l')], C_ll'(R)=C0_ll'(R-s).
!> R-s is wrapped into the finite FFT cell with its transfer-mesh Bloch twist. Half-cell
!> ties average adjacent integer images, retaining Hermiticity and cell-wrapping covariance.
!> The mirrored conjugate-transpose contribution enforces reciprocity also on Nyquist faces.
!> No grid-pair angular tensor is formed. first_pair>0 selects flattened column-major pairs;
!> otherwise values_R owns rows beginning at first_row and all grid columns.
!> \param bs_env ...
!> \param cells ...
!> \param icell ...
!> \param head_scalar ...
!> \param head_vector ...
!> \param first_row ...
!> \param first_pair ...
!> \param values_R ...
!> \param wing_left ...
!> \param wing_right ...
! **************************************************************************************************
   SUBROUTINE add_gamma_corr_pair_cell(bs_env, cells, icell, head_scalar, head_vector, &
                                       first_row, first_pair, values_R, wing_left, wing_right)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: cells(:, :), icell
      COMPLEX(KIND=dp), INTENT(IN)                       :: head_scalar, head_vector(:)
      INTEGER, INTENT(IN)                                :: first_row, first_pair
      REAL(KIND=dp), INTENT(INOUT)                       :: values_R(:, :, :)
      COMPLEX(KIND=dp), INTENT(IN), OPTIONAL             :: wing_left(:), wing_right(:)

      COMPLEX(KIND=dp)                                   :: amplitude(2), twist(27, 2)
      INTEGER :: destination(27, 2), high(3), idir, iimage, il, ilocal, ilp, imirror, ipair, ix, &
         iy, iz, jlocal, low(3), lower(3), ngrid, nmesh(3), shift(3), translated(3), wrapped(3)
      LOGICAL                                            :: valid_layout
      REAL(KIND=dp)                                      :: delta(3), half_boundary, qoffset(3), &
                                                            tolerance, weight
      REAL(KIND=dp), ALLOCATABLE                         :: fractional_grid(:, :)

      CPASSERT(PRESENT(wing_left) .EQV. PRESENT(wing_right))
      ngrid = SIZE(head_vector)
      CPASSERT(ngrid > 0)
      CPASSERT(ALLOCATED(bs_env%ri_rs%grid_points))
      CPASSERT(ASSOCIATED(bs_env%ri_rs%cell))
      CPASSERT(SIZE(bs_env%ri_rs%grid_points, 1) == 3)
      CPASSERT(SIZE(bs_env%ri_rs%grid_points, 2) == ngrid)
      valid_layout = SIZE(cells, 1) == 3 .AND. icell >= 1 .AND. icell <= SIZE(cells, 2)
      CPASSERT(valid_layout)
      IF (PRESENT(wing_left)) THEN
         CPASSERT(SIZE(wing_left) == ngrid .AND. SIZE(wing_right) == ngrid)
      END IF
      IF (first_pair > 0) THEN
         CPASSERT(ngrid <= HUGE(1)/ngrid)
         CPASSERT(SIZE(values_R, 2) == 1)
         CPASSERT(first_pair + SIZE(values_R, 1) - 1 <= ngrid*ngrid)
      ELSE
         valid_layout = first_row >= 1 .AND. first_row + SIZE(values_R, 1) - 1 <= ngrid
         CPASSERT(valid_layout)
         CPASSERT(SIZE(values_R, 2) == ngrid)
      END IF
      nmesh = bs_env%kpoints_scf_desymm%nkp_grid
      CPASSERT(ALL(nmesh > 0))
      lower = -nmesh/2
      CPASSERT(SIZE(cells, 2) == PRODUCT(nmesh))
      CPASSERT(SIZE(values_R, 3) == SIZE(cells, 2))
      qoffset = bs_env%kpoints_chi_eps_W%xkp(1:3, 1)
      ALLOCATE (fractional_grid(3, ngrid))
      fractional_grid(:, :) = MATMUL(bs_env%ri_rs%cell%h_inv, bs_env%ri_rs%grid_points)
      DO idir = 1, 3
         IF (bs_env%periodic(idir) == 0) fractional_grid(idir, :) = 0.0_dp
         valid_layout = MAXVAL(ABS(fractional_grid(idir, :))) <= 0.5_dp + 1.0E-10_dp
         CPASSERT(valid_layout)
      END DO
      DO iz = -1, 1
         DO iy = -1, 1
            DO ix = -1, 1
               shift = [ix, iy, iz]
               iimage = 1 + ix + 1 + 3*(iy + 1) + 9*(iz + 1)
               DO imirror = 1, 2
                  translated = (3 - 2*imirror)*cells(:, icell) + shift
                  wrapped = MODULO(translated - lower, nmesh) + lower
                  destination(iimage, imirror) = 1 + (wrapped(3) - lower(3)) + &
                                                 nmesh(3)*(wrapped(2) - lower(2) + nmesh(2)*(wrapped(1) - lower(1)))
                  valid_layout = ALL(cells(:, destination(iimage, imirror)) == wrapped)
                  CPASSERT(valid_layout)
                  twist(iimage, imirror) = EXP(CMPLX(0.0_dp, &
                                                     twopi*DOT_PRODUCT(qoffset, REAL(translated - wrapped, dp)), dp))
               END DO
            END DO
         END DO
      END DO
      DO jlocal = 1, SIZE(values_R, 2)
         DO ilocal = 1, SIZE(values_R, 1)
            IF (first_pair > 0) THEN
               ipair = first_pair + ilocal - 1
               il = MODULO(ipair - 1, ngrid) + 1
               ilp = (ipair - 1)/ngrid + 1
            ELSE
               il = first_row + ilocal - 1
               ilp = jlocal
            END IF
            delta = fractional_grid(:, il) - fractional_grid(:, ilp)
            low = NINT(delta)
            high = low
            DO idir = 1, 3
               half_boundary = ANINT(delta(idir) - 0.5_dp) + 0.5_dp
               tolerance = 64.0_dp*EPSILON(1.0_dp)*MAX(1.0_dp, ABS(delta(idir)))
               IF (ABS(delta(idir) - half_boundary) <= tolerance) THEN
                  low(idir) = NINT(half_boundary - 0.5_dp)
                  high(idir) = low(idir) + 1
               END IF
            END DO
            weight = 0.5_dp/REAL(PRODUCT(high - low + 1), dp)
            amplitude(1) = head_scalar*head_vector(il)*CONJG(head_vector(ilp))
            amplitude(2) = CONJG(head_scalar)*head_vector(il)*CONJG(head_vector(ilp))
            IF (PRESENT(wing_left)) THEN
               amplitude(1) = amplitude(1) + &
                              head_vector(il)*wing_right(ilp) + wing_left(il)*CONJG(head_vector(ilp))
               amplitude(2) = amplitude(2) + &
                              CONJG(head_vector(ilp)*wing_right(il) + wing_left(ilp)*CONJG(head_vector(il)))
            END IF
            DO iz = low(3), high(3)
               DO iy = low(2), high(2)
                  DO ix = low(1), high(1)
                     iimage = 1 + ix + 1 + 3*(iy + 1) + 9*(iz + 1)
                     DO imirror = 1, 2
                        values_R(ilocal, jlocal, destination(iimage, imirror)) = &
                           values_R(ilocal, jlocal, destination(iimage, imirror)) + &
                           weight*REAL(twist(iimage, imirror)*amplitude(imirror), dp)
                     END DO
                  END DO
               END DO
            END DO
         END DO
      END DO
   END SUBROUTINE add_gamma_corr_pair_cell

! **************************************************************************************************
!> \brief Form band Bloch orbitals once instead of communicating them for every self-energy.
!> \param bs_env ...
!> \param phi_R ...
!> \param orbitals ...
!> \param phi_native_k ...
! **************************************************************************************************
   SUBROUTINE collect_band_bloch_orbitals(bs_env, phi_R, orbitals, phi_native_k)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(KIND=dp), INTENT(IN)                          :: phi_R(:, :, :)
      COMPLEX(dp), ALLOCATABLE, INTENT(INOUT)            :: orbitals(:, :, :, :)
      COMPLEX(dp), INTENT(IN), OPTIONAL                  :: phi_native_k(:, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'collect_band_bloch_orbitals'
      REAL(dp), PARAMETER :: progress_fraction(9) = [0.001_dp, 0.003_dp, 0.01_dp, 0.03_dp, 0.1_dp, &
         0.3_dp, 0.6_dp, 0.9_dp, 1.0_dp]

      COMPLEX(KIND=dp)                                   :: phase
      COMPLEX(KIND=dp), ALLOCATABLE                      :: coefficients(:, :, :, :), phi_k(:, :), &
                                                            phi_plane(:, :), psi_k(:, :)
      INTEGER :: handle, i_kp_eval, i_local, i_native, i_node, i_owned, i_progress, i_R, i_root, &
         i_row, ispin, n_grid, n_local, n_local_max, n_node, n_owned
      REAL(KIND=dp)                                      :: available_GB, delta_k(3), orbital_bytes, &
                                                            start_time
      TYPE(mp_comm_type)                                 :: node_comm

      CALL timeset(routineN, handle)

      n_grid = SIZE(phi_R, 1)
      IF (PRESENT(phi_native_k)) THEN
         CPASSERT(SIZE(phi_native_k, 1) == n_grid)
         CPASSERT(SIZE(phi_native_k, 2) == bs_env%n_ao)
         n_owned = (bs_env%nkp_scf_desymm + bs_env%para_env%num_pe - 1 - bs_env%para_env%mepos)/ &
                   bs_env%para_env%num_pe
         CPASSERT(SIZE(phi_native_k, 3) == n_owned)
      END IF
      CPASSERT(SIZE(phi_R, 2) == bs_env%n_ao)
      CPASSERT(SIZE(phi_R, 3) == SIZE(bs_env%index_to_cell_3c, 2))
      CPASSERT(bs_env%para_env%num_pe > 0)
      CPASSERT(bs_env%para_env%mepos >= 0)
      CPASSERT(bs_env%para_env%mepos < bs_env%para_env%num_pe)
      CPASSERT(.NOT. ALLOCATED(orbitals))
      CALL collect_band_mo_coefficients(bs_env, coefficients)
      CALL node_comm%from_split_type(bs_env%para_env, mp_comm_split_type_shared, key=bs_env%para_env%mepos)
      n_node = node_comm%num_pe
      i_node = node_comm%mepos
      n_local_max = n_grid/n_node + MIN(1, MODULO(n_grid, n_node))
      n_local = n_grid/n_node + MERGE(1, 0, i_node < MODULO(n_grid, n_node))
      ! Charge each rank its complete orbital copy and its own work arrays.
      orbital_bytes = 16.0_dp*REAL(n_grid, dp)*REAL(bs_env%n_ao, dp)* &
                      REAL(bs_env%nkp_bs_and_DOS, dp)*REAL(bs_env%n_spin, dp) + &
                      32.0_dp*REAL(n_local_max, dp)*REAL(bs_env%n_ao, dp)
      IF (PRESENT(phi_native_k)) orbital_bytes = orbital_bytes + &
                                                 16.0_dp*REAL(n_grid, dp)*REAL(bs_env%n_ao, dp)
      ! Already allocated coefficients and interpolation data are included in the
      ! measured usage. Leave half the available memory for subsequent workspaces.
      CALL mp_mem_avail_per_rank_GB(bs_env%para_env, available_GB)
      CALL bs_env%para_env%min(available_GB)
      IF (orbital_bytes > 0.5_dp*MAX(0.0_dp, available_GB)*1.0E9_dp) THEN
         CPABORT("Insufficient memory for GW band-path orbitals")
      END IF
      ALLOCATE (orbitals(n_grid, bs_env%n_ao, bs_env%nkp_bs_and_DOS, bs_env%n_spin), SOURCE=z_zero)
      ALLOCATE (phi_k(n_local, bs_env%n_ao), psi_k(n_local, bs_env%n_ao))
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T69,F12.1)') &
            'GW band-path orbitals: GB per rank', 16.0_dp*REAL(SIZE(orbitals, KIND=int_8), dp)*1.0E-9_dp
         FLUSH (bs_env%unit_nr)
      END IF
      ! Node-local ranks compute disjoint rows and exchange each completed band.
      IF (PRESENT(phi_native_k)) ALLOCATE (phi_plane(n_grid, bs_env%n_ao))
      start_time = m_walltime()
      i_progress = 1
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A)') 'GW: evaluating band-path orbitals on the RI-RS grid'
         FLUSH (bs_env%unit_nr)
      END IF
      DO i_kp_eval = 1, bs_env%nkp_bs_and_DOS
         i_native = 0
         IF (PRESENT(phi_native_k) .AND. i_kp_eval <= bs_env%nkp_scf_desymm) THEN
            delta_k = bs_env%kpoints_DOS%xkp(:, i_kp_eval) - bs_env%kpoints_scf_desymm%xkp(:, i_kp_eval)
            IF (MAXVAL(ABS(delta_k - ANINT(delta_k))) <= 64.0_dp*EPSILON(1.0_dp)) i_native = i_kp_eval
         END IF
         phi_k = z_zero
         IF (i_native > 0) THEN
            i_root = MODULO(i_native - 1, bs_env%para_env%num_pe)
            i_owned = (i_native - 1)/bs_env%para_env%num_pe + 1
            IF (bs_env%para_env%mepos == i_root) phi_plane(:, :) = phi_native_k(:, :, i_owned)
            CALL bs_env%para_env%bcast(phi_plane, i_root)
            DO i_local = 1, n_local
               i_row = i_node + 1 + (i_local - 1)*n_node
               phi_k(i_local, :) = phi_plane(i_row, :)
            END DO
         ELSE
            DO i_R = 1, SIZE(phi_R, 3)
               phase = EXP(CMPLX(0.0_dp, twopi*SUM(bs_env%kpoints_DOS%xkp(:, i_kp_eval)* &
                                                   bs_env%index_to_cell_3c(:, i_R)), KIND=dp))
               DO i_local = 1, n_local
                  i_row = i_node + 1 + (i_local - 1)*n_node
                  phi_k(i_local, :) = phi_k(i_local, :) + phase*CMPLX(phi_R(i_row, :, i_R), 0.0_dp, dp)
               END DO
            END DO
         END IF
         IF (n_local > 0) THEN
            DO ispin = 1, bs_env%n_spin
               CALL zgemm('N', 'N', n_local, bs_env%n_ao, bs_env%n_ao, z_one, phi_k, n_local, &
                          coefficients(:, :, i_kp_eval, ispin), bs_env%n_ao, z_zero, psi_k, n_local)
               DO i_local = 1, n_local
                  i_row = i_node + 1 + (i_local - 1)*n_node
                  orbitals(i_row, :, i_kp_eval, ispin) = psi_k(i_local, :)
               END DO
            END DO
         END IF
         DO ispin = 1, bs_env%n_spin
            CALL node_comm%sum(orbitals(:, :, i_kp_eval, ispin))
         END DO
         IF (i_progress <= SIZE(progress_fraction)) THEN
            IF (REAL(i_kp_eval, dp) >= progress_fraction(i_progress)*REAL(bs_env%nkp_bs_and_DOS, dp)) THEN
            IF (bs_env%unit_nr > 0) THEN
               WRITE (bs_env%unit_nr, '(T2,A,T26,I5,A,I0,T40,F5.1,A,T55,A,T69,F10.1,A)') &
                  'GW band-path k-points ', i_kp_eval, ' / ', bs_env%nkp_bs_and_DOS, &
                  100.0_dp*REAL(i_kp_eval, dp)/REAL(bs_env%nkp_bs_and_DOS, dp), ' %', &
                  'Total time:', m_walltime() - start_time, ' s'
               FLUSH (bs_env%unit_nr)
            END IF
            DO WHILE (i_progress <= SIZE(progress_fraction))
               IF (REAL(i_kp_eval, dp) < progress_fraction(i_progress)*REAL(bs_env%nkp_bs_and_DOS, dp)) EXIT
               i_progress = i_progress + 1
            END DO
            END IF
         END IF
      END DO
      IF (ALLOCATED(phi_plane)) DEALLOCATE (phi_plane)
      CALL node_comm%free()
      DEALLOCATE (phi_k, psi_k, coefficients)

      CALL timestop(handle)

   END SUBROUTINE collect_band_bloch_orbitals

! **************************************************************************************************
!> \brief Build the band/q interpolation geometry once for all imaginary-time projections.
!> \param bs_env ...
!> \param interpolation ...
! **************************************************************************************************
   SUBROUTINE build_projection_interpolation(bs_env, interpolation)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(projection_interpolation_type), INTENT(OUT)   :: interpolation

      CHARACTER(LEN=*), PARAMETER :: routineN = 'build_projection_interpolation'

      INTEGER                                            :: handle, i_kp_eval, iq, n_offmesh, &
                                                            native_iq
      REAL(KIND=dp)                                      :: delta_k(3)

      CALL timeset(routineN, handle)

      ALLOCATE (interpolation%native_iq(bs_env%nkp_bs_and_DOS), &
                interpolation%offmesh_index(bs_env%nkp_bs_and_DOS))
      interpolation%native_iq = 0
      interpolation%offmesh_index = 0
      n_offmesh = 0
      DO i_kp_eval = 1, bs_env%nkp_bs_and_DOS
         native_iq = 0
         IF (i_kp_eval <= bs_env%nkp_scf_desymm) THEN
            delta_k = bs_env%kpoints_DOS%xkp(:, i_kp_eval) - &
                      bs_env%kpoints_scf_desymm%xkp(:, i_kp_eval)
            IF (MAXVAL(ABS(delta_k - ANINT(delta_k))) <= 64.0_dp*EPSILON(1.0_dp)) THEN
               native_iq = i_kp_eval
            END IF
         ELSE
            DO iq = 1, bs_env%nkp_scf_desymm
               delta_k = bs_env%kpoints_DOS%xkp(:, i_kp_eval) - &
                         bs_env%kpoints_scf_desymm%xkp(:, iq)
               IF (MAXVAL(ABS(delta_k - ANINT(delta_k))) <= 64.0_dp*EPSILON(1.0_dp)) THEN
                  native_iq = iq
                  EXIT
               END IF
            END DO
         END IF
         interpolation%native_iq(i_kp_eval) = native_iq
         IF (native_iq == 0) THEN
            n_offmesh = n_offmesh + 1
            interpolation%offmesh_index(i_kp_eval) = n_offmesh
         END IF
      END DO
      ! Bound the replicated geometry cache; large band paths keep the original evaluation.
      IF (8.0_dp*REAL(bs_env%nkp_scf_desymm, dp)*REAL(n_offmesh, dp) <= 256.0_dp*1024.0_dp**2) THEN
         ALLOCATE (interpolation%weights(bs_env%nkp_scf_desymm, n_offmesh))
         DO i_kp_eval = 1, bs_env%nkp_bs_and_DOS
            IF (interpolation%offmesh_index(i_kp_eval) == 0) CYCLE
            DO iq = 1, bs_env%nkp_scf_desymm
               interpolation%weights(iq, interpolation%offmesh_index(i_kp_eval)) = &
                  cardinal_weight(bs_env%kpoints_DOS%xkp(:, i_kp_eval) - &
                                  bs_env%kpoints_scf_desymm%xkp(:, iq), &
                                  bs_env%kpoints_scf_desymm%nkp_grid)
            END DO
         END DO
      END IF

      CALL timestop(handle)

   END SUBROUTINE build_projection_interpolation

! **************************************************************************************************
!> \brief Replicate band-k MO coefficients once before imaginary-time work is split into groups.
!> \param bs_env ...
!> \param band_coefficients ...
! **************************************************************************************************
   SUBROUTINE collect_band_mo_coefficients(bs_env, band_coefficients)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: band_coefficients(:, :, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'collect_band_mo_coefficients'

      INTEGER                                            :: handle, i_col, i_kp_eval, i_row, ispin, &
                                                            ncol_local, nrow_local
      INTEGER, POINTER                                   :: col_indices(:), row_indices(:)

      CALL timeset(routineN, handle)

      ALLOCATE (band_coefficients(bs_env%n_ao, bs_env%n_ao, &
                                  bs_env%nkp_bs_and_DOS, bs_env%n_spin))
      band_coefficients = z_zero
      DO ispin = 1, bs_env%n_spin
         DO i_kp_eval = 1, bs_env%nkp_bs_and_DOS
            CALL cp_cfm_get_info(bs_env%cfm_mo_coeff_kp(i_kp_eval, ispin), &
                                 nrow_local=nrow_local, ncol_local=ncol_local, &
                                 row_indices=row_indices, col_indices=col_indices)
            DO i_col = 1, ncol_local
               DO i_row = 1, nrow_local
                  band_coefficients(row_indices(i_row), col_indices(i_col), i_kp_eval, ispin) = &
                     bs_env%cfm_mo_coeff_kp(i_kp_eval, ispin)%local_data(i_row, i_col)
               END DO
            END DO
         END DO
      END DO
      CALL bs_env%para_env%sum(band_coefficients)

      CALL timestop(handle)

   END SUBROUTINE collect_band_mo_coefficients

! **************************************************************************************************
!> \brief Interpolate Σx_ll′(k) and project onto band Bloch orbitals, Eqs. (10)-(12).
!> Σx_nn(k) = Σ_ll′ ψ_nk(r_l)* Σx_ll′(k) ψ_nk(r_l′).
!> Native mesh points use their existing matrix; off-mesh points use the
!> pair-centred cardinal interpolation prepared for the requested bands.
!> \param bs_env ...
!> \param sigma_grid ...
!> \param first_row ...
!> \param last_row ...
!> \param ispin ...
!> \param sigma_band ...
!> \param interpolation ...
!> \param band_orbitals ...
! **************************************************************************************************
   SUBROUTINE self_energy_RS_to_Bloch(bs_env, sigma_grid, first_row, last_row, ispin, sigma_band, &
                                      interpolation, band_orbitals)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(dp), INTENT(IN)                            :: sigma_grid(:, :, :)
      INTEGER, INTENT(IN)                                :: first_row, last_row, ispin
      REAL(dp), INTENT(OUT)                              :: sigma_band(:, :)
      TYPE(projection_interpolation_type), INTENT(IN)    :: interpolation
      COMPLEX(dp), INTENT(IN)                            :: band_orbitals(:, :, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'self_energy_RS_to_Bloch'

      COMPLEX(dp), ALLOCATABLE                           :: psi(:, :), sigma_k(:, :), work(:, :)
      INTEGER                                            :: handle, i_band, i_batch, i_first, &
                                                            i_kp_eval, i_last, n_batch, n_grid, &
                                                            native_iq, nrows, phase_handle
      LOGICAL                                            :: valid_layout

      CALL timeset(routineN, handle)

      n_grid = SIZE(band_orbitals, 1)
      nrows = last_row - first_row + 1
      CPASSERT(first_row >= 1 .AND. last_row <= n_grid .AND. nrows >= 0)
      valid_layout = SIZE(sigma_grid, 1) == nrows .AND. SIZE(sigma_grid, 2) == n_grid
      CPASSERT(valid_layout)
      CPASSERT(SIZE(sigma_grid, 3) == bs_env%nkp_scf_desymm)
      CPASSERT(SIZE(band_orbitals, 2) == bs_env%n_ao)
      CPASSERT(SIZE(band_orbitals, 3) == bs_env%nkp_bs_and_DOS)
      CPASSERT(SIZE(band_orbitals, 4) == bs_env%n_spin)
      CPASSERT(SIZE(sigma_band, 1) == bs_env%n_ao)
      CPASSERT(SIZE(sigma_band, 2) == bs_env%nkp_bs_and_DOS)
      n_batch = MIN(16, bs_env%n_ao)
      ALLOCATE (sigma_k(nrows, n_grid), psi(n_grid, n_batch), work(nrows, n_batch))
      sigma_band = 0.0_dp
      DO i_kp_eval = 1, bs_env%nkp_bs_and_DOS
         CALL timeset(routineN//'_q_interpolation', phase_handle)
         native_iq = interpolation%native_iq(i_kp_eval)
         IF (native_iq > 0) THEN
            sigma_k(:, :) = sigma_grid(:, :, native_iq)
         ELSE
            CALL interpolate_self_energy_rows(bs_env, sigma_grid, first_row, i_kp_eval, sigma_k, interpolation)
         END IF
         CALL timestop(phase_handle)
         DO i_first = 1, bs_env%n_ao, n_batch
            i_last = MIN(i_first + n_batch - 1, bs_env%n_ao)
            psi = z_zero
            psi(:, 1:i_last - i_first + 1) = band_orbitals(:, i_first:i_last, i_kp_eval, ispin)
            CALL timeset(routineN//'_apply_sigma', phase_handle)
            IF (nrows > 0) CALL zgemm('N', 'N', nrows, n_batch, n_grid, z_one, &
                                      sigma_k, nrows, psi, n_grid, z_zero, work, nrows)
            CALL timestop(phase_handle)
            CALL timeset(routineN//'_band_contraction', phase_handle)
            DO i_band = i_first, i_last
               i_batch = i_band - i_first + 1
               sigma_band(i_band, i_kp_eval) = REAL(SUM(CONJG(psi(first_row:last_row, i_batch))*work(:, i_batch)), dp)
            END DO
            CALL timestop(phase_handle)
         END DO
      END DO
      CALL timeset(routineN//'_final_reduction', phase_handle)
      CALL bs_env%para_env%sum(sigma_band)
      CALL timestop(phase_handle)
      DEALLOCATE (sigma_k, psi, work)

      CALL timestop(handle)

   END SUBROUTINE self_energy_RS_to_Bloch

! **************************************************************************************************
!> \brief Project ordered local Σ(l,l') pairs with cached band Bloch orbitals.
!>
!> Each owned pair contributes Re[ψ_n(l)* Σ_ll'(k) ψ_n(l')]. Missing entries
!> in the rank-local column block are zero. Batched matrix products retain
!> dense kernels without allocating a pair-by-band-by-band tensor.
!> \param bs_env ...
!> \param sigma_pairs ...
!> \param first_pair ...
!> \param ispin ...
!> \param sigma_band ...
!> \param sum_env ...
!> \param interpolation ...
!> \param band_orbitals ...
! **************************************************************************************************
   SUBROUTINE self_energy_pairs_to_Bloch(bs_env, sigma_pairs, first_pair, ispin, sigma_band, &
                                         sum_env, interpolation, band_orbitals)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: sigma_pairs(:, :, :)
      INTEGER, INTENT(IN)                                :: first_pair, ispin
      REAL(KIND=dp), INTENT(OUT)                         :: sigma_band(:, :)
      TYPE(mp_para_env_type), POINTER                    :: sum_env
      TYPE(projection_interpolation_type), INTENT(IN)    :: interpolation
      COMPLEX(dp), ALLOCATABLE, INTENT(IN)               :: band_orbitals(:, :, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'self_energy_pairs_to_Bloch'

      COMPLEX(KIND=dp), ALLOCATABLE                      :: sigma_columns(:, :), sigma_local(:, :), &
                                                            work(:, :)
      INTEGER :: first_col, first_kp_eval, handle, i_band, i_batch, i_first, i_kp_eval, i_l, &
         i_local, i_lp, image_offset(126), last_col, last_kp_eval, last_pair, local_kp_eval, &
         n_batch, n_grid, n_local, phase_handle
      INTEGER, ALLOCATABLE                               :: pair_order(:)
      LOGICAL                                            :: valid_layout

      CALL timeset(routineN, handle)

      CPASSERT(ALLOCATED(band_orbitals))
      n_grid = SIZE(band_orbitals, 1)
      n_local = SIZE(sigma_pairs, 1)
      CPASSERT(SIZE(sigma_pairs, 2) == 1)
      CPASSERT(SIZE(sigma_pairs, 3) == bs_env%nkp_scf_desymm)
      CPASSERT(SIZE(band_orbitals, 2) == bs_env%n_ao)
      valid_layout = SIZE(band_orbitals, 3) >= bs_env%nkp_bs_and_DOS
      CPASSERT(valid_layout)
      CPASSERT(SIZE(band_orbitals, 4) == bs_env%n_spin)
      valid_layout = first_pair >= 1 .AND. first_pair + n_local - 1 <= n_grid*n_grid
      CPASSERT(valid_layout)
      CPASSERT(SIZE(sigma_band, 1) == bs_env%n_ao)
      CPASSERT(SIZE(sigma_band, 2) == bs_env%nkp_bs_and_DOS)
      CPASSERT(SIZE(interpolation%native_iq) == bs_env%nkp_bs_and_DOS)
      sigma_band = 0.0_dp
      IF (n_local > 0) THEN
         last_pair = first_pair + n_local - 1
         first_col = (first_pair - 1)/n_grid + 1
         last_col = (last_pair - 1)/n_grid + 1
         n_batch = MIN(16, bs_env%n_ao)
         CALL order_pair_images(bs_env, first_pair, n_local, pair_order, image_offset)
         !$OMP PARALLEL DEFAULT(NONE) &
         !$OMP SHARED(bs_env, sigma_pairs, first_pair, ispin, sigma_band, interpolation, band_orbitals, &
         !$OMP        n_grid, n_local, first_col, last_col, n_batch, pair_order, image_offset) &
         !$OMP PRIVATE(sigma_columns, sigma_local, work, i_kp_eval, phase_handle, first_kp_eval, last_kp_eval, local_kp_eval, &
         !$OMP         i_local, i_l, i_lp, i_first, i_band, i_batch)
         ALLOCATE (sigma_columns(n_grid, last_col - first_col + 1), &
                   sigma_local(n_local, kpoint_batch_size), work(n_grid, n_batch))
         !$OMP DO SCHEDULE(STATIC, 1)
         DO first_kp_eval = 1, bs_env%nkp_bs_and_DOS, kpoint_batch_size
            last_kp_eval = MIN(first_kp_eval + kpoint_batch_size - 1, bs_env%nkp_bs_and_DOS)
            CALL timeset(routineN//'_q_interpolation', phase_handle)
            CALL interpolate_band_self_energy(bs_env, sigma_pairs, first_kp_eval, last_kp_eval, &
                                              pair_order, image_offset, sigma_local, interpolation)
            CALL timestop(phase_handle)
            DO i_kp_eval = first_kp_eval, last_kp_eval
               local_kp_eval = i_kp_eval - first_kp_eval + 1
               sigma_columns = z_zero
               DO i_local = 1, n_local
                  i_l = MODULO(first_pair + i_local - 2, n_grid) + 1
                  i_lp = (first_pair + i_local - 2)/n_grid + 1
                  sigma_columns(i_l, i_lp - first_col + 1) = sigma_local(i_local, local_kp_eval)
               END DO
               DO i_first = 1, bs_env%n_ao, n_batch
                  CALL timeset(routineN//'_band_contraction', phase_handle)
                  CALL project_self_energy_columns(n_grid, last_col - first_col + 1, &
                                                   MIN(n_batch, bs_env%n_ao - i_first + 1), first_col, i_first, &
                                                   sigma_columns, band_orbitals(:, :, i_kp_eval, ispin), work)
                  DO i_band = i_first, MIN(i_first + n_batch - 1, bs_env%n_ao)
                     i_batch = i_band - i_first + 1
                     sigma_band(i_band, i_kp_eval) = &
                        REAL(SUM(CONJG(band_orbitals(:, i_band, i_kp_eval, ispin))* &
                                 work(:, i_batch)), dp)
                  END DO
                  CALL timestop(phase_handle)
               END DO
            END DO
         END DO
         !$OMP END DO
         DEALLOCATE (sigma_columns, sigma_local, work)
         !$OMP END PARALLEL
         DEALLOCATE (pair_order)
      END IF
      CALL timeset(routineN//'_final_reduction', phase_handle)
      CALL sum_env%sum(sigma_band)
      CALL timestop(phase_handle)

      CALL timestop(handle)

   END SUBROUTINE self_energy_pairs_to_Bloch

! **************************************************************************************************
!> \brief Evaluate the product cardinal kernel in Eq. (1), including Nyquist terms.
!> \param delta_k ...
!> \param n_grid ...
!> \return ...
! **************************************************************************************************
   FUNCTION cardinal_weight(delta_k, n_grid) RESULT(weight)
      REAL(KIND=dp), INTENT(IN)                          :: delta_k(3)
      INTEGER, INTENT(IN)                                :: n_grid(3)
      REAL(KIND=dp)                                      :: weight

      INTEGER                                            :: i_dim, i_frequency
      REAL(KIND=dp)                                      :: factor

      weight = 1.0_dp
      DO i_dim = 1, 3
         factor = 1.0_dp
         DO i_frequency = 1, (n_grid(i_dim) - 1)/2
            factor = factor + 2.0_dp*COS(twopi*delta_k(i_dim)*i_frequency)
         END DO
         IF (MODULO(n_grid(i_dim), 2) == 0) factor = factor + COS(0.5_dp*twopi*delta_k(i_dim)*n_grid(i_dim))
         weight = weight*factor/REAL(n_grid(i_dim), dp)
      END DO
   END FUNCTION cardinal_weight

! **************************************************************************************************
!> \brief Build Eq. (1)'s image phases, averaging adjacent images at half-cell boundaries.
!> Index 1+(2s_x+2)+5(2s_y+2)+25(2s_z+2) includes integer shifts and half-image averages.
!> \param bs_env ...
!> \param i_kp_eval ...
!> \param factors ...
!> \param fractional_grid ...
!> \param interpolation ...
! **************************************************************************************************
   SUBROUTINE interpolation_pair_factors(bs_env, i_kp_eval, factors, fractional_grid, interpolation)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: i_kp_eval
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: factors(:, :)
      REAL(KIND=dp), ALLOCATABLE, INTENT(OUT)            :: fractional_grid(:, :)
      TYPE(projection_interpolation_type), INTENT(IN), &
         OPTIONAL                                        :: interpolation

      INTEGER                                            :: idir, iq
      LOGICAL                                            :: valid_layout
      REAL(KIND=dp)                                      :: delta_k(3), weight

      CPASSERT(ALLOCATED(bs_env%ri_rs%grid_points))
      CPASSERT(ASSOCIATED(bs_env%ri_rs%cell))
      ALLOCATE (fractional_grid(3, SIZE(bs_env%ri_rs%grid_points, 2)))
      fractional_grid(:, :) = MATMUL(bs_env%ri_rs%cell%h_inv, bs_env%ri_rs%grid_points)
      DO idir = 1, 3
         IF (bs_env%periodic(idir) == 0) fractional_grid(idir, :) = 0.0_dp
         valid_layout = MAXVAL(ABS(fractional_grid(idir, :))) <= 0.5_dp + 1.0E-10_dp
         CPASSERT(valid_layout)
      END DO
      ALLOCATE (factors(125, bs_env%nkp_scf_desymm))
      DO iq = 1, bs_env%nkp_scf_desymm
         delta_k = bs_env%kpoints_DOS%xkp(:, i_kp_eval) - bs_env%kpoints_scf_desymm%xkp(:, iq)
         IF (PRESENT(interpolation)) THEN
            IF (ALLOCATED(interpolation%weights)) THEN
               weight = interpolation%weights(iq, interpolation%offmesh_index(i_kp_eval))
            ELSE
               weight = cardinal_weight(delta_k, bs_env%kpoints_scf_desymm%nkp_grid)
            END IF
         ELSE
            weight = cardinal_weight(delta_k, bs_env%kpoints_scf_desymm%nkp_grid)
         END IF
         CALL interpolation_image_factors(delta_k, weight, factors(:, iq))
      END DO
   END SUBROUTINE interpolation_pair_factors

! **************************************************************************************************
!> \brief Form Eq. (1)'s separable image phases and symmetric half-image averages.
!> For θ=2πδk, the half-image factors are (1+exp(±iθ))/2=exp(±iθ/2)cos(θ/2).
!> Three Cartesian phase pairs suffice for all 125 displacement classes; no spatial cutoff is added.
!> \param delta_k ...
!> \param weight ...
!> \param factors ...
! **************************************************************************************************
   PURE SUBROUTINE interpolation_image_factors(delta_k, weight, factors)
      REAL(KIND=dp), INTENT(IN)                          :: delta_k(3), weight
      COMPLEX(KIND=dp), INTENT(OUT)                      :: factors(125)

      COMPLEX(KIND=dp)                                   :: phase(-2:2, 3), phase_yz
      INTEGER                                            :: i_shift, idir, ix, iy, iz
      REAL(KIND=dp)                                      :: angle

      DO idir = 1, 3
         angle = twopi*delta_k(idir)
         phase(2, idir) = CMPLX(COS(angle), SIN(angle), KIND=dp)
         phase(-2, idir) = CONJG(phase(2, idir))
         phase(0, idir) = CMPLX(1.0_dp, 0.0_dp, KIND=dp)
         phase(1, idir) = CMPLX(0.5_dp, 0.0_dp, KIND=dp)*(CMPLX(1.0_dp, 0.0_dp, KIND=dp) + phase(2, idir))
         phase(-1, idir) = CONJG(phase(1, idir))
      END DO
      i_shift = 0
      DO iz = -2, 2
         DO iy = -2, 2
            phase_yz = CMPLX(weight, 0.0_dp, KIND=dp)*phase(iy, 2)*phase(iz, 3)
            DO ix = -2, 2
               i_shift = i_shift + 1
               factors(i_shift) = phase(ix, 1)*phase_yz
            END DO
         END DO
      END DO
   END SUBROUTINE interpolation_image_factors

! **************************************************************************************************
!> \brief Map Eq. (1)'s pair displacement to an image or a symmetric half-image average.
!> \param delta ...
!> \return ...
! **************************************************************************************************
   PURE INTEGER FUNCTION interpolation_pair_image(delta) RESULT(i_shift)
      REAL(KIND=dp), INTENT(IN)                          :: delta(3)

      INTEGER                                            :: idir, shift(3)
      REAL(KIND=dp)                                      :: half_boundary, tolerance

      shift = 2*NINT(delta)
      DO idir = 1, 3
         half_boundary = ANINT(delta(idir) - 0.5_dp) + 0.5_dp
         tolerance = 64.0_dp*EPSILON(1.0_dp)*MAX(1.0_dp, ABS(delta(idir)))
         IF (ABS(delta(idir) - half_boundary) <= tolerance) shift(idir) = NINT(2.0_dp*half_boundary)
      END DO
      i_shift = 1 + (shift(1) + 2) + 5*(shift(2) + 2) + 25*(shift(3) + 2)
   END FUNCTION interpolation_pair_image

! **************************************************************************************************
!> \brief Apply Eq. (1) to distributed rows, reusing the image labels of one grid column.
!> \param bs_env ...
!> \param sigma_grid ...
!> \param first_row ...
!> \param i_kp_eval ...
!> \param sigma_k ...
!> \param interpolation ...
! **************************************************************************************************
   SUBROUTINE interpolate_self_energy_rows(bs_env, sigma_grid, first_row, i_kp_eval, sigma_k, interpolation)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: sigma_grid(:, :, :)
      INTEGER, INTENT(IN)                                :: first_row, i_kp_eval
      COMPLEX(KIND=dp), INTENT(OUT)                      :: sigma_k(:, :)
      TYPE(projection_interpolation_type), INTENT(IN), &
         OPTIONAL                                        :: interpolation

      COMPLEX(KIND=dp), ALLOCATABLE                      :: factors(:, :)
      INTEGER                                            :: icol, iq, irow
      INTEGER, ALLOCATABLE                               :: pair_image(:)
      REAL(KIND=dp), ALLOCATABLE                         :: fractional_grid(:, :)

      CALL interpolation_pair_factors(bs_env, i_kp_eval, factors, fractional_grid, interpolation)
      ALLOCATE (pair_image(SIZE(sigma_grid, 1)))
      sigma_k = z_zero
      DO icol = 1, SIZE(sigma_grid, 2)
         DO irow = 1, SIZE(sigma_grid, 1)
            pair_image(irow) = interpolation_pair_image( &
                               fractional_grid(:, first_row + irow - 1) - fractional_grid(:, icol))
         END DO
         DO iq = 1, SIZE(sigma_grid, 3)
            sigma_k(:, icol) = sigma_k(:, icol) + factors(pair_image, iq)*sigma_grid(:, icol, iq)
         END DO
      END DO
   END SUBROUTINE interpolate_self_energy_rows

! **************************************************************************************************
!> \brief Bound simultaneous band interpolation and band-projection storage in bytes.
!> Shared image labels and thread-private packed Σ, factors and output blocks are included.
!> \param n_grid ...
!> \param n_pairs ...
!> \param n_k ...
!> \param n_ao ...
!> \param n_threads ...
!> \return ...
! **************************************************************************************************
   FUNCTION projection_workspace_bytes(n_grid, n_pairs, n_k, n_ao, n_threads) RESULT(bytes)
      INTEGER, INTENT(IN)                                :: n_grid
      REAL(dp), INTENT(IN)                               :: n_pairs
      INTEGER, INTENT(IN)                                :: n_k, n_ao, n_threads
      REAL(dp)                                           :: bytes

      bytes = 4.0_dp*n_pairs + 24.0_dp*REAL(n_grid, dp) + 4096.0_dp + &
              REAL(n_threads, dp)*(16.0_dp*(REAL(kpoint_batch_size + 1, dp)*n_pairs + &
                                            REAL(n_grid, dp)*REAL(MIN(16, n_ao) + 2, dp) + &
                                       REAL(n_k, dp)*REAL(125*kpoint_batch_size + pair_batch_size + kpoint_batch_size + 125, dp) + &
                                            REAL(pair_batch_size*kpoint_batch_size, dp)) + 24.0_dp*REAL(n_grid, dp) + 4096.0_dp)
   END FUNCTION projection_workspace_bytes

! **************************************************************************************************
!> \brief Partition owned pairs by Eq. (1)'s 125 image classes once for all band k-points.
!> \param bs_env ...
!> \param first_pair ...
!> \param n_local ...
!> \param pair_order ...
!> \param image_offset ...
! **************************************************************************************************
   SUBROUTINE order_pair_images(bs_env, first_pair, n_local, pair_order, image_offset)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: first_pair, n_local
      INTEGER, ALLOCATABLE, INTENT(OUT)                  :: pair_order(:)
      INTEGER, INTENT(OUT)                               :: image_offset(126)

      INTEGER                                            :: counts(125), cursor(125), i_class, i_l, &
                                                            i_local, i_lp, idir, n_grid
      LOGICAL                                            :: valid_layout
      REAL(dp), ALLOCATABLE                              :: fractional_grid(:, :)

      CPASSERT(ALLOCATED(bs_env%ri_rs%grid_points))
      CPASSERT(ASSOCIATED(bs_env%ri_rs%cell))
      n_grid = SIZE(bs_env%ri_rs%grid_points, 2)
      ALLOCATE (fractional_grid(3, n_grid), pair_order(n_local))
      fractional_grid(:, :) = MATMUL(bs_env%ri_rs%cell%h_inv, bs_env%ri_rs%grid_points)
      DO idir = 1, 3
         IF (bs_env%periodic(idir) == 0) fractional_grid(idir, :) = 0.0_dp
         valid_layout = MAXVAL(ABS(fractional_grid(idir, :))) <= 0.5_dp + 1.0E-10_dp
         CPASSERT(valid_layout)
      END DO
      counts = 0
      DO i_local = 1, n_local
         i_l = MODULO(first_pair + i_local - 2, n_grid) + 1
         i_lp = (first_pair + i_local - 2)/n_grid + 1
         i_class = interpolation_pair_image(fractional_grid(:, i_l) - fractional_grid(:, i_lp))
         counts(i_class) = counts(i_class) + 1
      END DO
      image_offset(1) = 1
      DO i_class = 1, 125
         image_offset(i_class + 1) = image_offset(i_class) + counts(i_class)
      END DO
      cursor = image_offset(:125)
      DO i_local = 1, n_local
         i_l = MODULO(first_pair + i_local - 2, n_grid) + 1
         i_lp = (first_pair + i_local - 2)/n_grid + 1
         i_class = interpolation_pair_image(fractional_grid(:, i_l) - fractional_grid(:, i_lp))
         pair_order(cursor(i_class)) = i_local
         cursor(i_class) = cursor(i_class) + 1
      END DO
      DEALLOCATE (fractional_grid)
   END SUBROUTINE order_pair_images

! **************************************************************************************************
!> \brief Evaluate Eq. (1) for bounded pair and band batches with the same image factors.
!> Native bands are copied unchanged; each image class uses Σ_(pair,q) times factors_(q,k).
!> \param bs_env ...
!> \param sigma_pairs ...
!> \param first_kp_eval ...
!> \param last_kp_eval ...
!> \param pair_order ...
!> \param image_offset ...
!> \param sigma_k ...
!> \param interpolation ...
! **************************************************************************************************
   SUBROUTINE interpolate_band_self_energy(bs_env, sigma_pairs, first_kp_eval, last_kp_eval, &
                                           pair_order, image_offset, sigma_k, interpolation)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(dp), INTENT(IN)                            :: sigma_pairs(:, :, :)
      INTEGER, INTENT(IN)                                :: first_kp_eval, last_kp_eval, &
                                                            pair_order(:), image_offset(126)
      COMPLEX(dp), INTENT(OUT)                           :: sigma_k(:, :)
      TYPE(projection_interpolation_type), INTENT(IN)    :: interpolation

      COMPLEX(dp), ALLOCATABLE                           :: band_factors(:, :), block(:, :), &
                                                            factors(:, :, :), PRODUCT(:, :), &
                                                            weights(:, :)
      INTEGER                                            :: i_class, i_first, i_kp_eval, i_off, &
                                                            i_pair, iq, n_off, n_pairs, n_q, &
                                                            native_iq, &
                                                            offmesh_kp(kpoint_batch_size)
      REAL(dp), ALLOCATABLE                              :: fractional_grid(:, :)

      n_off = 0
      DO i_kp_eval = first_kp_eval, last_kp_eval
         native_iq = interpolation%native_iq(i_kp_eval)
         IF (native_iq > 0) THEN
            sigma_k(:, i_kp_eval - first_kp_eval + 1) = sigma_pairs(:, 1, native_iq)
         ELSE
            n_off = n_off + 1
            offmesh_kp(n_off) = i_kp_eval
         END IF
      END DO
      IF (n_off == 0) RETURN
      n_q = SIZE(sigma_pairs, 3)
      ALLOCATE (factors(125, n_q, n_off), block(pair_batch_size, n_q), &
                weights(n_q, n_off), PRODUCT(pair_batch_size, n_off))
      DO i_off = 1, n_off
         CALL interpolation_pair_factors(bs_env, offmesh_kp(i_off), band_factors, fractional_grid, interpolation)
         factors(:, :, i_off) = band_factors
         DEALLOCATE (band_factors, fractional_grid)
      END DO
      DO i_class = 1, 125
         IF (image_offset(i_class) == image_offset(i_class + 1)) CYCLE
         DO i_off = 1, n_off
            weights(:, i_off) = factors(i_class, :, i_off)
         END DO
         DO i_first = image_offset(i_class), image_offset(i_class + 1) - 1, pair_batch_size
            n_pairs = MIN(pair_batch_size, image_offset(i_class + 1) - i_first)
            DO iq = 1, n_q
               DO i_pair = 1, n_pairs
                  block(i_pair, iq) = sigma_pairs(pair_order(i_first + i_pair - 1), 1, iq)
               END DO
            END DO
            CALL zgemm('N', 'N', n_pairs, n_off, n_q, z_one, block, pair_batch_size, &
                       weights, n_q, z_zero, product, pair_batch_size)
            DO i_off = 1, n_off
               DO i_pair = 1, n_pairs
                  sigma_k(pair_order(i_first + i_pair - 1), offmesh_kp(i_off) - first_kp_eval + 1) = &
                     PRODUCT(i_pair, i_off)
               END DO
            END DO
         END DO
      END DO
      DEALLOCATE (factors, block, weights, product)
   END SUBROUTINE interpolate_band_self_energy

! **************************************************************************************************
!> \brief Transform a periodic lattice product in a cell centred on each physical grid pair.
!> Products of two half-shifted lattice transforms are periodic. For s nearest to
!> H^{-1}(r_l-r_l'), F_ll'(k)=exp(2πiks) Σ_R exp(2πikR) F_ll'(R+s).
!> The even-mesh Nyquist endpoints and half-cell choices of s are averaged symmetrically.
!> On integer/half-shifted band meshes these averages are real separable endpoint weights,
!> allowing one FFT and one cell-vector scratch. values_R is consumed in place.
!> first_pair>0 denotes flattened column-major pairs; otherwise first_row denotes owned rows.
!> \param bs_env ...
!> \param cells ...
!> \param xkp ...
!> \param values_R ...
!> \param values_k ...
!> \param first_row ...
!> \param first_pair ...
! **************************************************************************************************
   SUBROUTINE product_to_k_pair_centered(bs_env, cells, xkp, values_R, values_k, first_row, first_pair)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: cells(:, :)
      REAL(KIND=dp), INTENT(IN)                          :: xkp(:, :)
      REAL(KIND=dp), INTENT(INOUT)                       :: values_R(:, :, :)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: values_k(:, :, :)
      INTEGER, INTENT(IN)                                :: first_row, first_pair

      CHARACTER(LEN=*), PARAMETER :: routineN = 'product_to_k_pair_centered'

      INTEGER                                            :: handle, icell, idir, ik, ilocal, &
                                                            isource, jlocal, lower(3), ngrid, &
                                                            nmesh(3), phase_handle, shift(3), &
                                                            wrapped(3)
      LOGICAL                                            :: tie(3), valid_layout
      REAL(KIND=dp)                                      :: delta(3), endpoint(3), weight
      REAL(KIND=dp), ALLOCATABLE                         :: fractional_grid(:, :), scratch(:)

      CALL timeset(routineN, handle)

      CALL timeset("product_pair_centered_prepare", phase_handle)
      CPASSERT(ALLOCATED(bs_env%ri_rs%grid_points))
      CPASSERT(ASSOCIATED(bs_env%ri_rs%cell))
      CPASSERT(SIZE(bs_env%ri_rs%grid_points, 1) == 3)
      CPASSERT(SIZE(bs_env%ri_rs%grid_points, 2) > 0)
      CPASSERT(SIZE(xkp, 1) >= 3 .AND. SIZE(xkp, 2) > 0)
      CPASSERT(SIZE(cells, 1) == 3)
      CPASSERT(SIZE(values_k, 1) == SIZE(values_R, 1))
      CPASSERT(SIZE(values_k, 2) == SIZE(values_R, 2))
      CPASSERT(SIZE(values_k, 3) == SIZE(xkp, 2))
      ngrid = SIZE(bs_env%ri_rs%grid_points, 2)
      nmesh = bs_env%kpoints_scf_desymm%nkp_grid
      CPASSERT(ALL(nmesh > 0))
      IF (first_pair > 0) THEN
         CPASSERT(ngrid <= HUGE(1)/ngrid)
         CPASSERT(SIZE(values_R, 2) == 1)
         CPASSERT(first_pair + SIZE(values_R, 1) - 1 <= ngrid*ngrid)
      ELSE
         valid_layout = first_row >= 1 .AND. first_row + SIZE(values_R, 1) - 1 <= ngrid
         CPASSERT(valid_layout)
         CPASSERT(SIZE(values_R, 2) == ngrid)
      END IF
      lower = -nmesh/2
      CPASSERT(SIZE(cells, 2) == PRODUCT(nmesh))
      CPASSERT(SIZE(values_R, 3) == SIZE(cells, 2))
      endpoint = COS(twopi*xkp(1:3, 1)*REAL(nmesh, dp))
      CPASSERT(ALL(ABS(ABS(endpoint) - 1.0_dp) < 1.0E-10_dp))
      DO ik = 1, SIZE(xkp, 2)
         delta = COS(twopi*xkp(1:3, ik)*REAL(nmesh, dp)) - endpoint
         CPASSERT(MAXVAL(ABS(delta)) < 1.0E-10_dp)
      END DO
      ALLOCATE (fractional_grid(3, ngrid))
      fractional_grid(:, :) = MATMUL(bs_env%ri_rs%cell%h_inv, bs_env%ri_rs%grid_points)
      DO idir = 1, 3
         IF (bs_env%periodic(idir) == 0) fractional_grid(idir, :) = 0.0_dp
      END DO
      !$OMP PARALLEL DEFAULT(NONE) &
      !$OMP SHARED(values_R, cells, nmesh, lower, endpoint, first_pair, first_row, ngrid, fractional_grid) &
      !$OMP PRIVATE(jlocal, ilocal, shift, tie, scratch, icell, wrapped, isource, weight, idir)
      ALLOCATE (scratch(SIZE(cells, 2)))
      !$OMP DO COLLAPSE(2) SCHEDULE(STATIC)
      DO jlocal = 1, SIZE(values_R, 2)
         DO ilocal = 1, SIZE(values_R, 1)
            CALL product_pair_shift(ilocal, jlocal, first_pair, first_row, ngrid, fractional_grid, shift, tie)
            scratch(:) = values_R(ilocal, jlocal, :)
            DO icell = 1, SIZE(cells, 2)
               wrapped = MODULO(cells(:, icell) + shift - lower, nmesh) + lower
               isource = 1 + wrapped(3) - lower(3) + &
                         nmesh(3)*(wrapped(2) - lower(2) + nmesh(2)*(wrapped(1) - lower(1)))
               weight = 1.0_dp
               DO idir = 1, 3
                  IF (MODULO(nmesh(idir), 2) == 0) THEN
                     IF (cells(idir, icell) == lower(idir)) THEN
                        IF (tie(idir)) THEN
                           weight = weight*(1.0_dp + 3.0_dp*endpoint(idir))/4.0_dp
                        ELSE
                           weight = weight*(1.0_dp + endpoint(idir))/2.0_dp
                        END IF
                     ELSE IF (tie(idir) .AND. cells(idir, icell) == lower(idir) + 1) THEN
                        weight = weight*(3.0_dp + endpoint(idir))/4.0_dp
                     END IF
                  ELSE IF (tie(idir) .AND. cells(idir, icell) == lower(idir)) THEN
                     weight = weight*(1.0_dp + endpoint(idir))/2.0_dp
                  END IF
               END DO
               values_R(ilocal, jlocal, icell) = weight*scratch(isource)
            END DO
         END DO
      END DO
      !$OMP END DO
      DEALLOCATE (scratch)
      !$OMP END PARALLEL
      CALL timestop(phase_handle)
      CALL cell_to_k_grid_fft(values_R, cells, xkp, nmesh, values_k)
      CALL timeset("product_pair_centered_phase", phase_handle)
      !$OMP PARALLEL DO COLLAPSE(2) DEFAULT(NONE) SCHEDULE(STATIC) &
      !$OMP SHARED(values_k, xkp, first_pair, first_row, ngrid, fractional_grid) &
      !$OMP PRIVATE(jlocal, ilocal, shift, tie, ik)
      DO jlocal = 1, SIZE(values_k, 2)
         DO ilocal = 1, SIZE(values_k, 1)
            CALL product_pair_shift(ilocal, jlocal, first_pair, first_row, ngrid, fractional_grid, shift, tie)
            DO ik = 1, SIZE(xkp, 2)
               values_k(ilocal, jlocal, ik) = values_k(ilocal, jlocal, ik)* &
                                              EXP(CMPLX(0.0_dp, twopi*DOT_PRODUCT(xkp(1:3, ik), REAL(shift, dp)), dp))
            END DO
         END DO
      END DO
      !$OMP END PARALLEL DO
      CALL timestop(phase_handle)

      CALL timestop(handle)

   END SUBROUTINE product_to_k_pair_centered

! **************************************************************************************************
!> \brief Select the lattice image of a grid-point pair and identify half-cell ties.
!> \param i_local ...
!> \param j_local ...
!> \param first_pair ...
!> \param first_row ...
!> \param ngrid ...
!> \param fractional_grid ...
!> \param pair_shift ...
!> \param pair_tie ...
! **************************************************************************************************
   SUBROUTINE product_pair_shift(i_local, j_local, first_pair, first_row, ngrid, fractional_grid, &
                                 pair_shift, pair_tie)
      INTEGER, INTENT(IN)                                :: i_local, j_local, first_pair, first_row, &
                                                            ngrid
      REAL(KIND=dp), INTENT(IN)                          :: fractional_grid(:, :)
      INTEGER, INTENT(OUT)                               :: pair_shift(3)
      LOGICAL, INTENT(OUT)                               :: pair_tie(3)

      INTEGER                                            :: idir, il, ilp, ipair
      REAL(KIND=dp)                                      :: delta(3), half_boundary, tolerance

      IF (first_pair > 0) THEN
         ipair = first_pair + i_local - 1
         il = MODULO(ipair - 1, ngrid) + 1
         ilp = (ipair - 1)/ngrid + 1
      ELSE
         il = first_row + i_local - 1
         ilp = j_local
      END IF
      delta = fractional_grid(:, il) - fractional_grid(:, ilp)
      pair_shift = NINT(delta)
      pair_tie = .FALSE.
      DO idir = 1, 3
         half_boundary = ANINT(delta(idir) - 0.5_dp) + 0.5_dp
         tolerance = 64.0_dp*EPSILON(1.0_dp)*MAX(1.0_dp, ABS(delta(idir)))
         IF (ABS(delta(idir) - half_boundary) <= tolerance) THEN
            pair_shift(idir) = NINT(half_boundary - 0.5_dp)
            pair_tie(idir) = .TRUE.
         END IF
      END DO
   END SUBROUTINE product_pair_shift

! **************************************************************************************************
!> \brief Contract a contiguous column block of Σ with strided band Bloch orbitals.
!> \param n_grid ...
!> \param n_columns ...
!> \param n_bands ...
!> \param first_col ...
!> \param i_first ...
!> \param sigma_columns ...
!> \param orbitals ...
!> \param work ...
! **************************************************************************************************
   SUBROUTINE project_self_energy_columns(n_grid, n_columns, n_bands, first_col, i_first, sigma_columns, orbitals, work)
      INTEGER, INTENT(IN)                                :: n_grid, n_columns, n_bands, first_col, &
                                                            i_first
      COMPLEX(dp), INTENT(IN) :: sigma_columns(n_grid, n_columns), orbitals(n_grid, *)
      COMPLEX(dp), INTENT(OUT)                           :: work(n_grid, n_bands)

      CALL zgemm('N', 'N', n_grid, n_bands, n_columns, z_one, sigma_columns, n_grid, &
                 orbitals(first_col, i_first), n_grid, z_zero, work, n_grid)
   END SUBROUTINE project_self_energy_columns

! **************************************************************************************************
!> \brief Accumulate one term of X_PQ(q,j)=sum_i T_ji Y_PQ(q,i).
!> T is the existing real minimax cosine transform, in either direction.
!> The caller supplies one already projected and symmetrized Y(q,i); only
!> its time weights are applied here, without repeating spatial projection.
!> \param block ...
!> \param matrix ...
!> \param weights ...
! **************************************************************************************************
   SUBROUTINE accumulate_ri_q_cosine(block, matrix, weights)
      TYPE(ri_q_matrix_block_type), INTENT(INOUT)        :: block
      COMPLEX(KIND=dp), INTENT(IN)                       :: matrix(:, :)
      REAL(KIND=dp), INTENT(IN)                          :: weights(:)

      INTEGER                                            :: icol, irow, itime, n

      CPASSERT(ALLOCATED(block%values))
      n = SIZE(matrix, 1)
      CPASSERT(SIZE(matrix, 2) == n)
      CPASSERT(SIZE(block%values, 1) == n)
      CPASSERT(SIZE(block%values, 2) == n)
      CPASSERT(SIZE(block%values, 3) == SIZE(weights))
      !$OMP PARALLEL DO DEFAULT(NONE) COLLAPSE(2) SCHEDULE(STATIC) &
      !$OMP SHARED(block, matrix, weights, n) PRIVATE(itime, icol, irow)
      DO itime = 1, SIZE(weights)
         DO icol = 1, n
            !$OMP SIMD
            DO irow = 1, n
               block%values(irow, icol, itime) = block%values(irow, icol, itime) + &
                                                 weights(itime)*matrix(irow, icol)
            END DO
         END DO
      END DO
      !$OMP END PARALLEL DO
   END SUBROUTINE accumulate_ri_q_cosine

! **************************************************************************************************
!> \brief Count live complex elements across independently allocated q blocks.
!> \param blocks ...
!> \return ...
! **************************************************************************************************
   FUNCTION ri_q_storage_elements(blocks) RESULT(nelements)
      TYPE(ri_q_matrix_block_type), INTENT(IN)           :: blocks(:)
      INTEGER(KIND=int_8)                                :: nelements

      INTEGER                                            :: iq

      nelements = 0_int_8
      DO iq = 1, SIZE(blocks)
         IF (ALLOCATED(blocks(iq)%values)) THEN
            nelements = nelements + SIZE(blocks(iq)%values, KIND=int_8)
         END IF
      END DO
   END FUNCTION ri_q_storage_elements

! **************************************************************************************************
!> \brief Take the real part of the existing inverse lattice FFT with one row of workspace.
!> \param values_k Reciprocal-space matrices.
!> \param xkp Fractional k points.
!> \param nkp_grid Mesh dimensions.
!> \param cells Lattice vectors.
!> \param values_R Real lattice matrices.
! **************************************************************************************************
   SUBROUTINE gw_k_grid_to_real_cells(values_k, xkp, nkp_grid, cells, values_R)
      COMPLEX(KIND=dp), INTENT(IN)                       :: values_k(:, :, :)
      REAL(KIND=dp), INTENT(IN)                          :: xkp(:, :)
      INTEGER, INTENT(IN)                                :: nkp_grid(3), cells(:, :)
      REAL(KIND=dp), INTENT(OUT)                         :: values_R(:, :, :)

      COMPLEX(KIND=dp), ALLOCATABLE                      :: row_R(:, :, :)
      INTEGER                                            :: irow
      TYPE(k_grid_to_cell_work_type)                     :: work

      ALLOCATE (row_R(1, SIZE(values_R, 2), SIZE(values_R, 3)))
      CALL k_grid_to_cell_prepare(work, xkp, nkp_grid, cells)
      DO irow = 1, SIZE(values_R, 1)
         CALL k_grid_to_cell_execute(work, values_k(irow:irow, :, :), row_R)
         values_R(irow, :, :) = REAL(row_R(1, :, :), dp)
      END DO
      CALL k_grid_to_cell_release(work)
      DEALLOCATE (row_R)

   END SUBROUTINE gw_k_grid_to_real_cells

! **************************************************************************************************
!> \brief Integrate each auxiliary function for the Gamma charge-mode projection.
!> For a primitive x^(2m) exp(-α x²), I_0=sqrt(π/α), I_2m=(2m-1)I_(2m-2)/(2α).
!> Odd Cartesian moments vanish; the existing spherical contraction supplies normalization.
!> \param qs_env Atomic auxiliary bases.
!> \param charge Integrals of the reference auxiliary functions.
! **************************************************************************************************
   SUBROUTINE gamma_auxiliary_charges(qs_env, charge)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      COMPLEX(KIND=dp), INTENT(OUT)                      :: charge(:)

      INTEGER                                            :: first, iatom, ico, il, ipgf, iset, isgf, &
                                                            lmax, ncart, offset
      REAL(KIND=dp)                                      :: alpha
      REAL(KIND=dp), ALLOCATABLE                         :: moments(:), primitive(:)
      TYPE(gto_basis_set_type), POINTER                  :: basis
      TYPE(particle_type), POINTER                       :: particles(:)
      TYPE(qs_kind_type), POINTER                        :: qs_kinds(:)

      CALL get_qs_env(qs_env, particle_set=particles, qs_kind_set=qs_kinds)
      charge = z_zero
      first = 0
      DO iatom = 1, SIZE(particles)
         CALL get_qs_kind(qs_kinds(particles(iatom)%atomic_kind%kind_number), &
                          basis_set=basis, basis_type='RI_AUX')
         DO iset = 1, basis%nset
            lmax = basis%lmax(iset)
            ncart = ncoset(lmax)
            ALLOCATE (moments(0:lmax), primitive(ncart*basis%npgf(iset)))
            primitive = 0.0_dp
            DO ipgf = 1, basis%npgf(iset)
               alpha = basis%zet(ipgf, iset)
               moments = 0.0_dp
               moments(0) = SQRT(pi/alpha)
               DO il = 2, lmax, 2
                  moments(il) = REAL(il - 1, dp)*moments(il - 2)/(2.0_dp*alpha)
               END DO
               DO ico = ncoset(basis%lmin(iset) - 1) + 1, ncart
                  primitive((ipgf - 1)*ncart + ico) = PRODUCT(moments(indco(:, ico)))
               END DO
            END DO
            offset = basis%first_sgf(1, iset)
            DO isgf = offset, offset + basis%nsgf_set(iset) - 1
               charge(first + isgf) = CMPLX(SUM(basis%sphi(1:SIZE(primitive), isgf)*primitive), 0.0_dp, dp)
            END DO
            DEALLOCATE (moments, primitive)
         END DO
         first = first + basis%nsgf
      END DO
      CPASSERT(first == SIZE(charge))

   END SUBROUTINE gamma_auxiliary_charges

! **************************************************************************************************
!> \brief Expand saved sparse Gamma-correction AO images using their prepared atom-pair traversal.
!> This preserves the block and mirror conventions of replicate_rs_matrices after QS release.
!> \param matrices Distributed sparse AO image matrices.
!> \param plan Prepared source image and atom-pair traversal.
!> \param values Dense AO images used by the existing Gamma-correction contraction.
!> \param cell_to_index Common Gamma-correction image map.
! **************************************************************************************************
   SUBROUTINE gamma_corr_ao_images(matrices, plan, values, cell_to_index)
      TYPE(dbcsr_p_type), POINTER                        :: matrices(:, :)
      TYPE(kp_transform_plan_type), INTENT(IN)           :: plan
      REAL(dp), INTENT(INOUT)                            :: values(:, :, :, :)
      INTEGER, POINTER                                   :: cell_to_index(:, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'gamma_corr_ao_images'

      CHARACTER                                          :: symmetry
      INTEGER                                            :: cell(3), handle, i_col, i_entry, &
                                                            i_image, i_mirror, i_row, i_source, &
                                                            i_spin, i_swap, n_col, n_row
      INTEGER, POINTER                                   :: col_offsets(:), row_offsets(:)
      LOGICAL                                            :: found
      REAL(dp)                                           :: mirror_sign
      REAL(dp), POINTER                                  :: block(:, :)
      TYPE(mp_comm_type)                                 :: comm

      CALL timeset(routineN, handle)

      CALL dbcsr_get_info(matrices(1, 1)%matrix, group=comm, matrix_type=symmetry, &
                          row_blk_offset=row_offsets, col_blk_offset=col_offsets)
      mirror_sign = MERGE(-1.0_dp, 1.0_dp, symmetry == dbcsr_type_antisymmetric)
      DO i_spin = 1, SIZE(matrices, 1)
         DO i_entry = 1, plan%nentry
            i_source = plan%image(i_entry)
            i_row = plan%row(i_entry)
            i_col = plan%col(i_entry)
            cell = plan%cell(:, i_entry)
            IF (plan%symmetry_sign(i_entry) < 0.0_dp) THEN
               i_swap = i_row
               i_row = i_col
               i_col = i_swap
            END IF
            IF (symmetry /= dbcsr_type_no_symmetry .AND. i_row > i_col) THEN
               i_swap = i_row
               i_row = i_col
               i_col = i_swap
               cell = -cell
            END IF
            CALL dbcsr_get_readonly_block_p(matrices(i_spin, i_source)%matrix, row=i_row, col=i_col, &
                                            block=block, found=found, row_size=n_row, col_size=n_col)
            IF (.NOT. found) CYCLE
            i_image = cell_to_index(cell(1), cell(2), cell(3))
            CPASSERT(i_image > 0)
            values(i_spin, row_offsets(i_row):row_offsets(i_row) + n_row - 1, &
                   col_offsets(i_col):col_offsets(i_col) + n_col - 1, i_image) = block
            IF (symmetry /= dbcsr_type_no_symmetry .AND. i_row /= i_col) THEN
               i_mirror = cell_to_index(-cell(1), -cell(2), -cell(3))
               CPASSERT(i_mirror > 0)
               values(i_spin, col_offsets(i_col):col_offsets(i_col) + n_col - 1, &
                      row_offsets(i_row):row_offsets(i_row) + n_row - 1, i_mirror) = mirror_sign*TRANSPOSE(block)
            END IF
         END DO
      END DO
      CALL comm%sum(values)

      CALL timestop(handle)

   END SUBROUTINE gamma_corr_ao_images

! **************************************************************************************************
!> \brief Return the contiguous grid-row range owned inside an MPI group.
!> \param n_grid ...
!> \param para_env ...
!> \param first_row ...
!> \param last_row ...
!> \param nrows_local ...
! **************************************************************************************************
   SUBROUTINE local_grid_row_range(n_grid, para_env, first_row, last_row, nrows_local)
      INTEGER, INTENT(IN)                                :: n_grid
      TYPE(mp_para_env_type), POINTER                    :: para_env
      INTEGER, INTENT(OUT)                               :: first_row, last_row, nrows_local

      first_row = n_grid*para_env%mepos/para_env%num_pe + 1
      last_row = n_grid*(para_env%mepos + 1)/para_env%num_pe
      nrows_local = MAX(0, last_row - first_row + 1)

   END SUBROUTINE local_grid_row_range

! **************************************************************************************************
!> \brief Assign contiguous ordered (l,l') grid pairs and prepare q-to-pair transfer counts.
!> \param plan ...
!> \param n_grid ...
!> \param n_q ...
!> \param owner_offset ...
!> \param para_env ...
! **************************************************************************************************
   SUBROUTINE initialize_pair_transfer(plan, n_grid, n_q, owner_offset, para_env)
      TYPE(pair_transfer_plan_type), INTENT(OUT)         :: plan
      INTEGER, INTENT(IN)                                :: n_grid, n_q, owner_offset
      TYPE(mp_para_env_type), POINTER                    :: para_env

      INTEGER                                            :: i_rank, n_rank, n_total_pairs

      CPASSERT(n_grid > 0 .AND. n_q > 0)
      CPASSERT(n_grid <= HUGE(1)/n_grid)
      n_total_pairs = n_grid*n_grid
      n_rank = para_env%num_pe
      plan%n_grid = n_grid
      plan%n_q = n_q
      plan%owner_offset = owner_offset
      ALLOCATE (plan%first_pair_rank(0:n_rank - 1), plan%n_pair_rank(0:n_rank - 1))
      DO i_rank = 0, n_rank - 1
         plan%first_pair_rank(i_rank) = &
            INT(INT(n_total_pairs, int_8)*INT(i_rank, int_8)/INT(n_rank, int_8)) + 1
         plan%n_pair_rank(i_rank) = &
            INT(INT(n_total_pairs, int_8)*INT(i_rank + 1, int_8)/INT(n_rank, int_8)) - &
            plan%first_pair_rank(i_rank) + 1
      END DO
      plan%first_pair = plan%first_pair_rank(para_env%mepos)
      plan%n_local = plan%n_pair_rank(para_env%mepos)
   END SUBROUTINE initialize_pair_transfer

! **************************************************************************************************
!> \brief Transfer q-local full matrices to owners of ordered grid pairs.
!> \param q_matrices ...
!> \param para_env ...
!> \param plan ...
!> \param fft_pairs ...
! **************************************************************************************************
   SUBROUTINE q_matrices_to_fft_pairs(q_matrices, para_env, plan, fft_pairs)
      COMPLEX(KIND=dp), INTENT(IN)                       :: q_matrices(:, :, :)
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(pair_transfer_plan_type), INTENT(IN)          :: plan
      COMPLEX(KIND=dp), INTENT(OUT)                      :: fft_pairs(:, :, :)

      INTEGER                                            :: handle, i_dest, i_q, i_q_local, &
                                                            i_source, n_rank
      INTEGER(KIND=int_8)                                :: i_stream
      INTEGER(KIND=int_8), ALLOCATABLE                   :: send_total(:)
      TYPE(bounded_transfer_type)                        :: transfer

      n_rank = para_env%num_pe
      CPASSERT(SIZE(q_matrices, 1) == plan%n_grid)
      CPASSERT(SIZE(q_matrices, 2) == plan%n_grid)
      CPASSERT(SIZE(fft_pairs, 1) == plan%n_local)
      CPASSERT(SIZE(fft_pairs, 2) == 1 .AND. SIZE(fft_pairs, 3) == plan%n_q)
      i_q_local = 0
      DO i_q = 1, plan%n_q
         IF (MODULO(i_q + plan%owner_offset, n_rank) == para_env%mepos) i_q_local = i_q_local + 1
      END DO
      CPASSERT(SIZE(q_matrices, 3) == i_q_local)
      ALLOCATE (send_total(0:n_rank - 1))
      send_total(:) = INT(plan%n_pair_rank, int_8)*INT(SIZE(q_matrices, 3), int_8)
      CALL initialize_bounded_transfer(transfer, para_env, send_total)
      DO WHILE (next_transfer_block(transfer))
         CALL timeset('gw_pair_transfer_pack', handle)
         DO i_dest = 0, n_rank - 1
            i_stream = 0_int_8
            DO i_q_local = 1, SIZE(q_matrices, 3)
               CALL pack_transfer_matrix(transfer, i_dest, q_matrices(:, :, i_q_local), i_stream, &
                                         first_element=INT(plan%first_pair_rank(i_dest), int_8), &
                                         n_elements=INT(plan%n_pair_rank(i_dest), int_8))
               i_stream = i_stream + INT(plan%n_pair_rank(i_dest), int_8)
            END DO
         END DO
         CALL timestop(handle)
         CALL timeset('gw_pair_transfer_alltoall', handle)
         CALL exchange_transfer_block(transfer, para_env)
         CALL timestop(handle)
         CALL timeset('gw_pair_transfer_unpack', handle)
         DO i_source = 0, n_rank - 1
            i_stream = 0_int_8
            DO i_q = 1, plan%n_q
               IF (MODULO(i_q + plan%owner_offset, n_rank) /= i_source) CYCLE
               CALL unpack_transfer_matrix(transfer, i_source, fft_pairs(:, :, i_q), i_stream)
               i_stream = i_stream + INT(plan%n_local, int_8)
            END DO
         END DO
         CALL timestop(handle)
      END DO
   END SUBROUTINE q_matrices_to_fft_pairs

! **************************************************************************************************
!> \brief Redistribute complete χ grid matrices from FFT rows to cyclic q owners.
!> \param fft_rows ...
!> \param n_grid ...
!> \param para_env ...
!> \param q_matrices ...
! **************************************************************************************************
   SUBROUTINE fft_rows_to_q_matrices(fft_rows, n_grid, para_env, q_matrices)
      COMPLEX(KIND=dp), INTENT(IN)                       :: fft_rows(:, :, :)
      INTEGER, INTENT(IN)                                :: n_grid
      TYPE(mp_para_env_type), POINTER                    :: para_env
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: q_matrices(:, :, :)

      INTEGER                                            :: i_dest, i_first, i_last, i_q, i_q_local, &
                                                            i_source, n_rank, n_rows
      INTEGER(KIND=int_8)                                :: i_stream, n_matrix
      INTEGER(KIND=int_8), ALLOCATABLE                   :: positions(:), send_total(:)
      TYPE(bounded_transfer_type)                        :: transfer

      n_rank = para_env%num_pe
      CALL local_grid_row_range(n_grid, para_env, i_first, i_last, n_rows)
      CPASSERT(SIZE(fft_rows, 1) == n_rows .AND. SIZE(fft_rows, 2) == n_grid)
      ALLOCATE (send_total(0:n_rank - 1), positions(0:n_rank - 1))
      send_total = 0_int_8
      n_matrix = INT(n_rows, int_8)*INT(n_grid, int_8)
      i_q_local = 0
      DO i_q = 1, SIZE(fft_rows, 3)
         i_dest = MODULO(i_q, n_rank)
         send_total(i_dest) = send_total(i_dest) + n_matrix
         IF (i_dest == para_env%mepos) i_q_local = i_q_local + 1
      END DO
      ALLOCATE (q_matrices(n_grid, n_grid, i_q_local))
      CALL initialize_bounded_transfer(transfer, para_env, send_total)
      DO WHILE (next_transfer_block(transfer))
         positions = 0_int_8
         DO i_q = 1, SIZE(fft_rows, 3)
            i_dest = MODULO(i_q, n_rank)
            CALL pack_transfer_matrix(transfer, i_dest, fft_rows(:, :, i_q), positions(i_dest))
            positions(i_dest) = positions(i_dest) + n_matrix
         END DO
         CALL exchange_transfer_block(transfer, para_env)
         DO i_source = 0, n_rank - 1
            i_first = INT(INT(n_grid, int_8)*INT(i_source, int_8)/INT(n_rank, int_8)) + 1
            i_last = INT(INT(n_grid, int_8)*INT(i_source + 1, int_8)/INT(n_rank, int_8))
            i_stream = 0_int_8
            DO i_q_local = 1, SIZE(q_matrices, 3)
               CALL unpack_transfer_matrix(transfer, i_source, q_matrices(:, :, i_q_local), i_stream, &
                                           first_row=i_first, n_rows=i_last - i_first + 1)
               i_stream = i_stream + INT(i_last - i_first + 1, int_8)*INT(n_grid, int_8)
            END DO
         END DO
      END DO
   END SUBROUTINE fft_rows_to_q_matrices

! **************************************************************************************************
!> \brief Form V_ll'(q) = Σ_PQ Z_lP(q) V_PQ(q) Z_l'Q(q)* and distribute FFT rows.
!> \param Z_q ...
!> \param interaction_q ...
!> \param n_q ...
!> \param para_env ...
!> \param fft_rows ...
! **************************************************************************************************
   SUBROUTINE project_auxiliary_interaction_q_to_fft_rows(Z_q, interaction_q, n_q, para_env, fft_rows)
      COMPLEX(KIND=dp), INTENT(IN)                       :: Z_q(:, :, :), interaction_q(:, :, :)
      INTEGER, INTENT(IN)                                :: n_q
      TYPE(mp_para_env_type), POINTER                    :: para_env
      COMPLEX(KIND=dp), INTENT(OUT)                      :: fft_rows(:, :, :)

      COMPLEX(KIND=dp), ALLOCATABLE                      :: q_matrices(:, :, :)
      INTEGER                                            :: i_q_local, n_grid

      n_grid = SIZE(Z_q, 1)
      CPASSERT(SIZE(interaction_q, 1) == SIZE(Z_q, 2))
      CPASSERT(SIZE(interaction_q, 2) == SIZE(Z_q, 2))
      CPASSERT(SIZE(interaction_q, 3) == SIZE(Z_q, 3))
      ALLOCATE (q_matrices(n_grid, n_grid, SIZE(Z_q, 3)))
      !$OMP PARALLEL DO DEFAULT(NONE) SCHEDULE(STATIC) &
      !$OMP SHARED(Z_q, interaction_q, q_matrices) PRIVATE(i_q_local)
      DO i_q_local = 1, SIZE(Z_q, 3)
         CALL contract_auxiliary_interaction_q(Z_q(:, :, i_q_local), &
                                               interaction_q(:, :, i_q_local), &
                                               q_matrices(:, :, i_q_local))
      END DO
      !$OMP END PARALLEL DO
      CALL q_matrices_to_fft_rows(q_matrices, n_q, para_env, fft_rows)

   END SUBROUTINE project_auxiliary_interaction_q_to_fft_rows

! **************************************************************************************************
!> \brief Form V_ll'(q) = Σ_PQ Z_lP(q) V_PQ(q) Z_l'Q(q)* and transfer FFT pairs.
!> \param Z_q ...
!> \param interaction_q ...
!> \param itime ...
!> \param para_env ...
!> \param plan ...
!> \param fft_pairs ...
! **************************************************************************************************
   SUBROUTINE project_auxiliary_time_q_to_fft_pairs(Z_q, interaction_q, itime, para_env, &
                                                    plan, fft_pairs)
      COMPLEX(KIND=dp), INTENT(IN)                       :: Z_q(:, :, :)
      TYPE(ri_q_matrix_block_type), INTENT(IN)           :: interaction_q(:)
      INTEGER, INTENT(IN)                                :: itime
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(pair_transfer_plan_type), INTENT(IN)          :: plan
      COMPLEX(KIND=dp), INTENT(OUT)                      :: fft_pairs(:, :, :)

      COMPLEX(KIND=dp), ALLOCATABLE                      :: interaction_time(:, :, :), &
                                                            q_matrices(:, :, :)
      INTEGER                                            :: i_q_local, n_grid, n_q_local, n_RI

      n_grid = SIZE(Z_q, 1)
      n_RI = SIZE(Z_q, 2)
      n_q_local = SIZE(Z_q, 3)
      CPASSERT(plan%owner_offset == 0)
      CPASSERT(SIZE(interaction_q) == n_q_local)
      CALL expand_ri_q_time(interaction_q, itime, SIZE(fft_pairs, 3), n_RI, para_env, interaction_time)
      ALLOCATE (q_matrices(n_grid, n_grid, n_q_local))
      !$OMP PARALLEL DO DEFAULT(NONE) SCHEDULE(STATIC) &
      !$OMP SHARED(Z_q, interaction_time, q_matrices, n_q_local) PRIVATE(i_q_local)
      DO i_q_local = 1, n_q_local
         CALL contract_auxiliary_interaction_q(Z_q(:, :, i_q_local), &
                                               interaction_time(:, :, i_q_local), &
                                               q_matrices(:, :, i_q_local))
      END DO
      !$OMP END PARALLEL DO
      CALL q_matrices_to_fft_pairs(q_matrices, para_env, plan, fft_pairs)

   END SUBROUTINE project_auxiliary_time_q_to_fft_pairs

! **************************************************************************************************
!> \brief Form V_ll'(q) = Σ_PQ Z_lP(q) V_PQ(q) Z_l'Q(q)* for exchange or correlation.
!> The caller supplies bare V_PQ(q) or screened Wbar_PQ(q,iτ).
!> \param Z_q ...
!> \param V_PQ ...
!> \param V_grid_q ...
! **************************************************************************************************
   SUBROUTINE contract_auxiliary_interaction_q(Z_q, V_PQ, V_grid_q)
      COMPLEX(KIND=dp), INTENT(IN)                       :: Z_q(:, :), V_PQ(:, :)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: V_grid_q(:, :)

      COMPLEX(KIND=dp), ALLOCATABLE                      :: ZV(:, :)
      INTEGER                                            :: n_grid, n_RI

      n_grid = SIZE(Z_q, 1)
      n_RI = SIZE(Z_q, 2)
      CPASSERT(SIZE(V_PQ, 1) == n_RI .AND. SIZE(V_PQ, 2) == n_RI)
      CPASSERT(SIZE(V_grid_q, 1) == n_grid .AND. SIZE(V_grid_q, 2) == n_grid)
      ALLOCATE (ZV(n_grid, n_RI))
      CALL zgemm('N', 'N', n_grid, n_RI, n_RI, z_one, Z_q, n_grid, &
                 V_PQ, n_RI, z_zero, ZV, n_grid)
      CALL zgemm('N', 'C', n_grid, n_grid, n_RI, z_one, ZV, n_grid, &
                 Z_q, n_grid, z_zero, V_grid_q, n_grid)

   END SUBROUTINE contract_auxiliary_interaction_q

! **************************************************************************************************
!> \brief Plan bounded generation of q-owned products directly into FFT-row transfer buffers.
!> Each peer stream stores its grid rows fastest, then columns, then locally owned q-points.
!> The output retains every q-point; no Fourier transform is repeated for a transfer block.
!> \param transfer ...
!> \param para_env ...
!> \param n_grid ...
!> \param n_q ...
!> \param offset ...
!> \param n_inner ...
!> \param max_rows ...
! **************************************************************************************************
   SUBROUTINE initialize_product_row_transfer(transfer, para_env, n_grid, n_q, offset, n_inner, max_rows)
      TYPE(bounded_transfer_type), INTENT(OUT)           :: transfer
      TYPE(mp_para_env_type), POINTER                    :: para_env
      INTEGER, INTENT(IN)                                :: n_grid, n_q, offset, n_inner
      INTEGER, INTENT(OUT)                               :: max_rows

      INTEGER                                            :: i_dest, i_first, i_last, i_q, n_local
      INTEGER(KIND=int_8)                                :: row_bytes, workspace_bytes
      INTEGER(KIND=int_8), ALLOCATABLE                   :: send_total(:)
      REAL(KIND=dp)                                      :: available_GB

      n_local = 0
      DO i_q = 1, n_q
         IF (MODULO(i_q + offset, para_env%num_pe) == para_env%mepos) n_local = n_local + 1
      END DO
      ALLOCATE (send_total(0:para_env%num_pe - 1))
      DO i_dest = 0, para_env%num_pe - 1
         i_first = INT(INT(n_grid, int_8)*i_dest/para_env%num_pe) + 1
         i_last = INT(INT(n_grid, int_8)*(i_dest + 1)/para_env%num_pe)
         send_total(i_dest) = INT(i_last - i_first + 1, int_8)*n_grid*n_local
      END DO
      CALL initialize_bounded_transfer(transfer, para_env, send_total)
      CALL mp_mem_avail_per_rank_GB(para_env, available_GB)
      row_bytes = 16_int_8*(2_int_8*INT(n_grid, int_8) + INT(n_inner, int_8))
      workspace_bytes = INT(MIN(32.0_dp*1024.0_dp**2, MAX(0.0_dp, available_GB)*1.0E8_dp), int_8)
      workspace_bytes = workspace_bytes/MAX(1, omp_get_max_threads())
      max_rows = INT(MAX(1_int_8, MIN(INT(n_grid, int_8), workspace_bytes/MAX(1_int_8, row_bytes))))
   END SUBROUTINE initialize_product_row_transfer

! **************************************************************************************************
!> \brief Generate the active peer-stream fragment of L R† or L V R†.
!> Scratch consists of a bounded matrix block and, for V, its auxiliary-space intermediate.
!> No complete grid-pair matrix is allocated on the q owner before redistribution.
!> \param transfer ...
!> \param i_dest ...
!> \param left ...
!> \param right ...
!> \param n_grid ...
!> \param n_inner ...
!> \param i_q_local ...
!> \param max_rows ...
!> \param interaction ...
! **************************************************************************************************
   SUBROUTINE pack_product_row_block(transfer, i_dest, left, right, n_grid, n_inner, i_q_local, max_rows, interaction)
      TYPE(bounded_transfer_type), INTENT(INOUT)         :: transfer
      INTEGER, INTENT(IN)                                :: i_dest, n_grid, n_inner
      COMPLEX(KIND=dp), INTENT(IN)                       :: left(n_grid, n_inner), &
                                                            right(n_grid, n_inner)
      INTEGER, INTENT(IN)                                :: i_q_local, max_rows
      COMPLEX(KIND=dp), INTENT(IN), OPTIONAL             :: interaction(:, :)

      COMPLEX(KIND=dp), ALLOCATABLE                      :: column_block(:, :), work(:, :)
      INTEGER                                            :: i_buffer, i_col, i_first, i_last, i_row, &
                                                            n_col, n_rank, n_row, n_rows
      INTEGER(KIND=int_8)                                :: hi, i_element, lo, n_matrix, stream_start

      n_rank = SIZE(transfer%send_total)
      CPASSERT(SIZE(right, 1) == n_grid .AND. SIZE(right, 2) == n_inner)
      i_first = INT(INT(n_grid, int_8)*i_dest/n_rank) + 1
      i_last = INT(INT(n_grid, int_8)*(i_dest + 1)/n_rank)
      n_rows = i_last - i_first + 1
      n_matrix = INT(n_rows, int_8)*n_grid
      stream_start = INT(i_q_local - 1, int_8)*n_matrix
      lo = MAX(stream_start, transfer%offset)
      hi = MIN(stream_start + n_matrix, transfer%offset + INT(transfer%send_counts(i_dest), int_8))
      IF (hi <= lo) RETURN
      i_element = lo - stream_start
      i_buffer = transfer%send_displs(i_dest) + INT(lo - transfer%offset) + 1
      DO WHILE (i_element < hi - stream_start)
         i_row = INT(MODULO(i_element, INT(n_rows, int_8)))
         i_col = INT(i_element/INT(n_rows, int_8)) + 1
         n_row = INT(MIN(INT(MIN(max_rows, n_rows - i_row), int_8), hi - stream_start - i_element))
         n_col = 1
         IF (i_row == 0 .AND. n_row == n_rows) THEN
            n_col = INT(MIN(INT(n_grid - i_col + 1, int_8), (hi - stream_start - i_element)/n_rows))
         END IF
         ALLOCATE (column_block(n_row, n_col))
         IF (PRESENT(interaction)) THEN
            IF (.NOT. (SIZE(interaction, 1) == n_inner .AND. SIZE(interaction, 2) == n_inner)) THEN
               CPABORT("Internal GW array shape mismatch")
            END IF
            ALLOCATE (work(n_row, n_inner))
            CALL zgemm('N', 'N', n_row, n_inner, n_inner, z_one, left(i_first + i_row, 1), n_grid, &
                       interaction, n_inner, z_zero, work, n_row)
            CALL zgemm('N', 'C', n_row, n_col, n_inner, z_one, work, n_row, &
                       right(i_col, 1), n_grid, z_zero, column_block, n_row)
            DEALLOCATE (work)
         ELSE
            CALL zgemm('N', 'C', n_row, n_col, n_inner, z_one, left(i_first + i_row, 1), n_grid, &
                       right(i_col, 1), n_grid, z_zero, column_block, n_row)
         END IF
         transfer%send_buffer(i_buffer:i_buffer + n_row*n_col - 1) = RESHAPE(column_block, [n_row*n_col])
         DEALLOCATE (column_block)
         i_buffer = i_buffer + n_row*n_col
         i_element = i_element + INT(n_row, int_8)*n_col
      END DO
   END SUBROUTINE pack_product_row_block

! **************************************************************************************************
!> \brief Place the active generated fragments into their complete local Fourier axes.
!> \param transfer ...
!> \param para_env ...
!> \param n_q ...
!> \param offset ...
!> \param fft_rows ...
! **************************************************************************************************
   SUBROUTINE unpack_product_row_block(transfer, para_env, n_q, offset, fft_rows)
      TYPE(bounded_transfer_type), INTENT(IN)            :: transfer
      TYPE(mp_para_env_type), POINTER                    :: para_env
      INTEGER, INTENT(IN)                                :: n_q, offset
      COMPLEX(KIND=dp), INTENT(INOUT)                    :: fft_rows(:, :, :)

      INTEGER                                            :: i_q, i_source
      INTEGER(KIND=int_8)                                :: i_stream

      DO i_source = 0, para_env%num_pe - 1
         i_stream = 0_int_8
         DO i_q = 1, n_q
            IF (MODULO(i_q + offset, para_env%num_pe) /= i_source) CYCLE
            CALL unpack_transfer_matrix(transfer, i_source, fft_rows(:, :, i_q), i_stream)
            i_stream = i_stream + SIZE(fft_rows, 1, KIND=int_8)*SIZE(fft_rows, 2, KIND=int_8)
         END DO
      END DO
   END SUBROUTINE unpack_product_row_block

! **************************************************************************************************
!> \brief Redistribute cyclic q or k matrices to FFT row owners.
!> The transfer-q owner is iq mod P; the SCF-k owner is (ik-1) mod P.
!> \param q_matrices ...
!> \param n_q ...
!> \param para_env ...
!> \param fft_rows ...
!> \param owner_offset ...
! **************************************************************************************************
   SUBROUTINE q_matrices_to_fft_rows(q_matrices, n_q, para_env, fft_rows, owner_offset)
      COMPLEX(KIND=dp), INTENT(IN)                       :: q_matrices(:, :, :)
      INTEGER, INTENT(IN)                                :: n_q
      TYPE(mp_para_env_type), POINTER                    :: para_env
      COMPLEX(KIND=dp), INTENT(OUT)                      :: fft_rows(:, :, :)
      INTEGER, INTENT(IN), OPTIONAL                      :: owner_offset

      INTEGER                                            :: i_dest, i_first, i_last, i_q, i_q_local, &
                                                            i_source, n_grid, n_rank, offset
      INTEGER(KIND=int_8)                                :: i_stream, n_matrix
      INTEGER(KIND=int_8), ALLOCATABLE                   :: send_total(:)
      TYPE(bounded_transfer_type)                        :: transfer

      offset = 0
      IF (PRESENT(owner_offset)) offset = owner_offset
      n_grid = SIZE(q_matrices, 1)
      n_rank = para_env%num_pe
      CPASSERT(SIZE(q_matrices, 2) == n_grid)
      CPASSERT(SIZE(fft_rows, 2) == n_grid)
      CPASSERT(SIZE(fft_rows, 3) == n_q)
      CALL local_grid_row_range(n_grid, para_env, i_first, i_last, i_q_local)
      CPASSERT(SIZE(fft_rows, 1) == i_q_local)
      i_q_local = 0
      DO i_q = 1, n_q
         IF (MODULO(i_q + offset, n_rank) == para_env%mepos) i_q_local = i_q_local + 1
      END DO
      CPASSERT(SIZE(q_matrices, 3) == i_q_local)
      ALLOCATE (send_total(0:n_rank - 1))
      DO i_dest = 0, n_rank - 1
         i_first = INT(INT(n_grid, int_8)*INT(i_dest, int_8)/INT(n_rank, int_8)) + 1
         i_last = INT(INT(n_grid, int_8)*INT(i_dest + 1, int_8)/INT(n_rank, int_8))
         send_total(i_dest) = INT(i_last - i_first + 1, int_8)*INT(n_grid, int_8)*SIZE(q_matrices, 3, KIND=int_8)
      END DO
      CALL initialize_bounded_transfer(transfer, para_env, send_total)
      DO WHILE (next_transfer_block(transfer))
         DO i_dest = 0, n_rank - 1
            i_first = INT(INT(n_grid, int_8)*INT(i_dest, int_8)/INT(n_rank, int_8)) + 1
            i_last = INT(INT(n_grid, int_8)*INT(i_dest + 1, int_8)/INT(n_rank, int_8))
            n_matrix = INT(i_last - i_first + 1, int_8)*INT(n_grid, int_8)
            i_stream = 0_int_8
            DO i_q_local = 1, SIZE(q_matrices, 3)
               CALL pack_transfer_matrix(transfer, i_dest, q_matrices(:, :, i_q_local), i_stream, &
                                         first_row=i_first, n_rows=i_last - i_first + 1)
               i_stream = i_stream + n_matrix
            END DO
         END DO
         CALL exchange_transfer_block(transfer, para_env)
         DO i_source = 0, n_rank - 1
            i_stream = 0_int_8
            DO i_q = 1, n_q
               IF (MODULO(i_q + offset, n_rank) /= i_source) CYCLE
               CALL unpack_transfer_matrix(transfer, i_source, fft_rows(:, :, i_q), i_stream)
               i_stream = i_stream + SIZE(fft_rows, 1, KIND=int_8)*INT(n_grid, int_8)
            END DO
         END DO
      END DO
   END SUBROUTINE q_matrices_to_fft_rows

! **************************************************************************************************
!> \brief ...
!> \param blocks ...
!> \param itime ...
!> \param n_q ...
!> \param n_RI ...
!> \param para_env ...
!> \param matrices ...
!> Expand X(-q,tau)=X(q,tau)* for one time only, preserving compact persistent storage.
! **************************************************************************************************
   SUBROUTINE expand_ri_q_time(blocks, itime, n_q, n_RI, para_env, matrices)
      TYPE(ri_q_matrix_block_type), INTENT(IN)           :: blocks(:)
      INTEGER, INTENT(IN)                                :: itime, n_q, n_RI
      TYPE(mp_para_env_type), POINTER                    :: para_env
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: matrices(:, :, :)

      INTEGER                                            :: i_local, i_q
      INTEGER, ALLOCATABLE                               :: partner(:)

      ALLOCATE (matrices(n_RI, n_RI, SIZE(blocks)), SOURCE=z_zero)
      ALLOCATE (partner(n_q), SOURCE=0)
      i_local = 0
      DO i_q = 1, n_q
         IF (MODULO(i_q, para_env%num_pe) /= para_env%mepos) CYCLE
         i_local = i_local + 1
         CPASSERT(ALLOCATED(blocks(i_local)%values))
         matrices(:, :, i_local) = blocks(i_local)%values(:, :, itime)
      END DO
      CPASSERT(i_local == SIZE(blocks))
   END SUBROUTINE expand_ri_q_time

! **************************************************************************************************
!> \brief ...
!> \param distributed ...
!> \param owner_matrix ...
!> \param group_env ...
!> \param root ...
!> \param n ...
! **************************************************************************************************
   SUBROUTINE set_owner_matrix(distributed, owner_matrix, group_env, root, n)
      TYPE(cp_cfm_type), INTENT(IN)                      :: distributed
      COMPLEX(KIND=dp), INTENT(IN)                       :: owner_matrix(:, :)
      TYPE(mp_para_env_type), INTENT(INOUT)              :: group_env
      INTEGER, INTENT(IN)                                :: root, n

      COMPLEX(KIND=dp), ALLOCATABLE                      :: column_block(:, :)
      INTEGER                                            :: first, width

      IF (group_env%mepos == root) THEN
         CPASSERT(ALL(SHAPE(owner_matrix) == [n, n]))
      END IF
      ALLOCATE (column_block(n, column_block_cols))
      DO first = 1, n, column_block_cols
         width = MIN(column_block_cols, n - first + 1)
         IF (group_env%mepos == root) column_block(:, 1:width) = owner_matrix(:, first:first + width - 1)
         CALL group_env%bcast(column_block(:, 1:width), root)
         CALL cp_cfm_set_submatrix(distributed, column_block(:, 1:width), &
                                   start_row=1, start_col=first)
      END DO
      DEALLOCATE (column_block)
   END SUBROUTINE set_owner_matrix

! **************************************************************************************************
!> \brief ...
!> \param distributed ...
!> \param owner_matrix ...
!> \param group_env ...
!> \param root ...
!> \param n ...
! **************************************************************************************************
   SUBROUTINE get_owner_matrix(distributed, owner_matrix, group_env, root, n)
      TYPE(cp_cfm_type), INTENT(IN)                      :: distributed
      COMPLEX(KIND=dp), INTENT(OUT)                      :: owner_matrix(:, :)
      TYPE(mp_para_env_type), INTENT(INOUT)              :: group_env
      INTEGER, INTENT(IN)                                :: root, n

      COMPLEX(KIND=dp), ALLOCATABLE                      :: column_block(:, :)
      INTEGER                                            :: first, width

      IF (group_env%mepos == root) THEN
         CPASSERT(ALL(SHAPE(owner_matrix) == [n, n]))
      END IF
      ALLOCATE (column_block(n, column_block_cols))
      DO first = 1, n, column_block_cols
         width = MIN(column_block_cols, n - first + 1)
         CALL cp_cfm_get_submatrix(distributed, column_block(:, 1:width), &
                                   start_row=1, start_col=first)
         IF (group_env%mepos == root) owner_matrix(:, first:first + width - 1) = column_block(:, 1:width)
      END DO
      DEALLOCATE (column_block)
   END SUBROUTINE get_owner_matrix

! **************************************************************************************************
!> \brief Bound the combined complex send/receive storage by free memory on every rank.
!> Keep at most 10% of the available rank memory and never more than 128 MiB.
!> Totals and offsets are 64 bit; each MPI block has bounded default-integer counts.
!> \param transfer ...
!> \param para_env ...
!> \param send_total ...
! **************************************************************************************************
   SUBROUTINE initialize_bounded_transfer(transfer, para_env, send_total)
      TYPE(bounded_transfer_type), INTENT(OUT)           :: transfer
      TYPE(mp_para_env_type), POINTER                    :: para_env
      INTEGER(KIND=int_8), INTENT(IN)                    :: send_total(0:)

      INTEGER(KIND=int_8), PARAMETER :: default_cap = 128_int_8*1024_int_8**2

      INTEGER                                            :: i_rank, n_rank, n_recv, n_send
      INTEGER(KIND=int_8)                                :: budget, buffer_bytes, &
                                                            full_buffer_elements, largest
      REAL(KIND=dp)                                      :: available_GB

      n_rank = para_env%num_pe
      CPASSERT(SIZE(send_total) == n_rank .AND. ALL(send_total >= 0_int_8))
      ALLOCATE (transfer%send_total(0:n_rank - 1), transfer%recv_total(0:n_rank - 1))
      transfer%send_total(:) = send_total
      CALL para_env%alltoall(transfer%send_total, transfer%recv_total, 1)
      largest = MAX(MAXVAL(send_total), MAXVAL(transfer%recv_total))
      CALL para_env%max(largest)
      ALLOCATE (transfer%send_counts(0:n_rank - 1), transfer%recv_counts(0:n_rank - 1), &
                transfer%send_displs(0:n_rank - 1), transfer%recv_displs(0:n_rank - 1))
      IF (largest == 0_int_8) RETURN
      CALL mp_mem_avail_per_rank_GB(para_env, available_GB)
      budget = INT(MIN(REAL(default_cap, dp), MAX(0.0_dp, available_GB)*1.0E8_dp), int_8)
      CALL para_env%min(budget)
      IF (budget < 32_int_8*INT(n_rank, int_8)) THEN
         CPWARN('GW MPI memory estimate exhausted; continuing with small buffers')
         budget = MAX(1024_int_8**2, 32_int_8*INT(n_rank, int_8))
      END IF
      full_buffer_elements = MAX(1_int_8, SUM(send_total)) + MAX(1_int_8, SUM(transfer%recv_total))
      CALL para_env%max(full_buffer_elements)
      IF (full_buffer_elements <= budget/16_int_8) THEN
         transfer%peer_limit = INT(largest)
      ELSE
         transfer%peer_limit = INT(MIN(largest, budget/(32_int_8*INT(n_rank, int_8)), &
                                       INT(HUGE(1), int_8)/INT(n_rank, int_8)))
      END IF
      transfer%n_blocks = (largest - 1_int_8)/INT(transfer%peer_limit, int_8) + 1_int_8
      n_send = 0
      n_recv = 0
      DO i_rank = 0, n_rank - 1
         n_send = n_send + INT(MIN(send_total(i_rank), INT(transfer%peer_limit, int_8)))
         n_recv = n_recv + INT(MIN(transfer%recv_total(i_rank), INT(transfer%peer_limit, int_8)))
      END DO
      ALLOCATE (transfer%send_buffer(MAX(1, n_send)), transfer%recv_buffer(MAX(1, n_recv)))
      buffer_bytes = 16_int_8*(INT(MAX(1, n_send), int_8) + INT(MAX(1, n_recv), int_8))
      CPASSERT(buffer_bytes <= budget)
   END SUBROUTINE initialize_bounded_transfer

! **************************************************************************************************
!> \brief Select the same peer-stream interval on every rank; MPI counts remain bounded integers.
!> \param transfer ...
!> \return ...
! **************************************************************************************************
   LOGICAL FUNCTION next_transfer_block(transfer) RESULT(has_block)
      TYPE(bounded_transfer_type), INTENT(INOUT)         :: transfer

      INTEGER                                            :: i_rank, n_recv, n_send

      has_block = transfer%i_block < transfer%n_blocks
      IF (.NOT. has_block) RETURN
      transfer%offset = transfer%i_block*INT(transfer%peer_limit, int_8)
      transfer%i_block = transfer%i_block + 1_int_8
      n_send = 0
      n_recv = 0
      DO i_rank = 0, SIZE(transfer%send_total) - 1
         transfer%send_counts(i_rank) = INT(MAX(0_int_8, MIN(INT(transfer%peer_limit, int_8), &
                                                             transfer%send_total(i_rank) - transfer%offset)))
         transfer%recv_counts(i_rank) = INT(MAX(0_int_8, MIN(INT(transfer%peer_limit, int_8), &
                                                             transfer%recv_total(i_rank) - transfer%offset)))
         transfer%send_displs(i_rank) = n_send
         transfer%recv_displs(i_rank) = n_recv
         n_send = n_send + transfer%send_counts(i_rank)
         n_recv = n_recv + transfer%recv_counts(i_rank)
      END DO
   END FUNCTION next_transfer_block

! **************************************************************************************************
!> \brief Exchange the current bounded block through CP2K all-to-all communication.
!> \param transfer ...
!> \param para_env ...
! **************************************************************************************************
   SUBROUTINE exchange_transfer_block(transfer, para_env)
      TYPE(bounded_transfer_type), INTENT(INOUT)         :: transfer
      TYPE(mp_para_env_type), POINTER                    :: para_env

      CALL para_env%alltoall(transfer%send_buffer, transfer%send_counts, &
                             transfer%send_displs, transfer%recv_buffer, transfer%recv_counts, &
                             transfer%recv_displs)
   END SUBROUTINE exchange_transfer_block

! **************************************************************************************************
!> \brief Pack the intersection of a column-major matrix section with one peer's active stream block.
!> A row section contains n_rows*SIZE(matrix,2) elements, row index varying fastest.
!> first_element and n_elements optionally select consecutive ordered pairs within that section.
!> \param transfer ...
!> \param i_peer ...
!> \param matrix ...
!> \param stream_start ...
!> \param first_row ...
!> \param n_rows ...
!> \param first_element ...
!> \param n_elements ...
! **************************************************************************************************
   SUBROUTINE pack_transfer_matrix(transfer, i_peer, matrix, stream_start, first_row, n_rows, &
                                   first_element, n_elements)
      TYPE(bounded_transfer_type), INTENT(INOUT)         :: transfer
      INTEGER, INTENT(IN)                                :: i_peer
      COMPLEX(KIND=dp), INTENT(IN)                       :: matrix(:, :)
      INTEGER(KIND=int_8), INTENT(IN)                    :: stream_start
      INTEGER, INTENT(IN), OPTIONAL                      :: first_row, n_rows
      INTEGER(KIND=int_8), INTENT(IN), OPTIONAL          :: first_element, n_elements

      INTEGER                                            :: i_buffer, i_col, i_first, i_row, n_copy, &
                                                            nrow
      INTEGER(KIND=int_8)                                :: first, hi, i_element, i_end, lo, n

      i_first = 1
      nrow = SIZE(matrix, 1)
      IF (PRESENT(first_row)) i_first = first_row
      IF (PRESENT(n_rows)) nrow = n_rows
      first = 1_int_8
      n = INT(nrow, int_8)*SIZE(matrix, 2, KIND=int_8)
      IF (PRESENT(first_element)) first = first_element
      IF (PRESENT(n_elements)) n = n_elements
      CPASSERT(nrow >= 0 .AND. i_first >= 1)
      CPASSERT(i_first + nrow - 1 <= SIZE(matrix, 1))
      CPASSERT(first >= 1_int_8 .AND. n >= 0_int_8)
      IF (.NOT. (first - 1_int_8 + n <= INT(nrow, int_8)*SIZE(matrix, 2, KIND=int_8))) THEN
         CPABORT('Transfer slice exceeds matrix extent')
      END IF
      lo = MAX(stream_start, transfer%offset)
      hi = MIN(stream_start + n, transfer%offset + INT(transfer%send_counts(i_peer), int_8))
      IF (hi <= lo) RETURN
      i_buffer = transfer%send_displs(i_peer) + INT(lo - transfer%offset) + 1
      i_element = first - 1_int_8 + lo - stream_start
      i_end = i_element + hi - lo
      DO WHILE (i_element < i_end)
         i_row = INT(MODULO(i_element, INT(nrow, int_8)))
         i_col = INT(i_element/INT(nrow, int_8)) + 1
         n_copy = INT(MIN(i_end - i_element, INT(nrow - i_row, int_8)))
         transfer%send_buffer(i_buffer:i_buffer + n_copy - 1) = &
            matrix(i_first + i_row:i_first + i_row + n_copy - 1, i_col)
         i_buffer = i_buffer + n_copy
         i_element = i_element + INT(n_copy, int_8)
      END DO
   END SUBROUTINE pack_transfer_matrix

! **************************************************************************************************
!> \brief Unpack the same matrix-section stream; optionally apply conjugation or q-pair averaging.
!> Each destination element is written once, with no full reshape or temporary matrix.
!> \param transfer ...
!> \param i_peer ...
!> \param matrix ...
!> \param stream_start ...
!> \param first_row ...
!> \param n_rows ...
!> \param first_element ...
!> \param n_elements ...
!> \param conjugate ...
!> \param average ...
! **************************************************************************************************
   SUBROUTINE unpack_transfer_matrix(transfer, i_peer, matrix, stream_start, first_row, n_rows, &
                                     first_element, n_elements, conjugate, average)
      TYPE(bounded_transfer_type), INTENT(IN)            :: transfer
      INTEGER, INTENT(IN)                                :: i_peer
      COMPLEX(KIND=dp), INTENT(INOUT)                    :: matrix(:, :)
      INTEGER(KIND=int_8), INTENT(IN)                    :: stream_start
      INTEGER, INTENT(IN), OPTIONAL                      :: first_row, n_rows
      INTEGER(KIND=int_8), INTENT(IN), OPTIONAL          :: first_element, n_elements
      LOGICAL, INTENT(IN), OPTIONAL                      :: conjugate, average

      INTEGER                                            :: i_buffer, i_col, i_first, i_row, n_copy, &
                                                            nrow
      INTEGER(KIND=int_8)                                :: first, hi, i_element, i_end, lo, n
      LOGICAL                                            :: take_average, take_conjugate

      take_conjugate = .FALSE.
      take_average = .FALSE.
      IF (PRESENT(conjugate)) take_conjugate = conjugate
      IF (PRESENT(average)) take_average = average
      i_first = 1
      nrow = SIZE(matrix, 1)
      IF (PRESENT(first_row)) i_first = first_row
      IF (PRESENT(n_rows)) nrow = n_rows
      first = 1_int_8
      n = INT(nrow, int_8)*SIZE(matrix, 2, KIND=int_8)
      IF (PRESENT(first_element)) first = first_element
      IF (PRESENT(n_elements)) n = n_elements
      CPASSERT(nrow >= 0 .AND. i_first >= 1)
      CPASSERT(i_first + nrow - 1 <= SIZE(matrix, 1))
      CPASSERT(first >= 1_int_8 .AND. n >= 0_int_8)
      IF (.NOT. (first - 1_int_8 + n <= INT(nrow, int_8)*SIZE(matrix, 2, KIND=int_8))) THEN
         CPABORT('Transfer slice exceeds matrix extent')
      END IF
      lo = MAX(stream_start, transfer%offset)
      hi = MIN(stream_start + n, transfer%offset + INT(transfer%recv_counts(i_peer), int_8))
      IF (hi <= lo) RETURN
      i_buffer = transfer%recv_displs(i_peer) + INT(lo - transfer%offset) + 1
      i_element = first - 1_int_8 + lo - stream_start
      i_end = i_element + hi - lo
      DO WHILE (i_element < i_end)
         i_row = INT(MODULO(i_element, INT(nrow, int_8)))
         i_col = INT(i_element/INT(nrow, int_8)) + 1
         n_copy = INT(MIN(i_end - i_element, INT(nrow - i_row, int_8)))
         IF (take_average) THEN
            CPASSERT(take_conjugate)
            matrix(i_first + i_row:i_first + i_row + n_copy - 1, i_col) = &
               CMPLX(0.5_dp, 0.0_dp, KIND=dp)*(matrix(i_first + i_row:i_first + i_row + n_copy - 1, i_col) + &
                                               CONJG(transfer%recv_buffer(i_buffer:i_buffer + n_copy - 1)))
         ELSE IF (take_conjugate) THEN
            matrix(i_first + i_row:i_first + i_row + n_copy - 1, i_col) = &
               CONJG(transfer%recv_buffer(i_buffer:i_buffer + n_copy - 1))
         ELSE
            matrix(i_first + i_row:i_first + i_row + n_copy - 1, i_col) = &
               transfer%recv_buffer(i_buffer:i_buffer + n_copy - 1)
         END IF
         i_buffer = i_buffer + n_copy
         i_element = i_element + INT(n_copy, int_8)
      END DO
   END SUBROUTINE unpack_transfer_matrix

! **************************************************************************************************
!> \brief Keep balanced q groups contiguous in physical-node rank order.
!> Every MPI rank participates. Group sizes differ by at most one; at most
!> n_node-1 groups cross node boundaries, independent of the launch rank order.
!> \param para_env ...
!> \param n_qgroups ...
!> \param my_qgroup ...
!> \param q_roots ...
! **************************************************************************************************
   SUBROUTINE grouped_w_partition(para_env, n_qgroups, my_qgroup, q_roots)
      TYPE(mp_para_env_type), POINTER                    :: para_env
      INTEGER, INTENT(IN)                                :: n_qgroups
      INTEGER, INTENT(OUT)                               :: my_qgroup
      INTEGER, ALLOCATABLE, INTENT(OUT)                  :: q_roots(:)

      INTEGER                                            :: node_first_rank
      INTEGER, ALLOCATABLE                               :: node_ids(:), rank_groups(:)
      TYPE(mp_comm_type)                                 :: node_comm

      CALL node_comm%from_split_type(para_env, mp_comm_split_type_shared, key=para_env%mepos)
      node_first_rank = para_env%mepos
      CALL node_comm%min(node_first_rank)
      CALL node_comm%free()
      ALLOCATE (node_ids(para_env%num_pe), SOURCE=0)
      node_ids(para_env%mepos + 1) = node_first_rank
      CALL para_env%sum(node_ids)
      ALLOCATE (rank_groups(para_env%num_pe), q_roots(n_qgroups))
      CALL partition_w_ranks(node_ids, rank_groups, q_roots)
      my_qgroup = rank_groups(para_env%mepos + 1)
   END SUBROUTINE grouped_w_partition

! **************************************************************************************************
!> \brief Assign contiguous balanced intervals of node-ordered ranks to q groups.
!> The root is the smallest world rank in each group, matching communicator split.
!> \param node_ids ...
!> \param rank_groups ...
!> \param q_roots ...
! **************************************************************************************************
   SUBROUTINE partition_w_ranks(node_ids, rank_groups, q_roots)
      INTEGER, INTENT(IN)                                :: node_ids(:)
      INTEGER, INTENT(OUT)                               :: rank_groups(:), q_roots(:)

      INTEGER                                            :: igroup, inode, iposition, irank, &
                                                            n_groups, n_rank

      n_rank = SIZE(node_ids)
      n_groups = SIZE(q_roots)
      CPASSERT(n_rank > 0 .AND. n_groups > 0 .AND. n_groups <= n_rank)
      CPASSERT(SIZE(rank_groups) == n_rank)
      CPASSERT(ALL(node_ids >= 0) .AND. ALL(node_ids < n_rank))
      rank_groups = -1
      q_roots = n_rank
      iposition = 0
      DO inode = 0, n_rank - 1
         DO irank = 0, n_rank - 1
            IF (node_ids(irank + 1) /= inode) CYCLE
            igroup = INT(INT(iposition, int_8)*n_groups/n_rank)
            rank_groups(irank + 1) = igroup
            q_roots(igroup + 1) = MIN(q_roots(igroup + 1), irank)
            iposition = iposition + 1
         END DO
      END DO
      CPASSERT(iposition == n_rank .AND. ALL(rank_groups >= 0))
      CPASSERT(ALL(q_roots < n_rank))
   END SUBROUTINE partition_w_ranks

! **************************************************************************************************
!> \brief ...
!> \param matrices ...
!> \param blacs ...
!> \param matrix_struct ...
!> \param group_env ...
!> \param n ...
!> Construct block-cyclic matrices on the MPI ranks assigned to a q group.
! **************************************************************************************************
   SUBROUTINE grouped_w_create(matrices, blacs, matrix_struct, group_env, n)
      TYPE(cp_cfm_type), INTENT(INOUT)                   :: matrices(6)
      TYPE(cp_blacs_env_type), POINTER                   :: blacs
      TYPE(cp_fm_struct_type), POINTER                   :: matrix_struct
      TYPE(mp_para_env_type), INTENT(INOUT), TARGET      :: group_env
      INTEGER, INTENT(IN)                                :: n

      CPASSERT(n > 0)
      CALL cp_blacs_env_create(blacs, group_env)
      CALL cp_fm_struct_create(matrix_struct, context=blacs, &
                               nrow_global=n, ncol_global=n)
      CALL cp_cfm_create(matrices(w_chi), matrix_struct)
      CALL cp_cfm_create(matrices(w_metric), matrix_struct)
      CALL cp_cfm_create(matrices(w_coulomb), matrix_struct)
      CALL cp_cfm_create(matrices(w_tmp), matrix_struct)
      CALL cp_cfm_create(matrices(w_dielectric), matrix_struct)
      CALL cp_cfm_create(matrices(w_output), matrix_struct)
   END SUBROUTINE grouped_w_create

! **************************************************************************************************
!> \brief ...
!> \param world ...
!> \param n ...
!> \param source_rank ...
!> \param destination_rank ...
!> \param source ...
!> \param destination ...
!> Transfer a matrix between world owners without a full-size communication buffer.
! **************************************************************************************************
   SUBROUTINE grouped_w_copy_matrix(world, n, source_rank, destination_rank, source, destination)
      TYPE(mp_para_env_type), INTENT(INOUT)              :: world
      INTEGER, INTENT(IN)                                :: n, source_rank, destination_rank
      COMPLEX(KIND=dp), INTENT(IN)                       :: source(:, :)
      COMPLEX(KIND=dp), INTENT(INOUT)                    :: destination(:, :)

      INTEGER                                            :: first, last, sender, tag

      IF (source_rank == destination_rank) THEN
         IF (world%mepos == source_rank) destination = source
         RETURN
      END IF
      DO first = 1, n, column_block_cols
         last = MIN(first + column_block_cols - 1, n)
         IF (world%mepos == source_rank) THEN
            CALL world%send(source(:, first:last), destination_rank, 0)
         ELSE IF (world%mepos == destination_rank) THEN
            sender = source_rank
            tag = 0
            CALL world%recv(destination(:, first:last), sender, tag)
         END IF
      END DO
   END SUBROUTINE grouped_w_copy_matrix

! **************************************************************************************************
!> \brief ...
!> \param matrices ...
!> \param group_env ...
!> \param n ...
!> \param which ...
!> \param matrix ...
!> Install one frequency-independent factor, reusing the root's input buffer.
! **************************************************************************************************
   SUBROUTINE grouped_w_set_factor(matrices, group_env, n, which, matrix)
      TYPE(cp_cfm_type), INTENT(INOUT)                   :: matrices(6)
      TYPE(mp_para_env_type), INTENT(INOUT)              :: group_env
      INTEGER, INTENT(IN)                                :: n, which
      COMPLEX(KIND=dp), INTENT(IN)                       :: matrix(:, :)

      SELECT CASE (which)
      CASE (1)
         CALL set_owner_matrix(matrices(w_metric), matrix, group_env, 0, n)
      CASE (2)
         CALL set_owner_matrix(matrices(w_coulomb), matrix, group_env, 0, n)
      CASE DEFAULT
         CPABORT("Invalid W metric factor")
      END SELECT
   END SUBROUTINE grouped_w_set_factor

! **************************************************************************************************
!> \brief Evaluate screened W with a distributed positive-definite dielectric inverse.
!> For ε=I+BᴴχB after the unchanged positive χ filter, Cholesky gives ε^-1.
!> min_dielectric and max_dielectric bound the spectrum by 1 and 1+||BᴴχB||_F;
!> they are diagnostics only and do not alter the interaction.
!> Reuse K=M^-1 V^1/2 to form W=K(ε^-1-I)Kᴴ. The optional norm diagnostic
!> is evaluated only when requested; its output is zero otherwise.
!> \param matrices ...
!> \param group_env ...
!> \param n ...
!> \param root ...
!> \param chi_owner ...
!> \param cutoff ...
!> \param output_owner ...
!> \param metric_chi_norm ...
!> \param min_dielectric ...
!> \param max_dielectric ...
!> \param screened_norm ...
!> \param compute_screened_norm ...
! **************************************************************************************************
   SUBROUTINE grouped_w_solve(matrices, group_env, n, root, chi_owner, cutoff, output_owner, &
                              metric_chi_norm, min_dielectric, max_dielectric, screened_norm, compute_screened_norm)
      TYPE(cp_cfm_type), INTENT(INOUT)                   :: matrices(6)
      TYPE(mp_para_env_type), INTENT(INOUT)              :: group_env
      INTEGER, INTENT(IN)                                :: n, root
      COMPLEX(KIND=dp), INTENT(IN)                       :: chi_owner(:, :)
      REAL(KIND=dp), INTENT(IN)                          :: cutoff
      COMPLEX(KIND=dp), INTENT(OUT)                      :: output_owner(:, :)
      REAL(KIND=dp), INTENT(OUT)                         :: metric_chi_norm, min_dielectric, &
                                                            max_dielectric, screened_norm
      LOGICAL, INTENT(IN), OPTIONAL                      :: compute_screened_norm

      INTEGER                                            :: info
      LOGICAL                                            :: need_screened_norm

      CPASSERT(root >= 0 .AND. root < group_env%num_pe)
      CALL set_owner_matrix(matrices(w_chi), chi_owner, group_env, root, n)

      ! Match the dense reference: symmetrize before each Hermitian eigensolve.
      CALL cp_cfm_transpose(matrices(w_chi), 'C', matrices(w_output))
      CALL cp_cfm_scale_and_add(CMPLX(0.5_dp, 0.0_dp, KIND=dp), matrices(w_chi), &
                                CMPLX(0.5_dp, 0.0_dp, KIND=dp), matrices(w_output))
      CALL cp_cfm_power(matrices(w_chi), exponent=1.0_dp, threshold=cutoff)
      CALL parallel_gemm('C', 'N', n, n, n, z_one, matrices(w_metric), &
                         matrices(w_chi), z_zero, matrices(w_tmp))
      CALL parallel_gemm('N', 'N', n, n, n, z_one, matrices(w_tmp), &
                         matrices(w_metric), z_zero, matrices(w_dielectric))
      metric_chi_norm = cp_cfm_norm(matrices(w_dielectric), 'F')
      CALL cp_cfm_add_on_diag(matrices(w_dielectric), z_one)
      CALL cp_cfm_transpose(matrices(w_dielectric), 'C', matrices(w_output))
      CALL cp_cfm_scale_and_add(CMPLX(0.5_dp, 0.0_dp, KIND=dp), matrices(w_dielectric), &
                                CMPLX(0.5_dp, 0.0_dp, KIND=dp), matrices(w_output))
      ! χ is positive semidefinite after filtering: ε=I+BᴴχB has λ_min>=1.
      ! The Frobenius norm gives λ_max<=1+||BᴴχB||_F; these are bounds, not eigenvalues.
      min_dielectric = 1.0_dp
      max_dielectric = 1.0_dp + metric_chi_norm
      CALL cp_cfm_cholesky_decompose(matrices(w_dielectric), info_out=info)
      IF (info /= 0) THEN
         CPABORT("Screened dielectric Cholesky factorization failed")
      END IF
      CALL cp_cfm_cholesky_invert(matrices(w_dielectric), info_out=info)
      IF (info /= 0) CPABORT("Screened dielectric Cholesky inversion failed")
      CALL cp_cfm_uplo_to_full(matrices(w_dielectric), matrices(w_output), uplo='U')
      CALL cp_cfm_add_on_diag(matrices(w_dielectric), -z_one)

      need_screened_norm = .TRUE.
      IF (PRESENT(compute_screened_norm)) need_screened_norm = compute_screened_norm
      screened_norm = 0.0_dp
      IF (need_screened_norm) THEN
         CALL parallel_gemm('N', 'N', n, n, n, z_one, matrices(w_coulomb), &
                            matrices(w_dielectric), z_zero, matrices(w_tmp))
         CALL parallel_gemm('N', 'C', n, n, n, z_one, matrices(w_tmp), &
                            matrices(w_coulomb), z_zero, matrices(w_output))
         screened_norm = cp_cfm_norm(matrices(w_output), 'F')
      END IF
      ! With K=M^-1 V^1/2 and Hermitian M^-1, W=K(ε^-1-I)Kᴴ.
      ! The existing factor removes two matrix products at every later frequency.
      CALL parallel_gemm('N', 'N', n, n, n, z_one, matrices(w_metric), &
                         matrices(w_dielectric), z_zero, matrices(w_tmp))
      CALL parallel_gemm('N', 'C', n, n, n, z_one, matrices(w_tmp), &
                         matrices(w_metric), z_zero, matrices(w_output))
      CALL get_owner_matrix(matrices(w_output), output_owner, group_env, root, n)
   END SUBROUTINE grouped_w_solve

! **************************************************************************************************
!> \brief Release the distributed dielectric workspaces for a q-point group.
!> \param matrices Distributed matrix storage.
!> \param matrix_struct ...
!> \param blacs ...
! **************************************************************************************************
   SUBROUTINE grouped_w_release(matrices, matrix_struct, blacs)
      TYPE(cp_cfm_type), INTENT(INOUT)                   :: matrices(6)
      TYPE(cp_fm_struct_type), POINTER                   :: matrix_struct
      TYPE(cp_blacs_env_type), POINTER                   :: blacs

      CALL cp_cfm_release(matrices(w_chi))
      CALL cp_cfm_release(matrices(w_metric))
      CALL cp_cfm_release(matrices(w_coulomb))
      CALL cp_cfm_release(matrices(w_tmp))
      CALL cp_cfm_release(matrices(w_dielectric))
      CALL cp_cfm_release(matrices(w_output))
      CALL cp_fm_struct_release(matrix_struct)
      CALL cp_blacs_env_release(blacs)
   END SUBROUTINE grouped_w_release

! **************************************************************************************************
!> \brief Access RI-RS full-k work data owned by the band-structure environment.
!> \param bs_env Band-structure environment owning the allocation.
!> \param gwkp_env Typed reference valid until release_gw_work.
! **************************************************************************************************
   SUBROUTINE get_gw_work(bs_env, gwkp_env)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gwkp_type), POINTER                           :: gwkp_env

      CPASSERT(ALLOCATED(bs_env%gwkp_env))
      SELECT TYPE (work => bs_env%gwkp_env)
      TYPE IS (gwkp_type)
         gwkp_env => work
      CLASS DEFAULT
         CPABORT('Invalid RI-RS full-k work type')
      END SELECT
   END SUBROUTINE get_gw_work

! **************************************************************************************************
!> \brief Release all periodic RI-RS work data after the last quasiparticle iteration.
!> \param bs_env Band-structure environment owning the work data.
! **************************************************************************************************
   SUBROUTINE release_gw_work(bs_env)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'release_gw_work'

      INTEGER                                            :: handle
      TYPE(gwkp_type), POINTER                           :: gwkp_env

      CALL timeset(routineN, handle)

      CALL get_gw_work(bs_env, gwkp_env)
      CALL mp_para_env_release(gwkp_env%group_env)
      IF (ALLOCATED(gwkp_env%eigenvalues_G)) DEALLOCATE (gwkp_env%eigenvalues_G)
      ! AO image pointers are released immediately after the Gamma contraction.
      CPASSERT(.NOT. ASSOCIATED(gwkp_env%gamma_corr_d))
      CPASSERT(.NOT. ASSOCIATED(gwkp_env%gamma_corr_h))
      CPASSERT(.NOT. ASSOCIATED(gwkp_env%gamma_corr_s))
      CPASSERT(.NOT. ASSOCIATED(gwkp_env%gamma_corr_index_to_cell))
      CPASSERT(.NOT. ASSOCIATED(gwkp_env%gamma_corr_cell_to_index))
      DEALLOCATE (bs_env%gwkp_env)
      NULLIFY (gwkp_env)

      CALL timestop(handle)

   END SUBROUTINE release_gw_work

! **************************************************************************************************
!> \brief Collect SCF and position-operator image cells, including mirrored images.
!> \param kp_a ...
!> \param kp_b ...
!> \param index_to_cell ...
!> \param cell_to_index ...
! **************************************************************************************************
   SUBROUTINE gw_gamma_cell_index_union(kp_a, kp_b, index_to_cell, cell_to_index)
      TYPE(kpoint_type), POINTER                         :: kp_a, kp_b
      INTEGER, DIMENSION(:, :), POINTER                  :: index_to_cell
      INTEGER, DIMENSION(:, :, :), POINTER               :: cell_to_index

      INTEGER                                            :: i, ic, isgn, n
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: list
      INTEGER, DIMENSION(3)                              :: image, max_image
      INTEGER, DIMENSION(:, :), POINTER                  :: itc_a, itc_b

      NULLIFY (itc_a, itc_b)
      CALL get_kpoint_info(kp_a, index_to_cell=itc_a)
      CALL get_kpoint_info(kp_b, index_to_cell=itc_b)
      CPASSERT(ASSOCIATED(itc_a) .AND. ASSOCIATED(itc_b))
      DO i = 1, 3
         max_image(i) = MAX(MAXVAL(ABS(itc_a(i, :))), MAXVAL(ABS(itc_b(i, :))))
      END DO
      ALLOCATE (cell_to_index(-max_image(1):max_image(1), -max_image(2):max_image(2), &
                              -max_image(3):max_image(3)), source=0)
      ALLOCATE (list(3, 2*(SIZE(itc_a, 2) + SIZE(itc_b, 2))))
      n = 0
      DO isgn = 1, -1, -2
         DO ic = 1, SIZE(itc_a, 2)
            image = isgn*itc_a(:, ic)
            IF (cell_to_index(image(1), image(2), image(3)) /= 0) CYCLE
            n = n + 1
            cell_to_index(image(1), image(2), image(3)) = n
            list(:, n) = image
         END DO
      END DO
      DO isgn = 1, -1, -2
         DO ic = 1, SIZE(itc_b, 2)
            image = isgn*itc_b(:, ic)
            IF (cell_to_index(image(1), image(2), image(3)) /= 0) CYCLE
            n = n + 1
            cell_to_index(image(1), image(2), image(3)) = n
            list(:, n) = image
         END DO
      END DO
      ALLOCATE (index_to_cell(3, n))
      index_to_cell = list(:, 1:n)
      DEALLOCATE (list)
   END SUBROUTINE gw_gamma_cell_index_union

! **************************************************************************************************
!> \brief Matrix power and optional spectral statistics.
!> \param matrix ...
!> \param exponent ...
!> \param eps ...
!> \param cond_nr ...
!> \param min_ev ...
!> \param max_ev ...
!> \param relative_eps Cutoff relative to the largest eigenvalue.
! **************************************************************************************************
   SUBROUTINE power(matrix, exponent, eps, cond_nr, min_ev, max_ev, &
                    relative_eps)
      COMPLEX(KIND=dp), DIMENSION(:, :)                  :: matrix
      REAL(KIND=dp)                                      :: exponent, eps
      REAL(KIND=dp), OPTIONAL                            :: cond_nr, min_ev, max_ev, relative_eps

      CHARACTER(len=*), PARAMETER                        :: routineN = 'power'

      COMPLEX(KIND=dp), ALLOCATABLE, DIMENSION(:, :)     :: eigenvectors
      INTEGER                                            :: handle, i, n
      REAL(KIND=dp)                                      :: eigenvalue_cutoff, pos_eval
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: eigenvalues

      IF (.NOT. PRESENT(relative_eps)) THEN
         CALL local_complex_power(matrix, exponent, eps, cond_nr, min_ev, max_ev)
         RETURN
      END IF

      CALL timeset(routineN, handle)

      ! make matrix perfectly Hermitian
      matrix(:, :) = 0.5_dp*(matrix(:, :) + CONJG(TRANSPOSE(matrix(:, :))))

      n = SIZE(matrix, 1)
      ALLOCATE (eigenvalues(n), eigenvectors(n, n))
      CALL diag_complex(matrix, eigenvectors, eigenvalues)

      eigenvalue_cutoff = MAX(eps, relative_eps*MAXVAL(eigenvalues))

      IF (PRESENT(cond_nr)) cond_nr = MAXVAL(ABS(eigenvalues))/MINVAL(ABS(eigenvalues))
      IF (PRESENT(min_ev)) min_ev = MINVAL(ABS(eigenvalues))
      IF (PRESENT(max_ev)) max_ev = MAXVAL(ABS(eigenvalues))

      DO i = 1, n
         IF (eigenvalue_cutoff < eigenvalues(i)) THEN
            pos_eval = (eigenvalues(i))**(0.5_dp*exponent)
         ELSE
            pos_eval = 0.0_dp
         END IF
         eigenvectors(:, i) = eigenvectors(:, i)*pos_eval
      END DO

      CALL ZGEMM("N", "C", n, n, n, z_one, eigenvectors, n, eigenvectors, n, z_zero, matrix, n)

      DEALLOCATE (eigenvalues, eigenvectors)

      CALL timestop(handle)

   END SUBROUTINE power

END MODULE gw_ri_rs_small_cell_full_kp_utils
