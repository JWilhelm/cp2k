!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Work states, quadrature and distributed-matrix helpers for periodic RI-RS GW.
!> \author Jan Wilhelm (with help of ChatGPT, GPT-6 Sol and GPT-6 Astra)
! **************************************************************************************************
MODULE gw_ri_rs_small_cell_full_kp_utils
   USE atomic_kind_types,               ONLY: atomic_kind_type,&
                                              get_atomic_kind_set
   USE basis_set_types,                 ONLY: gto_basis_set_type
   USE cell_types,                      ONLY: cell_type,&
                                              get_cell,&
                                              pbc
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
   USE ieee_arithmetic,                 ONLY: ieee_is_finite
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
                                              diamat_all,&
                                              gemm_square,&
                                              inv_3x3
   USE message_passing,                 ONLY: mp_comm_split_type_shared,&
                                              mp_comm_type,&
                                              mp_mem_avail_per_rank_GB,&
                                              mp_para_env_release,&
                                              mp_para_env_type,&
                                              mp_print_mem_per_rank,&
                                              mp_request_type,&
                                              mp_waitall
   USE mp2_eri_gpw,                     ONLY: mp2_eri_2c_integrate_gpw
   USE omp_lib,                         ONLY: omp_get_max_threads,&
                                              omp_in_parallel
   USE orbital_pointers,                ONLY: indco,&
                                              ncoset,&
                                              nso
   USE parallel_gemm_api,               ONLY: parallel_gemm
   USE particle_methods,                ONLY: get_particle_set
   USE particle_types,                  ONLY: particle_type
   USE physcon,                         ONLY: angstrom
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
   USE ri_rs_utils,                     ONLY: evaluate_ao_basis_on_points,&
                                              precompute_ri_rs_radii
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
   INTEGER, PARAMETER :: tile_cols = 32
   INTEGER, PARAMETER :: target_batch_size = 8, pair_batch_size = 256
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

   TYPE :: uniform_settings_type
      REAL(dp) :: accuracy = 0.0_dp, spacing = 0.0_dp, padding = 0.0_dp, &
                  l_block_edge = 0.0_dp
      INTEGER :: counts(3) = 0
      REAL(dp) :: lower(3) = 0.0_dp, origin(3) = 0.0_dp, step(3, 3) = 0.0_dp, weight = 0.0_dp
   END TYPE uniform_settings_type

   TYPE :: l_block_partition_type
      INTEGER :: nblocks = 0
      INTEGER, ALLOCATABLE :: first(:), indices(:)
      REAL(dp), ALLOCATABLE :: lower(:, :), upper(:, :)
   END TYPE l_block_partition_type

   TYPE :: ao_image_type
      TYPE(gto_basis_set_type), POINTER :: basis => NULL()
      REAL(dp) :: center(3) = 0.0_dp
      REAL(dp) :: bound = 0.0_dp, decay = 0.0_dp, radius2 = HUGE(0.0_dp)
      REAL(dp), ALLOCATABLE :: primary(:, :)
      INTEGER, ALLOCATABLE :: primary_l(:)
      REAL(dp), ALLOCATABLE :: envelope_exponents(:), radial_envelope(:, :)
   END TYPE ao_image_type

   ! Per-atom interpolation tables for the existing libint_potential_type
   ! Coulomb/erfc kernel. These are numerical cache data, not another kernel.
   TYPE :: radial_interpolation_type
      REAL(dp), ALLOCATABLE :: short_abscissae(:)
      REAL(dp), ALLOCATABLE :: coefficients(:, :), series(:, :, :), short_series(:, :, :), short_moments(:, :, :, :)
      ! Radial value and derivative, indexed (2, primitive/angular channel, squared-radius index).
      REAL(dp), ALLOCATABLE :: mesh_moments(:, :, :)
      INTEGER, ALLOCATABLE :: mesh_first(:, :), spherical_mesh_first(:, :)
      ! An optional exact factorization of primitive Cartesian rows through spherical RI functions.
      REAL(dp), ALLOCATABLE :: spherical_selection(:, :), spherical_angular(:, :, :, :), &
                               spherical_primitive(:, :, :)
      LOGICAL :: spherical_factorized = .FALSE.
      REAL(dp) :: mesh_spacing = 0.0_dp, mesh_offset = 0.0_dp, mesh_exponent = 0.0_dp
      REAL(dp) :: cutoff = 0.0_dp, attenuation = 0.0_dp
   END TYPE radial_interpolation_type

   TYPE :: radial_work_type
      REAL(dp), ALLOCATABLE :: jet(:), gaussian(:), derivative(:), bracket(:)
   END TYPE radial_work_type

   ! Work data shared by the periodic GW stages; input settings remain in bs_env.
   TYPE :: gwkp_type
      REAL(KIND=dp), ALLOCATABLE                 :: eigenvalues_G(:, :, :)
      COMPLEX(dp), ALLOCATABLE                  :: metric_coulomb_factor(:, :, :), psi_lm_k(:, :, :, :), &
                                                   Z_lP_q(:, :, :), Z_lP_zero(:, :, :)
      REAL(dp), ALLOCATABLE                     :: sigma_c_bands(:, :, :, :, :), sigma_x_bands(:, :, :)
      REAL(dp), ALLOCATABLE                     :: ao_values(:, :, :)
      COMPLEX(dp), ALLOCATABLE                  :: target_orbitals(:, :, :, :)
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
      CALL compute_Z_lP_small_cell_full_kp(bs_env, bs_env%ri_rs%grid_points, &
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
      CPASSERT(.NOT. ALLOCATED(gwkp_env%target_orbitals))
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
!> \brief Evaluate exchange and prepare target orbitals for correlation.
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

      CALL build_cell_ao_values(bs_env, bs_env%ri_rs%grid_points, gwkp_env%ao_values)
      CALL build_all_phi_k(bs_env, gwkp_env%ao_values, phi_lmu_k)
      CALL build_projection_interpolation(bs_env, gwkp_env%interpolation)
      CALL collect_target_bloch_orbitals(bs_env, gwkp_env%ao_values, &
                                         gwkp_env%target_orbitals, phi_lmu_k)
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
!> W remains in the auxiliary basis. A single q panel is broadcast during this
!> one-time redistribution; each auxiliary W(q,τ) goes only to its owning rank.
!> \param bs_env Band-structure settings.
!> \param gwkp_env Work arrays and their MPI distribution.
! **************************************************************************************************
   SUBROUTINE prepare_self_energy_groups(bs_env, gwkp_env)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(gwkp_type), INTENT(INOUT)                     :: gwkp_env

      CHARACTER(LEN=*), PARAMETER :: routineN = 'prepare_self_energy_groups'

      COMPLEX(KIND=dp), ALLOCATABLE                      :: psi_group(:, :, :, :), &
                                                            psi_panel(:, :, :), Z_group(:, :, :), &
                                                            Z_panel(:, :)
      INTEGER :: handle, i_dest, i_group_size, i_root, i_source, i_tag, ik, ik_local, iq, &
         iq_local, it, it_local, n_ao, n_aux, n_grid, n_k, n_local_k, n_local_q, n_local_time, &
         n_spin, n_stored_q, n_threads, n_time, n_world, phase_handle
      REAL(KIND=dp) :: available_GB, budget_bytes, grid_bytes, input_bytes, n_pairs, panel_bytes, &
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
      panel_bytes = 16.0_dp*REAL(n_grid, dp)*REAL(n_ao*n_spin + n_aux, dp)
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
         ! q matrices, band projection, and Green panels belong to separate stages.
         grid_bytes = n_pairs*32.0_dp*REAL(n_k, dp) + &
                      MAX(16.0_dp*REAL(n_grid, dp)**2*REAL(n_local_k, dp) + &
                          16.0_dp*REAL(n_aux, dp)**2*REAL(n_local_k, dp) + &
                          16.0_dp*REAL(n_threads*n_grid*(n_ao + n_aux), dp), &
                          projection_workspace_bytes(n_grid, n_pairs, n_k, n_ao, n_threads)) + &
                      32.0_dp*REAL(n_threads*n_k, dp) + 256.0_dp*1024.0_dp**2
         peak_bytes = MAX(persistent_bytes + panel_bytes, &
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
                psi_panel(n_grid, n_ao, n_spin), Z_panel(n_grid, n_aux))
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
         IF (column_env%mepos == i_root) psi_panel(:, :, :) = gwkp_env%psi_lm_k(:, :, (ik - 1)/n_world + 1, :)
         CALL column_env%bcast(psi_panel, i_root)
         ik_local = ik_local + 1
         psi_group(:, :, ik_local, :) = psi_panel
      END DO
      CALL timestop(phase_handle)
      CALL timeset('Sigma_groups_redistribute_ZW', phase_handle)
      DO iq = MODULO(gwkp_env%group_env%mepos - 1, i_group_size) + 1, n_k, i_group_size
         i_root = MODULO(iq, n_world)/i_group_size
         IF (column_env%mepos == i_root) Z_panel(:, :) = gwkp_env%Z_lP_q(:, :, (iq - 1)/n_world + 1)
         CALL column_env%bcast(Z_panel, i_root)
         iq_local = iq_local + 1
         Z_group(:, :, iq_local) = Z_panel
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
      DEALLOCATE (psi_panel, Z_panel)

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
         chi_q(:, :, :), G_grid_k(:, :, :), work(:, :), Z_panel(:, :)
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
      CPASSERT(.NOT. ALLOCATED(gwkp_env%target_orbitals))
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
         ALLOCATE (Z_group(n_grid, n_aux, n_local), Z_panel(n_grid, n_aux))
         iq_local = 0
         DO iq = MODULO(chi_env%mepos - 1, i_size) + 1, bs_env%nkp_scf_desymm, i_size
            i_root = MODULO(iq, n_world)/i_size
            IF (column_env%mepos == i_root) Z_panel(:, :) = gwkp_env%Z_lP_q(:, :, (iq - 1)/n_world + 1)
            CALL column_env%bcast(Z_panel, i_root)
            iq_local = iq_local + 1
            Z_group(:, :, iq_local) = Z_panel
         END DO
         DEALLOCATE (Z_panel)
         Z_time => Z_group
      END IF
      IF (n_groups > 1) THEN
         n_k_local = (bs_env%nkp_scf_desymm + i_size - 1 - chi_env%mepos)/i_size
         ALLOCATE (psi_time(n_grid, bs_env%n_ao, n_k_local, bs_env%n_spin))
         ALLOCATE (Z_panel(n_grid, bs_env%n_ao))
         DO ispin = 1, bs_env%n_spin
            iq_local = 0
            DO iq = chi_env%mepos + 1, bs_env%nkp_scf_desymm, i_size
               i_root = MODULO(iq - 1, n_world)/i_size
               IF (column_env%mepos == i_root) Z_panel(:, :) = gwkp_env%psi_lm_k(:, :, (iq - 1)/n_world + 1, ispin)
               CALL column_env%bcast(Z_panel, i_root)
               iq_local = iq_local + 1
               psi_time(:, :, iq_local, ispin) = Z_panel
            END DO
         END DO
         DEALLOCATE (Z_panel)
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
                                      gwkp_env%target_orbitals)
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

      CPASSERT(ALLOCATED(gwkp_env%target_orbitals))
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
      IF (orbitals_cover_scf(bs_env, gwkp_env%target_orbitals)) THEN
         scf_values => gwkp_env%target_orbitals
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
                                            sigma_band, gwkp_env%group_env, gwkp_env%interpolation, gwkp_env%target_orbitals)
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
                                            sigma_band, gwkp_env%group_env, gwkp_env%interpolation, gwkp_env%target_orbitals)
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
!> \brief Return exp(i 2 pi k.R).
!> \param kpoint ...
!> \param cell ...
!> \return ...
! **************************************************************************************************
   PURE FUNCTION lattice_phase(kpoint, cell) RESULT(phase)
      REAL(KIND=dp), DIMENSION(3), INTENT(IN)            :: kpoint
      INTEGER, DIMENSION(3), INTENT(IN)                  :: cell
      COMPLEX(KIND=dp)                                   :: phase

      REAL(KIND=dp)                                      :: angle

      angle = twopi*SUM(kpoint*REAL(cell, dp))
      phase = CMPLX(COS(angle), SIN(angle), KIND=dp)

   END FUNCTION lattice_phase

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
!> H is Hermitian. The root supplies H in bounded column tiles; eigensolver,
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
!> \brief Build explicit-cell AO values and periodic RI-RS coefficients Z_lP(q).
!> \param bs_env Band-structure environment.
!> \param grid_points Reference-cell RI-RS points r_l.
!> \param ao_values ...
!> \param Z_lP_q Gaussian RI-RS coefficients on the shifted SCF mesh.
!> \param Z_lP_zero ...
! **************************************************************************************************
   SUBROUTINE compute_Z_lP_small_cell_full_kp(bs_env, grid_points, ao_values, &
                                              Z_lP_q, Z_lP_zero)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: grid_points
      REAL(dp), ALLOCATABLE, INTENT(INOUT), TARGET       :: ao_values(:, :, :)
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: Z_lP_q(:, :, :), Z_lP_zero(:, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'compute_Z_lP_small_cell_full_kp'

      COMPLEX(KIND=dp), ALLOCATABLE                      :: charge_lq(:, :)
      INTEGER                                            :: first_l, handle, last_l, n_D_cells, &
                                                            n_grid, n_target_cells
      INTEGER, ALLOCATABLE                               :: D_cells(:, :), target_cells(:, :)
      REAL(KIND=dp)                                      :: elapsed, start_time
      REAL(KIND=dp), ALLOCATABLE                         :: D_llprime_A(:, :, :), &
                                                            qpoints_head(:, :), qpoints_scf(:, :), &
                                                            target_R(:, :, :)
      REAL(KIND=dp), POINTER                             :: phi_lmu_R(:, :, :)

      CALL timeset(routineN, handle)

      start_time = m_walltime()
      IF (bs_env%unit_nr > 0) WRITE (bs_env%unit_nr, '(T2,A)') "RI-RS: Compute Z_lP(k); required: D_ll'^R and d_lP^R"

      IF (bs_env%ri_rs%tikhonov < 0.0_dp) THEN
         CPABORT("RI_RS%TIKHONOV must not be negative")
      END IF

      n_grid = SIZE(grid_points, 2)
      CPASSERT(n_grid > 0)
      CPASSERT(bs_env%nkp_scf_desymm > 0)

      CALL build_cell_ao_values(bs_env, grid_points, ao_values)
      phi_lmu_R => ao_values
      CALL build_target_cell_set(bs_env, grid_points, target_cells)
      CALL build_D_cell_set(bs_env, phi_lmu_R, D_cells)
      n_target_cells = SIZE(target_cells, 2)
      n_D_cells = SIZE(D_cells, 2)
      CPASSERT(n_target_cells > 0)
      CPASSERT(n_D_cells > 0)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') 'RI-RS d_lP^R cells R:', n_target_cells
         WRITE (bs_env%unit_nr, '(T2,A,T72,I9)') "RI-RS D_ll'^R cells R:", n_D_cells
         FLUSH (bs_env%unit_nr)
      END IF

      ! Each rank stores a rectangular (l,l') block for every translation A.
      first_l = (n_grid*bs_env%para_env%mepos)/bs_env%para_env%num_pe + 1
      last_l = (n_grid*(bs_env%para_env%mepos + 1))/bs_env%para_env%num_pe
      ALLOCATE (D_llprime_A(last_l - first_l + 1, n_grid, n_D_cells))
      CALL build_D_llprime_lattice(bs_env, D_cells, phi_lmu_R, first_l, last_l, D_llprime_A)
      CALL mp_print_mem_per_rank(bs_env%para_env, bs_env%unit_nr, "After distribution of D_ll'^R:")
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A)') 'RI-RS three-center projection d_lP^R: starting'
         FLUSH (bs_env%unit_nr)
      END IF
      ALLOCATE (qpoints_scf(3, bs_env%nkp_scf_desymm))
      qpoints_scf(:, :) = bs_env%kpoints_chi_eps_W%xkp(:, 1:bs_env%nkp_scf_desymm)
      ALLOCATE (qpoints_head(3, 1), SOURCE=0.0_dp)
      CALL build_dlp_selected(bs_env, grid_points, target_cells, target_R, qpoints_head, charge_lq)
      CALL mp_print_mem_per_rank(bs_env%para_env, bs_env%unit_nr, 'After RI-RS three-center projection d_lP^R')

      CALL solve_Z_mesh(bs_env, D_cells, target_cells, D_llprime_A, target_R, first_l, last_l, qpoints_scf, &
                        Z_lP_q, distributed=.TRUE.)
      CALL solve_Z_mesh(bs_env, D_cells, target_cells, D_llprime_A, target_R, first_l, last_l, &
                        qpoints_head, Z_lP_zero, distributed=.FALSE., charge_lq=charge_lq)

      DEALLOCATE (target_cells, D_cells, D_llprime_A, qpoints_scf, target_R)
      DEALLOCATE (qpoints_head)

      elapsed = m_walltime() - start_time
      CALL bs_env%para_env%max(elapsed)
      IF (bs_env%unit_nr > 0) WRITE (bs_env%unit_nr, '(T2,A,T69,F10.1,A)') &
         'Completed computing Z_lP(q), Exec. time:', elapsed, ' s'
      IF (bs_env%unit_nr > 0) FLUSH (bs_env%unit_nr)

      CALL timestop(handle)

   END SUBROUTINE compute_Z_lP_small_cell_full_kp

! **************************************************************************************************
!> \brief Reconstruct the same cutoff AO lattice images without repeating the RI solve.
!> \param bs_env ...
!> \param grid_points ...
!> \param ao_values ...
! **************************************************************************************************
   SUBROUTINE build_cell_ao_values(bs_env, grid_points, ao_values)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(KIND=dp), INTENT(IN)                          :: grid_points(:, :)
      REAL(dp), ALLOCATABLE, INTENT(INOUT), TARGET       :: ao_values(:, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'build_cell_ao_values'

      INTEGER                                            :: handle, i_cell, ready
      REAL(dp)                                           :: available_GB, required_bytes
      TYPE(mp_comm_type)                                 :: node_comm

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
         CALL fill_ao_translations(bs_env, grid_points, bs_env%index_to_cell_3c, ao_values)
      END IF
      DO i_cell = 1, SIZE(ao_values, 3)
         CALL node_comm%bcast(ao_values(:, :, i_cell))
      END DO
      CALL node_comm%free()

      CALL timestop(handle)

   END SUBROUTINE build_cell_ao_values

! **************************************************************************************************
!> \brief Solve the periodic RI-RS equations at the supplied momentum-transfer points.
!> At zero Tikhonov parameter the Hermitian D matrix is solved in the
!> Moore-Penrose sense, yielding the minimum-norm unregularized fit.
!> \param bs_env ...
!> \param D_cells ...
!> \param target_cells ...
!> \param D_llprime_A ...
!> \param target_R ...
!> \param first_l ...
!> \param last_l ...
!> \param qpoints ...
!> \param Z_lP_q ...
!> \param distributed ...
!> \param charge_lq ...
! **************************************************************************************************
   SUBROUTINE solve_Z_mesh(bs_env, D_cells, target_cells, D_llprime_A, target_R, first_l, last_l, qpoints, &
                           Z_lP_q, distributed, charge_lq)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: D_cells, target_cells
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(IN)      :: D_llprime_A
      REAL(KIND=dp), CONTIGUOUS, INTENT(IN)              :: target_R(:, :, :)
      INTEGER, INTENT(IN)                                :: first_l, last_l
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: qpoints
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT)         :: Z_lP_q(:, :, :)
      LOGICAL, INTENT(IN)                                :: distributed
      COMPLEX(KIND=dp), INTENT(IN), OPTIONAL             :: charge_lq(:, :)

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'solve_Z_mesh'

      COMPLEX(KIND=dp)                                   :: phase
      COMPLEX(KIND=dp), ALLOCATABLE :: D_llprime_q(:, :), D_llprime_q_local(:, :), &
         D_llprime_q_packed(:), rhs(:, :), target_q(:, :), target_q_reference(:, :)
      INTEGER :: first_l_rank, handle, i_A, i_batch, i_rank, info, iq, iq_batch, iq_local, &
         iq_store, l, last_l_rank, lp, n_grid, n_q, n_q_local, n_RI, n_RI_reference, n_RI_solve, &
         next_progress
      INTEGER(KIND=int_8)                                :: n_done, n_local
      INTEGER, ALLOCATABLE                               :: displs(:), recvcounts(:)
      LOGICAL, ALLOCATABLE                               :: active_grid_point(:), required_q(:)
      REAL(KIND=dp)                                      :: antihermitian_error, &
                                                            diagonal_imag_error, max_scale, &
                                                            metric_head, progress_start, &
                                                            qpoint(3), volume
      REAL(KIND=dp), ALLOCATABLE                         :: scale(:)

      CALL timeset(routineN, handle)

      n_grid = SIZE(D_llprime_A, 2)
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
      ALLOCATE (D_llprime_q(n_grid, n_grid), D_llprime_q_local(last_l - first_l + 1, n_grid), &
                D_llprime_q_packed(n_grid*n_grid), recvcounts(bs_env%para_env%num_pe), &
                displs(bs_env%para_env%num_pe), &
                target_q(n_grid, n_RI_solve), target_q_reference(n_grid, n_RI_reference), &
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
            D_llprime_q_local = z_zero
            DO i_A = 1, SIZE(D_cells, 2)
               phase = lattice_phase(qpoints(:, iq_batch), D_cells(1:3, i_A))
               D_llprime_q_local(:, :) = D_llprime_q_local(:, :) + phase*D_llprime_A(:, :, i_A)
            END DO
            i_rank = MODULO(iq_batch - 1, bs_env%para_env%num_pe)
            IF (distributed) i_rank = MODULO(iq_batch, bs_env%para_env%num_pe)
            CALL bs_env%para_env%gatherv(RESHAPE(D_llprime_q_local, [SIZE(D_llprime_q_local)]), &
                                         D_llprime_q_packed, recvcounts, displs, i_rank)
            IF (bs_env%para_env%mepos == i_rank) THEN
               DO l = 0, bs_env%para_env%num_pe - 1
                  first_l_rank = (n_grid*l)/bs_env%para_env%num_pe + 1
                  last_l_rank = (n_grid*(l + 1))/bs_env%para_env%num_pe
                  D_llprime_q(first_l_rank:last_l_rank, :) = RESHAPE( &
                                                          D_llprime_q_packed(displs(l + 1) + 1:displs(l + 1) + recvcounts(l + 1)), &
                                                             [last_l_rank - first_l_rank + 1, n_grid])
               END DO
            END IF
         END DO
         CALL dlp_target_q_batch(bs_env, target_cells, target_R, qpoints, i_batch, &
                                 distributed, required_q, target_q_reference, iq)
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
            ! The Gamma-correction scalar head is separate. Keep the full Gaussian target.
            CPASSERT(n_RI_reference == n_RI)
            target_q(:, 2:) = target_q_reference
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
            target_q(:, 1) = metric_head/SQRT(volume)*charge_lq(:, iq)
         ELSE
            CPASSERT(n_RI_reference == n_RI)
            target_q(:, :) = target_q_reference
         END IF

         antihermitian_error = MAXVAL(ABS(D_llprime_q - CONJG(TRANSPOSE(D_llprime_q))))
         diagonal_imag_error = 0.0_dp
         DO l = 1, n_grid
            diagonal_imag_error = MAX(diagonal_imag_error, ABS(AIMAG(D_llprime_q(l, l))))
         END DO
         IF (antihermitian_error > 100.0_dp*bs_env%eps_filter .OR. &
             diagonal_imag_error > 100.0_dp*bs_env%eps_filter) THEN
            CPABORT("Non-Hermitian periodic RI-RS D(q)")
         END IF
         D_llprime_q(:, :) = 0.5_dp*(D_llprime_q(:, :) + CONJG(TRANSPOSE(D_llprime_q(:, :))))

         DO l = 1, n_grid
            IF (REAL(D_llprime_q(l, l), KIND=dp) < -100.0_dp*bs_env%eps_filter) THEN
               CPABORT("Periodic RI-RS D(q) has a negative diagonal element")
            END IF
            SCALE(l) = SQRT(MAX(REAL(D_llprime_q(l, l), KIND=dp), TINY(1.0_dp)))
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
                  D_llprime_q(l, lp) = D_llprime_q(l, lp)/(SCALE(l)*SCALE(lp))
               ELSE
                  D_llprime_q(l, lp) = z_zero
               END IF
            END DO
         END DO
         DO l = 1, n_grid
            D_llprime_q(l, l) = D_llprime_q(l, l) + bs_env%ri_rs%tikhonov*z_one
            IF (active_grid_point(l)) THEN
               rhs(l, :) = target_q(l, :)/SCALE(l)
            ELSE
               rhs(l, :) = z_zero
               SCALE(l) = 1.0_dp
            END IF
         END DO

         IF (bs_env%ri_rs%tikhonov <= 0.0_dp) THEN
            CALL solve_unregularized_D(D_llprime_q, rhs)
         ELSE
            CALL zpotrf('L', n_grid, D_llprime_q, n_grid, info)
            IF (info == 0) CALL zpotrs('L', n_grid, n_RI_solve, D_llprime_q, n_grid, rhs, n_grid, info)
            IF (info /= 0) CPABORT("Periodic RI-RS regularized solve failed")
         END IF

         DO l = 1, n_grid
            Z_lP_q(l, :, iq_store) = rhs(l, :)/SCALE(l)
         END DO
         IF (bs_env%para_env%mepos == 0 .AND. n_local > 1) THEN
            n_done = n_done + 1_int_8
            CALL write_ri_rs_phase_progress(bs_env%unit_nr, 'Z_lP(q) q-point', &
                                            n_done, n_local, next_progress, progress_start)
         END IF
      END DO
      IF (distributed) THEN
         CPASSERT(iq_local == n_q_local)
      END IF
      IF (.NOT. distributed) CALL bs_env%para_env%sum(Z_lP_q)
      DEALLOCATE (D_llprime_q, D_llprime_q_local, D_llprime_q_packed, recvcounts, displs, &
                  target_q, target_q_reference, rhs, scale, active_grid_point, required_q)

      CALL timestop(handle)

   END SUBROUTINE solve_Z_mesh

! **************************************************************************************************
!> \brief Solve a possibly rank-deficient Hermitian D matrix without Tikhonov regularization.
!> Only eigenvalues indistinguishable from zero at matrix-roundoff precision
!> are excluded from the Moore-Penrose inverse.
!> \param D_llprime_q ...
!> \param rhs ...
! **************************************************************************************************
   SUBROUTINE solve_unregularized_D(D_llprime_q, rhs)
      COMPLEX(KIND=dp), DIMENSION(:, :), INTENT(INOUT)   :: D_llprime_q, rhs

      COMPLEX(KIND=dp), ALLOCATABLE                      :: projected_rhs(:, :), work(:)
      INTEGER                                            :: ieigenvalue, info, ngrid
      REAL(KIND=dp)                                      :: rank_tolerance
      REAL(KIND=dp), ALLOCATABLE                         :: eigenvalues(:), rwork(:)

      ngrid = SIZE(D_llprime_q, 1)
      CPASSERT(ngrid > 0)
      CPASSERT(SIZE(D_llprime_q, 2) == ngrid)
      CPASSERT(SIZE(rhs, 1) == ngrid)
      ALLOCATE (eigenvalues(ngrid), rwork(MAX(1, 3*ngrid - 2)), &
                work(MAX(1, 2*ngrid - 1)), projected_rhs(ngrid, SIZE(rhs, 2)))
      CALL zheev('V', 'L', ngrid, D_llprime_q, ngrid, eigenvalues, work, SIZE(work), rwork, info)
      IF (info /= 0) CPABORT('Unregularized periodic RI-RS eigensolver failed')
      rank_tolerance = EPSILON(1.0_dp)*REAL(ngrid, dp)*MAXVAL(ABS(eigenvalues))
      IF (MINVAL(eigenvalues) < -10.0_dp*rank_tolerance) THEN
         CPABORT('Unregularized RI-RS D matrix is not positive semidefinite')
      END IF
      IF (.NOT. ANY(eigenvalues > rank_tolerance)) THEN
         CPABORT('Unregularized RI-RS D matrix has zero numerical rank')
      END IF

      projected_rhs(:, :) = MATMUL(CONJG(TRANSPOSE(D_llprime_q)), rhs)
      DO ieigenvalue = 1, ngrid
         IF (eigenvalues(ieigenvalue) > rank_tolerance) THEN
            projected_rhs(ieigenvalue, :) = &
               projected_rhs(ieigenvalue, :)/eigenvalues(ieigenvalue)
         ELSE
            projected_rhs(ieigenvalue, :) = z_zero
         END IF
      END DO
      rhs(:, :) = MATMUL(D_llprime_q, projected_rhs)
   END SUBROUTINE solve_unregularized_D

! **************************************************************************************************
!> \brief Fill an existing AO array using translated basis values.
!> \param bs_env ...
!> \param grid_points ...
!> \param cells ...
!> \param phi_lmu_R ...
! **************************************************************************************************
   SUBROUTINE fill_ao_translations(bs_env, grid_points, cells, phi_lmu_R)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: grid_points
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: cells
      REAL(KIND=dp), INTENT(OUT)                         :: phi_lmu_R(:, :, :)

      INTEGER                                            :: first_ao, i_atom, i_cell, i_kind, last_ao
      REAL(KIND=dp)                                      :: cutoff_squared, source_position(3)
      TYPE(cell_type), POINTER                           :: cell
      TYPE(gto_basis_set_type), POINTER                  :: basis
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set

      cell => bs_env%ri_rs%cell
      particle_set => bs_env%ri_rs%particle_set
      phi_lmu_R = 0.0_dp

      DO i_cell = 1, SIZE(cells, 2)
         DO i_atom = 1, bs_env%n_atom
            i_kind = particle_set(i_atom)%atomic_kind%kind_number
            basis => bs_env%basis_set_AO(i_kind)%gto_basis_set
            first_ao = bs_env%i_ao_start_from_atom(i_atom)
            last_ao = bs_env%i_ao_end_from_atom(i_atom)
            source_position = pbc(particle_set(i_atom)%r, cell) + &
                              MATMUL(bs_env%hmat, REAL(cells(:, i_cell), dp))
            cutoff_squared = bs_env%ri_rs%radius_ao_per_atom(i_atom)**2
            IF (bs_env%ri_rs%cutoff_radius_ri_ao > 0.0_dp) THEN
               cutoff_squared = bs_env%ri_rs%cutoff_radius_ri_ao**2
            END IF
            CALL evaluate_ao_basis_on_points(phi_lmu_R(:, first_ao:last_ao, i_cell), &
                                             grid_points, basis, source_position, cell, &
                                             cutoff_squared=cutoff_squared, minimum_image=.FALSE.)
         END DO
      END DO

   END SUBROUTINE fill_ao_translations

! **************************************************************************************************
!> \brief Enumerate translations T that survive the R_cut^(lP) criterion for at least one (l,P).
!> \param bs_env ...
!> \param grid_points ...
!> \param target_cells ...
! **************************************************************************************************
   SUBROUTINE build_target_cell_set(bs_env, grid_points, target_cells)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: grid_points
      INTEGER, ALLOCATABLE, INTENT(OUT)                  :: target_cells(:, :)

      INTEGER                                            :: cell_index(3), i_atom, i_l, ix, iy, iz, &
                                                            n_max(3), n_target, n_work
      INTEGER, ALLOCATABLE                               :: cells_work(:, :)
      LOGICAL                                            :: keep
      REAL(KIND=dp)                                      :: center(3), cutoff, max_cutoff
      REAL(KIND=dp), ALLOCATABLE                         :: automatic_cutoffs(:)
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set

      particle_set => bs_env%ri_rs%particle_set
      CALL get_dlp_target_cutoffs(bs_env, automatic_cutoffs)
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
      n_target = 0
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
               n_target = n_target + 1
               cells_work(:, n_target) = cell_index
            END DO
         END DO
      END DO

      ALLOCATE (target_cells(3, n_target))
      target_cells(:, :) = cells_work(:, 1:n_target)
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T69,F10.1,A)') &
            'Periodic RI-RS automatic R_cut^(lP):', &
            max_cutoff*angstrom, ' Å'
      END IF
      DEALLOCATE (automatic_cutoffs, cells_work)

   END SUBROUTINE build_target_cell_set

! **************************************************************************************************
!> \brief Enumerate the translations A required by the AO-product D matrix D_ll'^A.
!>
!> Unlike the right-hand side d_lP^T, D_ll'^A is independent of R_cut^(lP). A translation can
!> contribute only if it is the difference of two image cells with nonzero AO values on the
!> reference-cell grid. Keeping a separate cell set prevents Eq. (10) screening from truncating
!> the least-squares metric and avoids enumerating empty three-center image cells.
!> \param bs_env ...
!> \param phi_lmu_R ...
!> \param D_cells ...
! **************************************************************************************************
   SUBROUTINE build_D_cell_set(bs_env, phi_lmu_R, D_cells)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(IN)      :: phi_lmu_R
      INTEGER, ALLOCATABLE, INTENT(OUT)                  :: D_cells(:, :)

      INTEGER                                            :: cell_index(3), i, i_R, i_R_prime, &
                                                            n_active, n_D_cells
      INTEGER, ALLOCATABLE                               :: active_cells(:), cells_work(:, :)
      LOGICAL                                            :: found

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

      ALLOCATE (D_cells(3, n_D_cells))
      D_cells(:, :) = cells_work(:, 1:n_D_cells)
      DEALLOCATE (active_cells, cells_work)

   END SUBROUTINE build_D_cell_set

! **************************************************************************************************
!> \brief Construct D_ll'^R = [Σ_{S-S'=R,μ} ϕ_μ^S(r_L) ϕ_μ^S'(r_L')]².
!> The progress count enumerates pairs of active cell images (S,S'), not Gaussian pairs.
!> \param bs_env ...
!> \param D_cells ...
!> \param phi_lmu_R ...
!> \param first_l ...
!> \param last_l ...
!> \param D_llprime_A ...
! **************************************************************************************************
   SUBROUTINE build_D_llprime_lattice(bs_env, D_cells, phi_lmu_R, first_l, last_l, D_llprime_A)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: D_cells
      REAL(KIND=dp), INTENT(IN)                          :: phi_lmu_R(:, :, :)
      INTEGER, INTENT(IN)                                :: first_l, last_l
      REAL(KIND=dp), CONTIGUOUS, INTENT(OUT)             :: D_llprime_A(:, :, :)

      INTEGER                                            :: cell_A(3), handle, i_A, i_R, i_R_prime, &
                                                            next_progress, nrows
      INTEGER(KIND=int_8)                                :: n_done, n_total
      LOGICAL                                            :: found
      LOGICAL, ALLOCATABLE                               :: active_cell(:)
      REAL(KIND=dp)                                      :: progress_start
      REAL(KIND=dp), ALLOCATABLE                         :: phi_local(:, :)
      INTERFACE
         SUBROUTINE dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            IMPORT dp
            CHARACTER(LEN=1), INTENT(IN) :: transa, transb
            INTEGER, INTENT(IN) :: m, n, k, lda, ldb, ldc
            REAL(dp), INTENT(IN) :: alpha, beta, a(lda, *), b(ldb, *)
            REAL(dp), INTENT(INOUT) :: c(ldc, *)
         END SUBROUTINE dgemm
      END INTERFACE

      CALL timeset('build_D_llprime_lattice', handle)

      ALLOCATE (active_cell(SIZE(phi_lmu_R, 3)), SOURCE=.FALSE.)
      DO i_R = 1, SIZE(phi_lmu_R, 3)
         active_cell(i_R) = ANY(phi_lmu_R(:, :, i_R) /= 0.0_dp)
      END DO
      progress_start = m_walltime()
      next_progress = 1
      n_done = 0_int_8
      n_total = INT(COUNT(active_cell), int_8)**2
      ! Sum the AO images for each owned (l,l') rectangle before squaring.
      D_llprime_A = 0.0_dp
      nrows = MAX(0, last_l - first_l + 1)
      ALLOCATE (phi_local(nrows, SIZE(phi_lmu_R, 2)))
      IF (first_l <= last_l) THEN
         DO i_R = 1, SIZE(phi_lmu_R, 3)
            IF (.NOT. active_cell(i_R)) CYCLE
            phi_local(:, :) = phi_lmu_R(first_l:last_l, :, i_R)
            DO i_R_prime = 1, SIZE(phi_lmu_R, 3)
               IF (.NOT. active_cell(i_R_prime)) CYCLE
               IF (n_done > 0) CALL write_ri_rs_phase_progress(bs_env%unit_nr, "D_ll'^R Cell-image pairs", &
                                                               n_done, n_total, next_progress, progress_start)
               n_done = n_done + 1
               cell_A = bs_env%index_to_cell_3c(:, i_R) - &
                        bs_env%index_to_cell_3c(:, i_R_prime)
               CALL find_cell(cell_A, D_cells, found, i_A)
               IF (.NOT. found) CYCLE
               CALL dgemm('N', 'T', nrows, SIZE(phi_lmu_R, 1), SIZE(phi_lmu_R, 2), &
                          1.0_dp, phi_local, MAX(1, nrows), phi_lmu_R(:, :, i_R_prime), &
                          MAX(1, SIZE(phi_lmu_R, 1)), 1.0_dp, D_llprime_A(:, :, i_A), &
                          MAX(1, nrows))
            END DO
         END DO
      END IF
      IF (first_l <= last_l) CALL write_ri_rs_phase_progress(bs_env%unit_nr, "D_ll'^R Cell-image pairs", &
                                                             n_done, n_total, next_progress, progress_start)
      DO i_A = 1, SIZE(D_llprime_A, 3)
         D_llprime_A(:, :, i_A) = D_llprime_A(:, :, i_A)*D_llprime_A(:, :, i_A)
      END DO
      DEALLOCATE (active_cell, phi_local)

      CALL timestop(handle)

   END SUBROUTINE build_D_llprime_lattice

! **************************************************************************************************
!> \brief Find a lattice triplet in an explicit image-cell list.
!> \param cell ...
!> \param index_to_cell ...
!> \param found ...
!> \param index ...
! **************************************************************************************************
   SUBROUTINE find_cell(cell, index_to_cell, found, index)
      INTEGER, DIMENSION(3), INTENT(IN)                  :: cell
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: index_to_cell
      LOGICAL, INTENT(OUT)                               :: found
      INTEGER, INTENT(OUT)                               :: index

      INTEGER                                            :: i

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
!> \brief Distribute individual l rows evenly across MPI ranks; empty ranges are valid.
!> \param n_grid Number of RI-RS grid points.
!> \param n_ranks Number of MPI ranks.
!> \param i_rank MPI rank.
!> \param first_l First locally owned grid row.
!> \param last_l Last locally owned grid row.
! **************************************************************************************************
   SUBROUTINE dlp_l_range(n_grid, n_ranks, i_rank, first_l, last_l)
      INTEGER, INTENT(IN)                                :: n_grid, n_ranks, i_rank
      INTEGER, INTENT(OUT)                               :: first_l, last_l

      CPASSERT(i_rank >= 0 .AND. i_rank < n_ranks)
      first_l = INT(INT(n_grid, int_8)*INT(i_rank, int_8)/INT(n_ranks, int_8)) + 1
      last_l = INT(INT(n_grid, int_8)*INT(i_rank + 1, int_8)/INT(n_ranks, int_8))
   END SUBROUTINE dlp_l_range

! **************************************************************************************************
!> \brief Transform local l rows to target q and gather rows for the q-local solves.
!> d_lP(q)=Σ_T exp(iqT)d_lP(T); two send buffers overlap transfer with the next target transform.
!> Each rank holds all P and T for its l interval; each destination receives one complete q.
!> \param bs_env ...
!> \param target_cells ...
!> \param target_R ...
!> \param qpoints ...
!> \param i_batch ...
!> \param distributed ...
!> \param required_q ...
!> \param target_q ...
!> \param iq ...
! **************************************************************************************************
   SUBROUTINE dlp_target_q_batch(bs_env, target_cells, target_R, qpoints, i_batch, distributed, &
                                 required_q, target_q, iq)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: target_cells(:, :)
      REAL(dp), CONTIGUOUS, INTENT(IN), TARGET           :: target_R(:, :, :)
      REAL(dp), INTENT(IN)                               :: qpoints(:, :)
      INTEGER, INTENT(IN)                                :: i_batch
      LOGICAL, INTENT(IN)                                :: distributed, required_q(:)
      COMPLEX(dp), ASYNCHRONOUS, CONTIGUOUS, INTENT(OUT) :: target_q(:, :)
      INTEGER, INTENT(OUT)                               :: iq

      COMPLEX(dp)                                        :: phase
      COMPLEX(dp), ALLOCATABLE, ASYNCHRONOUS             :: receive_q(:), send_q(:, :, :)
      INTEGER :: cell_index(3), first_l, handle, i_dest, i_grid, i_l, i_P, i_q, i_rank, i_slot, &
         i_T, last_l, local_first_l, local_last_l, n_active_q, n_grid_P, n_l, n_P, n_ranks, &
         n_recv, position, q_column, q_offset
      INTEGER(KIND=int_8)                                :: work_bytes
      INTEGER, ALLOCATABLE                               :: q_indices(:)
      LOGICAL                                            :: receiving, sending(2), use_blas
      REAL(dp)                                           :: qpoint(3)
      REAL(dp), ALLOCATABLE                              :: phase_imag(:, :), phase_real(:, :), &
                                                            result_imag(:, :), result_real(:, :)
      REAL(dp), CONTIGUOUS, POINTER                      :: target_matrix(:, :)
      TYPE(mp_request_type)                              :: send_requests(2)
      TYPE(mp_request_type), ALLOCATABLE                 :: recv_requests(:)
      INTERFACE
         SUBROUTINE dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            IMPORT dp
            CHARACTER(LEN=1), INTENT(IN) :: transa, transb
            INTEGER, INTENT(IN) :: m, n, k, lda, ldb, ldc
            REAL(dp), INTENT(IN) :: alpha, beta, a(lda, *), b(ldb, *)
            REAL(dp), INTENT(INOUT) :: c(ldc, *)
         END SUBROUTINE dgemm
      END INTERFACE

      CALL timeset('d_lP_T_to_q', handle)

      n_ranks = bs_env%para_env%num_pe
      i_rank = bs_env%para_env%mepos
      CPASSERT(SIZE(required_q) == SIZE(qpoints, 2))
      q_offset = 0
      IF (distributed) q_offset = 1
      iq = i_batch*n_ranks + MODULO(i_rank - q_offset, n_ranks) + 1
      CALL dlp_l_range(SIZE(target_q, 1), n_ranks, i_rank, local_first_l, local_last_l)
      CPASSERT(SIZE(target_R, 1) == local_last_l - local_first_l + 1)
      CPASSERT(SIZE(target_R, 2) == SIZE(target_q, 2))
      CPASSERT(SIZE(target_q, KIND=int_8) <= INT(HUGE(0), int_8))
      n_P = SIZE(target_q, 2)
      ALLOCATE (recv_requests(n_ranks), send_q(SIZE(target_R, 1), SIZE(target_R, 2), 2))
      target_q = z_zero
      n_recv = 0
      receiving = .FALSE.
      IF (iq <= SIZE(qpoints, 2)) receiving = required_q(iq)
      IF (receiving) THEN
         ! Packed rank blocks: offset=(first_l-1)*n_P, followed by local (l,P) order.
         ! Receive into contiguous storage before assigning the noncontiguous output rows.
         ALLOCATE (receive_q(SIZE(target_q)))
         DO i_dest = 0, n_ranks - 1
            IF (i_dest == i_rank) CYCLE
            CALL dlp_l_range(SIZE(target_q, 1), n_ranks, i_dest, first_l, last_l)
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
      work_bytes = INT(SIZE(target_R, 1), int_8)*INT(SIZE(target_R, 2), int_8)
      use_blas = work_bytes > 0_int_8 .AND. work_bytes <= INT(HUGE(n_grid_P), int_8)
      IF (use_blas) THEN
         n_grid_P = INT(work_bytes)
         work_bytes = 16_int_8*INT(n_active_q, int_8)* &
                      (work_bytes + INT(SIZE(target_cells, 2), int_8))
         use_blas = n_active_q >= 4 .AND. SIZE(target_cells, 2) >= 32 .AND. &
                    work_bytes <= 64_int_8*1024_int_8*1024_int_8
      END IF
      IF (use_blas) THEN
         ALLOCATE (phase_real(SIZE(target_cells, 2), n_active_q), &
                   phase_imag(SIZE(target_cells, 2), n_active_q), &
                   result_real(n_grid_P, n_active_q), result_imag(n_grid_P, n_active_q))
         DO q_column = 1, n_active_q
            qpoint(:) = qpoints(1:3, q_indices(q_column))
            DO i_T = 1, SIZE(target_cells, 2)
               cell_index(:) = target_cells(1:3, i_T)
               phase = lattice_phase(qpoint, cell_index)
               phase_real(i_T, q_column) = REAL(phase, KIND=dp)
               phase_imag(i_T, q_column) = AIMAG(phase)
            END DO
         END DO
         target_matrix(1:n_grid_P, 1:SIZE(target_cells, 2)) => target_R
         CALL dgemm('N', 'N', n_grid_P, n_active_q, SIZE(target_cells, 2), &
                    1.0_dp, target_matrix, n_grid_P, phase_real, SIZE(target_cells, 2), &
                    0.0_dp, result_real, n_grid_P)
         CALL dgemm('N', 'N', n_grid_P, n_active_q, SIZE(target_cells, 2), &
                    1.0_dp, target_matrix, n_grid_P, phase_imag, SIZE(target_cells, 2), &
                    0.0_dp, result_imag, n_grid_P)
      END IF
      sending = .FALSE.
      q_column = 0
      IF (SIZE(target_R, 1) > 0 .AND. SIZE(target_R, 2) > 0) THEN
         DO i_dest = 0, n_ranks - 1
            i_q = i_batch*n_ranks + MODULO(i_dest - q_offset, n_ranks) + 1
            IF (i_q > SIZE(qpoints, 2)) CYCLE
            IF (.NOT. required_q(i_q)) CYCLE
            q_column = q_column + 1
            i_slot = MODULO(i_dest, 2) + 1
            IF (sending(i_slot)) CALL mp_waitall(send_requests(i_slot:i_slot))
            sending(i_slot) = .FALSE.
            IF (use_blas) THEN
               DO i_P = 1, SIZE(target_R, 2)
                  DO i_grid = 1, SIZE(target_R, 1)
                     i_T = (i_P - 1)*SIZE(target_R, 1) + i_grid
                     send_q(i_grid, i_P, i_slot) = &
                        CMPLX(result_real(i_T, q_column), result_imag(i_T, q_column), KIND=dp)
                  END DO
               END DO
            ELSE
               send_q(:, :, i_slot) = z_zero
               DO i_T = 1, SIZE(target_cells, 2)
                  send_q(:, :, i_slot) = send_q(:, :, i_slot) + &
                                         lattice_phase(qpoints(:, i_q), target_cells(:, i_T))*target_R(:, :, i_T)
               END DO
            END IF
            IF (i_dest == i_rank) THEN
               target_q(local_first_l:local_last_l, :) = send_q(:, :, i_slot)
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
            CALL dlp_l_range(SIZE(target_q, 1), n_ranks, i_dest, first_l, last_l)
            n_l = last_l - first_l + 1
            DO i_P = 1, n_P
               position = (first_l - 1)*n_P + (i_P - 1)*n_l
               DO i_l = 1, n_l
                  target_q(first_l + i_l - 1, i_P) = receive_q(position + i_l)
               END DO
            END DO
         END DO
         DEALLOCATE (receive_q)
      END IF
      DEALLOCATE (send_q, recv_requests)

      CALL timestop(handle)

   END SUBROUTINE dlp_target_q_batch

! **************************************************************************************************
!> \brief Evaluate d_lP^T by real-space numerical quadrature.
!> \param bs_env ...
!> \param grid_points ...
!> \param target_cells ...
!> \param target_R ...
!> \param qpoints ...
!> \param charge_lq ...
! **************************************************************************************************
   SUBROUTINE build_dlp_selected(bs_env, grid_points, target_cells, target_R, &
                                 qpoints, charge_lq)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(dp), INTENT(IN)                               :: grid_points(:, :)
      INTEGER, INTENT(IN)                                :: target_cells(:, :)
      REAL(dp), ALLOCATABLE, INTENT(OUT)                 :: target_R(:, :, :)
      REAL(dp), INTENT(IN), OPTIONAL                     :: qpoints(:, :)
      COMPLEX(dp), ALLOCATABLE, INTENT(OUT), OPTIONAL    :: charge_lq(:, :)

      INTEGER                                            :: first_l, handle, last_l, n_P

      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,A9)') 'd_lP method', 'NUMERICAL'
      END IF
      CALL dlp_l_range(SIZE(grid_points, 2), bs_env%para_env%num_pe, &
                       bs_env%para_env%mepos, first_l, last_l)
      n_P = bs_env%n_RI
      IF (bs_env%auto_ri%enabled) n_P = SUM(bs_env%auto_ri%sizes_opt_RI)
      ALLOCATE (target_R(last_l - first_l + 1, n_P, SIZE(target_cells, 2)))

      CALL timeset("d_lP_uniform_grid", handle)

      IF (PRESENT(charge_lq)) THEN
         CPASSERT(PRESENT(qpoints))
         ALLOCATE (charge_lq(SIZE(grid_points, 2), SIZE(qpoints, 2)))
         CALL build_uniform_dlp(bs_env, grid_points, target_cells, target_R, first_l, &
                                qpoints, charge_lq)
      ELSE
         CALL build_uniform_dlp(bs_env, grid_points, target_cells, target_R, first_l)
      END IF

      CALL timestop(handle)

      IF (.NOT. ALL(ieee_is_finite(target_R))) THEN
         CPABORT("Nonfinite DLP cell value")
      END IF
   END SUBROUTINE build_dlp_selected

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

      COMPLEX(KIND=dp), ALLOCATABLE                      :: go(:, :), gv(:, :), panel(:, :), &
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
      ALLOCATE (go(nr, ng), gv(nr, ng), panel(nr, no))
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
                          gamma_corr(:, :, c), no, z_zero, panel, nr)
               state%wing_tau(:, c, t) = state%wing_tau(:, c, t) + &
                                         weight*SUM(CONJG(scaled(:, 1:no))*panel, DIM=2)
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
      DEALLOCATE (occ_factor, vir_factor, scaled, go, gv, panel)
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
!> \brief Form target Bloch orbitals once instead of communicating them for every self-energy.
!> \param bs_env ...
!> \param phi_R ...
!> \param orbitals ...
!> \param phi_native_k ...
! **************************************************************************************************
   SUBROUTINE collect_target_bloch_orbitals(bs_env, phi_R, orbitals, phi_native_k)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(KIND=dp), INTENT(IN)                          :: phi_R(:, :, :)
      COMPLEX(dp), ALLOCATABLE, INTENT(INOUT)            :: orbitals(:, :, :, :)
      COMPLEX(dp), INTENT(IN), OPTIONAL                  :: phi_native_k(:, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'collect_target_bloch_orbitals'
      REAL(dp), PARAMETER :: progress_fraction(9) = [0.001_dp, 0.003_dp, 0.01_dp, 0.03_dp, 0.1_dp, &
         0.3_dp, 0.6_dp, 0.9_dp, 1.0_dp]

      COMPLEX(KIND=dp)                                   :: phase
      COMPLEX(KIND=dp), ALLOCATABLE                      :: coefficients(:, :, :, :), phi_k(:, :), &
                                                            phi_plane(:, :), psi_k(:, :)
      INTEGER :: handle, i_local, i_native, i_node, i_owned, i_progress, i_R, i_root, i_row, &
         i_target, ispin, n_grid, n_local, n_local_max, n_node, n_owned
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
      CALL collect_target_mo_coefficients(bs_env, coefficients)
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
      ! Node-local ranks compute disjoint rows and exchange each completed target.
      IF (PRESENT(phi_native_k)) ALLOCATE (phi_plane(n_grid, bs_env%n_ao))
      start_time = m_walltime()
      i_progress = 1
      IF (bs_env%unit_nr > 0) THEN
         WRITE (bs_env%unit_nr, '(T2,A)') 'GW: evaluating band-path orbitals on the RI-RS grid'
         FLUSH (bs_env%unit_nr)
      END IF
      DO i_target = 1, bs_env%nkp_bs_and_DOS
         i_native = 0
         IF (PRESENT(phi_native_k) .AND. i_target <= bs_env%nkp_scf_desymm) THEN
            delta_k = bs_env%kpoints_DOS%xkp(:, i_target) - bs_env%kpoints_scf_desymm%xkp(:, i_target)
            IF (MAXVAL(ABS(delta_k - ANINT(delta_k))) <= 64.0_dp*EPSILON(1.0_dp)) i_native = i_target
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
               phase = EXP(CMPLX(0.0_dp, twopi*SUM(bs_env%kpoints_DOS%xkp(:, i_target)* &
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
                          coefficients(:, :, i_target, ispin), bs_env%n_ao, z_zero, psi_k, n_local)
               DO i_local = 1, n_local
                  i_row = i_node + 1 + (i_local - 1)*n_node
                  orbitals(i_row, :, i_target, ispin) = psi_k(i_local, :)
               END DO
            END DO
         END IF
         DO ispin = 1, bs_env%n_spin
            CALL node_comm%sum(orbitals(:, :, i_target, ispin))
         END DO
         IF (i_progress <= SIZE(progress_fraction)) THEN
            IF (REAL(i_target, dp) >= progress_fraction(i_progress)*REAL(bs_env%nkp_bs_and_DOS, dp)) THEN
            IF (bs_env%unit_nr > 0) THEN
               WRITE (bs_env%unit_nr, '(T2,A,T26,I5,A,I0,T40,F5.1,A,T55,A,T69,F10.1,A)') &
                  'GW band-path k-points ', i_target, ' / ', bs_env%nkp_bs_and_DOS, &
                  100.0_dp*REAL(i_target, dp)/REAL(bs_env%nkp_bs_and_DOS, dp), ' %', &
                  'Total time:', m_walltime() - start_time, ' s'
               FLUSH (bs_env%unit_nr)
            END IF
            DO WHILE (i_progress <= SIZE(progress_fraction))
               IF (REAL(i_target, dp) < progress_fraction(i_progress)*REAL(bs_env%nkp_bs_and_DOS, dp)) EXIT
               i_progress = i_progress + 1
            END DO
            END IF
         END IF
      END DO
      IF (ALLOCATED(phi_plane)) DEALLOCATE (phi_plane)
      CALL node_comm%free()
      DEALLOCATE (phi_k, psi_k, coefficients)

      CALL timestop(handle)

   END SUBROUTINE collect_target_bloch_orbitals

! **************************************************************************************************
!> \brief Build the target/q interpolation geometry once for all imaginary-time projections.
!> \param bs_env ...
!> \param interpolation ...
! **************************************************************************************************
   SUBROUTINE build_projection_interpolation(bs_env, interpolation)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(projection_interpolation_type), INTENT(OUT)   :: interpolation

      CHARACTER(LEN=*), PARAMETER :: routineN = 'build_projection_interpolation'

      INTEGER                                            :: handle, i_target, iq, n_offmesh, &
                                                            native_iq
      REAL(KIND=dp)                                      :: delta_k(3)

      CALL timeset(routineN, handle)

      ALLOCATE (interpolation%native_iq(bs_env%nkp_bs_and_DOS), &
                interpolation%offmesh_index(bs_env%nkp_bs_and_DOS))
      interpolation%native_iq = 0
      interpolation%offmesh_index = 0
      n_offmesh = 0
      DO i_target = 1, bs_env%nkp_bs_and_DOS
         native_iq = 0
         IF (i_target <= bs_env%nkp_scf_desymm) THEN
            delta_k = bs_env%kpoints_DOS%xkp(:, i_target) - &
                      bs_env%kpoints_scf_desymm%xkp(:, i_target)
            IF (MAXVAL(ABS(delta_k - ANINT(delta_k))) <= 64.0_dp*EPSILON(1.0_dp)) THEN
               native_iq = i_target
            END IF
         ELSE
            DO iq = 1, bs_env%nkp_scf_desymm
               delta_k = bs_env%kpoints_DOS%xkp(:, i_target) - &
                         bs_env%kpoints_scf_desymm%xkp(:, iq)
               IF (MAXVAL(ABS(delta_k - ANINT(delta_k))) <= 64.0_dp*EPSILON(1.0_dp)) THEN
                  native_iq = iq
                  EXIT
               END IF
            END DO
         END IF
         interpolation%native_iq(i_target) = native_iq
         IF (native_iq == 0) THEN
            n_offmesh = n_offmesh + 1
            interpolation%offmesh_index(i_target) = n_offmesh
         END IF
      END DO
      ! Bound the replicated geometry cache; large band paths keep the original evaluation.
      IF (8.0_dp*REAL(bs_env%nkp_scf_desymm, dp)*REAL(n_offmesh, dp) <= 256.0_dp*1024.0_dp**2) THEN
         ALLOCATE (interpolation%weights(bs_env%nkp_scf_desymm, n_offmesh))
         DO i_target = 1, bs_env%nkp_bs_and_DOS
            IF (interpolation%offmesh_index(i_target) == 0) CYCLE
            DO iq = 1, bs_env%nkp_scf_desymm
               interpolation%weights(iq, interpolation%offmesh_index(i_target)) = &
                  cardinal_weight(bs_env%kpoints_DOS%xkp(:, i_target) - &
                                  bs_env%kpoints_scf_desymm%xkp(:, iq), &
                                  bs_env%kpoints_scf_desymm%nkp_grid)
            END DO
         END DO
      END IF

      CALL timestop(handle)

   END SUBROUTINE build_projection_interpolation

! **************************************************************************************************
!> \brief Replicate target-k MO coefficients once before imaginary-time work is split into groups.
!> \param bs_env ...
!> \param target_coefficients ...
! **************************************************************************************************
   SUBROUTINE collect_target_mo_coefficients(bs_env, target_coefficients)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), ALLOCATABLE, INTENT(OUT) :: target_coefficients(:, :, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'collect_target_mo_coefficients'

      INTEGER                                            :: handle, i_col, i_row, i_target, ispin, &
                                                            ncol_local, nrow_local
      INTEGER, POINTER                                   :: col_indices(:), row_indices(:)

      CALL timeset(routineN, handle)

      ALLOCATE (target_coefficients(bs_env%n_ao, bs_env%n_ao, &
                                    bs_env%nkp_bs_and_DOS, bs_env%n_spin))
      target_coefficients = z_zero
      DO ispin = 1, bs_env%n_spin
         DO i_target = 1, bs_env%nkp_bs_and_DOS
            CALL cp_cfm_get_info(bs_env%cfm_mo_coeff_kp(i_target, ispin), &
                                 nrow_local=nrow_local, ncol_local=ncol_local, &
                                 row_indices=row_indices, col_indices=col_indices)
            DO i_col = 1, ncol_local
               DO i_row = 1, nrow_local
                  target_coefficients(row_indices(i_row), col_indices(i_col), i_target, ispin) = &
                     bs_env%cfm_mo_coeff_kp(i_target, ispin)%local_data(i_row, i_col)
               END DO
            END DO
         END DO
      END DO
      CALL bs_env%para_env%sum(target_coefficients)

      CALL timestop(handle)

   END SUBROUTINE collect_target_mo_coefficients

! **************************************************************************************************
!> \brief Interpolate Σx_ll′(k) and project onto target Bloch orbitals, Eqs. (10)-(12).
!> Σx_nn(k) = Σ_ll′ ψ_nk(r_l)* Σx_ll′(k) ψ_nk(r_l′).
!> Native mesh points use their existing matrix; off-mesh points use the
!> pair-centred cardinal interpolation prepared for the requested targets.
!> \param bs_env ...
!> \param sigma_grid ...
!> \param first_row ...
!> \param last_row ...
!> \param ispin ...
!> \param sigma_band ...
!> \param interpolation ...
!> \param target_orbitals ...
! **************************************************************************************************
   SUBROUTINE self_energy_RS_to_Bloch(bs_env, sigma_grid, first_row, last_row, ispin, sigma_band, &
                                      interpolation, target_orbitals)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(dp), INTENT(IN)                            :: sigma_grid(:, :, :)
      INTEGER, INTENT(IN)                                :: first_row, last_row, ispin
      REAL(dp), INTENT(OUT)                              :: sigma_band(:, :)
      TYPE(projection_interpolation_type), INTENT(IN)    :: interpolation
      COMPLEX(dp), INTENT(IN)                            :: target_orbitals(:, :, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'self_energy_RS_to_Bloch'

      COMPLEX(dp), ALLOCATABLE                           :: psi(:, :), sigma_k(:, :), work(:, :)
      INTEGER                                            :: handle, i_band, i_batch, i_first, &
                                                            i_last, i_target, n_batch, n_grid, &
                                                            native_iq, nrows, phase_handle
      LOGICAL                                            :: valid_layout

      CALL timeset(routineN, handle)

      n_grid = SIZE(target_orbitals, 1)
      nrows = last_row - first_row + 1
      CPASSERT(first_row >= 1 .AND. last_row <= n_grid .AND. nrows >= 0)
      valid_layout = SIZE(sigma_grid, 1) == nrows .AND. SIZE(sigma_grid, 2) == n_grid
      CPASSERT(valid_layout)
      CPASSERT(SIZE(sigma_grid, 3) == bs_env%nkp_scf_desymm)
      CPASSERT(SIZE(target_orbitals, 2) == bs_env%n_ao)
      CPASSERT(SIZE(target_orbitals, 3) == bs_env%nkp_bs_and_DOS)
      CPASSERT(SIZE(target_orbitals, 4) == bs_env%n_spin)
      CPASSERT(SIZE(sigma_band, 1) == bs_env%n_ao)
      CPASSERT(SIZE(sigma_band, 2) == bs_env%nkp_bs_and_DOS)
      n_batch = MIN(16, bs_env%n_ao)
      ALLOCATE (sigma_k(nrows, n_grid), psi(n_grid, n_batch), work(nrows, n_batch))
      sigma_band = 0.0_dp
      DO i_target = 1, bs_env%nkp_bs_and_DOS
         CALL timeset(routineN//'_q_interpolation', phase_handle)
         native_iq = interpolation%native_iq(i_target)
         IF (native_iq > 0) THEN
            sigma_k(:, :) = sigma_grid(:, :, native_iq)
         ELSE
            CALL interpolate_self_energy_rows(bs_env, sigma_grid, first_row, i_target, sigma_k, interpolation)
         END IF
         CALL timestop(phase_handle)
         DO i_first = 1, bs_env%n_ao, n_batch
            i_last = MIN(i_first + n_batch - 1, bs_env%n_ao)
            psi = z_zero
            psi(:, 1:i_last - i_first + 1) = target_orbitals(:, i_first:i_last, i_target, ispin)
            CALL timeset(routineN//'_apply_sigma', phase_handle)
            IF (nrows > 0) CALL zgemm('N', 'N', nrows, n_batch, n_grid, z_one, &
                                      sigma_k, nrows, psi, n_grid, z_zero, work, nrows)
            CALL timestop(phase_handle)
            CALL timeset(routineN//'_band_contraction', phase_handle)
            DO i_band = i_first, i_last
               i_batch = i_band - i_first + 1
               sigma_band(i_band, i_target) = REAL(SUM(CONJG(psi(first_row:last_row, i_batch))*work(:, i_batch)), dp)
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
!> \brief Project ordered local Σ(l,l') pairs with cached target Bloch orbitals.
!>
!> Each owned pair contributes Re[ψ_n(l)* Σ_ll'(k) ψ_n(l')]. Missing entries
!> in the rank-local column panel are zero. Batched matrix products retain
!> dense kernels without allocating a pair-by-target-by-band tensor.
!> \param bs_env ...
!> \param sigma_pairs ...
!> \param first_pair ...
!> \param ispin ...
!> \param sigma_band ...
!> \param sum_env ...
!> \param interpolation ...
!> \param target_orbitals ...
! **************************************************************************************************
   SUBROUTINE self_energy_pairs_to_Bloch(bs_env, sigma_pairs, first_pair, ispin, sigma_band, &
                                         sum_env, interpolation, target_orbitals)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: sigma_pairs(:, :, :)
      INTEGER, INTENT(IN)                                :: first_pair, ispin
      REAL(KIND=dp), INTENT(OUT)                         :: sigma_band(:, :)
      TYPE(mp_para_env_type), POINTER                    :: sum_env
      TYPE(projection_interpolation_type), INTENT(IN)    :: interpolation
      COMPLEX(dp), ALLOCATABLE, INTENT(IN)               :: target_orbitals(:, :, :, :)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'self_energy_pairs_to_Bloch'

      COMPLEX(KIND=dp), ALLOCATABLE                      :: sigma_columns(:, :), sigma_local(:, :), &
                                                            work(:, :)
      INTEGER :: first_col, handle, i_band, i_batch, i_first, i_l, i_local, i_lp, i_target, &
         i_target_first, i_target_last, i_target_local, image_offset(126), last_col, last_pair, &
         n_batch, n_grid, n_local, phase_handle
      INTEGER, ALLOCATABLE                               :: pair_order(:)
      LOGICAL                                            :: valid_layout

      CALL timeset(routineN, handle)

      CPASSERT(ALLOCATED(target_orbitals))
      n_grid = SIZE(target_orbitals, 1)
      n_local = SIZE(sigma_pairs, 1)
      CPASSERT(SIZE(sigma_pairs, 2) == 1)
      CPASSERT(SIZE(sigma_pairs, 3) == bs_env%nkp_scf_desymm)
      CPASSERT(SIZE(target_orbitals, 2) == bs_env%n_ao)
      valid_layout = SIZE(target_orbitals, 3) >= bs_env%nkp_bs_and_DOS
      CPASSERT(valid_layout)
      CPASSERT(SIZE(target_orbitals, 4) == bs_env%n_spin)
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
         !$OMP SHARED(bs_env, sigma_pairs, first_pair, ispin, sigma_band, interpolation, target_orbitals, &
         !$OMP        n_grid, n_local, first_col, last_col, n_batch, pair_order, image_offset) &
         !$OMP PRIVATE(sigma_columns, sigma_local, work, i_target, phase_handle, i_target_first, i_target_last, i_target_local, &
         !$OMP         i_local, i_l, i_lp, i_first, i_band, i_batch)
         ALLOCATE (sigma_columns(n_grid, last_col - first_col + 1), &
                   sigma_local(n_local, target_batch_size), work(n_grid, n_batch))
         !$OMP DO SCHEDULE(STATIC, 1)
         DO i_target_first = 1, bs_env%nkp_bs_and_DOS, target_batch_size
            i_target_last = MIN(i_target_first + target_batch_size - 1, bs_env%nkp_bs_and_DOS)
            CALL timeset(routineN//'_q_interpolation', phase_handle)
            CALL interpolate_target_batch(bs_env, sigma_pairs, i_target_first, i_target_last, &
                                          pair_order, image_offset, sigma_local, interpolation)
            CALL timestop(phase_handle)
            DO i_target = i_target_first, i_target_last
               i_target_local = i_target - i_target_first + 1
               sigma_columns = z_zero
               DO i_local = 1, n_local
                  i_l = MODULO(first_pair + i_local - 2, n_grid) + 1
                  i_lp = (first_pair + i_local - 2)/n_grid + 1
                  sigma_columns(i_l, i_lp - first_col + 1) = sigma_local(i_local, i_target_local)
               END DO
               DO i_first = 1, bs_env%n_ao, n_batch
                  CALL timeset(routineN//'_band_contraction', phase_handle)
                  CALL project_target_columns(n_grid, last_col - first_col + 1, &
                                              MIN(n_batch, bs_env%n_ao - i_first + 1), first_col, i_first, &
                                              sigma_columns, target_orbitals(:, :, i_target, ispin), work)
                  DO i_band = i_first, MIN(i_first + n_batch - 1, bs_env%n_ao)
                     i_batch = i_band - i_first + 1
                     sigma_band(i_band, i_target) = &
                        REAL(SUM(CONJG(target_orbitals(:, i_band, i_target, ispin))* &
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
!> \param i_target ...
!> \param factors ...
!> \param fractional_grid ...
!> \param interpolation ...
! **************************************************************************************************
   SUBROUTINE interpolation_pair_factors(bs_env, i_target, factors, fractional_grid, interpolation)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      INTEGER, INTENT(IN)                                :: i_target
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
         delta_k = bs_env%kpoints_DOS%xkp(:, i_target) - bs_env%kpoints_scf_desymm%xkp(:, iq)
         IF (PRESENT(interpolation)) THEN
            IF (ALLOCATED(interpolation%weights)) THEN
               weight = interpolation%weights(iq, interpolation%offmesh_index(i_target))
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
!> \param i_target ...
!> \param sigma_k ...
!> \param interpolation ...
! **************************************************************************************************
   SUBROUTINE interpolate_self_energy_rows(bs_env, sigma_grid, first_row, i_target, sigma_k, interpolation)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(KIND=dp), INTENT(IN)                       :: sigma_grid(:, :, :)
      INTEGER, INTENT(IN)                                :: first_row, i_target
      COMPLEX(KIND=dp), INTENT(OUT)                      :: sigma_k(:, :)
      TYPE(projection_interpolation_type), INTENT(IN), &
         OPTIONAL                                        :: interpolation

      COMPLEX(KIND=dp), ALLOCATABLE                      :: factors(:, :)
      INTEGER                                            :: icol, iq, irow
      INTEGER, ALLOCATABLE                               :: pair_image(:)
      REAL(KIND=dp), ALLOCATABLE                         :: fractional_grid(:, :)

      CALL interpolation_pair_factors(bs_env, i_target, factors, fractional_grid, interpolation)
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
!> \brief Bound simultaneous target interpolation and band-projection storage in bytes.
!> Shared image labels and thread-private packed Σ, factors and output panels are included.
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
              REAL(n_threads, dp)*(16.0_dp*(REAL(target_batch_size + 1, dp)*n_pairs + &
                                            REAL(n_grid, dp)*REAL(MIN(16, n_ao) + 2, dp) + &
                                       REAL(n_k, dp)*REAL(125*target_batch_size + pair_batch_size + target_batch_size + 125, dp) + &
                                            REAL(pair_batch_size*target_batch_size, dp)) + 24.0_dp*REAL(n_grid, dp) + 4096.0_dp)
   END FUNCTION projection_workspace_bytes

! **************************************************************************************************
!> \brief Partition owned pairs by Eq. (1)'s 125 image classes once for all target k-points.
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
!> \brief Evaluate Eq. (1) for bounded pair and target batches with the same image factors.
!> Native targets are copied unchanged; each image class uses Σ_(pair,q) times factors_(q,k).
!> \param bs_env ...
!> \param sigma_pairs ...
!> \param first_target ...
!> \param last_target ...
!> \param pair_order ...
!> \param image_offset ...
!> \param sigma_k ...
!> \param interpolation ...
! **************************************************************************************************
   SUBROUTINE interpolate_target_batch(bs_env, sigma_pairs, first_target, last_target, &
                                       pair_order, image_offset, sigma_k, interpolation)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      COMPLEX(dp), INTENT(IN)                            :: sigma_pairs(:, :, :)
      INTEGER, INTENT(IN)                                :: first_target, last_target, &
                                                            pair_order(:), image_offset(126)
      COMPLEX(dp), INTENT(OUT)                           :: sigma_k(:, :)
      TYPE(projection_interpolation_type), INTENT(IN)    :: interpolation

      COMPLEX(dp), ALLOCATABLE                           :: factors(:, :, :), panel(:, :), &
                                                            PRODUCT(:, :), target_factors(:, :), &
                                                            weights(:, :)
      INTEGER                                            :: i_class, i_first, i_off, i_pair, &
                                                            i_target, iq, n_off, n_pairs, n_q, &
                                                            native_iq, &
                                                            off_target(target_batch_size)
      REAL(dp), ALLOCATABLE                              :: fractional_grid(:, :)

      n_off = 0
      DO i_target = first_target, last_target
         native_iq = interpolation%native_iq(i_target)
         IF (native_iq > 0) THEN
            sigma_k(:, i_target - first_target + 1) = sigma_pairs(:, 1, native_iq)
         ELSE
            n_off = n_off + 1
            off_target(n_off) = i_target
         END IF
      END DO
      IF (n_off == 0) RETURN
      n_q = SIZE(sigma_pairs, 3)
      ALLOCATE (factors(125, n_q, n_off), panel(pair_batch_size, n_q), &
                weights(n_q, n_off), PRODUCT(pair_batch_size, n_off))
      DO i_off = 1, n_off
         CALL interpolation_pair_factors(bs_env, off_target(i_off), target_factors, fractional_grid, interpolation)
         factors(:, :, i_off) = target_factors
         DEALLOCATE (target_factors, fractional_grid)
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
                  panel(i_pair, iq) = sigma_pairs(pair_order(i_first + i_pair - 1), 1, iq)
               END DO
            END DO
            CALL zgemm('N', 'N', n_pairs, n_off, n_q, z_one, panel, pair_batch_size, &
                       weights, n_q, z_zero, product, pair_batch_size)
            DO i_off = 1, n_off
               DO i_pair = 1, n_pairs
                  sigma_k(pair_order(i_first + i_pair - 1), off_target(i_off) - first_target + 1) = &
                     PRODUCT(i_pair, i_off)
               END DO
            END DO
         END DO
      END DO
      DEALLOCATE (factors, panel, weights, product)
   END SUBROUTINE interpolate_target_batch

! **************************************************************************************************
!> \brief Transform a periodic lattice product in a cell centred on each physical grid pair.
!> Products of two half-shifted lattice transforms are periodic. For s nearest to
!> H^{-1}(r_l-r_l'), F_ll'(k)=exp(2πiks) Σ_R exp(2πikR) F_ll'(R+s).
!> The even-mesh Nyquist endpoints and half-cell choices of s are averaged symmetrically.
!> On integer/half-shifted target meshes these averages are real separable endpoint weights,
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
!> \brief Contract a contiguous column block of Σ with strided target Bloch orbitals.
!> \param n_grid ...
!> \param n_columns ...
!> \param n_bands ...
!> \param first_col ...
!> \param i_first ...
!> \param sigma_columns ...
!> \param orbitals ...
!> \param work ...
! **************************************************************************************************
   SUBROUTINE project_target_columns(n_grid, n_columns, n_bands, first_col, i_first, sigma_columns, orbitals, work)
      INTEGER, INTENT(IN)                                :: n_grid, n_columns, n_bands, first_col, &
                                                            i_first
      COMPLEX(dp), INTENT(IN) :: sigma_columns(n_grid, n_columns), orbitals(n_grid, *)
      COMPLEX(dp), INTENT(OUT)                           :: work(n_grid, n_bands)

      CALL zgemm('N', 'N', n_grid, n_bands, n_columns, z_one, sigma_columns, n_grid, &
                 orbitals(first_col, i_first), n_grid, z_zero, work, n_grid)
   END SUBROUTINE project_target_columns

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
!> Scratch consists of a bounded row tile and, for V, its auxiliary-space intermediate.
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

      COMPLEX(KIND=dp), ALLOCATABLE                      :: tile(:, :), work(:, :)
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
         ALLOCATE (tile(n_row, n_col))
         IF (PRESENT(interaction)) THEN
            IF (.NOT. (SIZE(interaction, 1) == n_inner .AND. SIZE(interaction, 2) == n_inner)) THEN
               CPABORT("Internal GW array shape mismatch")
            END IF
            ALLOCATE (work(n_row, n_inner))
            CALL zgemm('N', 'N', n_row, n_inner, n_inner, z_one, left(i_first + i_row, 1), n_grid, &
                       interaction, n_inner, z_zero, work, n_row)
            CALL zgemm('N', 'C', n_row, n_col, n_inner, z_one, work, n_row, &
                       right(i_col, 1), n_grid, z_zero, tile, n_row)
            DEALLOCATE (work)
         ELSE
            CALL zgemm('N', 'C', n_row, n_col, n_inner, z_one, left(i_first + i_row, 1), n_grid, &
                       right(i_col, 1), n_grid, z_zero, tile, n_row)
         END IF
         transfer%send_buffer(i_buffer:i_buffer + n_row*n_col - 1) = RESHAPE(tile, [n_row*n_col])
         DEALLOCATE (tile)
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

      COMPLEX(KIND=dp), ALLOCATABLE                      :: tile(:, :)
      INTEGER                                            :: first, width

      IF (group_env%mepos == root) THEN
         CPASSERT(ALL(SHAPE(owner_matrix) == [n, n]))
      END IF
      ALLOCATE (tile(n, tile_cols))
      DO first = 1, n, tile_cols
         width = MIN(tile_cols, n - first + 1)
         IF (group_env%mepos == root) tile(:, 1:width) = owner_matrix(:, first:first + width - 1)
         CALL group_env%bcast(tile(:, 1:width), root)
         CALL cp_cfm_set_submatrix(distributed, tile(:, 1:width), &
                                   start_row=1, start_col=first)
      END DO
      DEALLOCATE (tile)
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

      COMPLEX(KIND=dp), ALLOCATABLE                      :: tile(:, :)
      INTEGER                                            :: first, width

      IF (group_env%mepos == root) THEN
         CPASSERT(ALL(SHAPE(owner_matrix) == [n, n]))
      END IF
      ALLOCATE (tile(n, tile_cols))
      DO first = 1, n, tile_cols
         width = MIN(tile_cols, n - first + 1)
         CALL cp_cfm_get_submatrix(distributed, tile(:, 1:width), &
                                   start_row=1, start_col=first)
         IF (group_env%mepos == root) owner_matrix(:, first:first + width - 1) = tile(:, 1:width)
      END DO
      DEALLOCATE (tile)
   END SUBROUTINE get_owner_matrix

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
   SUBROUTINE write_ri_rs_phase_progress(unit_nr, phase, n_done, n_total, next_progress, phase_start)
      INTEGER, INTENT(IN)                                :: unit_nr
      CHARACTER(LEN=*), INTENT(IN)                       :: phase
      INTEGER(KIND=int_8), INTENT(IN)                    :: n_done, n_total
      INTEGER, INTENT(INOUT)                             :: next_progress
      REAL(KIND=dp), INTENT(IN)                          :: phase_start

      REAL(KIND=dp), PARAMETER :: milestones(9) = [0.1_dp, 0.3_dp, 1.0_dp, 3.0_dp, 10.0_dp, 30.0_dp&
         , 60.0_dp, 90.0_dp, 100.0_dp]

      CHARACTER(LEN=100)                                 :: progress_line
      CHARACTER(LEN=20)                                  :: count_text
      CHARACTER(LEN=32)                                  :: phase_text
      CHARACTER(LEN=64)                                  :: progress_format
      CHARACTER(LEN=8)                                   :: time_text
      INTEGER                                            :: count_width, previous_progress
      REAL(KIND=dp)                                      :: percent_done, phase_time

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
   END SUBROUTINE write_ri_rs_phase_progress

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
   SUBROUTINE prepare_dlp_groups(comm, nl, bs_env, nt, nresult, integration_comm, field_comm, ngroups)
      CLASS(mp_comm_type), INTENT(IN) :: comm
      TYPE(post_scf_bandstructure_type), POINTER :: bs_env
      INTEGER(int_8), INTENT(IN) :: nresult
      INTEGER, INTENT(IN) :: nl, nt
      TYPE(mp_comm_type), INTENT(OUT) :: integration_comm, field_comm
      INTEGER, INTENT(OUT) :: ngroups

      INTEGER :: maxnp
      REAL(dp) :: contraction_bytes, selection_bytes, local_bytes, reduction_bytes
      INTEGER :: np, igroup, integration_rank, nlocal_T
      REAL(dp) :: result_bytes
      REAL(dp) :: available_GB, cell_bytes, required_bytes, cyclic_bytes, workspace_bytes

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
         ! The caller allocated target_R before this call. Its bytes are already
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
   END SUBROUTINE prepare_dlp_groups

! **************************************************************************************************
!> \brief Sum Eq. (2) directly onto ranks owning cyclic T cells, T=r+1+j*N.
!> The first two indices are (L,P); contiguous T panels are pointer views, with no packing copy.
!> Reduce-scatter retains only each rank's output cells instead of replicating the full sum.
!> \param comm ...
!> \param partial ...
!> \param cells ...
! **************************************************************************************************
   SUBROUTINE sum_dlp_cyclic_cells(comm, partial, cells)
      CLASS(mp_comm_type), INTENT(IN)                    :: comm
      REAL(dp), CONTIGUOUS, TARGET, INTENT(IN)            :: partial(:, :, :)
      REAL(dp), ALLOCATABLE, TARGET, INTENT(OUT)          :: cells(:, :, :)

      INTEGER :: nl, np, nt, nranks, rank, nlocal, icycle, first_T, last_T, &
                 round_first, round_last, first_rank, last_rank, panel_cells
      INTEGER(int_8) :: plane_count, iround
      INTEGER :: recv_counts(comm%num_pe)
      REAL(dp), POINTER, CONTIGUOUS :: send_panel(:, :), receive_panel(:)
      REAL(dp), TARGET :: empty(0)

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
      panel_cells = INT(MAX(1_int_8, (32_int_8*1024_int_8**2/8_int_8)/plane_count))
      panel_cells = MIN(panel_cells, INT(INT(HUGE(0), int_8)/plane_count))
      icycle = 0
      DO iround = 1_int_8, INT(nt, int_8), INT(nranks, int_8)
         icycle = icycle + 1
         round_first = INT(iround)
         round_last = INT(MIN(INT(nt, int_8), iround + INT(nranks, int_8) - 1_int_8))
         first_T = round_first
         DO WHILE (first_T <= round_last)
            last_T = INT(MIN(INT(round_last, int_8), INT(first_T, int_8) + INT(panel_cells, int_8) - 1_int_8))
            first_rank = first_T - round_first
            last_rank = last_T - round_first
            recv_counts = 0
            recv_counts(first_rank + 1:last_rank + 1) = INT(plane_count)
            send_panel(1:INT(plane_count), 1:last_T - first_T + 1) => partial(:, :, first_T:last_T)
            receive_panel => empty
            IF (rank >= first_rank .AND. rank <= last_rank) THEN
               receive_panel(1:INT(plane_count)) => cells(:, :, icycle)
            END IF
            CALL comm%sum_scatter(send_panel, receive_panel, recv_counts)
            IF (last_T == round_last) EXIT
            first_T = last_T + 1
         END DO
      END DO
   END SUBROUTINE sum_dlp_cyclic_cells

! **************************************************************************************************
!> \brief Redistribute Eq. (2) from cyclic T cells to disjoint contiguous l rows.
!> Rank r initially owns T=r+1+j*N. The caller owns first_l:first_l+size(target_R,1)-1.
!> Only the local T slice and local l slice are materialized, never a global DLP replica.
!> \param comm ...
!> \param cell_values ...
!> \param first_l ...
!> \param target_R ...
! **************************************************************************************************
   SUBROUTINE distribute_dlp_rows(comm, cell_values, first_l, target_R)
      CLASS(mp_comm_type), INTENT(IN)                    :: comm
      REAL(dp), INTENT(IN)                               :: cell_values(:, :, :)
      INTEGER, INTENT(IN)                                :: first_l
      REAL(dp), INTENT(OUT)                              :: target_R(:, :, :)

      INTEGER :: il, ip, irank, it, local_it, nl, np, nlocal_l, nlocal_T, nranks, nt, position
      INTEGER(int_8) :: count_wide
      INTEGER, ALLOCATABLE :: first_rows(:), row_counts(:), cell_counts(:), &
                              send_counts(:), send_offsets(:), recv_counts(:), recv_offsets(:)
      REAL(dp), ALLOCATABLE :: send_values(:), receive_values(:)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'distribute_dlp_rows'
      INTEGER :: handle

      CALL timeset(routineN, handle)

      nranks = comm%num_pe
      nl = SIZE(cell_values, 1)
      np = SIZE(target_R, 2)
      nlocal_l = SIZE(target_R, 1)
      nt = SIZE(target_R, 3)
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
                  target_R(il, ip, it) = receive_values(position)
               END DO
            END DO
         END DO
         CPASSERT(position == recv_offsets(irank) + recv_counts(irank))
      END DO

      CALL timestop(handle)

   END SUBROUTINE distribute_dlp_rows

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
!> Quadrature tiles are MPI distributed and each AO field is evaluated once. Bounded batches
!> share these fields within memory groups whose ranks divide T; independent groups divide g.
!> Group sums assemble disjoint cyclic T cells. A final all-to-all
!> redistributes the completed d(T) from T ownership to the caller's l ownership.
!> Each memory group holds one full partial d(T), subject to the memory budget.
!> No global AO-pair matrix is formed.
!> EPS_FILTER/10 controls the grid, finite box and screening threshold together.
!> \param bs_env Periodic band-structure environment.
!> \param grid_points RI-RS points r_l, indexed (3,l).
!> \param target_cells Retained integer lattice vectors T, indexed (3,T).
!> \param target_R Real right-hand side indexed (local l,P,T).
!> \param first_l Global index of the first locally owned l row.
!> \param qpoints ...
!> \param charge_lq ...
! **************************************************************************************************
   SUBROUTINE build_uniform_dlp(bs_env, grid_points, target_cells, target_R, first_l, &
                                qpoints, charge_lq)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(dp), INTENT(IN)                               :: grid_points(:, :)
      INTEGER, INTENT(IN)                                :: target_cells(:, :)
      REAL(dp), INTENT(OUT)                              :: target_R(:, :, :)
      INTEGER, INTENT(IN)                                :: first_l
      REAL(dp), INTENT(IN), OPTIONAL                     :: qpoints(:, :)
      COMPLEX(dp), INTENT(OUT), OPTIONAL                 :: charge_lq(:, :)

      TYPE(uniform_settings_type)                        :: settings

      CALL read_uniform_settings(bs_env, grid_points, settings)
      CALL integrate_uniform_tiles(bs_env, grid_points, target_cells, settings, target_R, first_l, &
                                   qpoints, charge_lq)
   END SUBROUTINE build_uniform_dlp

! **************************************************************************************************
!> \brief Accumulate Q_lq += w SUM_g f_l(r_g)^2 exp(i q.r_g) with real BLAS products.
!> The real field is the existing f_l(r_g)^2 on one quadrature tile. Cosine and sine
!> columns share a DGEMM; OpenMP partitions disjoint l rows without a reduction.
!> The grid, phase convention, weights, and retained quadrature points are unchanged.
!> \param points ...
!> \param field ...
!> \param q_cart ...
!> \param weight ...
!> \param charge_lq ...
! **************************************************************************************************
   SUBROUTINE accumulate_fourier_charge(points, field, q_cart, weight, charge_lq)
      REAL(dp), INTENT(IN)                               :: points(:, :)
      REAL(dp), CONTIGUOUS, INTENT(IN)                   :: field(:, :)
      REAL(dp), INTENT(IN)                               :: q_cart(:, :), weight
      COMPLEX(dp), INTENT(INOUT)                         :: charge_lq(:, :)

      INTEGER, PARAMETER                                 :: row_block = 64

      COMPLEX(dp)                                        :: phase
      INTEGER                                            :: first_l, ig, il, iq, nblock, ng, nl, nq
      REAL(dp), ALLOCATABLE                              :: charge_parts(:, :), phase_parts(:, :)

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
   SUBROUTINE read_uniform_settings(bs_env, grid_points, settings, report)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(dp), INTENT(IN)                               :: grid_points(:, :)
      TYPE(uniform_settings_type), INTENT(OUT)           :: settings
      LOGICAL, INTENT(IN), OPTIONAL                      :: report

      INTEGER(int_8)                                     :: npoints
      LOGICAL                                            :: report_settings
      REAL(dp)                                           :: exponent_max, exponent_min, &
                                                            padding_safety, spacing_prefactor

      settings%accuracy = dlp_accuracy(bs_env)
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
      CALL build_uniform_mesh(bs_env, grid_points, settings)
      npoints = PRODUCT(INT(settings%counts, int_8))
      report_settings = .TRUE.
      IF (PRESENT(report)) report_settings = report
      IF (bs_env%unit_nr > 0 .AND. report_settings) THEN
         WRITE (bs_env%unit_nr, '(T2,A,T72,ES9.1E2)') 'd_lP uniform quadrature points', REAL(npoints, dp)
         WRITE (bs_env%unit_nr, '(T2,A,T72,ES9.1)') 'd_lP integration threshold', settings%accuracy
         WRITE (bs_env%unit_nr, '(T2,A,T69,F12.3)') 'd_lP largest AO exponent [bohr^-2]', exponent_max
         WRITE (bs_env%unit_nr, '(T2,A,T69,F12.3)') 'd_lP uniform spacing [bohr]', settings%spacing
      END IF
   END SUBROUTINE read_uniform_settings

! **************************************************************************************************
!> \brief Build Eq. (3) on a cell-commensurate lattice without coarsening its largest spacing.
!> The integration parallelepiped contains the original padded Cartesian box.
!> \param bs_env Band-structure environment with RI-RS cell and atoms.
!> \param grid_points ...
!> \param settings ...
! **************************************************************************************************
   SUBROUTINE build_uniform_mesh(bs_env, grid_points, settings)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(dp), INTENT(IN)                               :: grid_points(:, :)
      TYPE(uniform_settings_type), INTENT(INOUT)         :: settings

      INTEGER                                            :: idir, subdivisions(3)
      REAL(dp)                                           :: box_center(3), box_extent(3), &
                                                            eigenvalues(3), inverse_step(3, 3), &
                                                            metric(3, 3), scale, upper(3)
      TYPE(cell_type), POINTER                           :: cell
      TYPE(particle_type), POINTER                       :: particles(:)

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
   END SUBROUTINE build_uniform_mesh

! **************************************************************************************************
!> \brief Partition the RI-RS points r_l into Cartesian cubes used by Eq. (4).
!> \param grid_points ...
!> \param edge ...
!> \param blocks ...
! **************************************************************************************************
   SUBROUTINE build_l_blocks(grid_points, edge, blocks)
      REAL(dp), INTENT(IN)                               :: grid_points(:, :), edge
      TYPE(l_block_partition_type), INTENT(OUT)          :: blocks

      INTEGER                                            :: cube(3), iblock, idir, il, insert
      INTEGER, ALLOCATABLE                               :: block_of_l(:), counts(:), cubes(:, :), &
                                                            next(:)
      REAL(dp)                                           :: origin(3)

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
   END SUBROUTINE build_l_blocks

! **************************************************************************************************
!> \brief Return the smallest and largest positive AO primitive exponents for the grid heuristic.
!> \param bs_env ...
!> \param exponent_min ...
!> \param exponent_max ...
! **************************************************************************************************
   SUBROUTINE ao_exponent_range(bs_env, exponent_min, exponent_max)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(dp), INTENT(OUT)                              :: exponent_min, exponent_max

      INTEGER                                            :: ikind, ipgf, iset
      REAL(dp)                                           :: exponent
      TYPE(gto_basis_set_type), POINTER                  :: basis

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
   SUBROUTINE get_dlp_target_cutoffs(bs_env, cutoffs)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(dp), ALLOCATABLE, INTENT(OUT)                 :: cutoffs(:)

      INTEGER                                            :: iatom, ikind
      REAL(dp)                                           :: accuracy, ao_max, ao_min, ri_max, ri_min
      TYPE(gto_basis_set_type), POINTER                  :: basis
      TYPE(particle_type), POINTER                       :: particles(:)

      particles => bs_env%ri_rs%particle_set
      accuracy = dlp_accuracy(bs_env)
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
   END SUBROUTINE get_dlp_target_cutoffs

! **************************************************************************************************
!> \brief Set the DLP threshold below the sparse-filter threshold.
!> \param bs_env ...
!> \return ...
! **************************************************************************************************
   FUNCTION dlp_accuracy(bs_env) RESULT(accuracy)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(dp)                                           :: accuracy

      IF (.NOT. ieee_is_finite(bs_env%eps_filter) .OR. &
          bs_env%eps_filter <= 0.0_dp .OR. bs_env%eps_filter >= 1.0_dp) THEN
         CPABORT('EPS_FILTER must be between zero and one for periodic RI-RS DLP')
      END IF
      accuracy = bs_env%eps_filter/10.0_dp
      IF (accuracy <= 0.0_dp) CPABORT('EPS_FILTER is too small for DLP quadrature')
   END FUNCTION dlp_accuracy

! **************************************************************************************************
!> \brief Return the smallest and largest positive primitive exponent of one Gaussian basis.
!> \param basis ...
!> \param exponent_min ...
!> \param exponent_max ...
! **************************************************************************************************
   SUBROUTINE basis_exponent_range(basis, exponent_min, exponent_max)
      TYPE(gto_basis_set_type), POINTER                  :: basis
      REAL(dp), INTENT(OUT)                              :: exponent_min, exponent_max

      INTEGER                                            :: ipgf, iset
      REAL(dp)                                           :: exponent

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
!> \brief Distribute integration tiles in Eq. (2), reusing fields and potentials within each tile.
!> \param bs_env ...
!> \param grid_points ...
!> \param target_cells ...
!> \param settings ...
!> \param target_R ...
!> \param first_l ...
!> \param qpoints ...
!> \param charge_lq ...
! **************************************************************************************************
   SUBROUTINE integrate_uniform_tiles(bs_env, grid_points, target_cells, settings, target_R, first_l, &
                                      qpoints, charge_lq)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(dp), INTENT(IN)                               :: grid_points(:, :)
      INTEGER, INTENT(IN)                                :: target_cells(:, :)
      TYPE(uniform_settings_type), INTENT(IN)            :: settings
      REAL(dp), INTENT(OUT)                              :: target_R(:, :, :)
      INTEGER, INTENT(IN)                                :: first_l
      REAL(dp), INTENT(IN), OPTIONAL                     :: qpoints(:, :)
      COMPLEX(dp), INTENT(OUT), OPTIONAL                 :: charge_lq(:, :)

      INTEGER, PARAMETER                                 :: tile_edge = 8, tile_size = tile_edge**3

      INTEGER :: counts(3), handle, iatom, ibatch, iblock, idir, ig, igroup, il, &
         indices(tile_size), integration_rank, irank, it, last_rank, next_progress, ngroups, &
         nkeep, nl, np, npoints, nranks, nreceived, position, rank, ranks_per_batch, &
         tile_counts(3), tile_first(3), tile_points(3)
      INTEGER(int_8) :: block_contractions, coalesced_contractions, contraction_products, &
         index_grid, iround, itile, n_completed_tiles, n_local_tiles, nrounds, ntiles, ntotal, &
         potential_points, screened_blocks
      INTEGER, ALLOCATABLE                               :: kept_per_rank(:), local_cells(:, :), &
                                                            recv_counts(:), recv_offsets(:)
      LOGICAL, ALLOCATABLE                               :: target_blocks(:, :, :), &
                                                            target_selection(:, :, :)
      REAL(dp) :: bound, discarded, elapsed, field_bound(tile_size), lower(3), mesh_index(3), &
         omitted, points(3, tile_size), prep_start, target_center(3), threshold, weight
      REAL(dp), ALLOCATABLE :: cyclic_targets(:, :, :), envelopes(:, :), local_targets(:, :, :), &
         q_cart(:, :), receive_fields(:), send_fields(:), target_cutoffs(:)
      REAL(dp) :: field(tile_size, SIZE(grid_points, 2))
      TYPE(ao_image_type), ALLOCATABLE                   :: images(:)
      TYPE(cell_type), POINTER                           :: cell
      TYPE(l_block_partition_type)                       :: blocks
      TYPE(mp_comm_type)                                 :: field_comm, integration_comm
      TYPE(particle_type), POINTER                       :: particles(:)
      TYPE(radial_interpolation_type), ALLOCATABLE       :: aux_potentials(:)

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
      CALL build_l_blocks(grid_points, settings%l_block_edge, blocks)
      bound = MAXVAL(envelopes(1, :))
      target_R = 0.0_dp
      CALL prepare_dlp_groups(bs_env%para_env, nl, bs_env, SIZE(target_cells, 2), SIZE(target_R, KIND=int_8), &
                              integration_comm, field_comm, ngroups)
      igroup = field_comm%mepos
      integration_rank = integration_comm%mepos
      nranks = field_comm%num_pe
      rank = field_comm%mepos
      np = bs_env%n_RI
      IF (bs_env%auto_ri%enabled) np = SUM(bs_env%auto_ri%sizes_opt_RI)
      ALLOCATE (local_cells(3, SIZE(target_cells(:, igroup + 1::ngroups), 2)))
      local_cells(:, :) = target_cells(:, igroup + 1::ngroups)

      CALL timeset('d_lP_prepare_target_geometry', handle)

      cell => bs_env%ri_rs%cell
      particles => bs_env%ri_rs%particle_set
      CALL get_dlp_target_cutoffs(bs_env, target_cutoffs)
      ALLOCATE (target_selection(nl, bs_env%n_atom, SIZE(local_cells, 2)))
      ALLOCATE (target_blocks(blocks%nblocks, bs_env%n_atom, SIZE(local_cells, 2)), source=.FALSE.)
      DO it = 1, SIZE(local_cells, 2)
         DO iatom = 1, bs_env%n_atom
            target_center = pbc(particles(iatom)%r, cell) + MATMUL(bs_env%hmat, REAL(local_cells(:, it), dp))
            DO il = 1, nl
               target_selection(il, iatom, it) = SUM((grid_points(:, il) - target_center)**2) <= target_cutoffs(iatom)**2
            END DO
            DO iblock = 1, blocks%nblocks
               IF (distance_to_box_squared(target_center, blocks%lower(:, iblock), &
                                           blocks%upper(:, iblock)) > target_cutoffs(iatom)**2) CYCLE
               target_blocks(iblock, iatom, it) = &
                  ANY(target_selection(blocks%indices(blocks%first(iblock):blocks%first(iblock + 1) - 1), iatom, it))
            END DO
         END DO
      END DO
      DEALLOCATE (target_cutoffs)
      IF (bs_env%unit_nr > 0) THEN
         FLUSH (bs_env%unit_nr)
      END IF

      CALL timestop(handle)

      ALLOCATE (local_targets(nl, np, SIZE(local_cells, 2)), source=0.0_dp)
      ALLOCATE (kept_per_rank(nranks), recv_counts(nranks), recv_offsets(nranks))
      ! At most 32 MiB of shared fields per rank, independent of the MPI size.
      ranks_per_batch = MAX(1, INT(32.0_dp*1024.0_dp**2/ &
                                   (8.0_dp*REAL(nl + 3, dp)*REAL(tile_size, dp))))
      ranks_per_batch = MIN(nranks, ranks_per_batch)
      ALLOCATE (send_fields((nl + 3)*tile_size), &
                receive_fields((nl + 3)*tile_size*ranks_per_batch))
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
      tile_counts = (counts + tile_edge - 1)/tile_edge
      ntiles = PRODUCT(INT(tile_counts, int_8))
      nrounds = (ntiles + INT(bs_env%para_env%num_pe, int_8) - 1_int_8)/ &
                INT(bs_env%para_env%num_pe, int_8)
      next_progress = 1
      n_local_tiles = nrounds
      threshold = 0.5_dp*settings%accuracy/(REAL(ntotal, dp)*weight)
      CALL screen_ao_images(images, settings%accuracy/ &
                            (8.0_dp*weight*REAL(ntotal, dp)*MAX(1.0_dp, bound)))
      DO iround = 0_int_8, nrounds - 1_int_8
         itile = iround*INT(bs_env%para_env%num_pe, int_8) + &
                 INT(bs_env%para_env%mepos, int_8)
         nkeep = 0
         IF (itile < ntiles) THEN
            ! Compact tiles preserve the quadrature while tightening all spatial bounds.
            index_grid = itile
            DO idir = 1, 3
               tile_first(idir) = tile_edge*INT(MODULO(index_grid, INT(tile_counts(idir), int_8)))
               tile_points(idir) = MIN(tile_edge, counts(idir) - tile_first(idir))
               index_grid = index_grid/INT(tile_counts(idir), int_8)
            END DO
            npoints = PRODUCT(tile_points)
            DO ig = 1, npoints
               index_grid = INT(ig - 1, int_8)
               DO idir = 1, 3
                  mesh_index(idir) = lower(idir) + &
                                     REAL(tile_first(idir) + INT(MODULO(index_grid, INT(tile_points(idir), int_8))), dp)
                  index_grid = index_grid/INT(tile_points(idir), int_8)
               END DO
               points(:, ig) = settings%origin + MATMUL(settings%step, mesh_index)
            END DO
            CALL timeset('d_lP_f_l_squared', handle)
            CALL evaluate_field_squared(images, points(:, 1:npoints), field(1:npoints, :), &
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
               CALL timeset('d_lP_integrate_tiles', handle)
               CALL integrate_potentials(bs_env, grid_points, local_cells, target_selection, target_blocks, &
                                         points(:, 1:nkeep), field(1:nkeep, :), &
                                         weight, envelopes, blocks, threshold, omitted, potential_points, &
                                         block_contractions, screened_blocks, coalesced_contractions, &
                                         contraction_products, local_targets, aux_potentials)
               CALL timestop(handle)
            END DO
         END DO
         n_completed_tiles = iround + 1_int_8
         CALL write_ri_rs_phase_progress(bs_env%unit_nr, 'd_lP Integration batch done', &
                                         n_completed_tiles, n_local_tiles, next_progress, elapsed)
      END DO

      CALL timeset('d_lP_uniform_reduce', handle)

      DEALLOCATE (send_fields, receive_fields, kept_per_rank, recv_counts, recv_offsets)
      CALL apply_target_cutoff(bs_env, grid_points, local_cells, local_targets)
      CALL sum_dlp_cyclic_cells(integration_comm, local_targets, cyclic_targets)
      DEALLOCATE (local_targets, local_cells)
      CALL distribute_dlp_rows(bs_env%para_env, cyclic_targets, first_l, target_R)
      DEALLOCATE (cyclic_targets)
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
   END SUBROUTINE integrate_uniform_tiles

! **************************************************************************************************
!> \brief Form each atom-local auxiliary potential's spherical coefficients just once.
!> \param bs_env ...
!> \param settings ...
!> \param aux_potentials ...
! **************************************************************************************************
   SUBROUTINE prepare_auto_ri_potentials(bs_env, settings, aux_potentials)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(uniform_settings_type), INTENT(IN)            :: settings
      TYPE(radial_interpolation_type), ALLOCATABLE, &
         INTENT(OUT)                                     :: aux_potentials(:)

      CHARACTER(LEN=12)                                  :: spacing_text
      INTEGER                                            :: first_U, iatom, ikind, nmesh, np, nref
      REAL(dp)                                           :: cutoff, mesh_bytes, &
                                                            mesh_displacement(3), mesh_spacing
      REAL(dp), ALLOCATABLE                              :: U_Pp(:, :)
      TYPE(cell_type), POINTER                           :: cell
      TYPE(gto_basis_set_type), POINTER                  :: basis
      TYPE(particle_type), POINTER                       :: particles(:)

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
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(dp), INTENT(IN)                               :: grid_points(:, :)
      TYPE(ao_image_type), ALLOCATABLE, INTENT(OUT)      :: images(:)

      INTEGER                                            :: iatom, icell, iimage, ikind, maxao, nao
      REAL(dp)                                           :: center(3), cutoff2
      REAL(dp), ALLOCATABLE                              :: primary(:, :)
      TYPE(cell_type), POINTER                           :: cell
      TYPE(gto_basis_set_type), POINTER                  :: basis
      TYPE(particle_type), POINTER                       :: particles(:)

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
      TYPE(ao_image_type), INTENT(INOUT)                 :: image

      INTEGER                                            :: il, ipacked, nkeep, nl
      LOGICAL, ALLOCATABLE                               :: nonzero(:)
      REAL(dp), ALLOCATABLE                              :: primary(:, :)

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
      TYPE(ao_image_type), INTENT(INOUT)                 :: image

      INTEGER                                            :: ico, idir, il, ipgf, irow, iset, isgf, &
                                                            ncart, power
      REAL(dp)                                           :: amplitude, exponent, factor
      REAL(dp), ALLOCATABLE                              :: ao_bound(:)
      TYPE(gto_basis_set_type), POINTER                  :: basis

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
      TYPE(ao_image_type), INTENT(INOUT)                 :: image

      INTEGER                                            :: ico, idegree, idir, iexponent, il, ipgf, &
                                                            irow, iset, isgf, ncart, power
      REAL(dp)                                           :: amplitude, angular_factor
      REAL(dp), ALLOCATABLE                              :: ao_weight(:)
      TYPE(gto_basis_set_type), POINTER                  :: basis

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
      TYPE(ao_image_type), INTENT(IN)                    :: image
      REAL(dp), INTENT(IN)                               :: distance_squared
      REAL(dp)                                           :: bound

      INTEGER                                            :: idegree, iexponent, lmax
      REAL(dp)                                           :: exponent, polynomial, radius, &
                                                            radius_squared

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
!> Each omitted image contributes at most delta/N; box distances bound every point in a tile.
!> \param images ...
!> \param tolerance ...
! **************************************************************************************************
   SUBROUTINE screen_ao_images(images, tolerance)
      TYPE(ao_image_type), INTENT(INOUT)                 :: images(:)
      REAL(dp), INTENT(IN)                               :: tolerance

      INTEGER                                            :: iimage, nimages
      REAL(dp)                                           :: amplitude, delta

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
!> \brief Evaluate the complete squared AO field in Eq. (2) on one Cartesian tile.
!> \param images ...
!> \param points ...
!> \param field_squared ...
!> \param field_accuracy ...
! **************************************************************************************************
   SUBROUTINE evaluate_field_squared(images, points, field_squared, field_accuracy)
      TYPE(ao_image_type), INTENT(IN)                    :: images(:)
      REAL(dp), INTENT(IN)                               :: points(:, :)
      REAL(dp), INTENT(OUT)                              :: field_squared(:, :)
      REAL(dp), INTENT(IN), OPTIONAL                     :: field_accuracy

      INTEGER                                            :: first, last, nblock, ng, nthreads
      REAL(dp)                                           :: tolerance

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
         CALL evaluate_field_squared_block(images, points(:, first:last), field_squared(first:last, :), tolerance)
      END DO
      !$OMP END PARALLEL DO
   END SUBROUTINE evaluate_field_squared

! **************************************************************************************************
!> \brief Evaluate Eq. (1), then square, with a tile-local bound on the squared-field error.
!> For image bounds b_i and A=sum_i b_i, omit at most delta=sqrt(A²+epsilon)-A in amplitude;
!> then 2A*delta+delta²<=epsilon. Whole tiles with A²<=epsilon need no AO evaluation.
!> The legacy radius bound applies when no positive squared-field accuracy is supplied.
!> \param images ...
!> \param points ...
!> \param field_squared ...
!> \param field_accuracy ...
! **************************************************************************************************
   SUBROUTINE evaluate_field_squared_block(images, points, field_squared, field_accuracy)
      TYPE(ao_image_type), INTENT(IN)                    :: images(:)
      REAL(dp), INTENT(IN)                               :: points(:, :)
      REAL(dp), INTENT(OUT)                              :: field_squared(:, :)
      REAL(dp), INTENT(IN), OPTIONAL                     :: field_accuracy

      INTEGER                                            :: handle, iimage, il, nao, ng, nimages, &
                                                            nl, nprimary
      REAL(dp)                                           :: amplitude, delta, lower(3), &
                                                            root_tolerance, tolerance, upper(3)
      REAL(dp), ALLOCATABLE                              :: field(:, :), image_field(:, :), &
                                                            values(:, :)
      REAL(dp)                                           :: image_bounds(SIZE(images))

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
         CALL evaluate_uniform_ao_values(values, points, images(iimage)%basis, images(iimage)%center)
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
   END SUBROUTINE evaluate_field_squared_block

! **************************************************************************************************
!> \brief Evaluate the AO values in Eq. (1), sharing primitive factors before contraction.
!> Columns of primitive are Cartesian Gaussian primitives, grouped by basis set and exponent.
!> The original spherical contraction coefficients define exactly the same unscreened AO values.
!> \param values ...
!> \param points ...
!> \param basis ...
!> \param center ...
! **************************************************************************************************
   SUBROUTINE evaluate_uniform_ao_values(values, points, basis, center)
      REAL(dp), INTENT(OUT)                              :: values(:, :)
      REAL(dp), INTENT(IN)                               :: points(:, :)
      TYPE(gto_basis_set_type), POINTER                  :: basis
      REAL(dp), INTENT(IN)                               :: center(3)

      INTEGER                                            :: first_sgf, ico, ig, ipgf, ipower, iset, &
                                                            last_sgf, ncart, ng, nprimitive
      INTEGER, ALLOCATABLE                               :: offsets(:)
      LOGICAL                                            :: parallel_points
      REAL(dp)                                           :: relative(3)
      REAL(dp), ALLOCATABLE                              :: coordinate_power(:, :), exponential(:), &
                                                            monomial(:, :), primitive(:, :), &
                                                            radius2(:)

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
   END SUBROUTINE evaluate_uniform_ao_values

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
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      TYPE(radial_interpolation_type), ALLOCATABLE, &
         INTENT(IN)                                      :: aux_potentials(:)
      REAL(dp), ALLOCATABLE, INTENT(OUT)                 :: envelopes(:, :)

      INTEGER                                            :: iatom, ico, idir, ikind, ipgf, irow, &
                                                            irow_packed, iset, isgf, ishell, l, &
                                                            ncart, power
      LOGICAL, ALLOCATABLE                               :: angular_used(:)
      REAL(dp)                                           :: exponent, factor
      REAL(dp), ALLOCATABLE                              :: bounds(:)
      TYPE(gto_basis_set_type), POINTER                  :: basis
      TYPE(particle_type), POINTER                       :: particles(:)

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
!> \param target_cells ...
!> \param target_selection ...
!> \param target_blocks ...
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
!> \param target_R ...
!> \param aux_potentials ...
! **************************************************************************************************
   SUBROUTINE integrate_potentials(bs_env, grid_points, target_cells, target_selection, target_blocks, &
                                   points, field_squared, weight, &
                                   envelopes, blocks, threshold, omitted, potential_points, block_contractions, &
                                   screened_blocks, coalesced_contractions, contraction_products, target_R, &
                                   aux_potentials)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(dp), INTENT(IN)                               :: grid_points(:, :)
      INTEGER, INTENT(IN)                                :: target_cells(:, :)
      LOGICAL, INTENT(IN)                                :: target_selection(:, :, :), &
                                                            target_blocks(:, :, :)
      REAL(dp), INTENT(IN)                               :: points(:, :), field_squared(:, :), &
                                                            weight, envelopes(:, :)
      TYPE(l_block_partition_type), INTENT(IN)           :: blocks
      REAL(dp), INTENT(IN)                               :: threshold
      REAL(dp), INTENT(INOUT)                            :: omitted
      INTEGER(int_8), INTENT(INOUT)                      :: potential_points, block_contractions, &
                                                            screened_blocks, &
                                                            coalesced_contractions, &
                                                            contraction_products
      REAL(dp), INTENT(INOUT)                            :: target_R(:, :, :)
      TYPE(radial_interpolation_type), ALLOCATABLE, &
         INTENT(IN)                                      :: aux_potentials(:)

      INTEGER                                            :: firstp, handle, iatom, iblock, ig, &
                                                            ikind, ilocal, it, lastp, maxnp, &
                                                            nactive, nkeep, np, nthreads, nunion, &
                                                            row
      REAL(dp)                                           :: center(3), cutoff, decay, distance, &
                                                            estimate
      REAL(dp), ALLOCATABLE                              :: block_field_bound(:, :), &
                                                            block_target(:, :), field_work(:, :), &
                                                            point_bound(:), potential(:, :)
      LOGICAL                                            :: point_needed(SIZE(points, 2))
      INTEGER                                            :: union_indices(SIZE(points, 2)), &
                                                            active_l(SIZE(grid_points, 2))
      TYPE(cell_type), POINTER                           :: cell
      TYPE(gto_basis_set_type), POINTER                  :: basis
      LOGICAL                                            :: block_needed(blocks%nblocks)
      INTEGER :: block_active_last(blocks%nblocks), block_active_first(blocks%nblocks)
      TYPE(particle_type), POINTER                       :: particles(:)

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
      !$OMP SHARED(bs_env, grid_points, target_cells, points, field_squared, weight, envelopes, &
      !$OMP blocks, threshold, target_R, aux_potentials, cell, particles, cutoff, &
      !$OMP maxnp, block_field_bound, target_selection, target_blocks) &
      !$OMP PRIVATE(firstp, handle, iatom, iblock, ig, ikind, ilocal, it, lastp, nactive, nkeep, &
      !$OMP np, nunion, row, center, decay, distance, estimate, basis, &
      !$OMP block_target, field_work, point_bound, potential, point_needed, union_indices, &
      !$OMP active_l, block_needed, block_active_last, block_active_first) &
      !$OMP REDUCTION(+:omitted, potential_points, block_contractions, screened_blocks, &
      !$OMP coalesced_contractions, contraction_products)

      CALL timeset('d_lP_allocate_thread_buffers', handle)

      ALLOCATE (block_target(SIZE(grid_points, 2), maxnp), &
                field_work(SIZE(points, 2), SIZE(grid_points, 2)), point_bound(SIZE(points, 2)), &
                potential(SIZE(points, 2), maxnp))

      CALL timestop(handle)

      !$OMP DO COLLAPSE(2) SCHEDULE(DYNAMIC)
      DO it = 1, SIZE(target_cells, 2)
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
            center = pbc(particles(iatom)%r, cell) + MATMUL(bs_env%hmat, REAL(target_cells(:, it), dp))
            IF (.NOT. ANY(target_blocks(:, iatom, it))) CYCLE
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
               IF (.NOT. target_blocks(iblock, iatom, it)) CYCLE
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
               CALL evaluate_uniform_potential(potential(:, 1:np), &
                                               points(:, union_indices(1:nunion)), &
                                               basis, center, cutoff, aux_potentials(iatom), &
                                               attenuation=MAX(0.0_dp, bs_env%ri_metric%omega))
            ELSE
               CALL evaluate_uniform_potential(potential(1:nunion, 1:np), &
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
                  IF (target_selection(blocks%indices(ilocal), iatom, it)) THEN
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
                       potential, SIZE(potential, 1), 0.0_dp, block_target, SIZE(block_target, 1))
            DO ilocal = 1, nactive
               target_R(active_l(ilocal), firstp:lastp, it) = &
                  target_R(active_l(ilocal), firstp:lastp, it) + block_target(ilocal, 1:np)
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

      DEALLOCATE (block_target, field_work, point_bound, potential)

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
      REAL(dp), INTENT(IN)                               :: point(3), lower(3), upper(3)
      REAL(dp)                                           :: distance_squared

      INTEGER                                            :: idir
      REAL(dp)                                           :: delta

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
!> \param target_cells ...
!> \param target_R ...
! **************************************************************************************************
   SUBROUTINE apply_target_cutoff(bs_env, grid_points, target_cells, target_R)
      TYPE(post_scf_bandstructure_type), POINTER         :: bs_env
      REAL(dp), INTENT(IN)                               :: grid_points(:, :)
      INTEGER, INTENT(IN)                                :: target_cells(:, :)
      REAL(dp), INTENT(INOUT)                            :: target_R(:, :, :)

      INTEGER                                            :: firstp, iatom, il, it, lastp
      REAL(dp)                                           :: center(3), cutoff
      REAL(dp), ALLOCATABLE                              :: target_cutoffs(:)
      TYPE(cell_type), POINTER                           :: cell
      TYPE(particle_type), POINTER                       :: particles(:)

      cell => bs_env%ri_rs%cell
      particles => bs_env%ri_rs%particle_set
      CALL get_dlp_target_cutoffs(bs_env, target_cutoffs)
      DO it = 1, SIZE(target_cells, 2)
         DO iatom = 1, bs_env%n_atom
            center = pbc(particles(iatom)%r, cell) + MATMUL(bs_env%hmat, REAL(target_cells(:, it), dp))
            firstp = bs_env%i_RI_start_from_atom(iatom)
            lastp = bs_env%i_RI_end_from_atom(iatom)
            IF (bs_env%auto_ri%enabled) THEN
               firstp = 1 + SUM(bs_env%auto_ri%sizes_opt_RI(:iatom - 1))
               lastp = firstp + bs_env%auto_ri%sizes_opt_RI(iatom) - 1
            END IF
            cutoff = target_cutoffs(iatom)
            DO il = 1, SIZE(grid_points, 2)
               IF (NORM2(grid_points(:, il) - center) > cutoff) target_R(il, firstp:lastp, it) = 0.0_dp
            END DO
         END DO
      END DO
      DEALLOCATE (target_cutoffs)
   END SUBROUTINE apply_target_cutoff

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
   SUBROUTINE evaluate_uniform_potential(values, points, basis, center, cutoff, contracted, attenuation)
      REAL(dp), CONTIGUOUS, INTENT(OUT)                  :: values(:, :)
      REAL(dp), INTENT(IN)                               :: points(:, :)
      TYPE(gto_basis_set_type), POINTER                  :: basis
      REAL(dp), INTENT(IN)                               :: center(3), cutoff
      TYPE(radial_interpolation_type), INTENT(IN), &
         OPTIONAL                                        :: contracted
      REAL(dp), INTENT(IN), OPTIONAL                     :: attenuation

      CPASSERT(ASSOCIATED(basis))
      CPASSERT(SIZE(points, 1) == 3)
      IF (PRESENT(contracted)) THEN
         CPASSERT(SIZE(values, 1) >= SIZE(points, 2))
      ELSE
         CPASSERT(SIZE(values, 1) == SIZE(points, 2))
      END IF
      CALL contract_potentials(values, points, basis, center, cutoff, contracted, attenuation)
   END SUBROUTINE evaluate_uniform_potential

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
      TYPE(gto_basis_set_type), POINTER                  :: basis
      REAL(dp), INTENT(IN), OPTIONAL                     :: U_Pp(:, :)
      TYPE(radial_interpolation_type), INTENT(OUT)       :: contracted
      REAL(dp), INTENT(IN), OPTIONAL                     :: cutoff, attenuation, radial_table_bytes, &
                                                            mesh_step(3, 3), mesh_displacement(3)

      INTEGER                                            :: ico, iopt, ipgf, irow, irow_packed, &
                                                            iset, isgf, ishell, l, ncart, &
                                                            ninterval, np, nrows
      LOGICAL, ALLOCATABLE                               :: angular_used(:)
      REAL(dp)                                           :: bytes_per_node, table_bytes
      REAL(dp), ALLOCATABLE                              :: full_abscissae(:), moments(:, :)

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
      TYPE(gto_basis_set_type), POINTER                  :: basis
      REAL(dp), INTENT(IN)                               :: U_Pp(:, :)
      TYPE(radial_interpolation_type), INTENT(INOUT)     :: contracted

      INTEGER                                            :: c, first, ico, ipgf, iset, ishell, l, &
                                                            lmax, m, ncart, ref_c, ref_m, ref_p, &
                                                            ref_shell, row
      LOGICAL, ALLOCATABLE                               :: used(:)
      REAL(dp)                                           :: actual, expected, ref_abs, ref_value, &
                                                            scale
      REAL(dp), ALLOCATABLE                              :: angular_coeff(:, :, :, :), &
                                                            primitive_scale(:, :, :)

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
      TYPE(gto_basis_set_type), POINTER                  :: basis
      REAL(dp), INTENT(IN)                               :: cutoff, omega
      REAL(dp), ALLOCATABLE, INTENT(OUT)                 :: series(:, :, :), short_series(:, :, :)

      INTEGER                                            :: in, ipgf, iset, lmax
      REAL(dp)                                           :: beta_ratio, exponent

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
      REAL(dp), CONTIGUOUS, INTENT(OUT)                  :: values(:, :)
      REAL(dp), INTENT(IN)                               :: points(:, :)
      TYPE(gto_basis_set_type), POINTER                  :: basis
      REAL(dp), INTENT(IN)                               :: center(3), cutoff
      TYPE(radial_interpolation_type), INTENT(IN), &
         OPTIONAL                                        :: contracted
      REAL(dp), INTENT(IN), OPTIONAL                     :: attenuation

      INTEGER                                            :: ico, in, ipgf, ipoint, irow, iset, isgf, &
                                                            ishell, l, lmax, ncart
      LOGICAL                                            :: parallel_points, short
      REAL(dp)                                           :: beta_ratio, exponent, minimum_radius2, &
                                                            omega, polynomial, radius, &
                                                            relative(3), weight
      REAL(dp), ALLOCATABLE                              :: long_radial(:), radial(:), &
                                                            series(:, :, :), short_series(:, :, :)
      TYPE(radial_work_type)                             :: radial_work

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
      REAL(dp), CONTIGUOUS, INTENT(OUT)                  :: values(:, :)
      REAL(dp), INTENT(IN)                               :: points(:, :)
      TYPE(gto_basis_set_type), POINTER                  :: basis
      REAL(dp), INTENT(IN)                               :: center(3), cutoff
      TYPE(radial_interpolation_type), INTENT(IN), &
         TARGET                                          :: contracted
      REAL(dp), INTENT(IN), OPTIONAL                     :: attenuation

      INTEGER                                            :: handle, iangular, ico, ig, imesh, in, &
                                                            ipgf, irow, iset, isgf, ishell, l, &
                                                            lmax, m, nchannels, ng, np, nrows, &
                                                            nwork
      INTEGER, ALLOCATABLE                               :: angular_count(:, :, :), &
                                                            angular_index(:), &
                                                            angular_term(:, :, :, :), &
                                                            mesh_channel(:), minimum_l(:)
      LOGICAL                                            :: factorized, in_table, parallel_points, &
                                                            prepared, short, tabulated, use_mesh
      LOGICAL, ALLOCATABLE                               :: angular_used(:, :)
      REAL(dp) :: beta_ratio, exponent, mesh_delta, mesh_position, minimum_radius2, omega, &
         radial_sum, radius, radius2, relative(3), spherical_value, x
      REAL(dp), ALLOCATABLE :: angular_values(:, :, :), coordinate_power(:, :), long_radial(:), &
         monomial(:), radial(:), radial_channels(:, :, :), radial_mesh(:), work(:, :)
      REAL(dp), CONTIGUOUS, POINTER                      :: coefficient_matrix(:, :)
      REAL(dp), POINTER                                  :: series(:, :, :), short_series(:, :, :)
      TYPE(radial_work_type)                             :: radial_work

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
      REAL(dp), INTENT(IN)                               :: radius, exponent, omega, &
                                                            coefficients(0:), &
                                                            short_coefficients(0:)
      REAL(dp), INTENT(OUT)                              :: radial(0:)
      TYPE(radial_work_type), INTENT(INOUT)              :: radial_work
      REAL(dp), INTENT(INOUT)                            :: long_radial(0:)

      INTEGER                                            :: il
      REAL(dp)                                           :: beta, beta_ratio, factor, lower_exp, &
                                                            ratio_power, root_x, upper_exp, x

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
      REAL(dp), INTENT(IN)                               :: exponent, cutoff
      REAL(dp), INTENT(OUT)                              :: coefficients(0:)

      INTEGER                                            :: ij, ik, in
      REAL(dp) :: factorial, gamma_p, integral, term, weight, y, terms(0:UBOUND(coefficients, 1))

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
      REAL(dp), INTENT(IN)                               :: radius, exponent, cutoff, &
                                                            coefficients(0:)
      REAL(dp), INTENT(OUT)                              :: radial(0:)
      TYPE(radial_work_type), INTENT(INOUT)              :: radial_work

      INTEGER                                            :: ij, il, in, lmax
      REAL(dp)                                           :: displacement, factor, power, prefactor, &
                                                            value, x

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
      REAL(dp), INTENT(IN)                               :: xmax
      REAL(dp), ALLOCATABLE, INTENT(OUT)                 :: abscissae(:)

      INTEGER                                            :: ix, last

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
      REAL(dp), INTENT(IN)                               :: a, omega
      INTEGER, INTENT(IN)                                :: lmax
      REAL(dp), ALLOCATABLE, INTENT(OUT)                 :: table(:, :)
      INTEGER, INTENT(IN), OPTIONAL                      :: ninterval
      REAL(dp), INTENT(IN), OPTIONAL                     :: point_spacing, point_offset
      INTEGER, INTENT(IN), OPTIONAL                      :: last_moment
      REAL(dp), INTENT(IN), OPTIONAL                     :: abscissae(0:)

      INTEGER                                            :: ig, ij, il, iter, ix, last, order, &
                                                            quadrature_last
      REAL(dp)                                           :: b, derivative, dz, factor, lower, &
                                                            lower_exp, offset, p0, p1, p2, &
                                                            ratio_power, root_x, spacing, t, t2, &
                                                            term, upper_exp, w, width, x, z

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
      REAL(dp), INTENT(IN)                               :: x, abscissae(0:), table(0:, 0:)
      REAL(dp), INTENT(OUT)                              :: radial(0:)

      INTEGER::ix, il, ik, ileft, iright, imiddle
      REAL(dp)::delta, value
      REAL(dp), PARAMETER :: inverse(6) = [1.0_dp, 0.5_dp, 1.0_dp/3.0_dp, 0.25_dp, 0.2_dp, 1.0_dp/6.0_dp]
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
      TYPE(gto_basis_set_type), POINTER                  :: basis
      TYPE(radial_interpolation_type), INTENT(INOUT)     :: contracted
      REAL(dp), INTENT(IN)                               :: step(3, 3), displacement(3)
      REAL(dp), INTENT(INOUT)                            :: table_bytes

      INTEGER :: idir, ifirst, il, ipgf, iset, ishell, jdir, last, max_l, max_primitive, &
         minimum_l, nchannels, nshell_total, nstored, shell_index
      LOGICAL                                            :: direct_shells
      REAL(dp)                                           :: bytes_per_node, index_bytes, linear(3), &
                                                            metric(3, 3), offset, ratio, &
                                                            required_bytes, spacing
      REAL(dp), ALLOCATABLE                              :: moments(:, :)

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
!> \param target_rank ...
!> \param source ...
!> \param TARGET ...
!> Transfer a matrix between world owners without a full-size communication buffer.
! **************************************************************************************************
   SUBROUTINE grouped_w_copy_matrix(world, n, source_rank, target_rank, source, TARGET)
      TYPE(mp_para_env_type), INTENT(INOUT)              :: world
      INTEGER, INTENT(IN)                                :: n, source_rank, target_rank
      COMPLEX(KIND=dp), INTENT(IN)                       :: source(:, :)
      COMPLEX(KIND=dp), INTENT(INOUT)                    :: target(:, :)

      INTEGER                                            :: first, last, sender, tag

      IF (source_rank == target_rank) THEN
         IF (world%mepos == source_rank) TARGET = source
         RETURN
      END IF
      DO first = 1, n, tile_cols
         last = MIN(first + tile_cols - 1, n)
         IF (world%mepos == source_rank) THEN
            CALL world%send(source(:, first:last), target_rank, 0)
         ELSE IF (world%mepos == target_rank) THEN
            sender = source_rank
            tag = 0
            CALL world%recv(TARGET(:, first:last), sender, tag)
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
