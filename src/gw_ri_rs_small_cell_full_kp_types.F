!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Work data of periodic RI-RS GW with full k-point sampling.
!> \author Jan Wilhelm
! **************************************************************************************************
MODULE gw_ri_rs_small_cell_full_kp_types
   USE cp_dbcsr_api,                    ONLY: dbcsr_p_type
   USE kinds,                           ONLY: dp,&
                                              int_8
   USE kpoint_methods,                  ONLY: kp_transform_plan_type
   USE message_passing,                 ONLY: mp_para_env_type
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'gw_ri_rs_small_cell_full_kp_types'

   PUBLIC :: bounded_transfer_type, gamma_ao_time_state_type, gamma_atom_projection_type, &
             gamma_correction_type, gwkp_type, pair_transfer_plan_type, &
             projection_interpolation_type, ri_q_matrix_block_type

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

END MODULE gw_ri_rs_small_cell_full_kp_types
