!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Routines to compute the Coulomb integral V_(alpha beta)(k) for a k-point k using lattice
!>        summation in real space. These integrals are e.g. needed in periodic RI for RPA, GW
!> \par History
!>       2018.05 created [Jan Wilhelm]
!> \author Jan Wilhelm
! **************************************************************************************************
MODULE kpoint_coulomb_2c
   USE atomic_kind_types,               ONLY: atomic_kind_type,&
                                              get_atomic_kind_set
   USE basis_set_types,                 ONLY: gto_basis_set_type
   USE cell_types,                      ONLY: cell_type,&
                                              get_cell,&
                                              pbc
   USE constants_operator,              ONLY: operator_coulomb,&
                                              operator_verfc
   USE cp_dbcsr_api,                    ONLY: &
        dbcsr_create, dbcsr_init_p, dbcsr_iterator_blocks_left, dbcsr_iterator_next_block, &
        dbcsr_iterator_start, dbcsr_iterator_stop, dbcsr_iterator_type, dbcsr_p_type, &
        dbcsr_release_p, dbcsr_set, dbcsr_type, dbcsr_type_no_symmetry
   USE cp_dbcsr_contrib,                ONLY: dbcsr_reserve_all_blocks
   USE generic_shg_integrals,           ONLY: int_operators_r12_ab_shg
   USE generic_shg_integrals_init,      ONLY: contraction_matrix_shg
   USE kinds,                           ONLY: dp
   USE kpoint_types,                    ONLY: get_kpoint_info,&
                                              kpoint_type
   USE mathconstants,                   ONLY: fourpi,&
                                              gaussi,&
                                              pi,&
                                              twopi,&
                                              z_one,&
                                              z_zero
   USE mathlib,                         ONLY: gauss_legendre,&
                                              vector_product
   USE message_passing,                 ONLY: mp_para_env_type
   USE orbital_pointers,                ONLY: indco,&
                                              ncoset
   USE particle_types,                  ONLY: particle_type
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type
   USE qs_kind_types,                   ONLY: get_qs_kind,&
                                              qs_kind_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'kpoint_coulomb_2c'

   PUBLIC :: build_2c_coulomb_matrix_kp, build_2c_coulomb_matrix_kp_small_cell

! **************************************************************************************************

   TYPE two_d_util_type
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)  :: block
   END TYPE two_d_util_type

   ! Numerical Ewald splitting and conservative Gaussian-tail cutoff heuristics.
   REAL(KIND=dp), PARAMETER :: ewald_omega_default = 0.3_dp, &
                               ewald_tail_exponent = 50.0_dp, ewald_lmax_margin = 4.0_dp

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param matrix_v_kp ...
!> \param kpoints ...
!> \param basis_type ...
!> \param cell ...
!> \param particle_set ...
!> \param qs_kind_set ...
!> \param atomic_kind_set ...
!> \param size_lattice_sum ...
!> \param operator_type ...
!> \param ikp_start ...
!> \param ikp_end ...
! **************************************************************************************************
   SUBROUTINE build_2c_coulomb_matrix_kp(matrix_v_kp, kpoints, basis_type, cell, particle_set, qs_kind_set, &
                                         atomic_kind_set, size_lattice_sum, operator_type, ikp_start, ikp_end)

      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: matrix_v_kp
      TYPE(kpoint_type), POINTER                         :: kpoints
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type
      TYPE(cell_type), POINTER                           :: cell
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      INTEGER                                            :: size_lattice_sum, operator_type, &
                                                            ikp_start, ikp_end

      CHARACTER(LEN=*), PARAMETER :: routineN = 'build_2c_coulomb_matrix_kp'

      INTEGER                                            :: handle
      TYPE(dbcsr_type), POINTER                          :: matrix_v_L_tmp

      CALL timeset(routineN, handle)

      CALL allocate_tmp(matrix_v_L_tmp, matrix_v_kp, ikp_start)

      CALL lattice_sum(matrix_v_kp, kpoints, basis_type, cell, particle_set, &
                       qs_kind_set, atomic_kind_set, size_lattice_sum, matrix_v_L_tmp, &
                       operator_type, ikp_start, ikp_end)

      CALL deallocate_tmp(matrix_v_L_tmp)

      CALL timestop(handle)

   END SUBROUTINE build_2c_coulomb_matrix_kp

! **************************************************************************************************
!> \brief ...
!> \param matrix_v_kp ...
!> \param kpoints ...
!> \param basis_type ...
!> \param cell ...
!> \param particle_set ...
!> \param qs_kind_set ...
!> \param atomic_kind_set ...
!> \param size_lattice_sum ...
!> \param matrix_v_L_tmp ...
!> \param operator_type ...
!> \param ikp_start ...
!> \param ikp_end ...
! **************************************************************************************************
   SUBROUTINE lattice_sum(matrix_v_kp, kpoints, basis_type, cell, particle_set, &
                          qs_kind_set, atomic_kind_set, size_lattice_sum, matrix_v_L_tmp, &
                          operator_type, ikp_start, ikp_end)

      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: matrix_v_kp
      TYPE(kpoint_type), POINTER                         :: kpoints
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type
      TYPE(cell_type), POINTER                           :: cell
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      INTEGER                                            :: size_lattice_sum
      TYPE(dbcsr_type), POINTER                          :: matrix_v_L_tmp
      INTEGER                                            :: operator_type, ikp_start, ikp_end

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'lattice_sum'

      INTEGER :: factor, handle, handle2, i_block, i_x, i_x_inner, i_x_outer, ik, j_y, j_y_inner, &
         j_y_outer, k_z, k_z_inner, k_z_outer, x_max, x_min, y_max, y_min, z_max, z_min
      INTEGER, DIMENSION(3)                              :: nkp_grid
      REAL(KIND=dp)                                      :: coskl, sinkl
      REAL(KIND=dp), DIMENSION(3)                        :: vec_L, vec_s
      REAL(KIND=dp), DIMENSION(3, 3)                     :: hmat
      TYPE(two_d_util_type), ALLOCATABLE, DIMENSION(:)   :: blocks_v_L, blocks_v_L_store
      TYPE(two_d_util_type), ALLOCATABLE, &
         DIMENSION(:, :, :)                              :: blocks_v_kp

      CALL timeset(routineN, handle)

      CALL get_factor_and_xyz_min_max(cell, kpoints, size_lattice_sum, factor, hmat, &
                                      x_min, x_max, y_min, y_max, z_min, z_max, nkp_grid)

      CALL allocate_blocks_v_kp(blocks_v_kp, matrix_v_kp, ikp_start, ikp_end)
      CALL allocate_blocks_v_L(blocks_v_L, matrix_v_L_tmp)
      CALL allocate_blocks_v_L(blocks_v_L_store, matrix_v_L_tmp)

      DO i_x_inner = 0, 2*nkp_grid(1) - 1
         DO j_y_inner = 0, 2*nkp_grid(2) - 1
            DO k_z_inner = 0, 2*nkp_grid(3) - 1

               DO i_x_outer = x_min, x_max + nkp_grid(1), 2*nkp_grid(1)
                  DO j_y_outer = y_min, y_max + nkp_grid(2), 2*nkp_grid(2)
                     DO k_z_outer = z_min, z_max + nkp_grid(3), 2*nkp_grid(3)

                        i_x = i_x_inner + i_x_outer
                        j_y = j_y_inner + j_y_outer
                        k_z = k_z_inner + k_z_outer

                        IF (i_x > x_max .OR. i_x < x_min .OR. &
                            j_y > y_max .OR. j_y < y_min .OR. &
                            k_z > z_max .OR. k_z < z_min) CYCLE

                        vec_s = [REAL(i_x, dp), REAL(j_y, dp), REAL(k_z, dp)]

                        vec_L = MATMUL(hmat, vec_s)

                        ! Compute (P 0 | Q vec_L) and store it in matrix_v_L_tmp
                        CALL compute_v_transl(matrix_v_L_tmp, blocks_v_L, vec_L, particle_set, &
                                              qs_kind_set, atomic_kind_set, basis_type, cell, &
                                              operator_type)

!$OMP PARALLEL DO DEFAULT(NONE) &
!$OMP             SHARED(blocks_v_L, blocks_v_L_store) &
!$OMP             PRIVATE(i_block)
                        DO i_block = 1, SIZE(blocks_v_L)
                           blocks_v_L_store(i_block)%block(:, :) = blocks_v_L_store(i_block)%block(:, :) &
                                                                   + blocks_v_L(i_block)%block(:, :)
                        END DO
!$OMP END PARALLEL DO

                     END DO
                  END DO
               END DO

               CALL timeset(routineN//"_R_to_k", handle2)

               ! add exp(iq*vec_L) * (P 0 | Q vec_L) to V_PQ(q)
               DO ik = ikp_start, ikp_end

                  ! coskl and sinkl are identical for all i_x_outer, j_y_outer, k_z_outer
                  coskl = COS(twopi*DOT_PRODUCT(vec_s(1:3), kpoints%xkp(1:3, ik)))
                  sinkl = SIN(twopi*DOT_PRODUCT(vec_s(1:3), kpoints%xkp(1:3, ik)))

                  DO i_block = 1, SIZE(blocks_v_L)

                     blocks_v_kp(ik, 1, i_block)%block(:, :) = blocks_v_kp(ik, 1, i_block)%block(:, :) &
                                                               + coskl*blocks_v_L_store(i_block)%block(:, :)
                     blocks_v_kp(ik, 2, i_block)%block(:, :) = blocks_v_kp(ik, 2, i_block)%block(:, :) &
                                                               + sinkl*blocks_v_L_store(i_block)%block(:, :)

                  END DO

               END DO

               DO i_block = 1, SIZE(blocks_v_L)

                  blocks_v_L_store(i_block)%block(:, :) = 0.0_dp

               END DO

               CALL timestop(handle2)

            END DO
         END DO
      END DO

      CALL set_blocks_to_matrix_v_kp(matrix_v_kp, blocks_v_kp, ikp_start, ikp_end)

      CALL deallocate_blocks_v_kp(blocks_v_kp)
      CALL deallocate_blocks_v_L(blocks_v_L)
      CALL deallocate_blocks_v_L(blocks_v_L_store)

      CALL timestop(handle)

   END SUBROUTINE lattice_sum

! **************************************************************************************************
!> \brief ...
!> \param matrix_v_kp ...
!> \param blocks_v_kp ...
!> \param ikp_start ...
!> \param ikp_end ...
! **************************************************************************************************
   SUBROUTINE set_blocks_to_matrix_v_kp(matrix_v_kp, blocks_v_kp, ikp_start, ikp_end)

      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: matrix_v_kp
      TYPE(two_d_util_type), ALLOCATABLE, &
         DIMENSION(:, :, :)                              :: blocks_v_kp
      INTEGER                                            :: ikp_start, ikp_end

      CHARACTER(LEN=*), PARAMETER :: routineN = 'set_blocks_to_matrix_v_kp'

      INTEGER                                            :: col, handle, i_block, i_real_im, ik, row
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: data_block
      TYPE(dbcsr_iterator_type)                          :: iter

      CALL timeset(routineN, handle)

      DO ik = ikp_start, ikp_end

         DO i_real_im = 1, 2

            i_block = 1

            CALL dbcsr_iterator_start(iter, matrix_v_kp(ik, i_real_im)%matrix)

            DO WHILE (dbcsr_iterator_blocks_left(iter))

               CALL dbcsr_iterator_next_block(iter, row, col, data_block)

               data_block(:, :) = blocks_v_kp(ik, i_real_im, i_block)%block(:, :)

               i_block = i_block + 1

            END DO

            CALL dbcsr_iterator_stop(iter)

         END DO

      END DO

      CALL timestop(handle)

   END SUBROUTINE set_blocks_to_matrix_v_kp

! **************************************************************************************************
!> \brief ...
!> \param matrix_v_L_tmp ...
!> \param blocks_v_L ...
!> \param vec_L ...
!> \param particle_set ...
!> \param qs_kind_set ...
!> \param atomic_kind_set ...
!> \param basis_type ...
!> \param cell ...
!> \param operator_type ...
! **************************************************************************************************
   SUBROUTINE compute_v_transl(matrix_v_L_tmp, blocks_v_L, vec_L, particle_set, &
                               qs_kind_set, atomic_kind_set, basis_type, cell, operator_type)

      TYPE(dbcsr_type), POINTER                          :: matrix_v_L_tmp
      TYPE(two_d_util_type), ALLOCATABLE, DIMENSION(:)   :: blocks_v_L
      REAL(KIND=dp), DIMENSION(3)                        :: vec_L
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type
      TYPE(cell_type), POINTER                           :: cell
      INTEGER                                            :: operator_type

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'compute_v_transl'

      INTEGER                                            :: col, handle, i_block, kind_a, kind_b, row
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: kind_of
      REAL(dp), DIMENSION(3)                             :: ra, rab_L, rb
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: data_block
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: contr_a, contr_b
      TYPE(dbcsr_iterator_type)                          :: iter
      TYPE(gto_basis_set_type), POINTER                  :: basis_set_a, basis_set_b

      CALL timeset(routineN, handle)

      NULLIFY (basis_set_a, basis_set_b, data_block)

      CALL get_atomic_kind_set(atomic_kind_set, kind_of=kind_of)

      CALL dbcsr_set(matrix_v_L_tmp, 0.0_dp)

      i_block = 1

      CALL dbcsr_iterator_start(iter, matrix_v_L_tmp)

      DO WHILE (dbcsr_iterator_blocks_left(iter))

         CALL dbcsr_iterator_next_block(iter, row, col, data_block)

         kind_a = kind_of(row)
         kind_b = kind_of(col)

         CALL get_qs_kind(qs_kind=qs_kind_set(kind_a), basis_set=basis_set_a, basis_type=basis_type)
         CALL get_qs_kind(qs_kind=qs_kind_set(kind_b), basis_set=basis_set_b, basis_type=basis_type)

         ra(1:3) = pbc(particle_set(row)%r(1:3), cell)
         rb(1:3) = pbc(particle_set(col)%r(1:3), cell)

         rab_L(1:3) = rb(1:3) - ra(1:3) + vec_L(1:3)

         CALL contraction_matrix_shg(basis_set_a, contr_a)
         CALL contraction_matrix_shg(basis_set_b, contr_b)

         blocks_v_L(i_block)%block = 0.0_dp

         CALL int_operators_r12_ab_shg(operator_type, blocks_v_L(i_block)%block, rab=rab_L, &
                                       fba=basis_set_a, fbb=basis_set_b, scona_shg=contr_a, sconb_shg=contr_b, &
                                       calculate_forces=.FALSE.)

         i_block = i_block + 1

         DEALLOCATE (contr_a, contr_b)

      END DO

      CALL dbcsr_iterator_stop(iter)

      DEALLOCATE (kind_of)

      CALL timestop(handle)

   END SUBROUTINE compute_v_transl

! **************************************************************************************************
!> \brief ...
!> \param blocks_v_kp ...
! **************************************************************************************************
   SUBROUTINE deallocate_blocks_v_kp(blocks_v_kp)
      TYPE(two_d_util_type), ALLOCATABLE, &
         DIMENSION(:, :, :)                              :: blocks_v_kp

      CHARACTER(LEN=*), PARAMETER :: routineN = 'deallocate_blocks_v_kp'

      INTEGER                                            :: handle, i_block, i_real_img, ik

      CALL timeset(routineN, handle)

      DO ik = LBOUND(blocks_v_kp, 1), UBOUND(blocks_v_kp, 1)
         DO i_real_img = 1, SIZE(blocks_v_kp, 2)
            DO i_block = 1, SIZE(blocks_v_kp, 3)
               DEALLOCATE (blocks_v_kp(ik, i_real_img, i_block)%block)
            END DO
         END DO
      END DO

      DEALLOCATE (blocks_v_kp)

      CALL timestop(handle)

   END SUBROUTINE deallocate_blocks_v_kp

! **************************************************************************************************
!> \brief ...
!> \param blocks_v_L ...
! **************************************************************************************************
   SUBROUTINE deallocate_blocks_v_L(blocks_v_L)
      TYPE(two_d_util_type), ALLOCATABLE, DIMENSION(:)   :: blocks_v_L

      CHARACTER(LEN=*), PARAMETER :: routineN = 'deallocate_blocks_v_L'

      INTEGER                                            :: handle, i_block

      CALL timeset(routineN, handle)

      DO i_block = 1, SIZE(blocks_v_L, 1)
         DEALLOCATE (blocks_v_L(i_block)%block)
      END DO

      DEALLOCATE (blocks_v_L)

      CALL timestop(handle)

   END SUBROUTINE deallocate_blocks_v_L

! **************************************************************************************************
!> \brief ...
!> \param blocks_v_L ...
!> \param matrix_v_L_tmp ...
! **************************************************************************************************
   SUBROUTINE allocate_blocks_v_L(blocks_v_L, matrix_v_L_tmp)
      TYPE(two_d_util_type), ALLOCATABLE, DIMENSION(:)   :: blocks_v_L
      TYPE(dbcsr_type), POINTER                          :: matrix_v_L_tmp

      CHARACTER(LEN=*), PARAMETER :: routineN = 'allocate_blocks_v_L'

      INTEGER                                            :: col, handle, i_block, nblocks, row
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: data_block
      TYPE(dbcsr_iterator_type)                          :: iter

      CALL timeset(routineN, handle)

      nblocks = 0

      CALL dbcsr_iterator_start(iter, matrix_v_L_tmp)

      DO WHILE (dbcsr_iterator_blocks_left(iter))

         CALL dbcsr_iterator_next_block(iter, row, col, data_block)

         nblocks = nblocks + 1

      END DO

      CALL dbcsr_iterator_stop(iter)

      ALLOCATE (blocks_v_L(nblocks))

      i_block = 1

      CALL dbcsr_iterator_start(iter, matrix_v_L_tmp)

      DO WHILE (dbcsr_iterator_blocks_left(iter))

         CALL dbcsr_iterator_next_block(iter, row, col, data_block)

         ALLOCATE (blocks_v_L(i_block)%block(SIZE(data_block, 1), SIZE(data_block, 2)))
         blocks_v_L(i_block)%block = 0.0_dp

         i_block = i_block + 1

      END DO

      CALL dbcsr_iterator_stop(iter)

      CALL timestop(handle)

   END SUBROUTINE allocate_blocks_v_L

! **************************************************************************************************
!> \brief ...
!> \param blocks_v_kp ...
!> \param matrix_v_kp ...
!> \param ikp_start ...
!> \param ikp_end ...
! **************************************************************************************************
   SUBROUTINE allocate_blocks_v_kp(blocks_v_kp, matrix_v_kp, ikp_start, ikp_end)
      TYPE(two_d_util_type), ALLOCATABLE, &
         DIMENSION(:, :, :)                              :: blocks_v_kp
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: matrix_v_kp
      INTEGER                                            :: ikp_start, ikp_end

      CHARACTER(LEN=*), PARAMETER :: routineN = 'allocate_blocks_v_kp'

      INTEGER                                            :: col, handle, i_block, i_real_img, ik, &
                                                            nblocks, row
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: data_block
      TYPE(dbcsr_iterator_type)                          :: iter

      CALL timeset(routineN, handle)

      nblocks = 0

      CALL dbcsr_iterator_start(iter, matrix_v_kp(ikp_start, 1)%matrix)

      DO WHILE (dbcsr_iterator_blocks_left(iter))

         CALL dbcsr_iterator_next_block(iter, row, col, data_block)

         nblocks = nblocks + 1

      END DO

      CALL dbcsr_iterator_stop(iter)

      ALLOCATE (blocks_v_kp(ikp_start:ikp_end, SIZE(matrix_v_kp, 2), nblocks))

      DO ik = ikp_start, ikp_end

         DO i_real_img = 1, SIZE(matrix_v_kp, 2)

            i_block = 1

            CALL dbcsr_iterator_start(iter, matrix_v_kp(ik, i_real_img)%matrix)

            DO WHILE (dbcsr_iterator_blocks_left(iter))

               CALL dbcsr_iterator_next_block(iter, row, col, data_block)

               ALLOCATE (blocks_v_kp(ik, i_real_img, i_block)%block(SIZE(data_block, 1), &
                                                                    SIZE(data_block, 2)))
               blocks_v_kp(ik, i_real_img, i_block)%block = 0.0_dp

               i_block = i_block + 1

            END DO

            CALL dbcsr_iterator_stop(iter)

         END DO

      END DO

      CALL timestop(handle)

   END SUBROUTINE allocate_blocks_v_kp

! **************************************************************************************************
!> \brief ...
!> \param cell ...
!> \param kpoints ...
!> \param size_lattice_sum ...
!> \param factor ...
!> \param hmat ...
!> \param x_min ...
!> \param x_max ...
!> \param y_min ...
!> \param y_max ...
!> \param z_min ...
!> \param z_max ...
!> \param nkp_grid ...
! **************************************************************************************************
   SUBROUTINE get_factor_and_xyz_min_max(cell, kpoints, size_lattice_sum, factor, hmat, &
                                         x_min, x_max, y_min, y_max, z_min, z_max, nkp_grid)

      TYPE(cell_type), POINTER                           :: cell
      TYPE(kpoint_type), POINTER                         :: kpoints
      INTEGER                                            :: size_lattice_sum, factor
      REAL(KIND=dp), DIMENSION(3, 3)                     :: hmat
      INTEGER                                            :: x_min, x_max, y_min, y_max, z_min, z_max
      INTEGER, DIMENSION(3)                              :: nkp_grid

      CHARACTER(LEN=*), PARAMETER :: routineN = 'get_factor_and_xyz_min_max'

      INTEGER                                            :: handle, nkp
      INTEGER, DIMENSION(3)                              :: periodic

      CALL timeset(routineN, handle)

      CALL get_kpoint_info(kpoints, nkp_grid=nkp_grid, nkp=nkp)
      CALL get_cell(cell=cell, h=hmat, periodic=periodic)

      IF (periodic(1) == 0) THEN
         CPASSERT(nkp_grid(1) == 1)
      END IF
      IF (periodic(2) == 0) THEN
         CPASSERT(nkp_grid(2) == 1)
      END IF
      IF (periodic(3) == 0) THEN
         CPASSERT(nkp_grid(3) == 1)
      END IF

      IF (MODULO(nkp_grid(1), 2) == 1) THEN
         factor = 3**(size_lattice_sum - 1)
      ELSE IF (MODULO(nkp_grid(1), 2) == 0) THEN
         factor = 2**(size_lattice_sum - 1)
      END IF

      IF (MODULO(nkp_grid(1), 2) == 1) THEN
         x_min = -(factor*nkp_grid(1) - 1)/2
         x_max = (factor*nkp_grid(1) - 1)/2
      ELSE IF (MODULO(nkp_grid(1), 2) == 0) THEN
         x_min = -factor*nkp_grid(1)/2
         x_max = factor*nkp_grid(1)/2 - 1
      END IF
      IF (periodic(1) == 0) THEN
         x_min = 0
         x_max = 0
      END IF

      IF (MODULO(nkp_grid(2), 2) == 1) THEN
         y_min = -(factor*nkp_grid(2) - 1)/2
         y_max = (factor*nkp_grid(2) - 1)/2
      ELSE IF (MODULO(nkp_grid(2), 2) == 0) THEN
         y_min = -factor*nkp_grid(2)/2
         y_max = factor*nkp_grid(2)/2 - 1
      END IF
      IF (periodic(2) == 0) THEN
         y_min = 0
         y_max = 0
      END IF

      IF (MODULO(nkp_grid(3), 2) == 1) THEN
         z_min = -(factor*nkp_grid(3) - 1)/2
         z_max = (factor*nkp_grid(3) - 1)/2
      ELSE IF (MODULO(nkp_grid(3), 2) == 0) THEN
         z_min = -factor*nkp_grid(3)/2
         z_max = factor*nkp_grid(3)/2 - 1
      END IF
      IF (periodic(3) == 0) THEN
         z_min = 0
         z_max = 0
      END IF

      CALL timestop(handle)

   END SUBROUTINE get_factor_and_xyz_min_max

! **************************************************************************************************
!> \brief ...
!> \param matrix_v_L_tmp ...
!> \param matrix_v_kp ...
!> \param ikp_start ...
! **************************************************************************************************
   SUBROUTINE allocate_tmp(matrix_v_L_tmp, matrix_v_kp, ikp_start)

      TYPE(dbcsr_type), POINTER                          :: matrix_v_L_tmp
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: matrix_v_kp
      INTEGER                                            :: ikp_start

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'allocate_tmp'

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      NULLIFY (matrix_v_L_tmp)
      CALL dbcsr_init_p(matrix_v_L_tmp)
      CALL dbcsr_create(matrix=matrix_v_L_tmp, &
                        template=matrix_v_kp(ikp_start, 1)%matrix, &
                        matrix_type=dbcsr_type_no_symmetry)
      CALL dbcsr_reserve_all_blocks(matrix_v_L_tmp)
      CALL dbcsr_set(matrix_v_L_tmp, 0.0_dp)

      CALL timestop(handle)

   END SUBROUTINE allocate_tmp

! **************************************************************************************************
!> \brief ...
!> \param matrix_v_L_tmp ...
! **************************************************************************************************
   SUBROUTINE deallocate_tmp(matrix_v_L_tmp)

      TYPE(dbcsr_type), POINTER                          :: matrix_v_L_tmp

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'deallocate_tmp'

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      CALL dbcsr_release_p(matrix_v_L_tmp)

      CALL timestop(handle)

   END SUBROUTINE deallocate_tmp

! **************************************************************************************************
!> \brief ...
!> \param V_k ...
!> \param qs_env ...
!> \param kpoints ...
!> \param size_lattice_sum range of the direct lattice sum; not used with use_ewald
!> \param basis_type ...
!> \param ikp_start ...
!> \param ikp_end ...
!> \param use_ewald replace the direct lattice sum by an Ewald summation
! **************************************************************************************************
   SUBROUTINE build_2c_coulomb_matrix_kp_small_cell(V_k, qs_env, kpoints, size_lattice_sum, &
                                                    basis_type, ikp_start, ikp_end, use_ewald)
      COMPLEX(KIND=dp), DIMENSION(:, :, :)               :: V_k
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(kpoint_type), POINTER                         :: kpoints
      INTEGER                                            :: size_lattice_sum
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type
      INTEGER                                            :: ikp_start, ikp_end
      LOGICAL, INTENT(IN), OPTIONAL                      :: use_ewald

      CHARACTER(LEN=*), PARAMETER :: routineN = 'build_2c_coulomb_matrix_kp_small_cell'

      INTEGER :: factor, handle, handle2, i_cell, i_x, i_x_inner, i_x_outer, ik, ikp_local, j_y, &
         j_y_inner, j_y_outer, k_z, k_z_inner, k_z_outer, n_atom, n_bf, x_max, x_min, y_max, &
         y_min, z_max, z_min
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: bf_end_from_atom, bf_start_from_atom
      INTEGER, DIMENSION(3)                              :: nkp_grid
      LOGICAL                                            :: do_ewald
      REAL(KIND=dp)                                      :: coskl, ewald_omega, sinkl
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: V_L
      REAL(KIND=dp), DIMENSION(3)                        :: vec_L, vec_s
      REAL(KIND=dp), DIMENSION(3, 3)                     :: hmat
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(cell_type), POINTER                           :: cell
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set

      CALL timeset(routineN, handle)

      CALL get_qs_env(qs_env=qs_env, &
                      para_env=para_env, &
                      particle_set=particle_set, &
                      cell=cell, &
                      qs_kind_set=qs_kind_set, &
                      atomic_kind_set=atomic_kind_set)

      CALL get_factor_and_xyz_min_max(cell, kpoints, size_lattice_sum, factor, hmat, &
                                      x_min, x_max, y_min, y_max, z_min, z_max, nkp_grid)

      CALL get_basis_sizes(qs_env, n_atom, basis_type, bf_start_from_atom, bf_end_from_atom, n_bf)

      do_ewald = .FALSE.
      IF (PRESENT(use_ewald)) do_ewald = use_ewald
      ewald_omega = 0.0_dp
      IF (do_ewald) THEN
         CALL ewald_real_space_bounds(cell, qs_kind_set, basis_type, ewald_omega, &
                                      x_min, x_max, y_min, y_max, z_min, z_max)
      END IF

      ALLOCATE (V_L(n_bf, n_bf))

      DO i_x_inner = 0, 2*nkp_grid(1) - 1
         DO j_y_inner = 0, 2*nkp_grid(2) - 1
            DO k_z_inner = 0, 2*nkp_grid(3) - 1

               V_L(:, :) = 0.0_dp
               i_cell = 0

               DO i_x_outer = x_min, x_max + nkp_grid(1), 2*nkp_grid(1)
                  DO j_y_outer = y_min, y_max + nkp_grid(2), 2*nkp_grid(2)
                     DO k_z_outer = z_min, z_max + nkp_grid(3), 2*nkp_grid(3)

                        i_x = i_x_inner + i_x_outer
                        j_y = j_y_inner + j_y_outer
                        k_z = k_z_inner + k_z_outer

                        IF (i_x > x_max .OR. i_x < x_min .OR. &
                            j_y > y_max .OR. j_y < y_min .OR. &
                            k_z > z_max .OR. k_z < z_min) CYCLE

                        i_cell = i_cell + 1

                        vec_s = [REAL(i_x, dp), REAL(j_y, dp), REAL(k_z, dp)]

                        IF (MODULO(i_cell, para_env%num_pe) /= para_env%mepos) CYCLE

                        vec_L = MATMUL(hmat, vec_s)

                        ! Compute (P 0 | Q vec_L) and add it to V_R
                        CALL add_V_L(V_L, vec_L, n_atom, bf_start_from_atom, bf_end_from_atom, &
                                     particle_set, qs_kind_set, atomic_kind_set, basis_type, cell, ewald_omega)

                     END DO
                  END DO
               END DO

               CALL para_env%sync()
               CALL para_env%sum(V_L)

               CALL timeset(routineN//"_R_to_k", handle2)

               ikp_local = 0

               ! add exp(iq*vec_L) * (P 0 | Q vec_L) to V_PQ(q)
               DO ik = 1, ikp_end

                  IF (MODULO(ik, para_env%num_pe) /= para_env%mepos) CYCLE

                  ikp_local = ikp_local + 1

                  IF (ik < ikp_start) CYCLE

                  ! coskl and sinkl are identical for all i_x_outer, j_y_outer, k_z_outer
                  coskl = COS(twopi*DOT_PRODUCT(vec_s(1:3), kpoints%xkp(1:3, ik)))
                  sinkl = SIN(twopi*DOT_PRODUCT(vec_s(1:3), kpoints%xkp(1:3, ik)))

                  V_k(:, :, ikp_local) = V_k(:, :, ikp_local) + z_one*coskl*V_L(:, :) + &
                                         gaussi*sinkl*V_L(:, :)

               END DO

               CALL timestop(handle2)

            END DO
         END DO
      END DO

      IF (do_ewald) THEN
         CALL add_ewald_reciprocal(V_k, qs_env, kpoints, basis_type, ewald_omega, &
                                   bf_start_from_atom, bf_end_from_atom, ikp_start, ikp_end)
      END IF
      CALL timestop(handle)

   END SUBROUTINE build_2c_coulomb_matrix_kp_small_cell

! **************************************************************************************************
!> \brief ...
!> \param qs_env ...
!> \param n_atom ...
!> \param basis_type ...
!> \param bf_start_from_atom ...
!> \param bf_end_from_atom ...
!> \param n_bf ...
! **************************************************************************************************
   SUBROUTINE get_basis_sizes(qs_env, n_atom, basis_type, bf_start_from_atom, bf_end_from_atom, n_bf)

      TYPE(qs_environment_type), POINTER                 :: qs_env
      INTEGER                                            :: n_atom
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: bf_start_from_atom, bf_end_from_atom
      INTEGER                                            :: n_bf

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'get_basis_sizes'

      INTEGER                                            :: handle, iatom, ikind, n_kind, nsgf
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: kind_of
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(gto_basis_set_type), POINTER                  :: basis_set_a
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set

      CALL timeset(routineN, handle)

      CALL get_qs_env(qs_env=qs_env, particle_set=particle_set, &
                      qs_kind_set=qs_kind_set, atomic_kind_set=atomic_kind_set)
      CALL get_atomic_kind_set(atomic_kind_set, kind_of=kind_of)

      n_atom = SIZE(particle_set)
      n_kind = SIZE(qs_kind_set)

      DO ikind = 1, n_kind
         CALL get_qs_kind(qs_kind=qs_kind_set(ikind), basis_set=basis_set_a, &
                          basis_type=basis_type)
         CPASSERT(ASSOCIATED(basis_set_a))
      END DO

      ALLOCATE (bf_start_from_atom(n_atom), bf_end_from_atom(n_atom))

      n_bf = 0
      DO iatom = 1, n_atom
         bf_start_from_atom(iatom) = n_bf + 1
         ikind = kind_of(iatom)
         CALL get_qs_kind(qs_kind=qs_kind_set(ikind), nsgf=nsgf, basis_type=basis_type)
         n_bf = n_bf + nsgf
         bf_end_from_atom(iatom) = n_bf
      END DO

      CALL timestop(handle)

   END SUBROUTINE get_basis_sizes

! **************************************************************************************************
!> \brief ...
!> \param V_L ...
!> \param vec_L ...
!> \param n_atom ...
!> \param bf_start_from_atom ...
!> \param bf_end_from_atom ...
!> \param particle_set ...
!> \param qs_kind_set ...
!> \param atomic_kind_set ...
!> \param basis_type ...
!> \param cell ...
!> \param ewald_omega erfc attenuation of the Ewald real-space sum; 0 for the plain Coulomb kernel
! **************************************************************************************************
   SUBROUTINE add_V_L(V_L, vec_L, n_atom, bf_start_from_atom, bf_end_from_atom, &
                      particle_set, qs_kind_set, atomic_kind_set, basis_type, cell, ewald_omega)

      REAL(KIND=dp), DIMENSION(:, :)                     :: V_L
      REAL(KIND=dp), DIMENSION(3)                        :: vec_L
      INTEGER                                            :: n_atom
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: bf_start_from_atom, bf_end_from_atom
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type
      TYPE(cell_type), POINTER                           :: cell
      REAL(KIND=dp), INTENT(IN)                          :: ewald_omega

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'add_V_L'

      INTEGER                                            :: a_1, a_2, atom_a, atom_b, b_1, b_2, &
                                                            handle, kind_a, kind_b
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: kind_of
      REAL(dp), DIMENSION(3)                             :: ra, rab_L, rb
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: V_L_ab
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: contr_a, contr_b
      TYPE(gto_basis_set_type), POINTER                  :: basis_set_a, basis_set_b

      CALL timeset(routineN, handle)

      NULLIFY (basis_set_a, basis_set_b)

      CALL get_atomic_kind_set(atomic_kind_set, kind_of=kind_of)

      DO atom_a = 1, n_atom

         DO atom_b = 1, n_atom

            kind_a = kind_of(atom_a)
            kind_b = kind_of(atom_b)

            CALL get_qs_kind(qs_kind=qs_kind_set(kind_a), basis_set=basis_set_a, &
                             basis_type=basis_type)
            CALL get_qs_kind(qs_kind=qs_kind_set(kind_b), basis_set=basis_set_b, &
                             basis_type=basis_type)

            ra(1:3) = pbc(particle_set(atom_a)%r(1:3), cell)
            rb(1:3) = pbc(particle_set(atom_b)%r(1:3), cell)

            rab_L(1:3) = rb(1:3) - ra(1:3) + vec_L(1:3)

            CALL contraction_matrix_shg(basis_set_a, contr_a)
            CALL contraction_matrix_shg(basis_set_b, contr_b)

            a_1 = bf_start_from_atom(atom_a)
            a_2 = bf_end_from_atom(atom_a)
            b_1 = bf_start_from_atom(atom_b)
            b_2 = bf_end_from_atom(atom_b)

            ALLOCATE (V_L_ab(a_2 - a_1 + 1, b_2 - b_1 + 1))

            IF (ewald_omega > 0.0_dp) THEN
               CALL int_operators_r12_ab_shg(operator_verfc, V_L_ab, rab=rab_L, &
                                             fba=basis_set_a, fbb=basis_set_b, &
                                             scona_shg=contr_a, sconb_shg=contr_b, &
                                             omega=ewald_omega, calculate_forces=.FALSE.)
            ELSE
               CALL int_operators_r12_ab_shg(operator_coulomb, V_L_ab, rab=rab_L, &
                                             fba=basis_set_a, fbb=basis_set_b, &
                                             scona_shg=contr_a, sconb_shg=contr_b, &
                                             calculate_forces=.FALSE.)
            END IF

            V_L(a_1:a_2, b_1:b_2) = V_L(a_1:a_2, b_1:b_2) + V_L_ab(:, :)

            DEALLOCATE (contr_a, contr_b, V_L_ab)

         END DO

      END DO

      DEALLOCATE (kind_of)

      CALL timestop(handle)

   END SUBROUTINE add_V_L

! **************************************************************************************************
!> \brief ...
!> \param cell ...
!> \param qs_kinds ...
!> \param basis_type ...
!> \param omega ...
!> \param x_min ...
!> \param x_max ...
!> \param y_min ...
!> \param y_max ...
!> \param z_min ...
!> \param z_max ...
! **************************************************************************************************
   SUBROUTINE ewald_real_space_bounds(cell, qs_kinds, basis_type, omega, &
                                      x_min, x_max, y_min, y_max, z_min, z_max)
      TYPE(cell_type), POINTER                           :: cell
      TYPE(qs_kind_type), POINTER                        :: qs_kinds(:)
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type
      REAL(KIND=dp), INTENT(OUT)                         :: omega
      INTEGER, INTENT(OUT)                               :: x_min, x_max, y_min, y_max, z_min, z_max

      INTEGER                                            :: bounds(3), idim, ikind, iset, lmax
      REAL(KIND=dp)                                      :: alpha_min, beta, radius
      TYPE(gto_basis_set_type), POINTER                  :: basis

      alpha_min = HUGE(0.0_dp)
      DO ikind = 1, SIZE(qs_kinds)
         CALL get_qs_kind(qs_kinds(ikind), basis_set=basis, basis_type=basis_type)
         DO iset = 1, basis%nset
            alpha_min = MIN(alpha_min, MINVAL(basis%zet(1:basis%npgf(iset), iset)))
         END DO
      END DO
      lmax = basis_lmax(qs_kinds, basis_type)
      CPASSERT(alpha_min > 0.0_dp)
      omega = ewald_omega_default
      beta = 1.0_dp/(2.0_dp/alpha_min + 1.0_dp/omega**2)
      radius = SQRT((ewald_tail_exponent + ewald_lmax_margin*lmax)/beta)
      DO idim = 1, 3
         bounds(idim) = 0
         IF (cell%perd(idim) /= 0) THEN
            bounds(idim) = CEILING(radius*NORM2(cell%h_inv(idim, :))) + 1
         END IF
      END DO
      x_min = -bounds(1)
      x_max = bounds(1)
      y_min = -bounds(2)
      y_max = bounds(2)
      z_min = -bounds(3)
      z_max = bounds(3)
   END SUBROUTINE ewald_real_space_bounds

! **************************************************************************************************
!> \brief ...
!> \param V_k ...
!> \param qs_env ...
!> \param kpoints ...
!> \param basis_type ...
!> \param omega ...
!> \param bf_start ...
!> \param bf_end ...
!> \param ikp_start ...
!> \param ikp_end ...
! **************************************************************************************************
   SUBROUTINE add_ewald_reciprocal(V_k, qs_env, kpoints, basis_type, omega, &
                                   bf_start, bf_end, ikp_start, ikp_end)
      COMPLEX(KIND=dp), INTENT(INOUT)                    :: V_k(:, :, :)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(kpoint_type), POINTER                         :: kpoints
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type
      REAL(KIND=dp), INTENT(IN)                          :: omega
      INTEGER, INTENT(IN)                                :: bf_start(:), bf_end(:), ikp_start, &
                                                            ikp_end

      COMPLEX(KIND=dp), ALLOCATABLE                      :: fourier_basis(:)
      INTEGER                                            :: bounds(3), idim, ikp, ikp_local, ix, iy, &
                                                            iz, n_bf
      REAL(KIND=dp)                                      :: cutoff, p(3), p2, prefactor, q(3), &
                                                            reciprocal(3, 3)
      TYPE(cell_type), POINTER                           :: cell
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(particle_type), POINTER                       :: particles(:)
      TYPE(qs_kind_type), POINTER                        :: qs_kinds(:)

      CALL get_qs_env(qs_env, cell=cell, para_env=para_env, particle_set=particles, qs_kind_set=qs_kinds)
      IF (COUNT(cell%perd /= 0) == 2) THEN
         CALL add_ewald_reciprocal_2d(V_k, qs_env, kpoints, basis_type, omega, &
                                      bf_start, bf_end, ikp_start, ikp_end)
      ELSE
         CPASSERT(COUNT(cell%perd /= 0) == 3)
         n_bf = SIZE(V_k, 1)
         ALLOCATE (fourier_basis(n_bf))
         reciprocal = twopi*TRANSPOSE(cell%h_inv)
         cutoff = 2.0_dp*omega*SQRT(ewald_tail_exponent + ewald_lmax_margin*basis_lmax(qs_kinds, basis_type))
         DO ikp = ikp_start, ikp_end
            IF (MODULO(ikp, para_env%num_pe) /= para_env%mepos) CYCLE
            ikp_local = (ikp - para_env%mepos)/para_env%num_pe + 1
            IF (para_env%mepos == 0) ikp_local = ikp/para_env%num_pe
            q = kpoints%xkp(:, ikp)
            DO idim = 1, 3
               bounds(idim) = CEILING(cutoff*NORM2(cell%hmat(:, idim))/twopi + ABS(q(idim)))
            END DO
            DO ix = -bounds(1), bounds(1)
               DO iy = -bounds(2), bounds(2)
                  DO iz = -bounds(3), bounds(3)
                     p = MATMUL(reciprocal, q + REAL([ix, iy, iz], dp))
                     p2 = SUM(p*p)
                     IF (p2 > cutoff**2) CYCLE
                     CALL gaussian_basis_fourier(p, particles, qs_kinds, cell, basis_type, &
                                                 bf_start, bf_end, fourier_basis)
                     IF (p2 <= 0.0_dp) THEN
                        prefactor = -pi/(cell%deth*omega**2)
                     ELSE
                        prefactor = fourpi*EXP(-p2/(4.0_dp*omega**2))/(cell%deth*p2)
                     END IF
                     CALL zgerc(n_bf, n_bf, CMPLX(prefactor, 0.0_dp, dp), fourier_basis, 1, &
                                fourier_basis, 1, V_k(:, :, ikp_local), n_bf)
                  END DO
               END DO
            END DO
         END DO
         DEALLOCATE (fourier_basis)
      END IF
   END SUBROUTINE add_ewald_reciprocal

! **************************************************************************************************
!> \brief Reciprocal 2D Ewald Coulomb matrix for any pair of periodic cell axes.
!> The momentum perpendicular to their plane is integrated continuously.
!> \param V_k ...
!> \param qs_env ...
!> \param kpoints ...
!> \param basis_type ...
!> \param omega ...
!> \param bf_start ...
!> \param bf_end ...
!> \param ikp_start ...
!> \param ikp_end ...
! **************************************************************************************************
   SUBROUTINE add_ewald_reciprocal_2d(V_k, qs_env, kpoints, basis_type, omega, &
                                      bf_start, bf_end, ikp_start, ikp_end)
      COMPLEX(KIND=dp), INTENT(INOUT)                    :: V_k(:, :, :)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(kpoint_type), POINTER                         :: kpoints
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type
      REAL(KIND=dp), INTENT(IN)                          :: omega
      INTEGER, INTENT(IN)                                :: bf_start(:), bf_end(:), ikp_start, &
                                                            ikp_end

      INTEGER, PARAMETER                                 :: n_kz = 192

      COMPLEX(KIND=dp), ALLOCATABLE                      :: delta_minus(:), delta_plus(:), &
                                                            delta_sum(:), f_minus(:), f_plus(:), &
                                                            f_zero(:)
      INTEGER                                            :: free_axis, ikp, ikp_local, inode, ix, &
                                                            iy, n_bf, nx, ny, periodic_axis(2)
      REAL(KIND=dp)                                      :: a1(3), a2(3), area, b1(3), b2(3), &
                                                            cutoff, factor, kz, nodes(n_kz), &
                                                            normal(3), p(3), pnorm, q(3), weight, &
                                                            weights(n_kz)
      TYPE(cell_type), POINTER                           :: cell
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(particle_type), POINTER                       :: particles(:)
      TYPE(qs_kind_type), POINTER                        :: qs_kinds(:)

      CALL get_qs_env(qs_env, cell=cell, para_env=para_env, &
                      particle_set=particles, qs_kind_set=qs_kinds)
      CPASSERT(COUNT(cell%perd /= 0) == 2)
      periodic_axis = PACK([1, 2, 3], cell%perd /= 0)
      free_axis = FINDLOC(cell%perd, 0, DIM=1)
      a1 = cell%hmat(:, periodic_axis(1))
      a2 = cell%hmat(:, periodic_axis(2))
      normal = vector_product(a1, a2)
      area = NORM2(normal)
      CPASSERT(area > 0.0_dp)
      normal = normal/area
      b1 = twopi*vector_product(a2, normal)/area
      b2 = twopi*vector_product(normal, a1)/area
      n_bf = SIZE(V_k, 1)
      ALLOCATE (f_zero(n_bf), f_plus(n_bf), f_minus(n_bf), &
                delta_plus(n_bf), delta_minus(n_bf), delta_sum(n_bf))
      cutoff = 2.0_dp*omega*SQRT(ewald_tail_exponent + ewald_lmax_margin*basis_lmax(qs_kinds, basis_type))
      ! Legendre nodes and weights on [0, cutoff] for the free momentum.
      CALL gauss_legendre(n_kz, nodes, weights)
      nodes(:) = 0.5_dp*cutoff*(nodes + 1.0_dp)
      weights(:) = 0.5_dp*cutoff*weights
      DO ikp = ikp_start, ikp_end
         IF (MODULO(ikp, para_env%num_pe) /= para_env%mepos) CYCLE
         ikp_local = (ikp - para_env%mepos)/para_env%num_pe + 1
         IF (para_env%mepos == 0) ikp_local = ikp/para_env%num_pe
         q = kpoints%xkp(:, ikp)
         IF (ABS(q(free_axis)) > 0.0_dp) THEN
            CPABORT('2D reciprocal transfer has a nonzero nonperiodic component')
         END IF
         nx = CEILING(cutoff*NORM2(a1)/twopi + ABS(q(periodic_axis(1))))
         ny = CEILING(cutoff*NORM2(a2)/twopi + ABS(q(periodic_axis(2))))
         DO ix = -nx, nx
            DO iy = -ny, ny
               p = b1*(q(periodic_axis(1)) + REAL(ix, dp)) + &
                   b2*(q(periodic_axis(2)) + REAL(iy, dp))
               pnorm = NORM2(p)
               IF (pnorm > cutoff) CYCLE
               IF (pnorm <= 0.0_dp) THEN
                  CPABORT('2D Coulomb at Gamma requires charge-channel separation')
               END IF
               CALL gaussian_basis_fourier(p, particles, qs_kinds, cell, basis_type, &
                                           bf_start, bf_end, f_zero)
               factor = twopi*ERFC(pnorm/(2.0_dp*omega))/(area*pnorm)
               CALL zgerc(n_bf, n_bf, CMPLX(factor, 0.0_dp, dp), f_zero, 1, &
                          f_zero, 1, V_k(:, :, ikp_local), n_bf)
               DO inode = 1, n_kz
                  kz = nodes(inode)
                  CALL gaussian_basis_fourier(p + kz*normal, particles, qs_kinds, cell, &
                                              basis_type, bf_start, bf_end, f_plus)
                  CALL gaussian_basis_fourier(p - kz*normal, particles, qs_kinds, cell, &
                                              basis_type, bf_start, bf_end, f_minus)
                  weight = 2.0_dp*weights(inode)* &
                           EXP(-(pnorm**2 + kz**2)/(4.0_dp*omega**2))/ &
                           (area*(pnorm**2 + kz**2))
                  ! Expand the even-in-kz residual before the outer products.
                  ! Direct subtraction of three large matrices loses precision
                  ! near q=0, where their common denominator is small.
                  delta_plus(:) = f_plus - f_zero
                  delta_minus(:) = f_minus - f_zero
                  delta_sum(:) = delta_plus + delta_minus
                  factor = weight
                  CALL zgerc(n_bf, n_bf, CMPLX(factor, 0.0_dp, dp), f_zero, 1, &
                             delta_sum, 1, V_k(:, :, ikp_local), n_bf)
                  CALL zgerc(n_bf, n_bf, CMPLX(factor, 0.0_dp, dp), delta_sum, 1, &
                             f_zero, 1, V_k(:, :, ikp_local), n_bf)
                  CALL zgerc(n_bf, n_bf, CMPLX(factor, 0.0_dp, dp), delta_plus, 1, &
                             delta_plus, 1, V_k(:, :, ikp_local), n_bf)
                  CALL zgerc(n_bf, n_bf, CMPLX(factor, 0.0_dp, dp), delta_minus, 1, &
                             delta_minus, 1, V_k(:, :, ikp_local), n_bf)
               END DO
            END DO
         END DO
      END DO
      DEALLOCATE (f_zero, f_plus, f_minus, delta_plus, delta_minus, delta_sum)
   END SUBROUTINE add_ewald_reciprocal_2d

! **************************************************************************************************
!> \brief Largest angular momentum of a basis type over all kinds.
!> \param qs_kinds ...
!> \param basis_type ...
!> \return ...
! **************************************************************************************************
   FUNCTION basis_lmax(qs_kinds, basis_type) RESULT(lmax)
      TYPE(qs_kind_type), POINTER                        :: qs_kinds(:)
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type
      INTEGER                                            :: lmax

      INTEGER                                            :: ikind
      TYPE(gto_basis_set_type), POINTER                  :: basis

      lmax = 0
      DO ikind = 1, SIZE(qs_kinds)
         CALL get_qs_kind(qs_kinds(ikind), basis_set=basis, basis_type=basis_type)
         lmax = MAX(lmax, MAXVAL(basis%lmax))
      END DO
   END FUNCTION basis_lmax

! **************************************************************************************************
!> \brief ...
!> \param p ...
!> \param particles ...
!> \param qs_kinds ...
!> \param cell ...
!> \param basis_type ...
!> \param bf_start ...
!> \param bf_end ...
!> \param values ...
! **************************************************************************************************
   SUBROUTINE gaussian_basis_fourier(p, particles, qs_kinds, cell, basis_type, &
                                     bf_start, bf_end, values)
      REAL(KIND=dp), INTENT(IN)                          :: p(3)
      TYPE(particle_type), POINTER                       :: particles(:)
      TYPE(qs_kind_type), POINTER                        :: qs_kinds(:)
      TYPE(cell_type), POINTER                           :: cell
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type
      INTEGER, INTENT(IN)                                :: bf_start(:), bf_end(:)
      COMPLEX(KIND=dp), INTENT(OUT)                      :: values(:)

      COMPLEX(KIND=dp)                                   :: phase
      COMPLEX(KIND=dp), ALLOCATABLE                      :: moments(:, :), primitive(:)
      INTEGER                                            :: iatom, ico, idim, ikind, il, ipgf, iset, &
                                                            isgf, lmax, ncart, offset
      REAL(KIND=dp)                                      :: alpha, centre(3)
      TYPE(gto_basis_set_type), POINTER                  :: basis

      values = z_zero
      DO iatom = 1, SIZE(particles)
         ikind = particles(iatom)%atomic_kind%kind_number
         CALL get_qs_kind(qs_kinds(ikind), basis_set=basis, basis_type=basis_type)
         centre = pbc(particles(iatom)%r, cell)
         phase = EXP(gaussi*SUM(p*centre))
         CPASSERT(bf_end(iatom) - bf_start(iatom) + 1 == basis%nsgf)
         DO iset = 1, basis%nset
            lmax = basis%lmax(iset)
            ncart = ncoset(lmax)
            ALLOCATE (moments(0:lmax, 3), primitive(ncart*basis%npgf(iset)))
            primitive = z_zero
            DO ipgf = 1, basis%npgf(iset)
               alpha = basis%zet(ipgf, iset)
               DO idim = 1, 3
                  moments(0, idim) = SQRT(pi/alpha)*EXP(-p(idim)**2/(4.0_dp*alpha))
                  IF (lmax > 0) moments(1, idim) = gaussi*p(idim)*moments(0, idim)/(2.0_dp*alpha)
                  DO il = 2, lmax
                     moments(il, idim) = (gaussi*p(idim)*moments(il - 1, idim) + &
                                          REAL(il - 1, dp)*moments(il - 2, idim))/(2.0_dp*alpha)
                  END DO
               END DO
               DO ico = ncoset(basis%lmin(iset) - 1) + 1, ncart
                  primitive((ipgf - 1)*ncart + ico) = &
                     moments(indco(1, ico), 1)*moments(indco(2, ico), 2)*moments(indco(3, ico), 3)
               END DO
            END DO
            offset = basis%first_sgf(1, iset)
            DO isgf = offset, offset + basis%nsgf_set(iset) - 1
               values(bf_start(iatom) + isgf - 1) = &
                  phase*SUM(basis%sphi(1:SIZE(primitive), isgf)*primitive)
            END DO
            DEALLOCATE (moments, primitive)
         END DO
      END DO
   END SUBROUTINE gaussian_basis_fourier

END MODULE kpoint_coulomb_2c
