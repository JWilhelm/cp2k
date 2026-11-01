!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Shared numerical operations for computing the RI-RS matrix Z_lP.
!> \par History
!>      09.2026 created Jan Wilhelm
! **************************************************************************************************
MODULE ri_rs_fitting
   USE cp_blacs_env,                    ONLY: cp_blacs_env_create,&
                                              cp_blacs_env_release,&
                                              cp_blacs_env_type
   USE cp_dbcsr_api,                    ONLY: dbcsr_put_block,&
                                              dbcsr_type
   USE cp_fm_cholesky,                  ONLY: cp_fm_cholesky_decompose,&
                                              cp_fm_cholesky_solve
   USE cp_fm_struct,                    ONLY: cp_fm_struct_create,&
                                              cp_fm_struct_release,&
                                              cp_fm_struct_type
   USE cp_fm_types,                     ONLY: cp_fm_create,&
                                              cp_fm_get_info,&
                                              cp_fm_get_submatrix,&
                                              cp_fm_release,&
                                              cp_fm_set_submatrix,&
                                              cp_fm_type
   USE kinds,                           ONLY: dp,&
                                              int_8
   USE message_passing,                 ONLY: mp_para_env_type
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'ri_rs_fitting'
   REAL(KIND=dp), PARAMETER, PRIVATE :: jacobi_floor = 1.0E-16_dp

   PUBLIC :: evaluate_ri_rs_fit, &
             solve_ri_rs_fit, &
             store_Z_lP_columns

CONTAINS

! **************************************************************************************************
!> \brief Computes the regularized RI-RS fit from an AO collocation matrix and a replicated RHS.
!>
!>          D_ll' = [Σ_μ ϕ_μ(r_l) ϕ_μ(r_l')]²,
!>          d_l = 1/sqrt(max(D_ll, jacobi_floor)),
!>          D'_ll' = d_l D_ll' d_l' + λ δ_ll',
!>          Σ_l' D'_ll' Z'_l'P = d_l d_lP,
!>          Z_lP = d_l Z'_lP.
!>
!>        The RI-RS least-squares construction and balancing are described by Duchemin and
!>        Blase, arXiv:1902.00066, Sec. II.3. Here jacobi_floor bounds D_ll from below in
!>        both the LAPACK and ScaLAPACK paths. The regularization λ is added after Jacobi
!>        scaling; in the unscaled system its diagonal contribution is λ/d_l².
!> \param phi_local AO values on the fitting grid, replicated within the solve subgroup
!> \param d_lp replicated RHS d_lP on entry and fitted coefficients Z_lP on return
!> \param tikhonov regularization parameter λ in the Jacobi-scaled space
!> \param para_env_sub optional subgroup communicator for the distributed solve
!> \param blacs_env_sub optional BLACS context, supplied together with para_env_sub
! **************************************************************************************************
   SUBROUTINE solve_ri_rs_fit(phi_local, d_lp, tikhonov, para_env_sub, blacs_env_sub)
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: phi_local
      REAL(KIND=dp), DIMENSION(:, :), INTENT(INOUT)      :: d_lp
      REAL(KIND=dp), INTENT(IN)                          :: tikhonov
      TYPE(mp_para_env_type), OPTIONAL, POINTER          :: para_env_sub
      TYPE(cp_blacs_env_type), OPTIONAL, POINTER         :: blacs_env_sub

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'solve_ri_rs_fit'

      INTEGER                                            :: handle, handle2, i, info, n_grid, n_rhs
      REAL(KIND=dp), ALLOCATABLE                         :: gram(:, :), jacobi_scale(:)

      CALL timeset(routineN, handle)

      n_grid = SIZE(phi_local, 1)
      n_rhs = SIZE(d_lp, 2)
      CPASSERT(SIZE(d_lp, 1) == n_grid)
      CPASSERT(PRESENT(para_env_sub) .EQV. PRESENT(blacs_env_sub))
      IF (PRESENT(para_env_sub)) THEN
         ! Only the row norms are replicated; the distributed solve builds its own Gram tiles.
         ALLOCATE (jacobi_scale(n_grid))
         !$OMP PARALLEL DO DEFAULT(NONE) &
         !$OMP SHARED(n_grid, phi_local, jacobi_scale) &
         !$OMP PRIVATE(i) SCHEDULE(STATIC)
         DO i = 1, n_grid
            jacobi_scale(i) = 1.0_dp/MAX(DOT_PRODUCT(phi_local(i, :), phi_local(i, :)), SQRT(jacobi_floor))
         END DO
         !$OMP END PARALLEL DO
      ELSE
         CALL factor_ri_rs_fit(phi_local, tikhonov, gram, jacobi_scale, info)
         IF (info /= 0) CPABORT('RI-RS Cholesky factorization failed')
      END IF

      ! Pre-scale the right-hand side: d_lP <- d_l d_lP.
      CALL scale_rows_by_diag(d_lp, jacobi_scale)
      IF (PRESENT(para_env_sub)) THEN
         CALL solve_D_lp_distributed(phi_local, jacobi_scale, d_lp, tikhonov, &
                                     para_env_sub, blacs_env_sub)
      ELSE
         ! Solve Σ_l' D'_ll' Z'_l'P = d_l d_lP, overwriting d_lp with Z'_lP.
         CALL timeset(routineN//'_dpotrs', handle2)
         CALL dpotrs('L', n_grid, n_rhs, gram, n_grid, d_lp, n_grid, info)
         CALL timestop(handle2)
         IF (info /= 0) CPABORT('RI-RS Cholesky solve failed')
      END IF
      ! Undo the conditioning on the solution: Z_lP = d_l Z'_lP.
      CALL scale_rows_by_diag(d_lp, jacobi_scale)

      CALL timestop(handle)

   END SUBROUTINE solve_ri_rs_fit

! **************************************************************************************************
!> \brief Build and factor the Jacobi-scaled squared AO Gram matrix, K = d D d + λ I.
!>        The lower triangle is used throughout; the factor is retained for subsequent solves.
!>        See solve_ri_rs_fit for the RI-RS balancing convention and reference.
!> \param phi AO values indexed (grid point, AO)
!> \param tikhonov regularization in the Jacobi-scaled space
!> \param factor lower Cholesky factor of K on successful return
!> \param jacobi_scale pair-feature scales d_l
!> \param info LAPACK factorization status; the caller chooses how to handle failure
!> \param overlap optional lower triangle of the scaled AO overlap V V^T, V_l = sqrt(d_l) phi_l
! **************************************************************************************************
   SUBROUTINE factor_ri_rs_fit(phi, tikhonov, factor, jacobi_scale, info, overlap)
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: phi
      REAL(KIND=dp), INTENT(IN)                          :: tikhonov
      REAL(KIND=dp), ALLOCATABLE, INTENT(OUT)            :: factor(:, :), jacobi_scale(:)
      INTEGER, INTENT(OUT)                               :: info
      REAL(KIND=dp), ALLOCATABLE, INTENT(OUT), OPTIONAL  :: overlap(:, :)

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'factor_ri_rs_fit'

      INTEGER                                            :: handle, handle2, i, j, n_grid

      CALL timeset(routineN, handle)
      n_grid = SIZE(phi, 1)
      ALLOCATE (factor(n_grid, n_grid), jacobi_scale(n_grid))
      ! First form S_ll' = Σ_μ phi_lμ phi_l'μ; D_ll' = S_ll'^2.
      CALL timeset(routineN//'_dsyrk', handle2)
      CALL dsyrk('L', 'N', n_grid, SIZE(phi, 2), 1.0_dp, phi, n_grid, 0.0_dp, factor, n_grid)
      CALL timestop(handle2)
      DO i = 1, n_grid
         jacobi_scale(i) = 1.0_dp/MAX(factor(i, i), SQRT(jacobi_floor))
      END DO
      IF (PRESENT(overlap)) THEN
         ALLOCATE (overlap(n_grid, n_grid))
         DO j = 1, n_grid
            DO i = j, n_grid
               overlap(i, j) = factor(i, j)*SQRT(jacobi_scale(i))*SQRT(jacobi_scale(j))
            END DO
         END DO
      END IF
      ! Add λ after Jacobi scaling; LAPACK needs only the lower triangle.
      !$OMP PARALLEL DO DEFAULT(NONE) &
      !$OMP SHARED(n_grid, factor, jacobi_scale, tikhonov) &
      !$OMP PRIVATE(j, i) SCHEDULE(DYNAMIC)
      DO j = 1, n_grid
         DO i = j, n_grid
            factor(i, j) = factor(i, j)**2*jacobi_scale(i)*jacobi_scale(j)
         END DO
         factor(j, j) = factor(j, j) + tikhonov
      END DO
      !$OMP END PARALLEL DO
      CALL timeset(routineN//'_dpotrf', handle2)
      CALL dpotrf('L', n_grid, factor, n_grid, info)
      CALL timestop(handle2)
      CALL timestop(handle)
   END SUBROUTINE factor_ri_rs_fit

! **************************************************************************************************
!> \brief Evaluate the normalized cluster residual and its grid-coordinate derivative.
!>
!>    Let ϕ_l contain the AO values ϕ_μ(r_l), and let B_P be the symmetric matrix of
!>    three-center integrals (μν|P). svec stores the upper triangle, multiplying off-diagonal
!>    entries by sqrt(2), so its Euclidean norm equals the full matrix Frobenius norm.
!>
!>    d_l = 1 / sqrt(max((Σ_μ ϕ_μ(r_l)²)², jacobi_floor)),  V_l = sqrt(d_l) ϕ_l,
!>    A_:l = svec(V_l V_l^T),  B_:P = svec(B_P),  S = V V^T,
!>    K = A^T A + λ I = S∘S + λ I,  b = A^T B,  K Z′ = b.
!>
!>    Here λ is tikhonov and ∘ denotes elementwise multiplication. The objective is
!>    E = ||B - A Z′||² / ||B||², without a penalty term. For λ > 0 the derivative
!>    includes the response of the regularized coefficients:
!>
!>    K Y = Z′,  R = B - A Z′,
!>    ∂E/∂A = 2 [λ (A Y) Z′^T - R (Z′ + λ Y)^T] / ||B||².
!>
!>    The coordinate derivative contracts ∂E/∂A with ∂A/∂r_lα, including the derivative
!>    of d_l above the floor. For λ = 0 the numerical pseudoinverse is used; the residual
!>    derivative applies on a fixed-rank branch. The balancing convention is in solve_ri_rs_fit.
!>    K is independent of P: factor it once and solve panels of RI columns. Each rank holds
!>    complete AO-pair columns for its RI functions. Fixed AO-pair blocks and ordered sums
!>    over RI columns keep the summation order independent of their MPI distribution.
!> \param Phi_l_mu Unscaled AO values ϕ_μ(r_l), indexed (grid point, AO).
!> \param dPhi_alpha_l_mu AO coordinate derivatives ∂ϕ_μ(r_l)/∂r_lα, indexed (α, grid point, AO).
!> \param Int_3c_munu_P Symmetry-weighted upper AO pairs for local RI columns, B above;
!>    pair index ν(ν-1)/2 + μ for 1 <= μ <= ν <= nAO, with nAO(nAO+1)/2 rows.
!> \param Int_3c_norm2 Squared norm ||B||² summed over all RI columns in the atom MPI group.
!> \param tikhonov Regularization λ in the Jacobi-scaled fitting equations.
!> \param atom_env MPI group distributing the cluster's RI columns and large fitting matrices.
!> \param available_GB Available memory estimate per rank, used to select the local factorization.
!> \param normalized_error Normalized squared residual E, without the regularization penalty.
!> \param coordinate_gradient Derivative ∂E/∂r_lα, indexed (α, grid point).
!> \param max_abs_error Largest absolute unweighted three-center-integral error.
!> \param fit_successful False for an empty target or a failed regularized factorization.
! **************************************************************************************************
   SUBROUTINE evaluate_ri_rs_fit(Phi_l_mu, dPhi_alpha_l_mu, Int_3c_munu_P, &
                                 Int_3c_norm2, tikhonov, atom_env, available_GB, &
                                 normalized_error, coordinate_gradient, max_abs_error, fit_successful)
      REAL(KIND=dp), INTENT(IN)                          :: Phi_l_mu(:, :), &
                                                            dPhi_alpha_l_mu(:, :, :), &
                                                            Int_3c_munu_P(:, :), Int_3c_norm2, &
                                                            tikhonov
      TYPE(mp_para_env_type), POINTER                    :: atom_env
      REAL(KIND=dp), INTENT(IN)                          :: available_GB
      REAL(KIND=dp), INTENT(OUT)                         :: normalized_error, &
                                                            coordinate_gradient(:, :), &
                                                            max_abs_error
      LOGICAL, INTENT(OUT)                               :: fit_successful

      CHARACTER(len=*), PARAMETER :: routineN = 'evaluate_ri_rs_fit'
      INTEGER, PARAMETER                                 :: local_cholesky_limit = 3072, &
                                                            pair_block_size = 128, &
                                                            ri_panel_size = 32

      INTEGER :: contribution_sum_handle, factor_handle, first_owned_RI, first_pair, first_RI, &
         fit_derivative_handle, fit_solve_handle, handle, i_ao_pair, icartesian, ieigenvalue, &
         igrid_point, imu, inu, ip, jp, lapack_info, last_owned_RI, last_pair, local_first_col, &
         local_first_RI, local_last_RI, n_grid_points, nAO, nblock, ncol_local, ncols, &
         ncols_local, npairs, nRI, nrow_local, rhs_sum_handle, ri_block_size
      INTEGER(KIND=int_8)                                :: available_bytes, factor_bytes
      INTEGER, ALLOCATABLE                               :: mu_of_pair(:), nu_of_pair(:)
      INTEGER, POINTER                                   :: col_indices(:), row_indices(:)
      LOGICAL                                            :: local_cholesky
      REAL(KIND=dp)                                      :: dPhi_munu_l, fit_rank_tolerance, &
                                                            jacobi_scale, Phi_munu_norm2, &
                                                            scaling_projection
      REAL(KIND=dp), ALLOCATABLE :: col_values(:, :), contributions(:, :), dE_dPhi_munu_l(:, :), &
         dPhi_munu_l_scaled(:), fit_eigenvalues(:), fit_eigenvectors(:, :), &
         fit_projected_rhs(:, :), fit_work(:), integral_panel(:, :), jacobi_scaling(:), &
         local_factor(:, :), pair_symmetry(:), Phi_munu_l(:, :), Phi_munu_l_scaled(:, :), &
         Phi_munu_norm2_l(:), residual(:, :), response(:, :), rhs(:, :), row_values(:, :), &
         Y_effective_lP(:, :), Y_lP(:, :), Z_prime_lP(:, :)
      REAL(KIND=dp), CONTIGUOUS, POINTER                 :: local_data(:, :)
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env
      TYPE(cp_fm_struct_type), POINTER                   :: fm_struct_b, fm_struct_D
      TYPE(cp_fm_type)                                   :: fm_b, fm_D

      CALL timeset(routineN, handle)

      n_grid_points = SIZE(Phi_l_mu, 1)
      nAO = SIZE(Phi_l_mu, 2)
      npairs = nAO*(nAO + 1)/2
      nRI = SIZE(Int_3c_munu_P, 2)
      CALL atom_env%sum(nRI)
      first_owned_RI = nRI*atom_env%mepos/atom_env%num_pe + 1
      last_owned_RI = nRI*(atom_env%mepos + 1)/atom_env%num_pe
      CPASSERT(SIZE(Int_3c_munu_P, 1) == npairs)
      CPASSERT(SIZE(Int_3c_munu_P, 2) == last_owned_RI - first_owned_RI + 1)
      CPASSERT(ALL(SHAPE(coordinate_gradient) == [3, n_grid_points]))
      normalized_error = HUGE(normalized_error)
      coordinate_gradient = 0.0_dp
      max_abs_error = 0.0_dp
      fit_successful = .FALSE.
      IF (n_grid_points < 1 .OR. nRI < 1 .OR. Int_3c_norm2 <= TINY(1.0_dp)) THEN
         CALL timestop(handle)
         RETURN
      END IF
      ALLOCATE (mu_of_pair(npairs), nu_of_pair(npairs), pair_symmetry(npairs))
      DO inu = 1, nAO
         DO imu = 1, inu
            i_ao_pair = inu*(inu - 1)/2 + imu
            mu_of_pair(i_ao_pair) = imu
            nu_of_pair(i_ao_pair) = inu
            pair_symmetry(i_ao_pair) = SQRT(REAL(2 - MERGE(1, 0, imu == inu), dp))
         END DO
      END DO
      available_bytes = INT(0.8_dp*MAX(0.0_dp, available_GB)*1.0E9_dp, int_8)
      factor_bytes = 8_int_8*INT(n_grid_points, int_8)**2
      local_cholesky = tikhonov > 0.0_dp .AND. n_grid_points <= local_cholesky_limit .AND. &
                       factor_bytes <= available_bytes/20_int_8
      NULLIFY (fm_struct_D, fm_struct_b, blacs_env)
      IF (.NOT. local_cholesky) THEN
         CALL cp_blacs_env_create(blacs_env=blacs_env, para_env=atom_env)
         CALL cp_fm_struct_create(fm_struct_D, para_env=atom_env, context=blacs_env, &
                                  nrow_global=n_grid_points, ncol_global=n_grid_points)
      END IF
      ri_block_size = MIN(ri_panel_size, nRI)
      ALLOCATE (jacobi_scaling(n_grid_points), Phi_munu_norm2_l(n_grid_points))
      DO igrid_point = 1, n_grid_points
         Phi_munu_norm2 = SUM(Phi_l_mu(igrid_point, :)**2)**2
         Phi_munu_norm2_l(igrid_point) = Phi_munu_norm2
         jacobi_scaling(igrid_point) = 1.0_dp/SQRT(MAX(Phi_munu_norm2, jacobi_floor))
      END DO
      IF (local_cholesky) THEN
         CALL timeset("rirs_grid_local_factor", factor_handle)

         ALLOCATE (local_factor(n_grid_points, n_grid_points))
         CALL dgemm('N', 'T', n_grid_points, n_grid_points, nAO, 1.0_dp, &
                    Phi_l_mu, n_grid_points, Phi_l_mu, n_grid_points, &
                    0.0_dp, local_factor, n_grid_points)
         ! D′_ll′ = s_l s_l′ [Σ_μ ϕ_μ(r_l) ϕ_μ(r_l′)]² + λ δ_ll′.
         ! Here s is jacobi_scaling and λ is tikhonov.
         DO ip = 1, n_grid_points
            DO igrid_point = 1, n_grid_points
               local_factor(igrid_point, ip) = local_factor(igrid_point, ip)**2* &
                                               jacobi_scaling(igrid_point)*jacobi_scaling(ip)
            END DO
            local_factor(ip, ip) = local_factor(ip, ip) + tikhonov
         END DO
         CALL dpotrf('U', n_grid_points, local_factor, n_grid_points, lapack_info)
         CALL timestop(factor_handle)
      ELSE
         CALL cp_fm_create(fm_D, fm_struct_D)
         CALL cp_fm_get_info(fm_D, nrow_local=nrow_local, ncol_local=ncol_local, &
                             row_indices=row_indices, col_indices=col_indices, &
                             local_data=local_data)
         IF (nrow_local > 0 .AND. ncol_local > 0) THEN
            ALLOCATE (row_values(nrow_local, nAO), col_values(nAO, ncol_local))
            DO imu = 1, nAO
               row_values(:, imu) = Phi_l_mu(row_indices(:nrow_local), imu)
               col_values(imu, :) = Phi_l_mu(col_indices(:ncol_local), imu)
            END DO
            CALL dgemm('N', 'N', nrow_local, ncol_local, nAO, 1.0_dp, &
                       row_values, nrow_local, col_values, nAO, 0.0_dp, &
                       local_data, SIZE(local_data, 1))
            DO ip = 1, ncol_local
               DO igrid_point = 1, nrow_local
                  local_data(igrid_point, ip) = local_data(igrid_point, ip)**2* &
                                                jacobi_scaling(row_indices(igrid_point))* &
                                                jacobi_scaling(col_indices(ip))
                  IF (row_indices(igrid_point) == col_indices(ip)) THEN
                     local_data(igrid_point, ip) = local_data(igrid_point, ip) + tikhonov
                  END IF
               END DO
            END DO
            DEALLOCATE (row_values, col_values)
         END IF
      END IF
      IF (tikhonov <= 0.0_dp) THEN
         ! The Moore-Penrose solution of D gives the unregularized
         ! minimum-norm least-squares fit even when the grid is redundant.
         ALLOCATE (fit_eigenvectors(n_grid_points, n_grid_points), &
                   fit_eigenvalues(n_grid_points), &
                   fit_work(MAX(1, 3*n_grid_points - 1)), &
                   fit_projected_rhs(n_grid_points, ri_block_size))
         CALL cp_fm_get_submatrix(fm_D, fit_eigenvectors)
         CALL dsyev('V', 'L', n_grid_points, fit_eigenvectors, n_grid_points, &
                    fit_eigenvalues, fit_work, SIZE(fit_work), lapack_info)
         IF (lapack_info /= 0) CPABORT('Unregularized RI-RS grid-fit eigensolver failed')
         fit_rank_tolerance = EPSILON(1.0_dp)*REAL(n_grid_points, dp)* &
                              MAXVAL(ABS(fit_eigenvalues))
         IF (MINVAL(fit_eigenvalues) < -10.0_dp*fit_rank_tolerance) THEN
            CPABORT('Unregularized RI-RS grid-fit matrix is not positive semidefinite')
         END IF
         IF (.NOT. ANY(fit_eigenvalues > fit_rank_tolerance)) THEN
            CPABORT('Unregularized RI-RS grid-fit matrix has zero numerical rank')
         END IF
      ELSE
         IF (.NOT. local_cholesky) THEN
            CALL cp_fm_cholesky_decompose(fm_D, n=n_grid_points, info_out=lapack_info)
         END IF
         IF (lapack_info /= 0) THEN
            IF (local_cholesky) THEN
               DEALLOCATE (local_factor)
            ELSE
               CALL cp_fm_release(fm_D)
               CALL cp_fm_struct_release(fm_struct_D)
               CALL cp_blacs_env_release(blacs_env)
            END IF
            CALL timestop(handle)

            RETURN
         END IF
      END IF
      IF (.NOT. local_cholesky) THEN
         CALL cp_fm_struct_create(fm_struct_b, template_fmstruct=fm_struct_D, &
                                  nrow_global=n_grid_points, ncol_global=ri_block_size, force_block=.TRUE.)
         CALL cp_fm_create(fm_b, fm_struct_b)
      END IF
      ALLOCATE (rhs(n_grid_points, ri_block_size), Z_prime_lP(n_grid_points, ri_block_size), &
                Y_lP(n_grid_points, ri_block_size), Y_effective_lP(n_grid_points, ri_block_size), &
                Phi_munu_l(pair_block_size, n_grid_points), Phi_munu_l_scaled(pair_block_size, n_grid_points), &
                residual(pair_block_size, ri_block_size), response(pair_block_size, ri_block_size), &
                dE_dPhi_munu_l(pair_block_size, n_grid_points), &
                dPhi_munu_l_scaled(pair_block_size), contributions(1 + 3*n_grid_points, ri_block_size), &
                integral_panel(pair_block_size, ri_block_size))
      ! K is independent of P. Reuse its factorization for each panel of right-hand sides.
      normalized_error = 0.0_dp
      DO first_RI = 1, nRI, ri_block_size
         ncols = MIN(ri_block_size, nRI - first_RI + 1)
         local_first_RI = MAX(first_RI, first_owned_RI)
         local_last_RI = MIN(first_RI + ncols - 1, last_owned_RI)
         ncols_local = MAX(0, local_last_RI - local_first_RI + 1)
         rhs = 0.0_dp
         IF (ncols_local > 0) THEN
            local_first_col = local_first_RI - first_RI + 1
            DO first_pair = 1, npairs, pair_block_size
               last_pair = MIN(first_pair + pair_block_size - 1, npairs)
               nblock = last_pair - first_pair + 1
               CALL fill_pair_products(Phi_l_mu, jacobi_scaling, mu_of_pair, nu_of_pair, &
                                       pair_symmetry, first_pair, nblock, Phi_munu_l, Phi_munu_l_scaled)
               integral_panel(1:nblock, 1:ncols_local) = &
                  Int_3c_munu_P(first_pair:last_pair, &
                                local_first_RI - first_owned_RI + 1:local_last_RI - first_owned_RI + 1)
               CALL dgemm('T', 'N', n_grid_points, ncols_local, nblock, 1.0_dp, &
                          Phi_munu_l_scaled, pair_block_size, integral_panel, pair_block_size, &
                          1.0_dp, rhs(:, local_first_col:), n_grid_points)
            END DO
         END IF
         IF (.NOT. local_cholesky) THEN
            CALL timeset('rirs_fit_rhs_sum', rhs_sum_handle)
            CALL atom_env%sum(rhs)
            CALL timestop(rhs_sum_handle)
         END IF
         Z_prime_lP = 0.0_dp
         Y_lP = 0.0_dp
         CALL timeset('rirs_fit_solve', fit_solve_handle)
         IF (tikhonov <= 0.0_dp) THEN
            DO ip = 1, ncols
               fit_projected_rhs(:, ip) = MATMUL(TRANSPOSE(fit_eigenvectors), rhs(:, ip))
               DO ieigenvalue = 1, n_grid_points
                  IF (fit_eigenvalues(ieigenvalue) > fit_rank_tolerance) THEN
                     fit_projected_rhs(ieigenvalue, ip) = &
                        fit_projected_rhs(ieigenvalue, ip)/fit_eigenvalues(ieigenvalue)
                  ELSE
                     fit_projected_rhs(ieigenvalue, ip) = 0.0_dp
                  END IF
               END DO
               Z_prime_lP(:, ip) = MATMUL(fit_eigenvectors, fit_projected_rhs(:, ip))
            END DO
         ELSE IF (local_cholesky) THEN
            IF (ncols_local > 0) THEN
               local_first_col = local_first_RI - first_RI + 1
               Z_prime_lP(:, local_first_col:local_first_col + ncols_local - 1) = &
                  rhs(:, local_first_col:local_first_col + ncols_local - 1)
               CALL dpotrs('U', n_grid_points, ncols_local, local_factor, n_grid_points, &
                           Z_prime_lP(:, local_first_col:), n_grid_points, lapack_info)
               IF (lapack_info /= 0) CPABORT('RI-RS grid fit Cholesky solve failed')
               Y_lP(:, local_first_col:local_first_col + ncols_local - 1) = &
                  Z_prime_lP(:, local_first_col:local_first_col + ncols_local - 1)
               CALL dpotrs('U', n_grid_points, ncols_local, local_factor, n_grid_points, &
                           Y_lP(:, local_first_col:), n_grid_points, lapack_info)
               IF (lapack_info /= 0) CPABORT('RI-RS grid fit response solve failed')
            END IF
         ELSE
            CALL cp_fm_set_submatrix(fm_b, rhs)
            CALL cp_fm_cholesky_solve(fm_D, fm_b, n=n_grid_points, info_out=lapack_info)
            IF (lapack_info /= 0) CPABORT('RI-RS grid fit Cholesky solve failed')
            CALL cp_fm_get_submatrix(fm_b, Z_prime_lP)
            CALL cp_fm_cholesky_solve(fm_D, fm_b, n=n_grid_points, info_out=lapack_info)
            IF (lapack_info /= 0) CPABORT('RI-RS grid fit response solve failed')
            CALL cp_fm_get_submatrix(fm_b, Y_lP)
         END IF
         CALL timestop(fit_solve_handle)
         ! Keep Z′ and Y = K^(-1) Z′ for the coefficient response in the residual derivative.
         Y_effective_lP(:, :) = Z_prime_lP + tikhonov*Y_lP
         contributions = 0.0_dp
         CALL timeset('rirs_fit_derivative', fit_derivative_handle)
         DO first_pair = 1, npairs, pair_block_size
            IF (ncols_local == 0) CYCLE
            last_pair = MIN(first_pair + pair_block_size - 1, npairs)
            nblock = last_pair - first_pair + 1
            CALL fill_pair_products(Phi_l_mu, jacobi_scaling, mu_of_pair, nu_of_pair, &
                                    pair_symmetry, first_pair, nblock, Phi_munu_l, Phi_munu_l_scaled)
            ! The AO-pair derivatives depend on the grid and pair block, not on the RI column.
            ! Store the local RI residuals before contracting each derivative only once.
            local_first_col = local_first_RI - first_RI + 1
            integral_panel(1:nblock, 1:ncols_local) = &
               Int_3c_munu_P(first_pair:last_pair, &
                             local_first_RI - first_owned_RI + 1:local_last_RI - first_owned_RI + 1)
            ! K Z′ = b implies ||B-AZ′||² = ||B||² - tr(Z′^T b) - λ ||Z′||².
            ! Form B-AZ′ explicitly here to avoid cancellation between these large terms.
            residual(1:nblock, local_first_col:local_first_col + ncols_local - 1) = &
               integral_panel(1:nblock, 1:ncols_local)
            CALL dgemm('N', 'N', nblock, ncols_local, n_grid_points, -1.0_dp, &
                       Phi_munu_l_scaled, pair_block_size, Z_prime_lP(:, local_first_col:), n_grid_points, &
                       1.0_dp, residual(:, local_first_col:), pair_block_size)
            CALL dgemm('N', 'N', nblock, ncols_local, n_grid_points, 1.0_dp, &
                       Phi_munu_l_scaled, pair_block_size, Y_lP(:, local_first_col:), n_grid_points, &
                       0.0_dp, response(:, local_first_col:), pair_block_size)
            DO ip = local_first_RI, local_last_RI
               jp = ip - first_RI + 1
               contributions(1, jp) = contributions(1, jp) + &
                                      SUM(residual(1:nblock, jp)**2)/Int_3c_norm2
               max_abs_error = MAX(max_abs_error, &
                                   MAXVAL(ABS(residual(1:nblock, jp))/pair_symmetry(first_pair:last_pair)))
            END DO
            ! Sum the RI response across the locally owned columns with matrix products.
            CALL dgemm('N', 'T', nblock, n_grid_points, ncols_local, &
                       2.0_dp*tikhonov/Int_3c_norm2, response(:, local_first_col:), &
                       pair_block_size, Z_prime_lP(:, local_first_col:), n_grid_points, &
                       0.0_dp, dE_dPhi_munu_l, pair_block_size)
            CALL dgemm('N', 'T', nblock, n_grid_points, ncols_local, &
                       -2.0_dp/Int_3c_norm2, residual(:, local_first_col:), &
                       pair_block_size, Y_effective_lP(:, local_first_col:), n_grid_points, &
                       1.0_dp, dE_dPhi_munu_l, pair_block_size)
            DO igrid_point = 1, n_grid_points
               jacobi_scale = jacobi_scaling(igrid_point)
               DO icartesian = 1, 3
                  DO i_ao_pair = 1, nblock
                     imu = mu_of_pair(first_pair + i_ao_pair - 1)
                     inu = nu_of_pair(first_pair + i_ao_pair - 1)
                     dPhi_munu_l = pair_symmetry(first_pair + i_ao_pair - 1)*( &
                                   dPhi_alpha_l_mu(icartesian, igrid_point, imu)*Phi_l_mu(igrid_point, inu) + &
                                   Phi_l_mu(igrid_point, imu)*dPhi_alpha_l_mu(icartesian, igrid_point, inu))
                     dPhi_munu_l_scaled(i_ao_pair) = jacobi_scale*dPhi_munu_l
                  END DO
                  IF (Phi_munu_norm2_l(igrid_point) > jacobi_floor) THEN
                     scaling_projection = 2.0_dp*SUM(Phi_l_mu(igrid_point, :)**2)* &
                                          SUM(Phi_l_mu(igrid_point, :)*dPhi_alpha_l_mu(icartesian, igrid_point, :))
                     dPhi_munu_l_scaled(1:nblock) = dPhi_munu_l_scaled(1:nblock) - &
                                                    jacobi_scale**3*Phi_munu_l(1:nblock, igrid_point)*scaling_projection
                  END IF
                  jp = local_first_RI - first_RI + 1
                  contributions(1 + 3*(igrid_point - 1) + icartesian, jp) = &
                     contributions(1 + 3*(igrid_point - 1) + icartesian, jp) + &
                     DOT_PRODUCT(dE_dPhi_munu_l(1:nblock, igrid_point), dPhi_munu_l_scaled(1:nblock))
               END DO
            END DO
         END DO
         CALL timestop(fit_derivative_handle)
         DO ip = 1, ncols
            normalized_error = normalized_error + contributions(1, ip)
            coordinate_gradient = coordinate_gradient + RESHAPE(contributions(2:, ip), SHAPE(coordinate_gradient))
         END DO
      END DO
      CALL timeset('rirs_fit_contribution_sum', contribution_sum_handle)
      contributions(1, 1) = normalized_error
      contributions(2:, 1) = RESHAPE(coordinate_gradient, [3*n_grid_points])
      CALL atom_env%sum(contributions(:, 1))
      normalized_error = contributions(1, 1)
      coordinate_gradient = RESHAPE(contributions(2:, 1), SHAPE(coordinate_gradient))
      CALL timestop(contribution_sum_handle)
      CALL atom_env%max(max_abs_error)
      fit_successful = .TRUE.
      IF (local_cholesky) THEN
         DEALLOCATE (local_factor)
      ELSE
         CALL cp_fm_release(fm_b)
         CALL cp_fm_release(fm_D)
         CALL cp_fm_struct_release(fm_struct_b)
         CALL cp_fm_struct_release(fm_struct_D)
         CALL cp_blacs_env_release(blacs_env)
      END IF

      CALL timestop(handle)

   END SUBROUTINE evaluate_ri_rs_fit

! **************************************************************************************************
!> \brief Evaluate AO-pair products for the current block of pairs.
!> \param Phi_l_mu ...
!> \param jacobi_scaling ...
!> \param mu_of_pair ...
!> \param nu_of_pair ...
!> \param pair_symmetry ...
!> \param first_pair ...
!> \param nblock ...
!> \param Phi_munu_l ...
!> \param Phi_munu_l_scaled ...
! **************************************************************************************************
   SUBROUTINE fill_pair_products(Phi_l_mu, jacobi_scaling, mu_of_pair, nu_of_pair, &
                                 pair_symmetry, first_pair, nblock, Phi_munu_l, Phi_munu_l_scaled)
      REAL(KIND=dp), INTENT(IN)                          :: Phi_l_mu(:, :), jacobi_scaling(:)
      INTEGER, INTENT(IN)                                :: mu_of_pair(:), nu_of_pair(:)
      REAL(KIND=dp), INTENT(IN)                          :: pair_symmetry(:)
      INTEGER, INTENT(IN)                                :: first_pair, nblock
      REAL(KIND=dp), INTENT(OUT)                         :: Phi_munu_l(:, :), Phi_munu_l_scaled(:, :)

      INTEGER                                            :: i_ao_pair, igrid_point, imu, inu

      DO igrid_point = 1, SIZE(Phi_l_mu, 1)
         DO i_ao_pair = 1, nblock
            imu = mu_of_pair(first_pair + i_ao_pair - 1)
            inu = nu_of_pair(first_pair + i_ao_pair - 1)
            Phi_munu_l(i_ao_pair, igrid_point) = &
               pair_symmetry(first_pair + i_ao_pair - 1)* &
               Phi_l_mu(igrid_point, imu)*Phi_l_mu(igrid_point, inu)
         END DO
         Phi_munu_l_scaled(1:nblock, igrid_point) = &
            jacobi_scaling(igrid_point)*Phi_munu_l(1:nblock, igrid_point)
      END DO
   END SUBROUTINE fill_pair_products

! **************************************************************************************************
!> \brief Multiplies every matrix row by the corresponding diagonal entry:
!>        A(l, :) <- d_l A(l, :).
!> \param matrix ...
!> \param diagonal ...
! **************************************************************************************************
   SUBROUTINE scale_rows_by_diag(matrix, diagonal)

      REAL(KIND=dp), DIMENSION(:, :), INTENT(INOUT)      :: matrix
      REAL(KIND=dp), DIMENSION(:), INTENT(IN)            :: diagonal

      CHARACTER(LEN=*), PARAMETER :: routineN = 'scale_rows_by_diag'

      INTEGER                                            :: handle, i, j

      CALL timeset(routineN, handle)
      CPASSERT(SIZE(diagonal) == SIZE(matrix, 1))

      !$OMP PARALLEL DO DEFAULT(NONE) &
      !$OMP SHARED(matrix, diagonal) &
      !$OMP PRIVATE(j, i) &
      !$OMP SCHEDULE(STATIC)
      DO j = 1, SIZE(matrix, 2)
         DO i = 1, SIZE(matrix, 1)
            matrix(i, j) = matrix(i, j)*diagonal(i)
         END DO
      END DO
      !$OMP END PARALLEL DO

      CALL timestop(handle)

   END SUBROUTINE scale_rows_by_diag

! **************************************************************************************************
!> \brief Stores dense Z_lP columns in the distributed block-sparse matrix.
!> \param mat_Z_lP ...
!> \param Z_local ...
!> \param local_grid_idx ...
!> \param n_local_grid ...
!> \param n_loc_ri ...
!> \param atom_P ...
!> \param r_blk_sizes ...
!> \param row_offset ...
!> \param eps_filter ...
! **************************************************************************************************
   SUBROUTINE store_Z_lP_columns(mat_Z_lP, Z_local, local_grid_idx, n_local_grid, n_loc_ri, &
                                 atom_P, r_blk_sizes, row_offset, eps_filter)

      TYPE(dbcsr_type), INTENT(INOUT)                    :: mat_Z_lP
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: Z_local
      INTEGER, DIMENSION(:), INTENT(IN)                  :: local_grid_idx
      INTEGER, INTENT(IN)                                :: n_local_grid, n_loc_ri, atom_P
      INTEGER, DIMENSION(:), INTENT(IN)                  :: r_blk_sizes, row_offset
      REAL(KIND=dp), INTENT(IN)                          :: eps_filter

      CHARACTER(LEN=*), PARAMETER :: routineN = 'store_Z_lP_columns'

      INTEGER                                            :: current_chunk_size, g_pt, handle, i_blk, &
                                                            loc_ptr, r_end, r_start
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: Z_blk

      CALL timeset(routineN, handle)

      ALLOCATE (Z_blk(MAXVAL(r_blk_sizes), n_loc_ri))
      ! local_grid_idx is ascending, so one walking pointer maps sphere rows to global grid blocks.
      loc_ptr = 1

      DO i_blk = 1, SIZE(r_blk_sizes)
         r_start = row_offset(i_blk) + 1
         r_end = row_offset(i_blk) + r_blk_sizes(i_blk)
         current_chunk_size = r_blk_sizes(i_blk)
         Z_blk = 0.0_dp

         DO WHILE (loc_ptr <= n_local_grid)
            g_pt = local_grid_idx(loc_ptr)
            IF (g_pt > r_end) EXIT
            Z_blk(g_pt - r_start + 1, 1:n_loc_ri) = Z_local(loc_ptr, 1:n_loc_ri)
            loc_ptr = loc_ptr + 1
         END DO

         ! DBCSR routes each retained block to its global owner when the caller finalizes mat_Z_lP.
         IF (MAXVAL(ABS(Z_blk(1:current_chunk_size, 1:n_loc_ri))) > eps_filter) THEN
            CALL dbcsr_put_block(mat_Z_lP, row=i_blk, col=atom_P, &
                                 block=Z_blk(1:current_chunk_size, 1:n_loc_ri))
         END IF
      END DO

      DEALLOCATE (Z_blk)

      CALL timestop(handle)

   END SUBROUTINE store_Z_lP_columns

! **************************************************************************************************
!> \brief Solves the Jacobi-scaled, regularized RI-RS system with ScaLAPACK.
!>
!>        Each rank builds its block-cyclic part of
!>
!>          D'_ll' = d_l [Σ_μ ϕ_μ(r_l) ϕ_μ(r_l')]² d_l' + λ δ_ll'.
!>
!>        The Cholesky solution Z'_lP is gathered into the replicated right-hand side.
!>        The caller then restores Z_lP = d_l Z'_lP.
!> \param phi_local replicated AO values on the fitting grid
!> \param d_vec Jacobi scaling vector d_l, with the same floor as in the dense solve
!> \param d_lp replicated scaled RHS d_l d_lP on entry, intermediate solution Z'_lP on return
!> \param tikhonov ...
!> \param para_env_sub ...
!> \param blacs_env_sub ...
! **************************************************************************************************
   SUBROUTINE solve_D_lp_distributed(phi_local, d_vec, d_lp, tikhonov, para_env_sub, blacs_env_sub)

      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: phi_local
      REAL(KIND=dp), DIMENSION(:), INTENT(IN)            :: d_vec
      REAL(KIND=dp), DIMENSION(:, :), INTENT(INOUT)      :: d_lp
      REAL(KIND=dp), INTENT(IN)                          :: tikhonov
      TYPE(mp_para_env_type), POINTER                    :: para_env_sub
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env_sub

      CHARACTER(LEN=*), PARAMETER :: routineN = 'solve_D_lp_distributed'

      INTEGER                                            :: handle, i_loc, ig, info, j_loc, jg, &
                                                            n_ao, n_loc, n_rhs, ncol_local, &
                                                            nrow_local
      INTEGER, DIMENSION(:), POINTER                     :: col_indices, row_indices
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :), &
         POINTER                                         :: local_data
      TYPE(cp_fm_struct_type), POINTER                   :: fm_struct_b, fm_struct_D
      TYPE(cp_fm_type)                                   :: fm_b, fm_D

      CALL timeset(routineN, handle)
      n_loc = SIZE(phi_local, 1)
      n_ao = SIZE(phi_local, 2)
      n_rhs = SIZE(d_lp, 2)

      NULLIFY (fm_struct_D, fm_struct_b)
      CALL cp_fm_struct_create(fm_struct_D, para_env=para_env_sub, &
                               context=blacs_env_sub, &
                               nrow_global=n_loc, ncol_global=n_loc)
      CALL cp_fm_struct_create(fm_struct_b, para_env=para_env_sub, &
                               context=blacs_env_sub, &
                               nrow_global=n_loc, ncol_global=n_rhs)
      CALL cp_fm_create(fm_D, fm_struct_D)
      CALL cp_fm_create(fm_b, fm_struct_b)

      CALL cp_fm_get_info(fm_D, nrow_local=nrow_local, ncol_local=ncol_local, &
                          row_indices=row_indices, col_indices=col_indices, &
                          local_data=local_data)

      ! Build only the rank-owned Gram tiles, avoiding a replicated n_grid × n_grid matrix.
      IF (nrow_local > 0 .AND. ncol_local > 0) THEN
         BLOCK
            INTEGER, PARAMETER :: ntile = 1024
            INTEGER :: ib, ie, jb, je, mb, kb, ti, tj, handle_dgemm
            REAL(KIND=dp), ALLOCATABLE :: gram_t(:, :), phi_cols_t(:, :), phi_rows_t(:, :)
            ALLOCATE (phi_rows_t(ntile, n_ao), phi_cols_t(n_ao, ntile), gram_t(ntile, ntile))
            ! Pack AO values for the grid rows/columns selected by the block-cyclic FM layout.
            DO ib = 1, nrow_local, ntile
               ie = MIN(ib + ntile - 1, nrow_local)
               mb = ie - ib + 1
               !$OMP PARALLEL DO DEFAULT(NONE) &
               !$OMP SHARED(mb, n_ao, phi_rows_t, phi_local, row_indices, ib) &
               !$OMP PRIVATE(ti, j_loc) SCHEDULE(STATIC)
               DO j_loc = 1, n_ao
                  DO ti = 1, mb
                     phi_rows_t(ti, j_loc) = phi_local(row_indices(ib + ti - 1), j_loc)
                  END DO
               END DO
               !$OMP END PARALLEL DO
               DO jb = 1, ncol_local, ntile
                  je = MIN(jb + ntile - 1, ncol_local)
                  kb = je - jb + 1
                  !$OMP PARALLEL DO DEFAULT(NONE) &
                  !$OMP SHARED(kb, n_ao, phi_cols_t, phi_local, col_indices, jb) &
                  !$OMP PRIVATE(tj, i_loc) SCHEDULE(STATIC)
                  DO tj = 1, kb
                     DO i_loc = 1, n_ao
                        phi_cols_t(i_loc, tj) = phi_local(col_indices(jb + tj - 1), i_loc)
                     END DO
                  END DO
                  !$OMP END PARALLEL DO
                  CALL timeset(routineN//'_dgemm', handle_dgemm)
                  CALL dgemm('N', 'N', mb, kb, n_ao, &
                             1.0_dp, phi_rows_t, ntile, phi_cols_t, n_ao, &
                             0.0_dp, gram_t, ntile)
                  CALL timestop(handle_dgemm)
                  ! Square the AO Gram entries, apply d_l d_l', and add λ on the global diagonal.
                  !$OMP PARALLEL DO DEFAULT(NONE) &
                  !$OMP SHARED(mb, kb, gram_t, d_vec, row_indices, col_indices, ib, jb) &
                  !$OMP SHARED(local_data, tikhonov) &
                  !$OMP PRIVATE(ti, tj, ig, jg) SCHEDULE(STATIC)
                  DO tj = 1, kb
                     jg = col_indices(jb + tj - 1)
                     DO ti = 1, mb
                        ig = row_indices(ib + ti - 1)
                        local_data(ib + ti - 1, jb + tj - 1) = &
                           gram_t(ti, tj)*gram_t(ti, tj)*d_vec(ig)*d_vec(jg)
                        IF (ig == jg) THEN
                           local_data(ib + ti - 1, jb + tj - 1) = &
                              local_data(ib + ti - 1, jb + tj - 1) + tikhonov
                        END IF
                     END DO
                  END DO
                  !$OMP END PARALLEL DO
               END DO
            END DO
            DEALLOCATE (phi_rows_t, phi_cols_t, gram_t)
         END BLOCK
      END IF

      ! Distribute the pre-scaled RHS and solve for Z'_lP within the BLACS subgroup.
      CALL cp_fm_set_submatrix(fm_b, d_lp)

      CALL cp_fm_cholesky_decompose(fm_D, n=n_loc, info_out=info)
      IF (info /= 0) CPABORT("pdpotrf failed in solve_D_lp_distributed")

      CALL cp_fm_cholesky_solve(fm_D, fm_b, n=n_loc, info_out=info)
      IF (info /= 0) CPABORT("pdpotrs failed in solve_D_lp_distributed")

      ! Replicate Z'_lP within the subgroup; the caller applies the final Jacobi rescaling.
      CALL cp_fm_get_submatrix(fm_b, d_lp)

      CALL cp_fm_release(fm_D)
      CALL cp_fm_release(fm_b)
      CALL cp_fm_struct_release(fm_struct_D)
      CALL cp_fm_struct_release(fm_struct_b)

      CALL timestop(handle)

   END SUBROUTINE solve_D_lp_distributed

END MODULE ri_rs_fitting
