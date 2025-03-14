!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief A common interface (wrapper) for a callback into the md_run loop.
!>        Currently this is only used by the glbopt machinery, but its meant
!>        to be extended if others need to control the md_run loop, too.
!>
!> \par History
!>      11.2012 created [Ole]
!> \author Ole
! **************************************************************************************************
MODULE mdctrl_methods
   USE glbopt_callback,                 ONLY: glbopt_md_callback
   USE md_environment_types,            ONLY: md_environment_type
   USE mdctrl_types,                    ONLY: mdctrl_type
#include "../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: mdctrl_callback

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'mdctrl_methods'

CONTAINS

! **************************************************************************************************
!> \brief This is called by md_run for each step during during its main-loop.
!> \param mdctrl data which is passed on to the wrapped client-routine
!> \param md_env contains the current state of the md_run
!> \param should_stop can be used to abort the md_run
! **************************************************************************************************
   SUBROUTINE mdctrl_callback(mdctrl, md_env, should_stop)
      TYPE(mdctrl_type), POINTER                         :: mdctrl
      TYPE(md_environment_type), POINTER                 :: md_env
      LOGICAL, INTENT(inout)                             :: should_stop

      CPASSERT(ASSOCIATED(md_env))
      CPASSERT(ASSOCIATED(mdctrl))

      IF (ASSOCIATED(mdctrl%glbopt)) THEN
         CALL glbopt_md_callback(mdctrl%glbopt, md_env, should_stop)

         !ELSE IF(ASSOCIATED(mdctrl%your_own_hook)) THEN ...

      ELSE
         CPABORT("mdctrl_callback: No hook found.")
      END IF

   END SUBROUTINE mdctrl_callback

END MODULE mdctrl_methods

