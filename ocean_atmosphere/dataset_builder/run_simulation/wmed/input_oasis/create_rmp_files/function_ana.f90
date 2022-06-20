SUBROUTINE function_ana(ni, nj, xcoor, ycoor, fnc_ana)

!**** *function ana*  - calculate analytical function
  !
  IMPLICIT NONE
  !
  INTEGER, PARAMETER :: wp = SELECTED_REAL_KIND(12,307) ! double
  !
  INTEGER, INTENT(IN) :: ni, nj
  REAL(kind=wp), DIMENSION(ni,nj), INTENT(IN)  :: xcoor, ycoor
  REAL(kind=wp), DIMENSION(ni,nj), INTENT(OUT) :: fnc_ana
  !
  REAL (kind=wp), PARAMETER    :: dp_pi=3.14159265359
  REAL (kind=wp), PARAMETER    :: dp_conv = dp_pi/180.
  REAL(kind=wp)  :: dp_length, coef, coefmult
  INTEGER             :: i,j
  CHARACTER(LEN=7) :: cl_anaftype="fcos"
  !
  DO j=1,nj
    DO i=1,ni
!
      SELECT CASE (cl_anaftype)
      CASE ("fcos")
        dp_length = 1.2*dp_pi
        coef = 2.
        coefmult = 1.
        fnc_ana(i,j) = coefmult*(coef - COS( dp_pi*(ACOS( COS(xcoor(i,j)*dp_conv)*COS(ycoor(i,j)*dp_conv) )/dp_length)) )
!
      CASE ("fcossin")
        dp_length = 1.d0*dp_pi
        coef = 21.d0
        coefmult = 3.846d0 * 20.d0
        fnc_ana(i,j) = coefmult*(coef - COS( dp_pi*(ACOS( COS(ycoor(i,j)*dp_conv)*COS(ycoor(i,j)*dp_conv) )/dp_length)) * &
                                        SIN( dp_pi*(ASIN( SIN(xcoor(i,j)*dp_conv)*SIN(ycoor(i,j)*dp_conv) )/dp_length)) )
      END SELECT
!
    ENDDO
  ENDDO
  !
END SUBROUTINE function_ana
