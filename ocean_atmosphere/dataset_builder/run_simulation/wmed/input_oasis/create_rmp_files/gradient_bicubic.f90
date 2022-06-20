subroutine gradient_bicubic(NX1, NY1, src_array, sou_mask, &
                                  src_latitudes, src_longitudes, &
                                  id_per, cd_per, &
                                  gradient_i, gradient_j, gradient_ij)
!****
!               *****************************
!               * OASIS ROUTINE  -  LEVEL ? *
!               * -------------     ------- *
!               *****************************
!
!**** *gradient_bicubic*  - calculate gradients for bicubic remapping
!
!     Purpose:
!     -------
!     Calculation of gradients for bicubic interpolation. In contrast to
!     the gradients of conservative remapping, these gradients are    
!     calculated with respect to grid rows and grid lines.
!
!**   Interface:
!     ---------
!       *CALL*  *gradient_bicubic*(NX1, NY1, src_array, sou_mask,
!                                  src_latitudes, src_longitudes, 
!                                  gradient_i, gradient_j, gradient_ij)
!
!     Input:
!     -----
!          NX1            : grid dimension in x-direction (integer)
!          NY1            : grid dimension in y-direction (integer)
!          src_array      : array on source grid (real 2D)
!          sou_mask       : source grid mask (integer 2D)
!          src_latitudes  : latitudes on source grid (real 2D)
!          src_longitudes : longitudes on source grid (real 2D)
!          id_per         : number of overlapping points for source grid
!          cd_per         : grip periodicity type
!
!     Output:
!     ------
!          gradient_i     : gradient in i-direction (real 2D)
!          gradient_j     : gradient in j-direction (real 2D)
!          gradient_ij    : gradient in ij-direction (real 2D)
!
!     History:
!     -------
!       Version   Programmer     Date        Description
!       -------   ----------     ----        -----------  
!       2.5       V. Gayler      2002/04/05  created
!
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

IMPLICIT NONE
      
  INTEGER, PARAMETER :: wp = SELECTED_REAL_KIND(12,307) ! double
!-----------------------------------------------------------------------
!     INTENT(IN)
!-----------------------------------------------------------------------
INTEGER, INTENT(IN) :: &
         NX1, NY1,  &           ! source grid dimensiones
         id_per                ! nbr of overlapping grid points

CHARACTER(len=8), INTENT(IN) ::  &
         cd_per                ! grip periodicity type 

REAL (kind=wp), DIMENSION(NX1,NY1), INTENT(IN) :: &
          src_array            ! array on source grid

INTEGER, DIMENSION(NX1,NY1), INTENT(IN) :: &
         sou_mask             ! source grid mask

REAL (kind=wp), DIMENSION(NX1,NY1), INTENT(IN) :: &
          src_latitudes,   &    ! source grid latitudes
          src_longitudes        ! source grid longitudes

!-----------------------------------------------------------------------
!     INTENT(OUT)
!-----------------------------------------------------------------------
REAL (kind=wp), DIMENSION(NX1,NY1), INTENT(OUT) :: &
          gradient_i,   &       ! gradient in i-direction (real 2D)
          gradient_j,   &       ! gradient in j-direction (real 2D)
          gradient_ij           ! gradient in ij-direction (real 2D)

!-----------------------------------------------------------------------
!     LOCAL VARIABLES
!-----------------------------------------------------------------------
INTEGER ::  &
          i, j,    &            ! looping indicees
          ip1, jp1, im1, jm1
     
REAL (kind=wp) ::  &
          di, dj,        &         ! factor depending on grid cell distance
          gradient_ij1,  &         ! gradient needed to calculate gradient_ij
          gradient_ij2             ! gradient needed to calculate gradient_ij

REAL (kind=wp), DIMENSION(NX1,NY1) :: &
          src_lon,   &          ! source grid longitudes [radiants]
          src_lat,   &          ! source grid latitudes [radiants]
          pi180                 ! conversion factor: deg -> rad

INTEGER, PARAMETER ::  il_maskval= 1 ! in our grids sea_value = 0 and land_value = 1

!----------------------------------------------------------------------
!
!     Transformation from degree to radiant
!     -------------------------------------
      pi180 = 1.74532925199432957692e-2 ! =PI/180

      src_lon = src_longitudes * pi180
      src_lat = src_latitudes * pi180

!     Initialization
!     --------------
      gradient_i  = 0.
      gradient_j  = 0. 
      gradient_ij = 0. 

!     calculate gradients
!     -------------------
      DO i = 1, NX1
         DO j = 1, NY1
                   
            IF (sou_mask (i,j) /= il_maskval) THEN

               di = 0.5
               dj = 0.5

               ip1 = i + 1
               im1 = i - 1
               IF (i == NX1) THEN
                   IF (cd_per == 'P') ip1 = 1 + id_per ! the 0-meridian
                   IF (cd_per == 'R') ip1 = NX1
               ENDIF
               IF (i == 1 )  THEN
                   IF (cd_per == 'P') im1 = NX1 - id_per
                   IF (cd_per == 'R') im1 = 1
               ENDIF
               jp1 = j + 1
               jm1 = j - 1
               IF (j == NY1) THEN ! treatment of the last..
                  jp1 = NY1 
                  dj = 1.
               ENDIF   
               IF (j == 1 ) THEN  ! .. and the first grid-row
                  jm1 = 1
                  dj = 1.
               ENDIF


!              gradient i
!              ----------
               IF (sou_mask(ip1,j) /= il_maskval .OR. &
                   sou_mask(im1,j) /= il_maskval) THEN
                  IF (sou_mask(ip1,j) == il_maskval) THEN
                     ip1 = i
                     di = 1.
                  ELSE IF (sou_mask(im1,j) == il_maskval) THEN
                     im1 = i
                     di = 1.
                  ENDIF
                  gradient_i(i,j) = di * (src_array(ip1,j) - src_array(im1,j))
               ENDIF

!              gradient j
!              ----------
               IF (sou_mask(i,jp1) /= il_maskval .OR. &
                   sou_mask(i,jm1) /= il_maskval) THEN
                  IF (sou_mask(i,jp1) == il_maskval) THEN
                     jp1 = j
                     dj = 1.
                  ELSE IF (sou_mask(i,jm1) == il_maskval) THEN
                     jm1 = j
                     dj = 1.
                  ENDIF
                  gradient_j(i,j) = dj * (src_array(i,jp1) - src_array(i,jm1))
               ENDIF
!
!              gradient ij
!              -----------
               di = 0.5
               dj = 0.5
               ip1 = i + 1
               im1 = i - 1
               IF (i == NX1) THEN
                   IF (cd_per == 'P') ip1 = 1 + id_per ! the 0-meridian
                   IF (cd_per == 'R') ip1 = NX1
               ENDIF
               IF (i == 1 )  THEN
                   IF (cd_per == 'P')  im1 = NX1 - id_per
                   IF (cd_per == 'R')  im1 = 1
               ENDIF
               jp1 = j + 1
               jm1 = j - 1
               IF (j == NY1) THEN ! treatment of the last..
                  jp1 = NY1 
                  dj = 1.
               ENDIF   
               IF (j == 1 ) THEN  ! .. and the first grid-row
                  jm1 = 1
                  dj = 1.
               ENDIF

               gradient_ij1 = 0.
               IF (sou_mask(ip1,jp1) /= il_maskval .OR. &
                   sou_mask(im1,jp1) /= il_maskval) THEN
                  IF (sou_mask(ip1,jp1) == il_maskval .AND. &
                      sou_mask(i,jp1) /= il_maskval) THEN
                     ip1 = i
                     di = 1.
                  ELSE IF (sou_mask(im1,jp1) == il_maskval .AND. &
                           sou_mask(i,jp1) /= il_maskval) THEN
                     im1 = i
                     di = 1.
                  ELSE
                     di = 0.
                  ENDIF
                  gradient_ij1 = di * (src_array(ip1,jp1) - src_array(im1,jp1))
               ENDIF

               di = 0.5
               ip1 = i + 1
               im1 = i - 1
               IF (i == NX1) THEN
                   IF (cd_per == 'P') ip1 = 1 + id_per ! the 0-meridian
                   IF (cd_per == 'R') ip1 = NX1
               ENDIF
               IF (i == 1)  THEN
                   IF (cd_per == 'P') im1 = NX1 - id_per
                   IF (cd_per == 'R') im1 = 1
               ENDIF
               gradient_ij2 = 0.
               IF (sou_mask(ip1,jm1) /= il_maskval .OR. &
                   sou_mask(im1,jm1) /= il_maskval) THEN
                  IF (sou_mask(ip1,jm1) == il_maskval .AND. &
                      sou_mask(i,jm1) /= il_maskval) THEN
                     ip1 = i
                     di = 1.
                  ELSE IF (sou_mask(im1,jm1) == il_maskval .AND. &
                          sou_mask(i,jm1) /= il_maskval) THEN
                     im1 = i
                     di = 1.
                  ELSE
                     di = 0.
                  ENDIF
                  gradient_ij2 = di * (src_array(ip1,jm1) - src_array(im1,jm1))
               ENDIF

               IF (gradient_ij1 /= 0. .AND. gradient_ij2 /= 0.) THEN
                  gradient_ij(i,j) = dj * (gradient_ij1 - gradient_ij2)
               ENDIF
            ENDIF
            
         ENDDO
      ENDDO
!
END SUBROUTINE gradient_bicubic
