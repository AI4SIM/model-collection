       FUNCTION distance_rad (lon1, lat1, lon2, lat2)
!****
!               *****************************
!               * OASIS ROUTINE  -  LEVEL ? *
!               * -------------     ------- *
!               *****************************
!
!**** *distance*  - calculate the distance between two points on a sphere
!
!     Purpose:
!     -------
!     Calculation of the distance between two points on a sphere
!       1. Transformation to x,y,z-coordinates
!       2. Calculating the distance
!       3. Calculating the distance on the sphere
!
!**   Interface:
!     ---------
!       *CALL*  *distance_rad*(lon1, lat1, lon2, lat2)
!
!     Input:
!     -----
!          lon1              : longitude of first point (rad)
!          lat1              : latitude of first point (rad)
!          lon2              : longitude of second point (rad)
!          lat2              : latitude of second point (rad)
!
!     Output:
!     ------
!          distance          : distance
!!
!     History:
!     -------
!       Version   Programmer     Date        Description
!       -------   ----------     ----        -----------  
!       2.5       V. Gayler      2001/09/20  created
!
!-----------------------------------------------------------------------
      USE constants
      USE kinds_mod

      IMPLICIT NONE
!-----------------------------------------------------------------------
!     INTENT(IN)
!-----------------------------------------------------------------------
      REAL (kind=real_kind), INTENT(IN) :: lon1, & ! longitude of first point (rad)
                                           lon2, & ! longitude of second point (rad)
                                           lat1, & ! latitude of first point (rad)
                                           lat2    ! latitude of second point (rad)

!-----------------------------------------------------------------------
!     LOCAL VARIABLES
!-----------------------------------------------------------------------
      REAL (kind=real_kind) :: x1, y1, z1, & ! coordinates of the first point
                               x2, y2, z2, & ! coordinates of the second point
                               distance_rad ! distance between the points (rad)

!-----------------------------------------------------------------------

!     Transformation to x,y,z-coordinates
!     -----------------------------------
      x1 = cos(lat1)*cos(lon1)
      y1 = cos(lat1)*sin(lon1)
      z1 = sin(lat1)

      x2 = cos(lat2)*cos(lon2)
      y2 = cos(lat2)*sin(lon2)
      z2 = sin(lat2)

!     Calculation of the distance
!     ---------------------------
!     direct distance:
      distance_rad = SQRT((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

!     distance along the surface:
      distance_rad= 2.d0*ASIN(distance_rad/2.d0)

!-----------------------------------------------------------------------
      RETURN 
      END FUNCTION distance_rad

