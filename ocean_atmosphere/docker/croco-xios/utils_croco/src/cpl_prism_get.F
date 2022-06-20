! $Id: cpl_prism_get.F 1574 2014-07-01 15:13:38Z gcambon $
!
!======================================================================
! CROCO is a branch of ROMS developped at IRD and INRIA, in France
! The two other branches from UCLA (Shchepetkin et al) 
! and Rutgers University (Arango et al) are under MIT/X style license.
! CROCO specific routines (nesting) are under CeCILL-C license.
! 
! CROCO website : http://www.croco-ocean.org
!======================================================================
!
      SUBROUTINE cpl_prism_get (rcvtime)
!     !---------------------------------------------------------------------
!     !              ***  ROUTINE cpl_prism_get  ***
!     !
!     ! ** Purpose : - 
!     !           
!     !
!     !----------------------------------------------------------------------
#include "cppdefs.h"
#if defined OA_COUPLING || defined OW_COUPLING
      USE mod_prism             ! OASIS3 prism module
# ifdef AGRIF
      use Agrif_Util
# endif
      IMPLICIT NONE

#include "cppdefs.h"

#include "param.h"
#include "forces.h"
#include "mpi_cpl.h"
#include "ocean3d.h"
#include "scalars.h"
#include "grid.h"

      INTEGER    :: rcvtime, i, j, io, jo
      INTEGER    :: iok_utau, iok_vtau, iok_etau, iok_ntau, iok_taum
      INTEGER    :: iok_srflx, iok_itemp , iok_isalt
      INTEGER    :: iok_psfc
      INTEGER    :: iok_wdry 
      INTEGER    :: iok_wt0m1, iok_whs, iok_wdir
      INTEGER    :: iok_wbhd, iok_wfoc, iok_wlm
      INTEGER    :: iok_wubrx, iok_wubry
      INTEGER    :: iok_wust, iok_wvst 
      INTEGER    :: iok_utwo, iok_vtwo, iok_etwo, iok_ntwo
      INTEGER    :: iok_utaw, iok_vtaw, iok_etaw, iok_ntaw 
      INTEGER    :: imin_halo, imax_halo, jmin_halo, jmax_halo
      INTEGER    :: level
      CHARACTER*9 rclevel

      real, parameter :: eps=1.e-20
      real,dimension(GLOBAL_2D_ARRAY) :: tab_t_tmp
      real,dimension(GLOBAL_2D_ARRAY) :: etau
      real,dimension(GLOBAL_2D_ARRAY) :: ntau
      real,dimension(GLOBAL_2D_ARRAY) :: etaw
      real,dimension(GLOBAL_2D_ARRAY) :: ntaw
      real,dimension(GLOBAL_2D_ARRAY) :: etwo
      real,dimension(GLOBAL_2D_ARRAY) :: ntwo
# if (!defined OA_GRID_UV)
      real,dimension(GLOBAL_2D_ARRAY) :: tab_u_tmp
      real,dimension(GLOBAL_2D_ARRAY) :: tab_v_tmp
# endif

# ifdef AGRIF
      level=Agrif_Fixed()
      write(rclevel,'(a,i1.1)') "_CPLMASK", level
# else
      level=0
      write(rclevel,'(a)') '         '
# endif

!      
!!$      if (level==0) then
!!$         MPI_master_only 
!!$     & write(*,*) 'CPL-CROCO : Get for parent at time', rcvtime
!!$      else		
!!$         MPI_master_only 
!!$     & write(*,*) 'CPL-CROCO : Get for child:', level,
!!$     &  'at time', rcvtime
!!$      endif
!
      imin_halo = 1  
      imax_halo = Lmmpi  
      jmin_halo = 1  
      jmax_halo = Mmmpi  

      if (WEST_INTER) then
         imin_halo = imin_halo - 1
      endif 
      if (EAST_INTER) then
         imax_halo = imax_halo + 1
      endif 
      if (SOUTH_INTER) then     
         jmin_halo = jmin_halo - 1
      endif
      if (NORTH_INTER) then
         jmax_halo = jmax_halo + 1
      endif 
!
      iok_srflx = 0
      iok_isalt = 0
      iok_itemp = 0
      iok_utau = 0
      iok_vtau = 0
      iok_etau = 0
      iok_ntau = 0
      iok_taum = 0
      iok_psfc = 0
      iok_wdry = 0
      iok_wt0m1 = 0
      iok_whs = 0
      iok_wdir = 0
      iok_wbhd = 0
      iok_wubrx = 0
      iok_wubry = 0
      iok_wust = 0
      iok_wvst = 0
      iok_wfoc = 0
      iok_wlm = 0
      iok_utwo = 0
      iok_vtwo = 0
      iok_etwo = 0
      iok_ntwo = 0
      iok_utaw = 0
      iok_vtaw = 0
      iok_etaw = 0
      iok_ntaw = 0
!
      DO i = 1, nmaxfld
# ifdef OA_COUPLING
!-------------------------------------------------
!------ OA_COUPLING -  ATM VARIABLES -------------
!-------------------------------------------------
!
!  -- Scalar variables --
!--------------------------------------------------
!        Solar radiation flux - srflx
         if (trim(srcv_clname(i))=='CROCO_SRFL'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_srflx) 
            if (iok_srflx /= 0) then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               srflx = tab_t_tmp / (rho0*Cp) 
            endif
         end if  
!-------------------------------------------------
!        Non-solar heat flux - stflx(:,:,itemp)
         if (trim(srcv_clname(i))=='CROCO_STFL'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_itemp) 
            if (iok_itemp /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               stflx(:,:,itemp) = tab_t_tmp / (rho0*Cp)
            endif     
         end if
!-------------------------------------------------
!        Long wave flux - shflx_rlw
         if (trim(srcv_clname(i))=='CROCO_LWFL'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_itemp)
            if (iok_itemp /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               shflx_rlw = tab_t_tmp / (rho0*Cp)
            endif
         end if
!-------------------------------------------------
!        Latent heat flux - shflx_lat
         if (trim(srcv_clname(i))=='CROCO_LHFL'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_itemp)
            if (iok_itemp /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               shflx_lat = tab_t_tmp / (rho0*Cp)
            endif
         end if
!-------------------------------------------------
!        Sensible heat flux - shflx_sen
         if (trim(srcv_clname(i))=='CROCO_SHFL'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_itemp)
            if (iok_itemp /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               shflx_sen = tab_t_tmp / (rho0*Cp)
            endif
         end if
!--------------------------------------------------
!        Evaporation - precipitation flux - stflx(:,:,isalt)
!        Changing Unit from [kg/m^2/s] (<--> [mm/s]) to [PSU m/s]  --> coeff= 1/1000.
#  ifdef SALINITY
         if (trim(srcv_clname(i))=='CROCO_EVPR'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_isalt)
            if (iok_isalt /= 0) then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               stflx(:,:,isalt) = tab_t_tmp * t(:,:,N,nstp,isalt) / 1000.
            endif
         end if
#  endif
!-------------------------------------------------
!        Module of wind stress - smstr
         if (trim(srcv_clname(i))=='CROCO_TAUM'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_taum)
            if (iok_taum /= 0) then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               smstr = tab_t_tmp / rho0
!              avoid slight negative values (after interpolation for example)...
               where ( smstr < 0. )
                  smstr = 0.
               end where
            endif
         end if
!-------------------------------------------------
!        Surface pressure [Pa] - patm2d
#  ifdef READ_PATM
         if (trim(srcv_clname(i))=='CROCO_PSFC'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_psfc)
            if (iok_psfc /= 0) then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               patm2d = tab_t_tmp
            endif
         end if
#  endif
!-------------------------------------------------
!
!  -- Vector variables --
!--------------------------------------------------
!        U component of wind stress - sustr
         if (trim(srcv_clname(i))=='CROCO_UTAU'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_utau) 
            if (iok_utau /= 0) then  
#  if (!defined OA_GRID_UV)  
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
!              Interpolate grid T --> U
               do jo=jmin_halo-1,jmax_halo+1
               do io=imin_halo,imax_halo+1
                  tab_u_tmp(io,jo)=0.5*(tab_t_tmp(io-1,jo)+tab_t_tmp(io,jo))
               enddo
               enddo
               sustr  = tab_u_tmp / rho0
               if (iok_taum == 0) tab_u_tmp = tab_t_tmp ! temporary storage...
#  else
               call exchange_u2d_tile (1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               sustr  = tab_t_tmp / rho0
#  endif   
            endif
         end if
!-------------------------------------------------
!        V component of wind stress - svstr
         if (trim(srcv_clname(i))=='CROCO_VTAU'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_vtau) 
            if (iok_vtau /= 0) then 
#  if (!defined OA_GRID_UV) 
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
!              Interpolate grid T --> U
               do jo=jmin_halo,jmax_halo+1
               do io=imin_halo-1,imax_halo+1
                  tab_v_tmp(io,jo)=0.5*(tab_t_tmp(io,jo-1)+tab_t_tmp(io,jo))
               enddo
               enddo
               svstr = tab_v_tmp(:,:) / rho0
               if (iok_taum == 0) tab_v_tmp = tab_t_tmp ! temporary storage...
#  else
               call exchange_v2d_tile (1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               svstr = tab_t_tmp(:,:) / rho0                
#  endif 
            endif
         end if
!-------------------------------------------------
!        Eastward component of wind stress - temporary etau
         if (trim(srcv_clname(i))=='CROCO_ETAU'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_etau)
            if (iok_etau /= 0) then
#  if (!defined OA_GRID_UV)  
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               etau = tab_t_tmp / rho0
#  else
               call exchange_u2d_tile (1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               etau  = tab_t_tmp / rho0
#  endif   
            endif
         end if
!-------------------------------------------------
!        Northward component of wind stress - temporary ntau
         if (trim(srcv_clname(i))=='CROCO_NTAU'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_ntau)
            if (iok_ntau /= 0) then
#  if (!defined OA_GRID_UV)  
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               ntau = tab_t_tmp / rho0
#  else
               call exchange_u2d_tile (1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               ntau  = tab_t_tmp / rho0
#  endif   
            endif
         end if
# endif

# ifdef OW_COUPLING
!-------------------------------------------------
!------ OW_COUPLING - WAVE VARIABLES -------------
!-------------------------------------------------
!
!  -- Scalar variables --
!--------------------------------------------------
!        mask to manage wet-drying
!        not tested/performed yet
         if (trim(srcv_clname(i))=='CROCO_DRY'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_wdry)
!            if (iok_wdry /= 0)  then
!               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
!            endif
         end if
!-------------------------------------------------
!        Significant wave height (m) --> wave hrm - whrm
         if (trim(srcv_clname(i))=='CROCO_HS'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_whs)
            if (iok_whs /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               whrm = rmask * tab_t_tmp * 0.70710678
            endif
         end if
!-------------------------------------------------
!        Wave mean period (s) --> wave freq - wfrq
         if (trim(srcv_clname(i))=='CROCO_T0M1'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_wt0m1)
            if (iok_wt0m1 /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               wfrq = rmask * (2*pi) / max(tab_t_tmp, eps)
!     gc
               where ( wfrq > 1000. )  
                 wfrq = 0.
               end where
!     End gc
            endif
         end if
!-------------------------------------------------
!        Wave mean direction (deg) 
!            + then we apply croco grid rotation (-angler)
!              and projection on xi/eta trigo croco_grid - wdrx, wdre
         if (trim(srcv_clname(i))=='CROCO_DIR'//trim(rclevel)) then
           call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &          tab_t_tmp, rcvtime, iok_wdir)
           if (iok_wdir /= 0)  then
             call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
             wdrx = rmask * cos(tab_t_tmp - angler)
             wdre = rmask * sin(tab_t_tmp - angler)
           endif
         endif
!------------------------------------------------- 
!        wave_bhd : wave-induced Bernoulli head pressure (bhd in N.m-1)
!        not tested/performed yet
         if (trim(srcv_clname(i))=='CROCO_BHD'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_wbhd)
            if (iok_wbhd /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               bhd = rmask * tab_t_tmp
            endif
         endif
!-------------------------------------------------
!        wave_phioc : Wave-to-ocean TKE flux (phioc in W.m-2)
         if (trim(srcv_clname(i))=='CROCO_FOC'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_wfoc)
            if (iok_wfoc /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               foc = rmask * max(tab_t_tmp,0.)
            endif
         endif
!-------------------------------------------------
!        wave_wlm : mean length wave (m)
         if (trim(srcv_clname(i))=='CROCO_LM'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_wlm)
            if (iok_wlm /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               wlm = rmask * tab_t_tmp
            endif
         endif
!
!  -- Vector variables converted to modules  --
!--------------------------------------------------
!        ORBITAL BOTTOM VELOCITY      
!        wave_ubrx : x-component rms amplitude of wave orbital bottom velocity(m/s)
         if (trim(srcv_clname(i))=='CROCO_UBRX'//trim(rclevel)) then
             call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_wubrx)
            if (iok_wubrx /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               ubr = tab_t_tmp
            endif
         endif
!        wave_ubry : y-component rms amplitude of wave orbital bottom velocity(m/s)
         if (trim(srcv_clname(i))=='CROCO_UBRY'//trim(rclevel)) then
             call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_wubry)
            if (iok_wubry /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               ubr = rmask * sqrt(ubr**2+tab_t_tmp**2)
            endif
         endif
!-----------------------------------------------------------------------------------------------------------------         
!        SURFACE STOKES DRIFT VELOCITY        
!        wave_ussx : x-component of stokes drift surface velocity
         if (trim(srcv_clname(i))=='CROCO_USSX'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_wust)
            if (iok_wust /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               ust_ext = tab_t_tmp
            endif
         endif
!        wave_ussy : y-component of stokes drift surface velocity
         if (trim(srcv_clname(i))=='CROCO_USSY'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_wvst)
            if (iok_wvst /= 0)  then
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               ust_ext = rmask * sqrt(ust_ext**2 + tab_t_tmp**2)
            endif
         endif
!
!  -- Vector variables --
!--------------------------------------------------
!        U component of the wave-to-ocean momentum flux (in m2.s-2) - temporary twox
         if (trim(srcv_clname(i))=='CROCO_UTWO'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_utwo)
            if (iok_utwo /= 0) then
#  if (!defined OA_GRID_UV)  
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
!              Interpolate grid T --> U
               do jo=jmin_halo-1,jmax_halo+1
               do io=imin_halo,imax_halo+1
                  twox(io,jo)=0.5*(tab_t_tmp(io-1,jo)+tab_t_tmp(io,jo))
               enddo
               enddo
#  else
               call exchange_u2d_tile (1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               twox  = tab_t_tmp
#  endif   
            endif
         endif
!-------------------------------------------------
!        V component of the wave-to-ocean momentum flux (in m2.s-2) - temporary twoy
         if (trim(srcv_clname(i))=='CROCO_VTWO'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_vtwo)
            if (iok_vtwo /= 0) then
#  if (!defined OA_GRID_UV) 
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
!              Interpolate grid T --> U
               do jo=jmin_halo,jmax_halo+1
               do io=imin_halo-1,imax_halo+1
                  twoy(io,jo)=0.5*(tab_t_tmp(io,jo-1)+tab_t_tmp(io,jo))
               enddo
               enddo
#  else
               call exchange_v2d_tile (1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               twoy = tab_t_tmp
#  endif 
            endif
         endif
!-------------------------------------------------
!        Eastward component of the wave-to-ocean momentum flux (in m2.s-2) - temporary etwo
         if (trim(srcv_clname(i))=='CROCO_ETWO'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_etwo)
            if (iok_etwo /= 0) then
#  if (!defined OA_GRID_UV)  
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               etwo = tab_t_tmp / rho0
#  else
               call exchange_u2d_tile (1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               etwo  = tab_t_tmp / rho0
#  endif   
            endif
         end if
!-------------------------------------------------
!        Northward component of the wave-to-ocean momentum flux (in m2.s-2) - temporary ntwo
         if (trim(srcv_clname(i))=='CROCO_NTWO'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_ntwo)
            if (iok_ntwo /= 0) then
#  if (!defined OA_GRID_UV)  
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               ntwo = tab_t_tmp / rho0
#  else
               call exchange_u2d_tile (1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               ntwo  = tab_t_tmp / rho0
#  endif   
            endif
         end if
!-------------------------------------------------
!        U component of atmosphere-to-wave stress (in m2.s-2) - temporary tawx
         if (trim(srcv_clname(i))=='CROCO_UTAW'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_utaw)
            if (iok_utaw /= 0) then
#  if (!defined OA_GRID_UV)  
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
!              Interpolate grid T --> U
               do jo=jmin_halo-1,jmax_halo+1
               do io=imin_halo,imax_halo+1
                  tawx(io,jo)=0.5*(tab_t_tmp(io-1,jo)+tab_t_tmp(io,jo))
               enddo
               enddo
#  else
               call exchange_u2d_tile (1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               tawx = tab_t_tmp
#  endif   
            endif
         endif
!-------------------------------------------------
!        V component of atmosphere-to-wave stress (in m2.s-2) - temporary tawy
         if (trim(srcv_clname(i))=='CROCO_VTAW'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_vtaw)
            if (iok_vtaw /= 0) then
#  if (!defined OA_GRID_UV) 
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
!              Interpolate grid T --> U
               do jo=jmin_halo,jmax_halo+1
               do io=imin_halo-1,imax_halo+1
                  tawy(io,jo)=0.5*(tab_t_tmp(io,jo-1)+tab_t_tmp(io,jo))
               enddo
               enddo
#  else
               call exchange_v2d_tile (1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               tawy = tab_t_tmp
#  endif 
            endif
         endif
!-------------------------------------------------
!        Eastward component of atmosphere-to-wave stress (in m2.s-2) - temporary etaw
         if (trim(srcv_clname(i))=='CROCO_ETAW'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_etaw)
            if (iok_etaw /= 0) then
#  if (!defined OA_GRID_UV)  
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               etaw = tab_t_tmp / rho0
#  else
               call exchange_u2d_tile (1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               etaw  = tab_t_tmp / rho0
#  endif   
            endif
         end if
!-------------------------------------------------
!        Northward component of atmosphere-to-wave stress (in m2.s-2) - temporary ntaw
         if (trim(srcv_clname(i))=='CROCO_NTAW'//trim(rclevel)) then
            call cpl_prism_getvar(srcv_nid(:,i), cplmsk, nmaxatm,
     &  tab_t_tmp, rcvtime, iok_ntaw)
            if (iok_ntaw /= 0) then
#  if (!defined OA_GRID_UV)  
               call exchange_r2d_tile(1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               ntaw = tab_t_tmp / rho0
#  else
               call exchange_u2d_tile (1,Lmmpi,1,Mmmpi,tab_t_tmp(START_2D_ARRAY))
               ntaw  = tab_t_tmp / rho0
#  endif   
            endif
         end if
# endif
      END DO
!--------------- end coupled variable exchanges --------
!
# ifdef OA_COUPLING
!-------------------------------------------------
!     Compute total heat flux
!-------------------------------------------------
      if  (iok_itemp /= 0) then
         stflx(:,:,itemp) = stflx(:,:,itemp) + srflx
      endif
!
      if  (iok_etau /= 0 .and. iok_ntau /= 0) then
!-------------------------------------------------
!     Compute CROCO surface stress u, v components
!       when east/northward components exchanged
!-------------------------------------------------
#  if (!defined OA_GRID_UV)
!        from east/north component to x/y grid component at rho point
         tab_u_tmp = etau * cos(angler) + ntau * sin(angler)
         tab_v_tmp = ntau * cos(angler) - etau * sin(angler)
!        to u, v grid
         do jo=jmin_halo-1,jmax_halo+1
         do io=imin_halo,imax_halo+1
            sustr(io,jo)=0.5*(tab_u_tmp(io-1,jo)+tab_u_tmp(io,jo))
         enddo
         enddo
         do jo=jmin_halo,jmax_halo+1
         do io=imin_halo-1,imax_halo+1
            svstr(io,jo)=0.5*(tab_v_tmp(io,jo-1)+tab_v_tmp(io,jo))
         enddo
         enddo
!        compute module if not exchanged
         if (iok_taum == 0) then
            smstr = sqrt(tab_u_tmp**2 + tab_v_tmp**2)
         endif
#  else
         write(*,*) 'OA_GRID_UV and grid rotation: case not coded...'
         return  !-->  EXIT
#  endif
      endif
! 
      IF  (iok_taum == 0 .and. iok_utau /= 0 .and. iok_vtau /= 0) THEN
!-------------------------------------------------
!     Compute CROCO module stress when not exchanged
!-------------------------------------------------
#  if (!defined OA_GRID_UV) 
         smstr = sqrt(tab_u_tmp**2 + tab_v_tmp**2)  / rho0
#  else
         do jo=jmin_halo-1,jmax_halo+1
         do io=imin_halo-1,imax_halo+1
             smstr(io,jo)=sqrt(0.25*(sustr(io,jo)+sustr(io+1,jo))**2+
     &                         0.25*(svstr(io,jo)+svstr(io,jo+1))**2) / rho0
         enddo
         enddo
#  endif 
      ENDIF
# endif
!
# ifdef OW_COUPLING
      if  (iok_etaw /= 0 .and. iok_ntaw /= 0) then
!-------------------------------------------------
!     Compute atmosphere-to-wave stress u, v components
!       when east/northward components exchanged
!-------------------------------------------------
#  if (!defined OA_GRID_UV)
!        from east/north component to x/y grid component at rho point
         tab_u_tmp = etaw * cos(angler) + ntaw * sin(angler)
         tab_v_tmp = ntaw * cos(angler) - etaw * sin(angler)
!        to u, v grid
         do jo=jmin_halo-1,jmax_halo+1
         do io=imin_halo,imax_halo+1
            tawx(io,jo)=0.5*(tab_u_tmp(io-1,jo)+tab_u_tmp(io,jo))
         enddo
         enddo
         do jo=jmin_halo,jmax_halo+1
         do io=imin_halo-1,imax_halo+1
            tawy(io,jo)=0.5*(tab_v_tmp(io,jo-1)+tab_v_tmp(io,jo))
         enddo
         enddo
#  else
         write(*,*) 'OA_GRID_UV and grid rotation: case not coded...'
         return  !-->  EXIT 
#  endif
      endif
!
      if  (iok_etwo /= 0 .and. iok_ntwo /= 0) then
!-------------------------------------------------
!     Compute wave-to-ocean stress u, v components
!       when east/northward components exchanged
!-------------------------------------------------
#  if (!defined OA_GRID_UV)
!        from east/north component to x/y grid component at rho point
         tab_u_tmp = etwo * cos(angler) + ntwo * sin(angler)
         tab_v_tmp = ntwo * cos(angler) - etwo * sin(angler)
!        to u, v grid
         do jo=jmin_halo-1,jmax_halo+1
         do io=imin_halo,imax_halo+1
            twox(io,jo)=0.5*(tab_u_tmp(io-1,jo)+tab_u_tmp(io,jo))
         enddo
         enddo
         do jo=jmin_halo,jmax_halo+1
         do io=imin_halo-1,imax_halo+1
            twoy(io,jo)=0.5*(tab_v_tmp(io,jo-1)+tab_v_tmp(io,jo))
         enddo
         enddo
#  else
         write(*,*) 'OA_GRID_UV and grid rotation: case not coded...'
         return  !-->  EXIT
#  endif
      endif
!
!-------------------------------------------------
!  Update CROCO wind stress accounting for wave bulk
!-------------------------------------------------
#  if defined WAVE_SMFLUX
         sustr = twox
         svstr = twoy
#  else
         sustr = sustr - tawx + twox
         svstr = svstr - tawy + twoy
#  endif
# endif 
!-------------------------------------------------
!      
!	
#else 
!---------------------------------------------------------------------------------------
!     Dummy subroutine                                                           NO Coupling
!---------------------------------------------------------------------------------------
#endif 
      END SUBROUTINE cpl_prism_get
