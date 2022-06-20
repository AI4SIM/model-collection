! $Id: set_avg.F 1458 2014-02-03 15:01:25Z gcambon $
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
#include "cppdefs.h"
#ifdef AVERAGES

      subroutine set_avg (tile)

      implicit none
# include "param.h"
# ifdef SOLVE3D
#  include "work.h"
#  include "ncscrum.h"
# endif
      integer tile, trd
C$    integer omp_get_thread_num
# include "compute_tile_bounds.h"
#  ifdef MRL_WCI
      call wstokes (tile)
#  endif
# if defined SOLVE3D && !defined NBQ
      if (wrtavg(indxW)) then
        call Wvlcty (tile, workr)
      endif
# endif
      call set_avg_tile (Istr,Iend,Jstr,Jend)
      return
      end

      subroutine set_avg_tile (Istr,Iend,Jstr,Jend)
!
! Compute time-averaged fields within a tile.
! ------- ------------- ------ ------ - -----
! Because of syncronization issues, the delayed mode averaging
! procedure is used. This procedure implies that all fields to be
! averaged are sampled during the next time step, rather than at
! the end of the time step when they were computed.
!
! Although this algorithm results in somewhat awkward controlling
! logic it has the advantage that all fields to be sampled
! correspond to exactly the same time, which is time step "n".
! Particularly, this is done this way because vertical velocity
! corresponding to the newly computed horizontal velocities
! becomes available only during the following time step.
! The same applies to the density field.
!
! The algorithm consists of three logical blocks: (1) initialization
! of the averages arrays: when mod(ilc-1,navg).eq.1 the target arrays
! are set to the first contribution; (2) accumulation of averaged
! data, when mod(ilc-1,navg).gt.1; and (3) adding the last
! contribution and scaling.
!
      implicit none
# include "param.h"
      integer Istr,Iend,Jstr,Jend, i,j, ilc, iout, indxWrk
      real cff, cff1, eps, stf_cff
      parameter (eps=1.D-20)
      parameter(stf_cff=86400/0.01)
# ifdef SOLVE3D
      integer itrc,k
#  ifdef SEDIMENT
      integer ilay
#  endif
#  include "work.h"
# endif
# include "scalars.h"
# include "ncscrum.h"
# include "grid.h"
# include "ocean2d.h"
# include "ocean3d.h"
# include "averages.h"
# include "mixing.h"
# include "forces.h"
# ifdef SEDIMENT
#  include "sediment.h"
# endif
# ifdef WKB_WWAVE
#  include "wkb_wwave.h"
# endif
!
      integer IstrR,IendR,JstrR,JendR
# ifdef MPI
      if (Istr.eq.1 .and. ii.eq.0) then
# else
      if (Istr.eq.1) then
# endif
        IstrR=Istr-1
      else
        IstrR=Istr
      endif
# ifdef MPI
      if (Iend.eq.Lmmpi.and. ii.eq.NP_XI-1) then
# else
      if (Iend.eq.Lm) then
# endif
        IendR=Iend+1
      else
        IendR=Iend
      endif

# ifdef MPI
      if (Jstr.eq.1 .and. jj.eq.0) then
# else
      if (Jstr.eq.1) then
# endif
        JstrR=Jstr-1
      else
        JstrR=Jstr
      endif
# ifdef MPI
      if (Jend.eq.Mmmpi .and. jj.eq.NP_ETA-1) then
# else
      if (Jend.eq.Mm) then
# endif
        JendR=Jend+1
      else
        JendR=Jend
      endif
!
      ilc=1+iic-ntstart   ! number of time step since restart
!

      if (ilc.gt.ntsavg) then

        if (mod(ilc-ntsavg,navg).eq.1) then
          cff =1.0
          cff1=0.0
          if (ZEROTH_TILE) then
            time_avg=time
            !MPI_master_only write(*,*) 'start avg',iic,ntsavg,navg
          endif
        elseif (mod(ilc-ntsavg,navg).gt.1) then
          cff =1.0
          cff1=1.0
          if (ZEROTH_TILE) then
           time_avg=time_avg+time
          endif
        elseif (mod(ilc-ntsavg,navg).eq.0) then
          cff=1./float(navg)
          cff1=1.0
          if (ZEROTH_TILE) then
            time_avg=cff*(time_avg+time)
            !MPI_master_only write(*,*) 'finish avg',iic,ntsavg,navg
          endif
        endif

        if (wrtavg(indxZ)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              zeta_avg(i,j)=cff*( cff1*zeta_avg(i,j)
     &                                +zeta(i,j,fast_indx_out))
            enddo
          enddo
        endif
        if (wrtavg(indxUb)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              ubar_avg(i,j)=cff*( cff1*ubar_avg(i,j)
     &                                +ubar(i,j,fast_indx_out))
            enddo
          enddo
        endif
        if (wrtavg(indxVb)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              vbar_avg(i,j)=cff*( cff1*vbar_avg(i,j)
     &                                +vbar(i,j,fast_indx_out))
            enddo
          enddo
        endif
#ifdef MORPHODYN
        if (wrtavg(indxHm)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              h_avg(i,j)=cff*( cff1*h_avg(i,j)
     &                                 +h(i,j))
            enddo
          enddo
        endif
#endif
        if (wrtavg(indxBostr)) then
          do j=Jstr,Jend
            do i=Istr,Iend
              bostr_avg(i,j)=cff*( cff1*bostr_avg(i,j)+
     &                         0.5*sqrt((bustr(i,j)+bustr(i+1,j))**2
     &                                 +(bvstr(i,j)+bvstr(i,j+1))**2)
     &                                                          *rho0)
            enddo
          enddo
        endif
        if (wrtavg(indxBustr)) then
          do j=Jstr,Jend
            do i=Istr,Iend
              bustr_avg(i,j)=cff*( cff1*bustr_avg(i,j)+
     &                                  bustr(i,j)*rho0)
            enddo
          enddo
        endif
        if (wrtavg(indxBvstr)) then
          do j=Jstr,Jend
            do i=Istr,Iend
              bvstr_avg(i,j)=cff*( cff1*bvstr_avg(i,j)+
     &                                  bvstr(i,j)*rho0)
            enddo
          enddo
        endif
        if (wrtavg(indxWstr)) then
          do j=Jstr,Jend
            do i=Istr,Iend
              wstr_avg(i,j)=cff*( cff1*wstr_avg(i,j)+
# ifdef OA_COUPLING
     &                         smstr(i,j)*rho0)
# else
     &                         0.5*sqrt((sustr(i,j)+sustr(i+1,j))**2
     &                                 +(svstr(i,j)+svstr(i,j+1))**2)
     &                                                          *rho0)
# endif
            enddo
          enddo
        endif
        if (wrtavg(indxUWstr)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              sustr_avg(i,j)=cff*( cff1*sustr_avg(i,j)+
     &                                  sustr(i,j)*rho0)
            enddo
          enddo
        endif
        if (wrtavg(indxVWstr)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              svstr_avg(i,j)=cff*( cff1*svstr_avg(i,j)+
     &                                  svstr(i,j)*rho0)
            enddo
          enddo
        endif
# ifdef SOLVE3D
        if (wrtavg(indxU)) then
          do k=1,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                u_avg(i,j,k)=cff*(cff1*u_avg(i,j,k)+u(i,j,k,nstp))
              enddo
            enddo
          enddo
        endif
        if (wrtavg(indxV)) then
          do k=1,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                v_avg(i,j,k)=cff*(cff1*v_avg(i,j,k)+v(i,j,k,nstp))
              enddo
            enddo
          enddo
        endif
# ifdef TRACERS
        do itrc=1,NT
          if (wrtavg(indxV+itrc)) then
            do k=1,N
              do j=JstrR,JendR
                do i=IstrR,IendR
                  t_avg(i,j,k,itrc)=cff*( cff1*t_avg(i,j,k,itrc)
     &                                        +t(i,j,k,nstp,itrc))
                enddo
              enddo
            enddo
          endif
        enddo
# endif
        if (wrtavg(indxR)) then
          do k=1,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                rho_avg(i,j,k)=cff*(cff1*rho_avg(i,j,k)+rho(i,j,k))
              enddo
            enddo
          enddo
        endif
# if defined ANA_VMIX || defined BVF_MIXING \
  || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
  || defined GLS_MIXING
        if (wrtavg(indxbvf)) then
          do k=0,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                bvf_avg(i,j,k)=cff*(cff1*bvf_avg(i,j,k)+bvf(i,j,k))
              enddo
            enddo
          enddo
        endif
# endif
        if (wrtavg(indxO)) then
          do k=0,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                omega_avg(i,j,k)=cff*( cff1*omega_avg(i,j,k)
     &                                 +We(i,j,k)*pm(i,j)*pn(i,j)
# ifdef VADV_ADAPT_IMP
     &                                 +Wi(i,j,k)*pm(i,j)*pn(i,j)
# endif
     &                                                          )
              enddo
            enddo
          enddo
        endif
        if (wrtavg(indxW)) then
# ifdef NBQ
          do k=0,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                w_avg(i,j,k)=cff*(cff1*w_avg(i,j,k)+wz(i,j,k,nstp))
              enddo
            enddo
          enddo
# else
          do k=1,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                w_avg(i,j,k)=cff*(cff1*w_avg(i,j,k)+workr(i,j,k))
              enddo
            enddo
          enddo
# endif
        endif
# if defined LMD_SKPP || defined GLS_MIXING
        if (wrtavg(indxHbl)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
#  ifdef LMD_SKPP2005
              hbl_avg(i,j)=cff*( cff1*hbl_avg(i,j)
     &                               +hbls(i,j,nstp))
#  else
              hbl_avg(i,j)=cff*( cff1*hbl_avg(i,j)
     &                               +hbl(i,j))
# endif
            enddo
          enddo
        endif
#  endif
#  ifdef LMD_BKPP
        if (wrtavg(indxHbbl)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              hbbl_avg(i,j)=cff*( cff1*hbbl_avg(i,j)
     &                                +hbbl(i,j))
            enddo
          enddo
        endif
#  endif
#  ifdef GLS_MIXING
        if (wrtavg(indxTke)) then
          do k=0,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                tke_avg(i,j,k)=cff*(cff1*tke_avg(i,j,k)
     &                                  +trb(i,j,k,nstp,itke))
              enddo
            enddo
          enddo
        endif
        if (wrtavg(indxGls)) then
          do k=0,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                gls_avg(i,j,k)=cff*(cff1*gls_avg(i,j,k)
     &                                  +trb(i,j,k,nstp,igls))
              enddo
            enddo
          enddo
        endif
        if (wrtavg(indxLsc)) then
          do k=0,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                Lscale_avg(i,j,k)=cff*(cff1*Lscale_avg(i,j,k)
     &                                     +Lscale(i,j,k))
              enddo
            enddo
          enddo
        endif
#  endif
# ifdef TEMPERATURE
        if (wrtavg(indxShflx)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              stflx_avg(i,j,itemp)=cff*(cff1*stflx_avg(i,j,itemp)+
     &                                       stflx(i,j,itemp)*(rho0*Cp))
            enddo
          enddo
        endif
#  ifdef BHFLX
        if (wrtavg(indxBhflx)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              btflx_avg(i,j,itemp)=cff*(cff1*btflx_avg(i,j,itemp)+
     &                                       btflx(i,j,itemp)*(rho0*Cp))
            enddo
          enddo
        endif
# endif
#  endif


# ifdef SALINITY
        if (wrtavg(indxSwflx)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              stflx_avg(i,j,isalt)=cff*(cff1*stflx_avg(i,j,isalt)+
     &                               stf_cff*stflx(i,j,isalt) /
     &               ( max(eps,t(i,j,N,nstp,isalt)) ) )
            enddo
          enddo
        endif
#  ifdef BWFLUX
        if (wrtavg(indxBwflx)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              btflx_avg(i,j,isalt)=cff*(cff1*btflx_avg(i,j,isalt)+
     &                               stf_cff*btflx(i,j,isalt) /
     &               ( max(eps,t(i,j,1,nstp,isalt)) ) )
            enddo
          enddo
        endif
#  endif
#endif
        if (wrtavg(indxShflx_rsw)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
#   ifdef BULK_FLUX
              shflx_rsw_avg(i,j)=cff*( cff1*shflx_rsw_avg(i,j)+
     &                                 shflx_rsw(i,j)*rho0*Cp)
#  else
              srflx_avg(i,j)=cff*( cff1*srflx_avg(i,j)+
     &                             srflx(i,j)*rho0*Cp)
#  endif
            enddo
          enddo
        endif
#  if (defined BULK_FLUX || defined OA_COUPLING) && defined TEMPERATURE
        if (wrtavg(indxShflx_rlw)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              shflx_rlw_avg(i,j)=cff*( cff1*shflx_rlw_avg(i,j)+
     &                                 shflx_rlw(i,j)*rho0*Cp)
            enddo
          enddo
        endif
        if (wrtavg(indxShflx_lat)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              shflx_lat_avg(i,j)=cff*( cff1*shflx_lat_avg(i,j)+
     &                                 shflx_lat(i,j)*rho0*Cp)
            enddo
          enddo
        endif
        if (wrtavg(indxShflx_sen)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              shflx_sen_avg(i,j)=cff*( cff1*shflx_sen_avg(i,j)+
     &                                 shflx_sen(i,j)*rho0*Cp)
            enddo
          enddo
        endif
#  endif
#  if defined SST_SKIN && defined TEMPERATURE
        if (wrtavg(indxT)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              sst_skin_avg(i,j)=cff*( cff1*sst_skin_avg(i,j)+
     &                                     sst_skin(i,j) )
            enddo
          enddo
        endif
#  endif
#  ifdef VIS_COEF_3D
        if (wrtavg(indxVisc)) then
          do k=1,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                visc3d_avg(i,j,k)=cff*(cff1*visc3d_avg(i,j,k)
     &                                     +visc3d_r(i,j,k))
              enddo
            enddo
          enddo
        endif
#  endif
#  ifdef DIF_COEF_3D
        if (wrtavg(indxDiff)) then
          do k=1,N
            do j=Jstr,Jend
              do i=Istr,Iend
                diff3d_avg(i,j,k)=cff*(cff1*diff3d_avg(i,j,k)
#   ifdef TS_DIF2
#    ifdef TEMPERATURE
     &                      +diff2(i,j,itemp)
#    endif
#    ifdef TS_DIF_SMAGO
     &                     +diff3d_r(i,j,k)
#    endif
#   elif defined TS_DIF4
#    ifdef TEMPERATURE
     &                     + diff4(i,j,itemp)
#    endif
#    ifdef TS_DIF_SMAGO
     &                     +diff3d_r(i,j,k)*om_r(i,j)*on_r(i,j)
#    endif
     &                     +0.25*(diff3d_u(i,j,k)+diff3d_u(i+1,j,k)
     &                           +diff3d_v(i,j,k)+diff3d_v(i,j+1,k))
#   endif
     &                                  )
              enddo
            enddo
          enddo
        endif
#  endif
#  ifdef AVERAGES_K
        if (wrtavg(indxAkv)) then
          do k=0,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                Akv_avg(i,j,k)=cff*(cff1*Akv_avg(i,j,k)+Akv(i,j,k))
              enddo
            enddo
          enddo
        endif
#    ifdef TEMPERATURE
        if (wrtavg(indxAkt)) then
          do k=0,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                Akt_avg(i,j,k,itemp)=cff*(cff1*Akt_avg(i,j,k,itemp)
     &                                        +Akt(i,j,k,itemp))
              enddo
            enddo
          enddo
        endif
#   endif
#   ifdef SALINITY
        if (wrtavg(indxAks)) then
          do k=0,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                Akt_avg(i,j,k,isalt)=cff*(cff1*Akt_avg(i,j,k,isalt)
     &                                        +Akt(i,j,k,isalt))
              enddo
            enddo
          enddo
        endif
#   endif
#  endif
#  if defined BIOLOGY && !defined PISCES
        if (wrtavg(indxHel)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              hel_avg(i,j)=cff*( cff1*hel_avg(i,j)
     &                               +hel(i,j)     )
            enddo
          enddo
        endif
#   ifdef BIO_NChlPZD
        if (wrtavg(indxChC)) then
          do k=1,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                theta_avg(i,j,k)=cff*( cff1*theta_avg(i,j,k)
     &                                     +theta(i,j,k)     )
              enddo
            enddo
          enddo
        endif
#    ifdef OXYGEN
        if (wrtavg(indxU10)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              u10_avg(i,j)=cff*( cff1*u10_avg(i,j)+u10(i,j) )
            enddo
          enddo
        endif
        if (wrtavg(indxKvO2)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              Kv_O2_avg(i,j)=cff*( cff1*Kv_O2_avg(i,j)+Kv_O2(i,j) )
            enddo
          enddo
        endif
        if (wrtavg(indxO2sat)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              O2satu_avg(i,j)=cff*( cff1*O2satu_avg(i,j)
     &                                     +O2satu(i,j) )
            enddo
          enddo
        endif
#    endif /* OXYGEN */
#   elif defined BIO_BioEBUS
        if (wrtavg(indxAOU)) then
          do k=1,N
            do j=JstrR,JendR
              do i=IstrR,IendR
                AOU_avg(i,j,k)=cff*( cff1*AOU_avg(i,j,k)+AOU(i,j,k) )
              enddo
            enddo
          enddo
        endif
        if (wrtavg(indxWIND10)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              wind10_avg(i,j)=cff*( cff1*wind10_avg(i,j)+wind10(i,j) )
            enddo
          enddo
        endif
#   endif
#  endif /* BIOLOGY */
#  ifdef SEDIMENT
!
! sediment bed layer thickness, porosity, volume
! fraction of size class in sediment bed
!
        do ilay=1,NLAY
          do itrc=1,NST
            do j=JstrR,JendR
              do i=IstrR,IendR
                bed_frac_avg(i,j,ilay,itrc)=
     &                           cff*( cff1*bed_frac_avg(i,j,ilay,itrc)
     &                                     +bed_frac(i,j,ilay,itrc)   )
              enddo
            enddo
          enddo
        enddo
#   ifdef SUSPLOAD
        do itrc=1,NST
          indxWrk=indxDFLX(1)+itrc-1
!        if (wrtavg(indxWrk)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              settling_flux_avg(i,j,itrc)=
     &                           cff*( cff1*settling_flux_avg(i,j,itrc)
     &                                     +settling_flux(i,j,itrc) )
            enddo
          enddo
!        endif
        enddo
        do itrc=1,NST
          indxWrk=indxEFLX(1)+itrc-1
!         if (wrtavg(indxWrk)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              ero_flux_avg(i,j,itrc)=
     &                           cff*( cff1*ero_flux_avg(i,j,itrc)
     &                                     +ero_flux(i,j,itrc) )
            enddo
          enddo
!         endif
        enddo
#   endif
#   ifdef BEDLOAD
        do itrc=1,NST
          indxWrk=indxBDLU(1)+itrc-1
!        if (wrtavg(indxWrk)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              bedldu_avg(i,j,itrc)=
     &                           cff*( cff1*bedldu_avg(i,j,itrc)
     &                                     +bedldu(i,j,itrc)   )
            enddo
          enddo
!        endif
        enddo
        do itrc=1,NST
          indxWrk=indxBDLV(1)+itrc-1
!         if (wrtavg(indxWrk)) then
          do j=JstrR,JendR
            do i=IstrR,IendR
              bedldv_avg(i,j,itrc)=
     &                           cff*( cff1*bedldv_avg(i,j,itrc)
     &                                     +bedldv(i,j,itrc)   )
            enddo
          enddo
!         endif
        enddo
#   endif
#  endif /* SEDIMENT */
# endif /* SOLVE3D */

# ifdef WAVE_IO
#  ifdef WKB_WWAVE
        iout=wstp
#  endif
        if (wrtavg(indxHRM)) then
          do j=jstrR,jendR
            do i=istrR,iendR
#  ifdef WKB_WWAVE
              whrm_avg(i,j) = cff*( cff1*whrm_avg(i,j)+hrm(i,j,iout) )
#  else
              whrm_avg(i,j) = cff*( cff1*whrm_avg(i,j)+whrm(i,j) )
#  endif
            enddo
          enddo
        endif
        if (wrtavg(indxFRQ)) then
          do j=jstrR,jendR
            do i=istrR,iendR
#  ifdef WKB_WWAVE
              wfrq_avg(i,j) = cff*( cff1*wfrq_avg(i,j)+frq(i,j,iout) )
#  else
              wfrq_avg(i,j) = cff*( cff1*wfrq_avg(i,j)+wfrq(i,j) )
#  endif
            enddo
          enddo
        endif
#  ifdef WKB_WWAVE
        if (wrtavg(indxWAC)) then
          do j=jstrR,jendR
            do i=istrR,iendR
              wac_avg(i,j) = cff*( cff1*wac_avg(i,j)+wac(i,j,iout) )
            enddo
          enddo
        endif
#  endif
        if (wrtavg(indxWKX)) then
          do j=jstrR,jendR
            do i=istrR,iendR
#  ifdef WKB_WWAVE
              wkx_avg(i,j) = cff*( cff1*wkx_avg(i,j)+wkx(i,j,iout) )
#  else
              wkx_avg(i,j) = cff*( cff1*wkx_avg(i,j)+wwkx(i,j) )
#  endif
            enddo
          enddo
        endif
        if (wrtavg(indxWKE)) then
          do j=jstrR,jendR
            do i=istrR,iendR
#  ifdef WKB_WWAVE
              wke_avg(i,j) = cff*( cff1*wke_avg(i,j)+wke(i,j,iout) )
#  else
              wke_avg(i,j) = cff*( cff1*wke_avg(i,j)+wwke(i,j) )
#  endif
            enddo
          enddo
        endif
        if (wrtavg(indxEPB)) then
          do j=jstrR,jendR
            do i=istrR,iendR
              wepb_avg(i,j) = cff*( cff1*wepb_avg(i,j)+wepb(i,j) )
            enddo
          enddo
        endif
        if (wrtavg(indxEPD)) then
          do j=jstrR,jendR
            do i=istrR,iendR
              wepd_avg(i,j) = cff*( cff1*wepd_avg(i,j)+wepd(i,j) )
            enddo
          enddo
        endif
#  ifdef WAVE_ROLLER
#   ifdef WKB_WWAVE
        if (wrtavg(indxWAR)) then
          do j=jstrR,jendR
            do i=istrR,iendR
              war_avg(i,j) = cff*( cff1*war_avg(i,j)+war(i,j,iout) )
            enddo
          enddo
        endif
#   endif
        if (wrtavg(indxEPR)) then
          do j=jstrR,jendR
            do i=istrR,iendR
              wepr_avg(i,j) = cff*( cff1*wepr_avg(i,j)+wepr(i,j) )
            enddo
          enddo
        endif
#  endif
# endif  /* WKB_WAVE_IO */

# ifdef MRL_WCI
        if (wrthis(indxSUP)) then
          do j=jstrR,jendR
            do i=istrR,iendR
              sup_avg(i,j) = cff*( cff1*sup_avg(i,j)+sup(i,j) )
            enddo
          enddo
        endif
        if (wrthis(indxUST2D)) then
          do j=jstrR,jendR
            do i=istrR,iendR
              ust2d_avg(i,j) = cff*( cff1*ust2d_avg(i,j)+ust2d(i,j) )
            enddo
          enddo
        endif
        if (wrthis(indxVST2D)) then
          do j=jstrR,jendR
            do i=istrR,iendR
              vst2d_avg(i,j) = cff*( cff1*vst2d_avg(i,j)+vst2d(i,j) )
            enddo
          enddo
        endif
#  ifdef SOLVE3D
        if (wrthis(indxUST)) then
          do k=1,N
            do j=jstrR,jendR
              do i=istrR,iendR
                ust_avg(i,j,k)=cff * ( cff1*
     &                         ust_avg(i,j,k)+ust(i,j,k) )
              enddo
            enddo
          enddo
        endif
        if (wrthis(indxVST)) then
          do k=1,N
            do j=jstrR,jendR
              do i=istrR,iendR
                vst_avg(i,j,k)=cff * ( cff1*
     &                         vst_avg(i,j,k)+vst(i,j,k) )
              enddo
            enddo
          enddo
        endif
        if (wrthis(indxWST)) then
          do k=1,N
            do j=jstrR,jendR
              do i=istrR,iendR
                wst_avg(i,j,k)=cff * ( cff1*
     &                         wst_avg(i,j,k)+wst(i,j,k) )
              enddo
            enddo
          enddo
        endif
        if (wrthis(indxAkb)) then
          do k=0,N
            do j=jstrR,jendR
              do i=istrR,iendR
                akb_avg(i,j,k)=cff * ( cff1*
     &                         akb_avg(i,j,k)+Akb(i,j,k) )
              enddo
            enddo
          enddo
        endif
        if (wrthis(indxAkw)) then
          do k=0,N
            do j=jstrR,jendR
              do i=istrR,iendR
                akw_avg(i,j,k)=cff * ( cff1*
     &                         akw_avg(i,j,k)+Akw(i,j,k) )
              enddo
            enddo
          enddo
        endif
        if (wrthis(indxKVF)) then
          do k=1,N
            do j=jstrR,jendR
              do i=istrR,iendR
                kvf_avg(i,j,k)=cff * ( cff1*
     &                         kvf_avg(i,j,k)+kvf(i,j,k) )
              enddo
            enddo
          enddo
        endif
        if (wrthis(indxCALP)) then
          do j=jstrR,jendR
            do i=istrR,iendR
              calp_avg(i,j) = cff*( cff1*calp_avg(i,j)+calP(i,j) )
            enddo
          enddo
        endif
        if (wrthis(indxKAPS)) then
          do j=jstrR,jendR
            do i=istrR,iendR
              kaps_avg(i,j) = cff*( cff1*kaps_avg(i,j)+Kapsrf(i,j) )
            enddo
          enddo
        endif
#  endif  /* SOLVE3D */
# endif   /* MRL_WCI */

      endif
      return
      end
#else
      subroutine set_avg_empty
      end
#endif /* AVERAGES */
