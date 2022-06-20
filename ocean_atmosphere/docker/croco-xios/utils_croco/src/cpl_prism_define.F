! $Id: cpl_prism_define.F 1274 2013-09-16 09:24:57Z gcambon $
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
      SUBROUTINE cpl_prism_define
#include "cppdefs.h"
#if defined OA_COUPLING || defined OW_COUPLING
      USE mod_prism             ! OASIS3 prism module
      IMPLICIT NONE

#include "param.h"
#include "grid.h"
#include "scalars.h"
#include "mpi_cpl.h"
#include "ncscrum.h"
      integer i, j, ji, i0, j0, ja
      integer nproc, myproc, error_flg, ierr
      integer start_x, start_y
      real delta_lon, delta_lat
      integer ncid, varid, nf_fread
      INTEGER    :: level
      CHARACTER*2 clevel
      CHARACTER*9 rclevel
      CHARACTER*5 catm
      character fname*64
#include "netcdf.inc"

!     !-------------------------------------------------------------------
!     !             ***  ROUTINE cpl_prism_define  ***
!     !
!     ! ** Purpose :   Define grid and field information for ocean
!     !                exchange between CROCO and COUPLER. (OASIS3-MCT software)
!     !
!     ! ** Method  :   OASIS3 MPI communication
!     !--------------------------------------------------------------------

!------------------------------------------------------------------
!     Set the CROCO grid definition to PRISM system
!------------------------------------------------------------------

      start_x =  iminmpi
      start_y =  jminmpi
      
!     Set the CROCO partition definition to PRISM system
!------------------------------------------------------------------

      oasis_ig_paral(1) = 2     ! Box partition
      oasis_ig_paral(2) = (LLm)*(start_y-1) + (start_x-1)
      oasis_ig_paral(3) = Lmmpi
      oasis_ig_paral(4) = Mmmpi
      oasis_ig_paral(5) = LLm


      CALL prism_def_partition_proto(oasis_part_id, 
     &     oasis_ig_paral, comp_ierror, LLm*MMm)
      IF ( comp_ierror /= PRISM_Ok ) THEN
        CALL prism_abort_proto ( comp_id, 'cpl_prism_grid',
     &       'Failure in prism_def_partition')
      END IF

!------------------------------------------------------------------
!     Set the CROCO grid definition to PRISM system
!------------------------------------------------------------------

       CALL cpl_prism_grid(oasis_part_id)

!------------------------------------------------------------------
!     Set the WRF exchange variables definition to PRISM system
!------------------------------------------------------------------

      oasis_var_nodims(1) = 2   ! Rank of the field array is 2
      oasis_var_nodims(2) = 1   ! Bundles always 1 for OASIS3

      oasis_var_type = PRISM_Real

      oasis_var_shape(1) = 1
      oasis_var_shape(2) = Lmmpi !Lm+2+padd_X + 1  !Lm
      oasis_var_shape(3) = 1
      oasis_var_shape(4) = Mmmpi !Mm+2+padd_E + 1  !Mm

! default definiton
      ssnd_clname(:) = 'not defined' 
      srcv_clname(:) = 'not defined' 
#ifdef AGRIF
      level=Agrif_Fixed()
      write(clevel,'(a,i1.1)') "_", level
      write(rclevel,'(a,i1.1)') "_CPLMASK", level
#else
      level=0
      write(clevel,'(a)') '  '
      write(rclevel,'(a)') '         '
#endif

!------------------------------------------------------------------
!     Sent
!------------------------------------------------------------------
!     SST
      ssnd_clname(1) = 'CROCO_SST'//trim(clevel)
!     SSH
      ssnd_clname(2) = 'CROCO_SSH'//trim(clevel)
!     UOCE
      ssnd_clname(3) = 'CROCO_UOCE'//trim(clevel)
!     VOCE
      ssnd_clname(4) = 'CROCO_VOCE'//trim(clevel)
!     EOCE
      ssnd_clname(5) = 'CROCO_EOCE'//trim(clevel)
!     NOCE
      ssnd_clname(6) = 'CROCO_NOCE'//trim(clevel)

!------------------------------------------------------------------
!     Received
!------------------------------------------------------------------
!     Evaporation-Precipitation [kg/m^2/s]
      srcv_clname(1) = 'CROCO_EVPR'//trim(rclevel)
!     net surface SW radiation [W/m2]
      srcv_clname(2) = 'CROCO_SRFL'//trim(rclevel)
!     total non-solar heat flux [W/m2]
      srcv_clname(3) = 'CROCO_STFL'//trim(rclevel)
!     u component of wind stress [N/m2]
      srcv_clname(4) = 'CROCO_UTAU'//trim(rclevel)
!     v component of wind stress [N/m2]
      srcv_clname(5) = 'CROCO_VTAU'//trim(rclevel)
!     wind stress module [N/m2]
      srcv_clname(6) = 'CROCO_TAUM'//trim(rclevel)
!     zonal wind tress [N/m2]
      srcv_clname(7) = 'CROCO_ETAU'//trim(rclevel)
!     meridional wind tress [N/m2]
      srcv_clname(8) = 'CROCO_NTAU'//trim(rclevel)
!     Surface pressure [Pa]
      srcv_clname(9) = 'CROCO_PSFC'//trim(rclevel)

!     mask to manage wet-drying[]
      srcv_clname(10) = 'CROCO_DRY'//trim(rclevel)
!     mean Wave Period [s]
      srcv_clname(11) = 'CROCO_T0M1'//trim(rclevel)
!     significant wave height [m]
      srcv_clname(12) = 'CROCO_HS'//trim(rclevel)
!     mean wave direction [radian]
      srcv_clname(13) = 'CROCO_DIR'//trim(rclevel)
!     bhd: wave induced Bernoulli head pressure [N.m-1]
      srcv_clname(14) = 'CROCO_BHD'//trim(rclevel)
!     wave_phioc :  wave to ocean TKE flux  [W.m-2]
      srcv_clname(15) = 'CROCO_FOC'//trim(rclevel)
!     wave_wlm :  wave mean wave length  [m]
      srcv_clname(16) = 'CROCO_LM'//trim(rclevel)   
!     wave_ubrx: x-component rms amplitude of orbital bottom velocity of the waves [m.s-1]
      srcv_clname(17) = 'CROCO_UBRX'//trim(rclevel)
!     wave_ubry: y-component rms amplitude of orbital bottom velocity of the waves [m.s-1]
      srcv_clname(18) = 'CROCO_UBRY'//trim(rclevel)
!     wave_ussx: x-component of surface Stokes drift velocity [m.s-1]
      srcv_clname(19) = 'CROCO_USSX'//trim(rclevel)
!     wave_ussy: y-component of surface Stokes drift velocity [m.s-1]
      srcv_clname(20) = 'CROCO_USSY'//trim(rclevel)
!     wave_tauox: x-component of the wave-ocean momentum flux [m2.s-2]
      srcv_clname(21) = 'CROCO_UTWO'//trim(rclevel)  
!     wave_tauoy: y-component of the wave-ocean momentum flux [m2.s-2]
      srcv_clname(22) = 'CROCO_VTWO'//trim(rclevel)  
!     wave_tauox: eastward-component of the wave-ocean momentum flux [m2.s-2]
      srcv_clname(23) = 'CROCO_ETWO'//trim(rclevel) 
!     wave_tauoy: northward-component of the wave-ocean momentum flux [m2.s-2]
      srcv_clname(24) = 'CROCO_NTWO'//trim(rclevel)
!     wave_tauwx: Net wave-supported stress, x-component [m2.s-2]
      srcv_clname(25) = 'CROCO_UTAW'//trim(rclevel)  
!     wave_tauwy: Net wave-supported stress, y-component [m2.s-2]
      srcv_clname(26) = 'CROCO_VTAW'//trim(rclevel) 
!     wave_tauwx: Net wave-supported stress, eastward-component [m2.s-2]
      srcv_clname(27) = 'CROCO_ETAW'//trim(rclevel)   
!     wave_tauwy: Net wave-supported stress, northward-component [m2.s-2]
      srcv_clname(28) = 'CROCO_NTAW'//trim(rclevel)

!     long wave radiation flux [W/m2]
      srcv_clname(29) = 'CROCO_LWFL'//trim(rclevel)
!     latent heat flux [W/m2]
      srcv_clname(30) = 'CROCO_LHFL'//trim(rclevel)
!     sensible heat flux [W/m2]
      srcv_clname(31) = 'CROCO_SHFL'//trim(rclevel)

      DO ja = 0, nmaxatm

         IF ( ja == 0 ) THEN   ;   catm = ""
         ELSE                  ;   WRITE(catm,'(a,i1.1)') "_EXT", ja
         ENDIF
!     
!     ... Announce sent variables.
      DO ji = 1, nmaxfld
          CALL prism_def_var_proto ( ssnd_nid(ja,ji),
     &       trim(ssnd_clname(ji))//trim(catm),
     &       oasis_part_id, oasis_var_nodims,
     &       PRISM_Out, oasis_var_shape,
     &       oasis_var_type, comp_ierror)
          IF ( comp_ierror /= PRISM_Ok ) THEN
             CALL prism_abort_proto( ssnd_nid(ja,ji),'cpl_prism_define',
     &        'Failure in prism_def_var: '//ssnd_clname(ji)//trim(catm))
          ELSE
             IF ( ssnd_nid(ja,ji) /= -1 ) THEN
                MPI_master_only write(*,*) 
     & 'CPL-CROCO: sent ',trim(ssnd_clname(ji))//trim(catm),ssnd_nid(ja,ji)
             ENDIF
          ENDIF
      END DO

!     ... Announce received variables.
      DO ji = 1, nmaxfld
         CALL prism_def_var_proto ( srcv_nid(ja,ji),
     &       trim(srcv_clname(ji))//trim(catm),
     &       oasis_part_id, oasis_var_nodims, 
     &       PRISM_In, oasis_var_shape, 
     &       oasis_var_type, comp_ierror)
          IF ( comp_ierror /= PRISM_Ok ) THEN
             CALL prism_abort_proto( srcv_nid(ja,ji),'cpl_prism_define',
     &        'Failure in prism_def_var: '//trim(srcv_clname(ji))//trim(catm))
          ELSE
             IF ( srcv_nid(ja,ji) /= -1 ) THEN
                MPI_master_only write(*,*) 
     & 'CPL-CROCO: received ',trim(srcv_clname(ji))//trim(catm),srcv_nid(ja,ji)
             ENDIF
          ENDIF
      END DO

      END DO

!------------------------------------------------------------------
!     End of definition phase
!------------------------------------------------------------------
      CALL MPI_Barrier( MPI_COMM_WORLD, comp_ierror ) 
#ifdef AGRIF
      if (.not.Agrif_Root()) then
         if (Agrif_fixed() == Agrif_Nb_Fine_Grids()) then
#endif
            CALL prism_enddef_proto( comp_ierror )
            IF ( comp_ierror /= PRISM_Ok ) THEN
               CALL prism_abort_proto ( comp_id, 'cpl_prism_enddef', 
     &              'Failure in prism_enddef')
            ELSE
               MPI_master_only write(*,*) 
     &              'CPL-CROCO: ENDDEF OK'
            ENDIF
#ifdef AGRIF
         endif
      endif
#endif

      cplmsk(:,:,0) = 1.            ! default use, no coupling mask
      cplmsk(:,:,1:nmaxatm) = 0.    ! default definition when using coupling mask
      fname = 'coupling_masks'//trim(clevel)//'.nc'
# if defined MPI && defined PARALLEL_FILES
      ierr=0
      call insert_node (fname, len_trim(fname), mynode, NNODES, ierr)
# endif
      ierr=nf_open(trim(fname), nf_nowrite, ncid)
      if (ierr == NF_NOERR) then
          MPI_master_only write(*,*) 
     &    'CPL-CROCO: open coupling_masks'//trim(clevel)//'.nc ok'
          DO ja = 1, nmaxatm
             WRITE(catm,'(a,i1.1)') "_atm", ja
             ierr=nf_inq_varid (ncid, 'cplmsk'//catm, varid)
             if (ierr == NF_NOERR) then
      ierr=nf_fread (cplmsk(START_2D_ARRAY,ja), ncid, varid, 0, r2dvar)
                MPI_master_only write(*,*) 
     &               'CPL-CROCO: read cplmsk'//catm//' ok',
     &               minval(cplmsk(1:Lmmpi,1:Mmmpi,ja)),
     &               maxval(cplmsk(1:Lmmpi,1:Mmmpi,ja))
             endif
          enddo
          ierr=nf_close(ncid)
       else
          MPI_master_only write(*,*) 
     &        'CPL-CROCO: coupling_masks'//trim(clevel)//'.nc not found'
       endif


#else 
!---------------------------------------------------------------------------------------
!     Dummy subroutine                                                           NO Coupling
!---------------------------------------------------------------------------------------
#endif 

      END SUBROUTINE cpl_prism_define


