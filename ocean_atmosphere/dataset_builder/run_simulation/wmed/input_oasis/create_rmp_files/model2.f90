!------------------------------------------------------------------------
! Copyright 2018/03, CERFACS, Toulouse, France.
! All rights reserved. Use is subject to OASIS3 license terms.
!=============================================================================
!
PROGRAM model2
  !
  USE netcdf
  USE mod_oasis
  USE read_all_data
  USE write_all_fields
  !
  IMPLICIT NONE
  !
  INCLUDE 'mpif.h'
  !
  INTEGER, PARAMETER :: wp = SELECTED_REAL_KIND(12,307) ! double
  !
  CHARACTER(len=30), PARAMETER   :: data_gridname='grids.nc' ! file with the grids
  CHARACTER(len=30), PARAMETER   :: data_maskname='masks.nc' ! file with the masks
  CHARACTER(len=30)              :: data_filename, field_name
  !
  ! Component name (6 characters) same as in the namcouple
  CHARACTER(len=6)   :: comp_name = 'model2'
  CHARACTER(len=128) :: comp_out ! name of the output log file
  CHARACTER(len=3)   :: chout
  CHARACTER(len=4)   :: cl_grd_tgt ! name of the target grid
  !
  NAMELIST /grid_target_characteristics/cl_grd_tgt
  !
  ! Global grid parameters : 
  INTEGER :: nlon, nlat    ! dimensions in the 2 directions of space
  INTEGER :: il_size
  INTEGER, PARAMETER :: echelle=1            ! To calculate th delta error for plot
  REAL (kind=wp), DIMENSION(:,:), POINTER    :: gg_lon,gg_lat
  INTEGER, DIMENSION(:,:), POINTER           :: gg_mask ! mask, 0 == valid point, 1 == masked point 
  !
  INTEGER :: mype, npes ! rank and number of pe
  INTEGER :: localComm  ! local MPI communicator and Initialized
  INTEGER :: comp_id    ! component identification
  !
  INTEGER, DIMENSION(:), ALLOCATABLE :: il_paral ! Decomposition for each proc
  !
  INTEGER :: ierror, rank, w_unit
  INTEGER :: ic_nmsk, ic_nmskrv
  INTEGER :: FILE_Debug=2
  !
  ! Names of exchanged Fields
  CHARACTER(len=8), PARAMETER :: var_name = 'FRECVANA' ! 8 characters field received by the atmospheremodel2 from model1
  !
  ! Used in oasis_def_var and oasis_def_var
  INTEGER                       :: var_id 
  INTEGER                       :: var_nodims(2) 
  INTEGER                       :: var_type
  !
  REAL (kind=wp), PARAMETER     :: field_ini = -1. ! initialisation of received fields
  !
  ! Grid parameter definition
  INTEGER                       :: part_id  ! use to connect the partition to the variables
  INTEGER                       :: var_sh(4) ! local dimensions of the arrays; 2 x rank (=4)
  INTEGER :: ibeg, iend, jbeg, jend
  !
  ! Local fields arrays used in routines oasis_put and oasis_get
  REAL (kind=wp), POINTER       :: field_recv1d(:,:), field_recv(:,:), field_ana(:,:), gg_error(:,:)
  INTEGER, POINTER              :: mask_error(:,:) ! error mask, 0 == masked point, 1 == valid point 
  !
  ! Min and Max of the error of interpolation
  REAL (kind=wp)             :: min,max
  !
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !   INITIALISATION
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  CALL mpi_init(ierror)
  !
  CALL oasis_init_comp (comp_id, comp_name, ierror )
  IF (ierror /= 0) THEN
      WRITE(0,*) 'oasis_init_comp abort by model2 compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at line 109')
  ENDIF
  !
  ! Unit for output messages : one file for each process
  CALL MPI_Comm_Rank ( MPI_COMM_WORLD, rank, ierror )
  IF (ierror /= 0) THEN
      WRITE(0,*) 'MPI_Comm_Rank abort by model2 compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at line 116')
  ENDIF
  !
  !
  !!!!!!!!!!!!!!!!! OASIS_GET_LOCALCOMM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  CALL oasis_get_localcomm ( localComm, ierror )
  IF (ierror /= 0) THEN
      WRITE (0,*) 'oasis_get_localcomm abort by model2 compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at line 125')
  ENDIF
  !
  ! Get MPI size and rank
  CALL MPI_Comm_Size ( localComm, npes, ierror )
  IF (ierror /= 0) THEN
      WRITE(0,*) 'MPI_comm_size abort by model2 compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at line 132')
  ENDIF
  !
  CALL MPI_Comm_Rank ( localComm, mype, ierror )
  IF (ierror /= 0) THEN
      WRITE (0,*) 'MPI_Comm_Rank abort by model2 compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at line 138')
  ENDIF
  !
  IF ((FILE_Debug == 1) .AND. (mype == 0)) FILE_Debug=2
  !
  IF (FILE_Debug <= 1) THEN
      IF (mype == 0) THEN
          w_unit = 100 + rank
          WRITE(chout,'(I3)') w_unit
          comp_out=comp_name//'.root_'//chout
          OPEN(w_unit,file=TRIM(comp_out),form='formatted')
      ELSE
          w_unit = 15
          comp_out=comp_name//'.notroot'
          OPEN(w_unit,file=TRIM(comp_out),form='formatted',position='append')
      ENDIF
  ELSE
      w_unit = 100 + rank
      WRITE(chout,'(I3)') w_unit
      comp_out=comp_name//'.out_'//chout
      OPEN(w_unit,file=TRIM(comp_out),form='formatted')
  ENDIF
  !
  IF (FILE_Debug >= 2) THEN
      WRITE (w_unit,*) '-----------------------------------------------------------'
      WRITE (w_unit,*) TRIM(comp_name), ' running with reals compiled as kind ',wp
      WRITE (w_unit,*) 'I am component ', TRIM(comp_name), ' global rank :',rank
      WRITE (w_unit,*) '----------------------------------------------------------'
      WRITE(w_unit,*) 'I am the ', TRIM(comp_name), ' ', 'component identifier', comp_id, 'local rank', mype
      WRITE (w_unit,*) 'Number of processors :',npes
      WRITE (w_unit,*) '----------------------------------------------------------'
      CALL FLUSH(w_unit)
  ENDIF
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !  GRID DEFINITION 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  ! Reading global grids.nc and masks.nc netcdf files
  ! Get arguments giving source grid acronym and field type
  !
  OPEN(UNIT=70,FILE='name_grids.dat',FORM='FORMATTED')
  READ(UNIT=70,NML=grid_target_characteristics)
  CLOSE(70)
  !
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'Target grid name : ',cl_grd_tgt
      CALL flush(w_unit)
  ENDIF
  !
  ! Reading dimensions of the grid
  CALL read_dimgrid(nlon,nlat,data_gridname,cl_grd_tgt,w_unit,FILE_Debug)
  !
  ! Allocate grid arrays
  ALLOCATE(gg_lon(nlon,nlat), STAT=ierror )
  IF ( ierror /= 0 ) WRITE(w_unit,*) 'Error allocating gg_lon'
  ALLOCATE(gg_lat(nlon,nlat), STAT=ierror )
  IF ( ierror /= 0 ) WRITE(w_unit,*) 'Error allocating gg_lat'
  ALLOCATE(gg_mask(nlon,nlat), STAT=ierror )
  IF ( ierror /= 0 ) WRITE(w_unit,*) 'Error allocating indice_mask'
  ALLOCATE(gg_error(nlon,nlat),STAT=ierror )
  IF ( ierror /= 0 ) WRITE(w_unit,*) 'Error allocating gg_error'
  ALLOCATE(mask_error(nlon,nlat),STAT=ierror )
  IF ( ierror /= 0 ) WRITE(w_unit,*) 'Error allocating mask_error'
  !
  ! Read global grid longitudes, latitudes and mask 
  !
  CALL read_grid(nlon,nlat, data_gridname, cl_grd_tgt, w_unit, FILE_Debug, gg_lon,gg_lat)
  CALL read_mask(nlon,nlat, data_maskname, cl_grd_tgt, w_unit, FILE_Debug, gg_mask)
  !
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'After grid and mask reading'
      CALL FLUSH(w_unit)
  ENDIF
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !  PARTITION DEFINITION 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ !
  !
  il_size = 3
  ALLOCATE(il_paral(il_size))
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'After allocate il_paral, il_size', il_size
      CALL FLUSH(w_unit)
  ENDIF
  !
  il_paral(1)=0
  il_paral(2)=0
  IF (mype == 0) THEN
     il_paral(3)=nlon*nlat
  ELSE
     il_paral(3)=0
  END IF
  !       
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'il_paral = ', il_paral(:)
      CALL FLUSH(w_unit)
  ENDIF
  !       
  CALL oasis_def_partition (part_id, il_paral, ierror)
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'After oasis_def_partition'
      CALL FLUSH(w_unit)
  ENDIF
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! COUPLING LOCAL FIELD DECLARATION 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  var_nodims(1) = 2    ! Rank of the field array is 2
  var_nodims(2) = 1    ! Bundles always 1 for OASIS3
  var_type = OASIS_Real
  !
  var_sh(1) = 1
  var_sh(2) = il_paral(3)
  var_sh(3) = 1 
  var_sh(4) = 1
  !
  CALL oasis_def_var (var_id,var_name, part_id, &
     var_nodims, OASIS_In, var_sh, var_type, ierror)
  IF (ierror /= 0) THEN
      WRITE (w_unit,*) 'oasis_def_var abort by model2 compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at line 266')
  ENDIF
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'After def_var'
      CALL FLUSH(w_unit)
  ENDIF
  !
  DEALLOCATE(il_paral)
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !         TERMINATION OF DEFINITION PHASE 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  CALL oasis_enddef ( ierror )
  IF (ierror /= 0) THEN
      WRITE (w_unit,*) 'oasis_enddef abort by model2 compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at line 281')
  ENDIF
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'After enddef'
      CALL FLUSH(w_unit)
  ENDIF
  !
IF (mype == 0) THEN
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! RECEIVE ARRAYS AND CALCULATE THE ERROR 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  ! Allocate the fields send and received by the model
  !
  ALLOCATE(field_recv1d(var_sh(2), var_sh(4)), STAT=ierror )
  IF ( ierror /= 0 ) WRITE(w_unit,*) 'Error allocating field_recv1d'
  ALLOCATE(field_recv(nlon,nlat), STAT=ierror )
  IF ( ierror /= 0 ) WRITE(w_unit,*) 'Error allocating field_recv'
  ALLOCATE(field_ana(nlon,nlat),STAT=ierror )
  IF ( ierror /= 0 ) WRITE(w_unit,*) 'Error allocating field_ana'
  !
  CALL function_ana(nlon, nlat, gg_lon, gg_lat, field_ana)
  !
  ! Get the field FRECVANA
  field_recv1d=field_ini
  CALL oasis_get(var_id, 0, field_recv1d, ierror)
  field_recv = RESHAPE(field_recv1d,(/nlon,nlat/))
  DEALLOCATE(field_recv1d)
  !
  IF (ierror .NE. OASIS_Ok .AND. ierror .LT. OASIS_Recvd) THEN
      WRITE (w_unit,*) 'oasis_get abort by model2 compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at line 327')
  ENDIF
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'MINVAL(field_recv),MAXVAL(field_recv)=',MINVAL(field_recv),MAXVAL(field_recv)
  ENDIF
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! Modify the field so to have:
  !   - on masked points: value of 10000, error of -10000
  !   - on non-masked points that did not receive any interpolated value: value of 1.e20, error of -1.e20
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! 
  gg_error=-10000
  mask_error=1
  WHERE (gg_mask == 1)     ! masked points
  	field_recv=10000.
        mask_error=0
  ELSEWHERE                     ! non-masked points
	WHERE (field_recv /= 0.)   ! non-masked points that received an interpolated value
	   gg_error = ABS(((field_ana - field_recv)/field_recv))*100
        ELSEWHERE   ! non-masked points that did not receive an interpolated value
	   gg_error=-1.e20
           field_recv=1.e20
           mask_error=0
        END WHERE
  END WHERE   
  !
  IF (FILE_Debug >= 2) THEN
      WRITE (w_unit,*) 'After calculating the interpolation error'
      CALL FLUSH(w_unit)
  ENDIF
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !  Write the error and the field in a NetCDF file by proc0
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  data_filename='error_interp.nc'
  field_name='error_interp'
  CALL write_field(nlon, nlat, data_filename, field_name, w_unit, FILE_Debug, gg_lon, gg_lat, gg_error)
  !
  data_filename='FRECVANA.nc'
  field_name='FRECVANA' 
  CALL write_field(nlon, nlat, data_filename, field_name, w_unit, FILE_Debug, gg_lon, gg_lat, field_recv)
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! Calculate error min and max on non-masked points that received an interpolated value
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  min=MINVAL(gg_error, MASK=mask_error>0)
  IF (FILE_Debug >= 2) THEN
     WRITE(w_unit,*) 'Min (%) and its location in the error field : ',min
     WRITE(w_unit,*) MINLOC(gg_error)
     CALL FLUSH(w_unit)
  ENDIF
  !
  max=MAXVAL(gg_error, MASK=mask_error>0)
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*)'Max (%) and its location in the error field : ',max
      WRITE(w_unit,*) MAXLOC(gg_error)
      CALL FLUSH(w_unit)
  ENDIF
  !
  IF (FILE_Debug >= 2) THEN
      ic_nmsk=nlon*nlat-SUM(gg_mask)
      WRITE(w_unit,*) 'Number of non-masked points :',ic_nmsk
      ic_nmskrv=SUM(mask_error)
      WRITE(w_unit,*) 'Number of non-masked points that received a value :',ic_nmskrv
      WRITE(w_unit,*) 'Error mean on non masked points that received a value (%): ', SUM(ABS(gg_error), MASK=mask_error>0)/ic_nmskrv
      WRITE(w_unit,*) 'Delta error (%/echelle) :',(max - min)/echelle
      WRITE(w_unit,*) 'End calculation of stat on the error'
      CALL FLUSH(w_unit)
  ENDIF

  CALL SYSTEM("ln -s "//TRIM(comp_out)//" model2.out")
ENDIF ! of proc0
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !         TERMINATION 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  IF (FILE_Debug >= 2) THEN
      WRITE (w_unit,*) 'End of the program, before oasis_terminate'
      CALL FLUSH(w_unit)
  ENDIF
  !
  CALL oasis_terminate (ierror)
  IF (ierror /= 0) THEN
      WRITE (w_unit,*) 'oasis_terminate abort by model2 compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at line 477')
  ENDIF
  !
  CALL mpi_finalize(ierror)
  !
END PROGRAM MODEL2
!
