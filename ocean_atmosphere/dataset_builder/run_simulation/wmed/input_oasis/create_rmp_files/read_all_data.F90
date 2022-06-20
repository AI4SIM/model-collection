MODULE read_all_data
  !
  USE netcdf
  IMPLICIT NONE
  !
  !
  CONTAINS
!****************************************************************************************
SUBROUTINE read_dimgrid (nlon,nlat,data_filename,cl_grd,w_unit,FILE_Debug)
  !**************************************************************************************
  USE netcdf
  IMPLICIT NONE
  !
  INTEGER                  :: i,j,k,w_unit,FILE_Debug
  !
  INTEGER                  :: il_file_id,il_grid_id,il_lon_id, &
     il_lat_id,il_indice_id, &
     lon_dims,lat_dims,imask_dims
  !
  INTEGER, DIMENSION(NF90_MAX_VAR_DIMS) :: lon_dims_ids,lat_dims_ids,&
     imask_dims_ids,lon_dims_len,&
     lat_dims_len,imask_dims_len  
  !               
  INTEGER, INTENT(out)     :: nlon,nlat
  !
  CHARACTER(len=30)        :: data_filename
  CHARACTER(len=4)         :: cl_grd ! name of the source grid
  CHARACTER(len=8)         :: cl_nam ! cl_grd+.lon,+.lat ...
  !
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'Data_filename :',data_filename
      CALL FLUSH(w_unit)
  ENDIF
  !
  ! Dimensions
  !
  CALL hdlerr(NF90_OPEN(data_filename, NF90_NOWRITE, il_file_id), __LINE__ )
  !
  cl_nam=cl_grd//".lon" 
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'Longitudes :',cl_nam
      CALL FLUSH(w_unit)
  ENDIF
  CALL hdlerr( NF90_INQ_VARID(il_file_id, cl_nam,  il_lon_id),    __LINE__ )
  cl_nam=cl_grd//".lat" 
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'Latitudes :',cl_nam
      CALL FLUSH(w_unit)
  ENDIF
  CALL hdlerr( NF90_INQ_VARID(il_file_id, cl_nam,  il_lat_id),    __LINE__ )

  CALL hdlerr( NF90_INQUIRE_VARIABLE(il_file_id, varid=il_lon_id, ndims=lon_dims, dimids=lon_dims_ids), __LINE__ )
  !
  ! The value lon_dims_len(i) is obtained thanks to the lon_dims_ids ID already obtained from the file
  DO i=1,lon_dims
    CALL hdlerr( NF90_INQUIRE_DIMENSION(ncid=il_file_id,dimid=lon_dims_ids(i),len=lon_dims_len(i)), __LINE__ )
  ENDDO
  !
  nlon=lon_dims_len(1)
  nlat=lon_dims_len(2)
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  CALL hdlerr( NF90_INQUIRE_VARIABLE(ncid=il_file_id, varid=il_lat_id, ndims=lat_dims, dimids=lat_dims_ids), __LINE__ )
  !
  ! The value lat_dims_len(i) is obtained thanks to the lat_dims_ids ID already obtained from the file
  DO i=1,lat_dims
    CALL hdlerr( NF90_INQUIRE_DIMENSION(ncid=il_file_id,dimid=lat_dims_ids(i),len=lat_dims_len(i)), __LINE__ )
  ENDDO
  !
  IF ( (lat_dims_len(1) .NE. lon_dims_len(1)).OR.(lat_dims_len(2) .NE. lon_dims_len(2)) ) THEN
      WRITE(w_unit,*) 'Problem model1 in read_dimgrid'
      WRITE(w_unit,*) 'Dimensions of the latitude are not the same as the ones of the longitude'
      CALL flush(w_unit)
      STOP
  ENDIF
  !
  CALL hdlerr(NF90_CLOSE(il_file_id), __LINE__ )
  !
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'Reading input file ',data_filename
      WRITE(w_unit,*) 'Global dimensions nlon=',nlon,' nlat=',nlat
      CALL FLUSH(w_unit)
  ENDIF
  !
  !
END SUBROUTINE read_dimgrid

  !****************************************************************************************
  SUBROUTINE read_grid (nlon,nlat,&
                        data_filename, cl_grd, w_unit, FILE_Debug, &
                        gridlon,gridlat)
  !**************************************************************************************
  !
  INTEGER, PARAMETER :: wp = SELECTED_REAL_KIND(12,307) ! double
  !
  INTEGER                  :: i,j,k,w_unit,FILE_Debug
  !
  INTEGER                  :: il_file_id,il_lon_id, il_lat_id                                      
  !               
  INTEGER, INTENT(in)     :: nlon,nlat
  !
  CHARACTER(len=30)        :: data_filename
  CHARACTER(len=4)         :: cl_grd ! name of the source grid
  CHARACTER(len=8)         :: cl_nam ! cl_grd+.lon,+.lat ...
  !
  INTEGER,  DIMENSION(2)   :: ila_dim
  INTEGER,  DIMENSION(2)   :: ila_what
  !
  REAL (kind=wp), DIMENSION(nlon,nlat)  :: gridlon,gridlat
  !
  !
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'Data_filename :',data_filename
      CALL FLUSH(w_unit)
  ENDIF
  !
  ! Dimensions
  !
  CALL hdlerr(NF90_OPEN(data_filename, NF90_NOWRITE, il_file_id), __LINE__ )
  !
  cl_nam=cl_grd//".lon" 
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'Longitudes :',cl_nam
      CALL FLUSH(w_unit)
  ENDIF
  CALL hdlerr( NF90_INQ_VARID(il_file_id, cl_nam,  il_lon_id),    __LINE__ )
  cl_nam=cl_grd//".lat" 
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'Latitudes :',cl_nam
      CALL FLUSH(w_unit)
  ENDIF
  CALL hdlerr( NF90_INQ_VARID(il_file_id, cl_nam,  il_lat_id),    __LINE__ )
  !
  ila_what(:)=1
  !
  ila_dim(1)=nlon
  ila_dim(2)=nlat
  !
  ! Data
  !
  !
  CALL hdlerr( NF90_GET_VAR (il_file_id, il_lon_id, gridlon, &
     ila_what, ila_dim), __LINE__ )
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) '... global grid longitudes reading done'
      CALL FLUSH(w_unit)
  ENDIF
  !
  CALL hdlerr( NF90_GET_VAR (il_file_id, il_lat_id, gridlat, &
     ila_what, ila_dim), __LINE__ )
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) '... global grid latitudes reading done'
      CALL FLUSH(w_unit)
  ENDIF
  !
  !
  CALL hdlerr( NF90_CLOSE(il_file_id), __LINE__ )
  !
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'End of routine read_grid'
      CALL FLUSH(w_unit)
  ENDIF
  !
END SUBROUTINE read_grid

  !****************************************************************************************
  SUBROUTINE read_mask (nlon, nlat, &
                        data_filename, cl_grd, w_unit, FILE_Debug, &
                        indice_mask)
  !**************************************************************************************
  !
  INTEGER                  :: i,j,k,w_unit,FILE_Debug
  !
  INTEGER                  :: il_file_id, il_indice_id
  !               
  INTEGER, INTENT(in)     :: nlon,nlat
  !
  CHARACTER(len=30)        :: data_filename
  CHARACTER(len=4)         :: cl_grd ! name of the source grid
  CHARACTER(len=8)         :: cl_nam ! cl_grd+.lon,+.lat ...
  !
  INTEGER,  DIMENSION(2)   :: ila_dim
  INTEGER,  DIMENSION(2)   :: ila_what
  !
  INTEGER, DIMENSION(nlon,nlat)  :: indice_mask
  !
  !
  ! Dimensions
  !
  CALL hdlerr(NF90_OPEN(data_filename, NF90_NOWRITE, il_file_id), __LINE__ )
  !
  cl_nam=cl_grd//".msk" 
  CALL hdlerr( NF90_INQ_VARID(il_file_id, cl_nam,  il_indice_id),    __LINE__ )
  !
  ila_what(:)=1
  !
  ila_dim(1)=nlon
  ila_dim(2)=nlat
  !
  ! Data
  !
  CALL hdlerr( NF90_GET_VAR (il_file_id, il_indice_id, indice_mask, &
     ila_what, ila_dim), __LINE__ )
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) '... global grid mask reading done'
      CALL FLUSH(w_unit)
  ENDIF
  !
  CALL hdlerr( NF90_CLOSE(il_file_id), __LINE__ )
  !
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'End of routine read_mask'
      CALL FLUSH(w_unit)
  ENDIF
  !
END SUBROUTINE read_mask

  !****************************************************************************************
  SUBROUTINE read_area (nlon,nlat, &
                        data_filename, cl_grd, w_unit, FILE_Debug, &
                        gridsrf)
  !****************************************************************************************
  !
  INTEGER, PARAMETER :: wp = SELECTED_REAL_KIND(12,307) ! double
  !
  INTEGER                  :: i,j,k,w_unit,FILE_Debug
  !
  INTEGER                  :: il_file_id,  il_srf_id
  !               
  INTEGER, INTENT(in)     :: nlon,nlat
  !
  CHARACTER(len=30)        :: data_filename
  CHARACTER(len=4)         :: cl_grd ! name of the source grid
  CHARACTER(len=8)         :: cl_nam ! cl_grd+.lon,+.lat ...
  !
  INTEGER,  DIMENSION(2)   :: ila_dim
  INTEGER,  DIMENSION(2)   :: ila_what
  !
  REAL (kind=wp), DIMENSION(nlon,nlat)  :: gridsrf
  !
  !
  ! Dimensions
  !
  CALL hdlerr(NF90_OPEN(data_filename, NF90_NOWRITE, il_file_id), __LINE__ )
  !
  cl_nam=cl_grd//".srf" 
  CALL hdlerr( NF90_INQ_VARID(il_file_id, cl_nam, il_srf_id),    __LINE__ )
  !
  ila_what(:)=1
  !
  ila_dim(1)=nlon
  ila_dim(2)=nlat
  !
  ! Data
  !
  CALL hdlerr( NF90_GET_VAR (il_file_id, il_srf_id, gridsrf, &
     ila_what, ila_dim), __LINE__ )
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) '... global grid mask reading done'
      CALL FLUSH(w_unit)
  ENDIF
  !
  CALL hdlerr( NF90_CLOSE(il_file_id), __LINE__ )
  !
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'End of routine read_mask'
      CALL FLUSH(w_unit)
  ENDIF
  !
END SUBROUTINE read_area

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
END MODULE read_all_data
