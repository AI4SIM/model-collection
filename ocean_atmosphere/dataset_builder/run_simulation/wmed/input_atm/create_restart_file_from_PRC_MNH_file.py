#!/usr/bin/python
# -*- coding: utf-8 -*-
#
print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
print('&&&                                           ')
print('&&&   Execution de:')
print('&&&   '+__file__)
print('&&&                                           ')
print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
#
print('##############################################')
print('###')
#
import netCDF4
import numpy as np
import os
#
curdir_path=os.getcwd()+'/'
home_path=os.environ['HOME']
print('###   Dossier courant:', curdir_path)
#
print('###')
print('##############################################')
#
cfg_debug=False
#
if cfg_debug : print('++++++++++++++++++++++++++++++++++++++++++++++')
if cfg_debug : print('+++                                           ')
if cfg_debug : print('+++   0. Lecture des variables                ')
if cfg_debug : print('+++                                           ')
if cfg_debug : print('++++++++++++++++++++++++++++++++++++++++++++++')

file_MNH = netCDF4.Dataset(curdir_path+'ECMWF_20060101_00.nc')

LON_MNH=file_MNH.variables['LON'][1:-1,1:-1]
LAT_MNH=file_MNH.variables['LAT'][1:-1,1:-1]
U10_MNH=file_MNH.variables['UT'][0,2,1:-1,1:-1]
V10_MNH=file_MNH.variables['VT'][0,2,1:-1,1:-1]

try:
  EVAP_MNH=file_MNH.variables['EVAP3D'][1:-1,1:-1]
except KeyError:
  print('Ne trouve pas la variable EVAP3D... prise = a 0!')
  EVAP_MNH=np.zeros((np.shape(LON_MNH)[0],np.shape(LON_MNH)[1]))
else:
  print('ok')

try:
  RAIN_MNH=file_MNH.variables['INPRR3D'][1:-1,1:-1]
except KeyError:
  print('Ne trouve pas la variable INPRR3D... prise = a 0!')
  RAIN_MNH=np.zeros((np.shape(LON_MNH)[0],np.shape(LON_MNH)[1]))
else:
  print('ok')

try:
  FMU_MNH=file_MNH.variables['FMU'][1:-1,1:-1]
  FMV_MNH=file_MNH.variables['FMV'][1:-1,1:-1]
  H_MNH=file_MNH.variables['GFLUX'][1:-1,1:-1]
  LE_MNH=file_MNH.variables['LE'][1:-1,1:-1]
  LW_MNH=file_MNH.variables['LW'][1:-1,1:-1]
  RN_MNH=file_MNH.variables['RN'][1:-1,1:-1]
except KeyError:
  print('Ne trouve pas la variable FMU... prise = a 0!')
  FMU_MNH=np.zeros((np.shape(LON_MNH)[0],np.shape(LON_MNH)[1]))
  FMV_MNH=np.zeros((np.shape(LON_MNH)[0],np.shape(LON_MNH)[1]))
  H_MNH=np.zeros((np.shape(LON_MNH)[0],np.shape(LON_MNH)[1]))
  LE_MNH=np.zeros((np.shape(LON_MNH)[0],np.shape(LON_MNH)[1]))
  LW_MNH=np.zeros((np.shape(LON_MNH)[0],np.shape(LON_MNH)[1]))    
  RN_MNH=np.zeros((np.shape(LON_MNH)[0],np.shape(LON_MNH)[1]))
else:
  print('ok')

if cfg_debug : print('++++++++++++++++++++++++++++++++++++++++++++++')
if cfg_debug : print('+++                                           ')
if cfg_debug : print('+++   1. Creation du netcdf                   ')
if cfg_debug : print('+++                                           ')
if cfg_debug : print('++++++++++++++++++++++++++++++++++++++++++++++')

nlon_MNH=len(LON_MNH[0,:])
nlat_MNH=len(LAT_MNH[:,0])

if cfg_debug : print('nlat_MNH', nlat_MNH)
if cfg_debug : print('nlon_MNH', nlon_MNH)

#====================================================
#===   1.0 Create the netcdf restart file
#====================================================
fout=netCDF4.Dataset(curdir_path+'rstrt_SAVE.nc','w',format='NETCDF3_64BIT')
fout.Description='Restart file for MNH coupling'

#====================================================
#===   1.1 Create the dimensions of the files
#====================================================
fout.createDimension ('nlat_MNH', nlat_MNH)
fout.createDimension ('nlon_MNH', nlon_MNH)

#====================================================
#===   1.2 Create the variables
#====================================================
varout=fout.createVariable('MNH_TAUX','d',('nlat_MNH','nlon_MNH'),fill_value=999.)
varout=fout.createVariable('MNH_TAUY','d',('nlat_MNH','nlon_MNH'),fill_value=999.) 
varout=fout.createVariable('MNH_HEAT','d',('nlat_MNH','nlon_MNH'),fill_value=999.)
varout=fout.createVariable('MNH_SNET','d',('nlat_MNH','nlon_MNH'),fill_value=999.)
varout=fout.createVariable('MNH_EVAP','d',('nlat_MNH','nlon_MNH'),fill_value=999.) 
varout=fout.createVariable('MNH_RAIN','d',('nlat_MNH','nlon_MNH'),fill_value=999.)
varout=fout.createVariable('MNH_EVPR','d',('nlat_MNH','nlon_MNH'),fill_value=999.)
varout=fout.createVariable('MNH_PRES','d',('nlat_MNH','nlon_MNH'),fill_value=999.)
varout=fout.createVariable('MNH_LWFL','d',('nlat_MNH','nlon_MNH'),fill_value=999.)
varout=fout.createVariable('MNH_LHFL','d',('nlat_MNH','nlon_MNH'),fill_value=999.)
varout=fout.createVariable('MNH_SHFL','d',('nlat_MNH','nlon_MNH'),fill_value=999.)
varout=fout.createVariable('MNH__U10','d',('nlat_MNH','nlon_MNH'),fill_value=999.) 
varout=fout.createVariable('MNH__V10','d',('nlat_MNH','nlon_MNH'),fill_value=999.)

#====================================================
#===   1.3 Write out the data arrays into the file
#====================================================
fout.variables['MNH_TAUX'][:,:] = FMU_MNH[:,:]
fout.variables['MNH_TAUY'][:,:] = FMV_MNH[:,:]
fout.variables['MNH_HEAT'][:,:] = H_MNH[:,:]
fout.variables['MNH_SNET'][:,:] = RN_MNH[:,:]
fout.variables['MNH_EVAP'][:,:] = EVAP_MNH[:,:]
fout.variables['MNH_RAIN'][:,:] = RAIN_MNH[:,:]
fout.variables['MNH_EVPR'][:,:] = EVAP_MNH[:,:]-RAIN_MNH[:,:]
fout.variables['MNH_PRES'][:,:] = RAIN_MNH[:,:]
fout.variables['MNH_LWFL'][:,:] = LW_MNH[:,:]
fout.variables['MNH_LHFL'][:,:] = LE_MNH[:,:]
fout.variables['MNH_SHFL'][:,:] = H_MNH[:,:]
fout.variables['MNH__U10'][:,:] = U10_MNH[:,:]
fout.variables['MNH__V10'][:,:] = V10_MNH[:,:]

#====================================================
#===   1.4 Close the netcdf file
#====================================================
fout.close()

print('Ecriture netcdf finie')

print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
print('&&&                                           ')
print('&&&   Fin                                     ')
print('&&&                                           ')
print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
