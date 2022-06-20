#!/usr/bin/python
# -*- coding: utf-8 -*-
#
print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
print('&&&                                           ')
print('&&&   Execution de:                           ')
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

file_grd = netCDF4.Dataset(curdir_path+'croco_grd.nc')
file_ini = netCDF4.Dataset(curdir_path+'croco_iniVmodif.nc')

lon=file_grd.variables['x_rho'][1:-1,1:-1]
lat=file_grd.variables['y_rho'][1:-1,1:-1]

sst=file_ini.variables['temp'][0,-1,1:-1,1:-1]
ssh=file_ini.variables['zeta'][0,1:-1,1:-1]
ucur=file_ini.variables['u'][0,-1,1:-1,1:]
vcur=file_ini.variables['v'][0,-1,1:,1:-1]

if cfg_debug : print('np.shape(sst)=', np.shape(sst))
if cfg_debug : print('np.shape(ucur)=', np.shape(ucur))

if cfg_debug : print('++++++++++++++++++++++++++++++++++++++++++++++')
if cfg_debug : print('+++                                           ')
if cfg_debug : print('+++   1. Creation du netcdf                   ')
if cfg_debug : print('+++                                           ')
if cfg_debug : print('++++++++++++++++++++++++++++++++++++++++++++++')

nlon_CROCO=len(lon[0,:])
nlat_CROCO=len(lat[:,0])

print('nlon_CROCO', nlon_CROCO)
print('nlat_CROCO', nlat_CROCO)

#====================================================
#===   1.0 Create the netcdf restart file
#====================================================
fout=netCDF4.Dataset(curdir_path+'rstrt_SAVE.nc','w',format='NETCDF3_64BIT')
fout.Description='Restart file for CROCO coupling'

#====================================================
#===   1.1 Create the dimensions of the files
#====================================================
fout.createDimension ('nlat_CROCO', nlat_CROCO)
fout.createDimension ('nlon_CROCO', nlon_CROCO)

#====================================================
#===   1.2 Create the variables
#====================================================
varout=fout.createVariable('CROCO_SST','d',('nlat_CROCO','nlon_CROCO'),fill_value=999.)
varout=fout.createVariable('CROCO_SSH','d',('nlat_CROCO','nlon_CROCO'),fill_value=999.)
varout=fout.createVariable('CROCO_UOCE','d',('nlat_CROCO','nlon_CROCO'),fill_value=999.)
varout=fout.createVariable('CROCO_VOCE','d',('nlat_CROCO','nlon_CROCO'),fill_value=999.)

#====================================================
#===   1.3 Write out the data arrays into the file
#====================================================
fout.variables['CROCO_SST'][:,:] = sst[:,:]+273.15
fout.variables['CROCO_SSH'][:,:] = ssh[:,:]
fout.variables['CROCO_UOCE'][:,:] = ucur[:,:]
fout.variables['CROCO_VOCE'][:,:] = vcur[:,:]

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