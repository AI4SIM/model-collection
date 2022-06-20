#!/usr/bin/python
# -*- coding: utf-8 -*-
#
###################################################
#=================================================#
#  Creating grid and restart file for toy model
#  Author : J. Pianezze
#  Date   :        2020
#=================================================#
###################################################
#
import netCDF4
import numpy as np
import scipy
#import matplotlib.pyplot as plt
import math
from   pylab import *
import os
#from   mpl_toolkits.basemap import interp
#from   scipy.spatial import cKDTree
#
#
curdir_path=os.getcwd()+'/'
home_path=os.environ['HOME']
print('###   Dossier courant:', curdir_path)
#
script_name=os.path.splitext(os.path.basename(sys.argv[0]))[0]
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Importation des toolbox
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#sys.path.append(os.environ['dir_python3_tools'])
#try:
#  import toolbox_model
#except ImportError:
#  print('###   Pas de fichier toolbox_model.py')
#else:
#  print('###   Importation du module toolbox_model')
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

######### BEG USER PART ###############################

cfg_modif_mask = True
cfg_file_mnh   = '../../../input_atm/PGD_WMED_10km.nc'
cfg_file_croco = '../../../input_oce/croco_grd.nc'

######### END USER PART ###############################

# -------------------------------------------------
#  CROCO
# -------------------------------------------------

file_CROCO = netCDF4.Dataset(cfg_file_croco)

#------ Read variables
lon_CROCO      = file_CROCO.variables['lon_rho'] [1:-1,1:-1]
lat_CROCO      = file_CROCO.variables['lat_rho'] [1:-1,1:-1]
mask_CROCO_ori = file_CROCO.variables['mask_rho'][1:-1,1:-1]

nlon_CROCO  = np.size(lon_CROCO[0,:]) ;  print('nlon_CROCO =', nlon_CROCO)
nlat_CROCO  = np.size(lat_CROCO[:,0]) ;  print('nlat_CROCO =', nlat_CROCO)
ncorn_CROCO = 4                       ;  print('ncorn_CROCO=', ncorn_CROCO)

print('---- corners longitude/latitude')
clo_CROCO=np.zeros((ncorn_CROCO,nlat_CROCO,nlon_CROCO))
cla_CROCO=np.zeros((ncorn_CROCO,nlat_CROCO,nlon_CROCO))

deltax=lon_CROCO[0,1]-lon_CROCO[0,0] ; print('deltax=', deltax)
clo_CROCO[0,:,:]=lon_CROCO[:,:]+deltax/2.0
clo_CROCO[1,:,:]=lon_CROCO[:,:]-deltax/2.0
clo_CROCO[2,:,:]=lon_CROCO[:,:]-deltax/2.0
clo_CROCO[3,:,:]=lon_CROCO[:,:]+deltax/2.0

deltay=lat_CROCO[1,0]-lat_CROCO[0,0] ; print('deltay=', deltay)
cla_CROCO[0,:,:]=lat_CROCO[:,:]+deltay/2.0
cla_CROCO[1,:,:]=lat_CROCO[:,:]+deltay/2.0
cla_CROCO[2,:,:]=lat_CROCO[:,:]-deltay/2.0
cla_CROCO[3,:,:]=lat_CROCO[:,:]-deltay/2.0

print('---- area')
area_CROCO=np.zeros((nlat_CROCO,nlon_CROCO))
area_CROCO[:,:]=deltax*deltay


print('---- mask')
mask_CROCO=np.zeros((nlat_CROCO,nlon_CROCO))

for ind_lon in range(nlon_CROCO):
  for ind_lat in range(nlat_CROCO):
    if mask_CROCO_ori[ind_lat,ind_lon] == 1.0 :
      mask_CROCO[ind_lat,ind_lon]=0
    else:
      mask_CROCO[ind_lat,ind_lon]=1

# -------------------------------------------------
#  MNH
# -------------------------------------------------

file_orog = netCDF4.Dataset(cfg_file_mnh)

#------ Read variables
lon_MNH  = file_orog.variables['longitude'][1:-1,1:-1]
lat_MNH  = file_orog.variables['latitude'][1:-1,1:-1]
frac_sea_MNH = file_orog.variables['FRAC_SEA'][1:-1,1:-1]

nlon_MNH=np.size(lon_MNH[0,:]) ;  print('nlon_MNH=', nlon_MNH)
nlat_MNH=np.size(lat_MNH[:,0]) ;  print('nlat_MNH=', nlat_MNH)
ncorn_MNH=4                     ;  print('ncorn_MNH=', ncorn_MNH)

print('---- corners longitude/latitude')
clo_MNH=np.zeros((ncorn_MNH,nlat_MNH,nlon_MNH))
cla_MNH=np.zeros((ncorn_MNH,nlat_MNH,nlon_MNH))

deltax=lon_MNH[0,1]-lon_MNH[0,0] ; print('deltax=', deltax)
clo_MNH[0,:,:]=lon_MNH[:,:]+deltax/2.0
clo_MNH[1,:,:]=lon_MNH[:,:]-deltax/2.0
clo_MNH[2,:,:]=lon_MNH[:,:]-deltax/2.0
clo_MNH[3,:,:]=lon_MNH[:,:]+deltax/2.0

deltay=lat_MNH[1,0]-lat_MNH[0,0] ; print('deltay=', deltay)
cla_MNH[0,:,:]=lat_MNH[:,:]+deltay/2.0
cla_MNH[1,:,:]=lat_MNH[:,:]+deltay/2.0
cla_MNH[2,:,:]=lat_MNH[:,:]-deltay/2.0
cla_MNH[3,:,:]=lat_MNH[:,:]-deltay/2.0

print('---- area')
area_MNH=np.zeros((nlat_MNH,nlon_MNH))
area_MNH[:,:]=deltax*deltay


print('---- mask and var send by toy')
mask_MNH = np.zeros((nlat_MNH,nlon_MNH))

for ind_lon in range(nlon_MNH):
  for ind_lat in range(nlat_MNH):
    if frac_sea_MNH[ind_lat,ind_lon] == 0.0 or frac_sea_MNH[ind_lat,ind_lon]>=999.0 :
      mask_MNH[ind_lat,ind_lon]=1
    else:
      mask_MNH[ind_lat,ind_lon]=0  	

# ----------------------------------------------------
#  Modification du mask (suppression ocean atlantique)
# ----------------------------------------------------

if cfg_modif_mask:

  for ind_lon in range(nlon_MNH):
    for ind_lat in range(nlat_MNH):
      if ( ind_lon < 47 and ind_lat > 72 ):
        mask_MNH[ind_lat,ind_lon] = 1



##################################################
print('------------------------------------------')
print(' Creating netcdf file : grids.nc')

grids_file=netCDF4.Dataset(curdir_path+'grids.nc','w',format='NETCDF3_64BIT')
grids_file.Description='Grid file for OASIS coupling'

# ----------------------------------
# Create the dimensions of the files
# ----------------------------------
grids_file.createDimension ('x_atmt', nlon_MNH)
grids_file.createDimension ('y_atmt', nlat_MNH)
grids_file.createDimension ('crn_atmt', 4 )

grids_file.createDimension ('x_ocnt', nlon_CROCO)
grids_file.createDimension ('y_ocnt', nlat_CROCO)
grids_file.createDimension ('crn_ocnt', 4 )

# ----------------------------------
# Create the variables of the files
# ----------------------------------
varout=grids_file.createVariable('atmt.lon','d',('y_atmt','x_atmt'))
varout=grids_file.createVariable('atmt.lat','d',('y_atmt','x_atmt'))
varout=grids_file.createVariable('atmt.clo','d',('crn_atmt','y_atmt','x_atmt'))
varout=grids_file.createVariable('atmt.cla','d',('crn_atmt','y_atmt','x_atmt'))

varout=grids_file.createVariable('ocnt.lon','d',('y_ocnt','x_ocnt'))
varout=grids_file.createVariable('ocnt.lat','d',('y_ocnt','x_ocnt'))
varout=grids_file.createVariable('ocnt.clo','d',('crn_ocnt','y_ocnt','x_ocnt'))
varout=grids_file.createVariable('ocnt.cla','d',('crn_ocnt','y_ocnt','x_ocnt'))

# ---------------------------------------
# Write out the data arrays into the file
# ---------------------------------------
grids_file.variables['atmt.lon'][:,:] = lon_MNH[:,:]
grids_file.variables['atmt.lat'][:,:] = lat_MNH[:,:]
grids_file.variables['atmt.clo'][:,:] = clo_MNH[:,:,:]
grids_file.variables['atmt.cla'][:,:] = cla_MNH[:,:,:]

grids_file.variables['ocnt.lon'][:,:] = lon_CROCO[:,:]
grids_file.variables['ocnt.lat'][:,:] = lat_CROCO[:,:]
grids_file.variables['ocnt.clo'][:,:] = clo_CROCO[:,:,:]
grids_file.variables['ocnt.cla'][:,:] = cla_CROCO[:,:,:]

# ---------------------------------------
# close the file
# ---------------------------------------
grids_file.close()

print(' Closing netcdf file : grids.nc')
print('------------------------------------------')
##################################################

##################################################
print('------------------------------------------')
print(' Creating netcdf file : areas.nc')

areas_file=netCDF4.Dataset(curdir_path+'areas.nc','w',format='NETCDF3_64BIT')
areas_file.Description='Grid file for OASIS coupling'

# ----------------------------------
# Create the dimensions of the files
# ----------------------------------
areas_file.createDimension ('x_atmt', nlon_MNH)
areas_file.createDimension ('y_atmt', nlat_MNH)
areas_file.createDimension ('x_ocnt', nlon_CROCO)
areas_file.createDimension ('y_ocnt', nlat_CROCO)

# ----------------------------------
# Create the variables of the files
# ----------------------------------
varout=areas_file.createVariable('atmt.srf','d',('y_atmt','x_atmt'))
varout=areas_file.createVariable('ocnt.srf','d',('y_ocnt','x_ocnt'))

# ---------------------------------------
# Write out the data arrays into the file
# ---------------------------------------
areas_file.variables['atmt.srf'][:,:] = area_MNH[:,:]
areas_file.variables['ocnt.srf'][:,:] = area_CROCO[:,:]

# ---------------------------------------
# close the file
# ---------------------------------------
areas_file.close()

print(' Closing netcdf file : areas.nc')
print('-----------------------------------------')
####################################################

##################################################
print('------------------------------------------')
print(' Creating netcdf file : masks.nc')

masks_file=netCDF4.Dataset(curdir_path+'masks.nc','w',format='NETCDF3_64BIT')
masks_file.Description='Grid file for OASIS coupling'

# ----------------------------------
# Create the dimensions of the files
# ----------------------------------
masks_file.createDimension ('x_atmt', nlon_MNH)
masks_file.createDimension ('y_atmt', nlat_MNH)
masks_file.createDimension ('x_ocnt', nlon_CROCO)
masks_file.createDimension ('y_ocnt', nlat_CROCO)

# ----------------------------------
# Create the variables of the files
# ----------------------------------
varout=masks_file.createVariable('atmt.msk','i',('y_atmt','x_atmt'))
varout=masks_file.createVariable('ocnt.msk','i',('y_ocnt','x_ocnt'))

#varout=masks_file.createVariable('frac_sea_MNH','d',('y_atmt','x_atmt'))
#varout=masks_file.createVariable('mask_CROCO_on_MNH','d',('y_atmt','x_atmt'))

# ---------------------------------------
# Write out the data arrays into the file
# ---------------------------------------
masks_file.variables['atmt.msk'][:,:] = mask_MNH  [:,:]
masks_file.variables['ocnt.msk'][:,:] = mask_CROCO[:,:]

#masks_file.variables['frac_sea_MNH'][:,:] = frac_sea_MNH[:,:]
#masks_file.variables['mask_CROCO_on_MNH'][:,:] = mask_CROCO_on_MNH[:,:]

# ---------------------------------------
# close the file
# ---------------------------------------
masks_file.close()

print(' Closing netcdf file : masks.nc')
print('-----------------------------------------')
####################################################
