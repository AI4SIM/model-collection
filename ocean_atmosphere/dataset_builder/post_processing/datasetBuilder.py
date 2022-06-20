import xarray as xr
import numpy as np
import os
import glob
import logging
import shutil

from coare_croco import coare_croco

class DatasetBuilder:
    """
    Read the output files of an ocean-atmosphere coupled simulation and performs
    post processing. The atmosphere files contain ERA5 offline data in the
    simulation time-period and the ocean file contains the output of the coupled
    simulation. This post-processing aims at computing the ocean-atmosphere
    fluxes with the ocean being coupled to ERA5 data rather than a live
    atmosphere simulation. The results are stored in a folder containing two
    files per snapshot: oce_<datetime>.nc and atm_<datetime>.nc
    The post processing pipeline will:
        1- List the raw files to process in a given directory
        2- Read the grid layout of the ocean and the atmosphere
        3- Interpolate the atmospheric data to the ocean time
        4- Compute the fluxes between ocean and ERA5 atmosphere
    """
    def __init__(self,run_dir):
        """Create the PostProcessOA class.

        Args:
            run_dir (str): Path to the simulation run directory
        """
        self.logger = logging.getLogger(__name__)
        
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(run_dir)
        self.run_dir = run_dir

        # Read and list all ocean and atmosphere files in run_dir
        self._init_files()

        # Read the atmosphere geometry layout
        self.atm_geometry = self._read_atm_grid()

        # Read the ocean geometry layout
        self.oce_geometry = self._read_oce_geometry()



    def _init_files(self):
        """List ocean and atmosphere files in run_dir

        Args:
            run_dir (str): Path to the simulation run directory
        """
        
        self.oce_gridfile = os.path.join(self.run_dir, "croco_grd.nc")
        if not os.path.isfile(self.oce_gridfile):
            raise FileNotFoundError(self.oce_gridfile)
        self.logger.debug(f"Found ocean grid: {self.oce_gridfile}")
        
        oce_ncfiles = os.path.join(self.run_dir, "WMED_TEST_MODEL_*.nc")
        self.oce_ncfiles = sorted(glob.glob(oce_ncfiles))[:-1]
        
        if len(self.oce_ncfiles) == 0:
            raise FileNotFoundError(f"Could not find ocean file in {oce_ncfiles}")
        self.logger.debug(f"Found {len(self.oce_ncfiles)} ocean files")
        
        atm_ncfiles = os.path.join(self.run_dir, "ECMWF_*.nc")
        self.atm_ncfiles = sorted(glob.glob(atm_ncfiles))
        
        if len(self.atm_ncfiles) == 0:
            raise FileNotFoundError(f"Could not find atmosphere file in {oce_ncfiles}")
        self.logger.debug(f"Found {len(self.atm_ncfiles)} atmosphere files")
        

        
        
    def _read_oce_geometry(self):
        """
        Read the geometry layout of the ocean grid.

        Returns:
            ds (xarray.Dataset): the dataset containing the ocean grid
                layout
        """

        ds = xr.open_dataset(self.oce_gridfile, cache=False)

        variables_to_keep = [
            'lat_rho',
            'lon_rho',
            'lat_u',
            'lon_u',
            'lat_v',
            'lon_v',
            'angle'
            ]
        for v in ds.variables:
            if v not in variables_to_keep:
                ds = ds.drop_vars(v)
        # Remove duplicate coordinates (via rename to temporary name <>_)
        ds = ds.rename(
            {
                'xi_rho':  'oce_xr',
                'xi_v':  'oce_xr',
                'xi_u':  'oce_xu',
                'eta_rho': 'oce_yr',
                'eta_v': 'oce_yv',
                'eta_u': 'oce_yr',
            }
        )
        ds = ds.rename(
            {
                'lat_rho': 'oce_latr',
                'lat_u': 'oce_latu',
                'lat_v': 'oce_latv',
                'lon_rho': 'oce_lonr',
                'lon_u': 'oce_lonu',
                'lon_v': 'oce_lonv',
            }
        )

        ds['oce_xr'] = ds.oce_lonr[0,:]
        ds['oce_xu'] = ds.oce_lonu[0,:]
        ds['oce_yr'] = ds.oce_latr[:,0]
        ds['oce_yv'] = ds.oce_latv[:,0]

        self.logger.info(f"Extracted ocean geometry.")
        return ds

    def _read_atm_grid(self):
        """
        Read the geometry layout of the atmosphere grid.
        
        Returns:
            ds (xarray.Dataset): the dataset containing the atmosphere grid
                layout
        """

        ds = xr.open_dataset(self.atm_ncfiles[0], cache=False)
        # Remove coordinates for consistency
        ds = ds.rename(
            {
                'ni': 'atm_xr',
                'nj': 'atm_yr',
                'latitude': 'atm_latr',
                'longitude': 'atm_lonr',
                'time': 'atm_time',
            }
            )

        variables_to_keep = ['SSS', 'atm_latr', 'atm_lonr']
        for v in ds.variables:
            if v not in variables_to_keep:
                ds = ds.drop_vars(v)

        # Infer the ocean mask from ERA5 sea surface salinity (SSS)
        ds["atm_maskr"] = ds["SSS"] < 999
        ds = ds.drop_vars("SSS")

        # Change coordinates for consistency
        ds['atm_xr'] = ds.atm_lonr[0,:]
        ds['atm_yr'] = ds.atm_latr[:,0]
        ds = ds.reset_coords(['atm_latr', 'atm_lonr'])

        self.logger.info(f"Extracted atmosphere geometry.")
        return ds
    
    def _load_oce(self, file_number):
        """
        Load ocean fields.

        Args:
            file_number (int): the file number to open in the list of ocean files
        
        Returns:
            ds (xarray.Dataset): the dataset containing ocean fields
        """
        oce_file = self.oce_ncfiles[file_number]
        ds = xr.open_dataset(oce_file)
        variables_to_keep = {
            'time_counter': 'oce_time',
            'temp_surf':    'oce_sst',       # sea surface temperature
            'u_surf':       'oce_ssu',       # sea surface u current
            'v_surf':       'oce_ssv',       # sea surface v current
            'sustr':        'oce_sustr',     # u component of the wind stress
            'svstr':        'oce_svstr',     # v component of the wind stress 
            'shflx_sen':    'oce_shflx_sen', # sensible heat flux
            'shflx_lat':    'oce_shflx_lat', # latent heat flux
            'shflx_rlw':    'oce_shflx_rlw', # long-wavbe heat flux
        }

        for v in ds.variables:
            if v not in variables_to_keep.keys():
                ds = ds.drop_vars(v)
        
        # Remove duplicate coordinates (via rename to temporary name <>_)
        ds = ds.rename(
            {
                'x_rho':'_x_rho',
                'x_u': '_x_u',
                'x_v': '_x_rho',
                'y_rho': '_y_rho',
                'y_u': '_y_rho',
                'y_v': '_y_v',
            }
        )
        ds = ds.rename(
            {
                '_x_rho': 'oce_xr',
                '_x_u': 'oce_xu',
                '_y_rho': 'oce_yr',
                '_y_v': 'oce_yv',
            }
        )

        # Rename variables
        ds = ds.rename(variables_to_keep)

        ds = xr.merge([ds, self.oce_geometry])
        ds['oce_xr'] = ds.oce_lonr[0,:]
        ds['oce_xu'] = ds.oce_lonu[0,:]
        ds['oce_yr'] = ds.oce_latr[:,0]
        ds['oce_yv'] = ds.oce_latv[:,0]

        # Interpolate u and v grid to centered r grid
        ds = ds.interp(
            {
                'oce_xu': ds.oce_xr,
                'oce_yv': ds.oce_yr
            }
        )
        # No longer need u and v coordinates
        ds = ds.drop_vars(
            [
                'oce_lonu',
                'oce_latu',
                'oce_lonv',
                'oce_latv',
                'oce_xu',
                'oce_yv',
            ]
        )
        return ds
    
    def _load_atm(self, file_number_low, oce_time):
        """
        Interpolate atmospheric fields at times oce_time

        Args:
            file_number_low (int): the first atmosphere file to open
            oce_time (xarray.DataArray): the time at which atmospheric fields 
                will be interpolated

        Returns:
            ds (xarray.Dataset): the dataset containing the atmosphere fields
                layout
        """
        variables_to_keep = {
            'SST':   'atm_sst', # ERA5 sea surface temperature
            'UT':    'atm_u',   # u wind speed
            'VT':    'atm_v',   # u wind speed
            'THT':   'atm_t',   # air potential temperature
            'PABST': 'atm_p',   # ait surface pressure
            'RVT':   'atm_h',   # air huminidy
            }
        def _preprocess(ds):
            # Remove duplicate coordinates (via rename to temporary name <>_)
            ds = ds.rename_dims(
                {
                    'ni': '_ni',
                    'ni_u': '_ni',
                    'ni_v': '_ni',
                    'nj': '_nj',
                    'nj_u': '_nj',
                    'nj_v': '_nj',
                    'level': '_level',
                    'level_w': '_level'
                }
            )

            for v in ds.variables:
                if v not in variables_to_keep.keys():
                    if v != 'time':
                        ds = ds.drop_vars(v)

            ds = ds.rename(variables_to_keep)
            ds = ds.rename({'time':  'atm_time'})
            ds = ds.rename_dims(
                {
                    '_ni': 'atm_xr',
                    '_nj': 'atm_yr',
                    '_level': 'level'
                }
            )

            ds = xr.merge([ds, self.atm_geometry])
            ds = ds.isel(level=0)
            return ds
        # Open atmosphere data at time t and t+1. This is required in order to
        # interpolate temporally at ocean times
        ds0 = xr.open_dataset(self.atm_ncfiles[file_number_low])
        ds0 = _preprocess(ds0)
        ds1 = xr.open_dataset(self.atm_ncfiles[file_number_low+1])
        ds1 = _preprocess(ds1)

        ds = xr.combine_by_coords([ds0, ds1])

        ds0.close()
        ds1.close()

        # ERA5 data over continent has to be ignored for oceanic flux
        # calculations. We place nan values in non-relevant regions. These
        # values will be ignored during the spatal interpolation (before flux
        # calculation)
        cond = xr.where(ds.atm_maskr, 1, np.nan)
        for _, v in variables_to_keep.items():
            ds[v].values = ds[v]*cond
        
        # Interpolate atmospheric variable at oceanic time
        ds = ds.interp({'atm_time': oce_time}, method='linear')
        # Drop the atmospheric time coordinate
        ds = ds.drop_vars(['atm_time'])
        return ds
    
    def _load(self, oce_file_number):
        """
        Load ocean and atmosphere fields.

        Args:
            oce_file_number (int): the file number to open in the list of ocean files
        Returns:
            oce (xarray.Dataset): the dataset containing the ocean fields
            atm (xarray.Dataset): the dataset containing the atmosphere fields
        """
        oce = self._load_oce(oce_file_number)
        atm = self._load_atm(oce_file_number//6, oce.oce_time)

        self.logger.info(f"Loaded simulation snapshot: {oce.oce_time.data[0]} - {oce.oce_time.data[-1]}")
    
        oce = self.compute_fluxes(oce, atm)

        return oce, atm
        
    
    def compute_fluxes(self, oce, atm):
        """
        Compute the ocean atmosphere fluxes

        Args:
            oce (xr.Dataset): the ocean dataset to save
            atm (xr.Dataset): the atmosphere dataset to save
            dave_dor (str): the location to save netcdf files

        Returns:
            oce (xarray.Dataset): the updated ocean dataset containing fluxes
        """

        # Dictionaries for the interpolations
        atm_to_oce = {"atm_xr": oce.oce_xr, "atm_yr": oce.oce_yr}
        oce_u_to_r = {"oce_xu": oce.oce_xr}
        oce_v_to_r = {"oce_yv": oce.oce_yr}
        
        oce_sst = oce.oce_sst
        oce_ssu = oce.oce_ssu
        oce_ssv = oce.oce_ssv

        # Interpolate (spatially) atmospheric fields to the ocean grid
        atm_u = atm.atm_u.interp(atm_to_oce, method='linear')
        atm_t = atm.atm_t.interp(atm_to_oce, method='linear')
        atm_v = atm.atm_v.interp(atm_to_oce, method='linear')
        atm_p = atm.atm_p.interp(atm_to_oce, method='linear')
        atm_h = atm.atm_h.interp(atm_to_oce, method='linear')

        # Compute fluxes with the coare parametrization
        sustr, svstr = xr.apply_ufunc(
            coare_croco,
            atm_u,
            atm_v,
            oce_ssu,
            oce_ssv,
            atm_t,
            (atm_h*100.),
            10,
            oce_sst,
            3,
            output_core_dims=[[],[]])

        # Store the fluxes in the ocean dataset
        oce["atm_sustr"] = (('oce_time', 'oce_yr', 'oce_xr'), sustr)
        oce["atm_svstr"] = (('oce_time', 'oce_yr', 'oce_xr'), svstr)

        return oce

    def _save(self, oce, atm, save_dir):
        """
        Save the ocean and atmosphere datasets to the netcdf format

        Args:
            oce (xr.Dataset): the ocean dataset to save
            atm (xr.Dataset): the atmosphere dataset to save
            save_dir (str): the location to save netcdf files
        
        """
        time, dset = zip(*oce.groupby('oce_time'))
        path = [save_dir + f"/oce_{t}.nc" for t in time]
        xr.save_mfdataset(dset, path)

        time, dset = zip(*atm.groupby('oce_time'))
        path = [save_dir + f"/atm_{t}.nc" for t in time]
        xr.save_mfdataset(dset, path)

    def run(self, save_dir):
        """
        Run the post processing pipeline

        Args:
            save_dir (str): the location to save netcdf files
        
        """
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        for i in range(len(self.oce_ncfiles)):
            oce, atm = self._load(i)
            self._save(oce, atm, save_dir)
            oce.close()
            atm.close()



    #def set_ncg_lookup_tables(self):
    #    oce_rshape = self.oce_geometry.oce_maskr.shape
    #    self.oce_to_atm_ngcIndex = [[],[]]
    #    curr_i = 0
    #    ngc_i = np.zeros(oce_rshape[0], dtype=int)
    #    for i in range(ngc_i.shape[0]):
    #        if self.atm_geometry.atm_latr[curr_i+1, 0] < self.oce_geometry.oce_latr[i, 0]:
    #            curr_i += 1
    #        ngc_i[i] = curr_i
    #    self.oce_to_atm_ngcIndex[0] = ngc_i
    #    
    #    curr_j = 0
    #    ngc_j = np.zeros(oce_rshape[1], dtype=int)
    #    for j in range(ngc_j.shape[0]):
    #        if self.atm_geometry.atm_lonr[0, curr_j+1] < self.oce_geometry.oce_lonr[0,j]:
    #            curr_j += 1
    #        ngc_j[j] = curr_j
    #    self.oce_to_atm_ngcIndex[1] = ngc_j
    #           
    #def get_era5_ngc(self, i, j):
    #    ngc_i = self.oce_to_atm_ngcIndex[0][i]
    #    ngc_j = self.oce_to_atm_ngcIndex[1][j]
    #    return ngc_i, ngc_j
