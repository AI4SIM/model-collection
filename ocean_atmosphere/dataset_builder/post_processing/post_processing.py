"""
Define a PostProcessing class which can be used to run the post-processing of
a coupled ocean-atmosphere simulation.
"""
import os
import glob
import logging
import shutil
import xarray as xr
import xesmf as xe
import numpy as np

from coare_croco import coare_croco


class PostProcessing:
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
        3- Setup regridder xesmf interpolation instances
        3- Interpolate the atmospheric ERA5 data to the ocean time
        4- Compute the fluxes between ocean and ERA5 atmosphere
    """

    def __init__(self, run_dir):
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

        self.geometry = {}
        # Read the atmosphere geometry layout
        self.geometry["atm"] = self._read_atm_grid()

        # Read the ocean geometry layout
        self.geometry["oce"] = self._read_oce_geometry()

        # Compute interpolation maps
        self._prepare_regridder()

    def _init_files(self):
        """List ocean and atmosphere files in run_dir

        Args:
            run_dir (str): Path to the simulation run directory
        """

        self.oce_gridfile = os.path.join(self.run_dir, "croco_grd.nc")
        if not os.path.isfile(self.oce_gridfile):
            raise FileNotFoundError(self.oce_gridfile)
        self.logger.debug("Found ocean grid: %s", self.oce_gridfile)

        oce_ncfiles = os.path.join(self.run_dir, "WMED_TEST_MODEL_*.nc")
        self.oce_ncfiles = sorted(glob.glob(oce_ncfiles))[:-1]

        if len(self.oce_ncfiles) == 0:
            raise FileNotFoundError(f"Could not find ocean file in {oce_ncfiles}")
        self.logger.debug("Found %d ocean files", len(self.oce_ncfiles))

        atm_ncfiles = os.path.join(self.run_dir, "ECMWF_*.nc")
        self.atm_ncfiles = sorted(glob.glob(atm_ncfiles))

        if len(self.atm_ncfiles) == 0:
            raise FileNotFoundError(f"Could not find atmosphere file in {oce_ncfiles}")
        self.logger.debug("Found %d atmosphere files", len(self.atm_ncfiles))

    def _read_oce_geometry(self):
        """
        Read the geometry layout of the ocean grid.

        Returns:
            dataset (xarray.Dataset): the dataset containing the ocean grid
                layout
        """

        dataset = xr.open_dataset(self.oce_gridfile, cache=False)

        variables_to_keep = [
            "lat_rho",
            "lon_rho",
            "lat_u",
            "lon_u",
            "lat_v",
            "lon_v",
            "angle",
            "mask_u",
            "mask_v",
            "mask_rho",
        ]
        for var in dataset.variables:
            if var not in variables_to_keep:
                dataset = dataset.drop_vars(var)
        # Remove duplicate coordinates (via rename to temporary name <>_)
        dataset = dataset.rename(
            {
                "xi_rho": "oce_xr",
                "xi_v": "oce_xr",
                "xi_u": "oce_xu",
                "eta_rho": "oce_yr",
                "eta_v": "oce_yv",
                "eta_u": "oce_yr",
            }
        )
        dataset = dataset.rename(
            {
                "lat_rho": "oce_latr",
                "lat_u": "oce_latu",
                "lat_v": "oce_latv",
                "lon_rho": "oce_lonr",
                "lon_u": "oce_lonu",
                "lon_v": "oce_lonv",
            }
        )

        dataset["oce_xr"] = dataset.oce_lonr[0, :]
        dataset["oce_xu"] = dataset.oce_lonu[0, :]
        dataset["oce_yr"] = dataset.oce_latr[:, 0]
        dataset["oce_yv"] = dataset.oce_latv[:, 0]

        self.logger.info("Extracted ocean geometry.")
        return dataset

    def _read_atm_grid(self):
        """
        Read the geometry layout of the atmosphere grid.

        Returns:
            dataset (xarray.Dataset): the dataset containing the atmosphere grid
                layout
        """

        dataset = xr.open_dataset(self.atm_ncfiles[0], cache=False)
        # Remove coordinates for consistency
        dataset = dataset.rename(
            {
                "ni": "atm_xr",
                "nj": "atm_yr",
                "latitude": "atm_latr",
                "longitude": "atm_lonr",
                "time": "atm_time",
            }
        )

        variables_to_keep = ["SSS", "atm_latr", "atm_lonr"]
        for var in dataset.variables:
            if var not in variables_to_keep:
                dataset = dataset.drop_vars(var)

        # Infer the ocean mask from ERA5 sea surface salinity (SSS)
        dataset["atm_maskr"] = dataset["SSS"] < 999
        dataset = dataset.drop_vars("SSS")

        # Change coordinates for consistency
        dataset["atm_xr"] = dataset.atm_lonr[0, :]
        dataset["atm_yr"] = dataset.atm_latr[:, 0]
        dataset = dataset.reset_coords(["atm_latr", "atm_lonr"])

        self.logger.info("Extracted atmosphere geometry.")
        return dataset

    def _prepare_regridder(self):
        """
        Initiate xesmf.Regridder instances for future interpolations
        """

        grid_u_oce = {
            "lon": self.geometry["oce"].oce_xu,
            "lat": self.geometry["oce"].oce_yr,
            "mask": self.geometry["oce"].mask_u,
        }

        grid_v_oce = {
            "lon": self.geometry["oce"].oce_xr,
            "lat": self.geometry["oce"].oce_yv,
            "mask": self.geometry["oce"].mask_v,
        }

        grid_atm = {
            "lon": self.geometry["atm"].atm_xr,
            "lat": self.geometry["atm"].atm_yr,
            "mask": self.geometry["atm"].atm_maskr,
        }

        grid_r_oce = {
            "lon": self.geometry["oce"].oce_xr,
            "lat": self.geometry["oce"].oce_yr,
            "mask": self.geometry["oce"].mask_rho,
        }

        self.regridder = {
            "oce_u_to_r": xe.Regridder(
                ds_in=grid_u_oce,
                ds_out=grid_r_oce,
                method="bilinear",
                extrap_method="inverse_dist"),
            "oce_v_to_r": xe.Regridder(
                ds_in=grid_v_oce,
                ds_out=grid_r_oce,
                method="bilinear",
                extrap_method="inverse_dist"),
            "atm_to_oce": xe.Regridder(
                ds_in=grid_atm,
                ds_out=grid_r_oce,
                method="bilinear",
                extrap_method="inverse_dist"
            )
        }
        self.logger.info("Computed interpolation weights.")

    def _load_oce(self, file_number):
        """
        Load ocean fields.

        Args:
            file_number (int): the file number to open in the list of ocean files

        Returns:
            dataset (xarray.Dataset): the dataset containing ocean fields
        """
        oce_file = self.oce_ncfiles[file_number]
        dataset = xr.open_dataset(oce_file)
        variables_to_keep = {
            "time_counter": "oce_time",
            "temp_surf": "oce_sst",  # sea surface temperature
            "u_surf": "oce_ssu",  # sea surface u current
            "v_surf": "oce_ssv",  # sea surface v current
            "sustr": "oce_sustr",  # u component of the wind stress
            "svstr": "oce_svstr",  # v component of the wind stress
            "shflx_sen": "oce_shflx_sen",  # sensible heat flux
            "shflx_lat": "oce_shflx_lat",  # latent heat flux
            "shflx_rlw": "oce_shflx_rlw",  # long-wavbe heat flux
        }

        for var in dataset.variables:
            if variables_to_keep.get(var) is None:
                dataset = dataset.drop_vars(var)

        # Remove duplicate coordinates (via rename to temporary name <>_)
        dataset = dataset.rename(
            {
                "x_rho": "_x_rho",
                "x_u": "_x_u",
                "x_v": "_x_rho",
                "y_rho": "_y_rho",
                "y_u": "_y_rho",
                "y_v": "_y_v",
            }
        )
        dataset = dataset.rename(
            {
                "_x_rho": "oce_xr",
                "_x_u": "oce_xu",
                "_y_rho": "oce_yr",
                "_y_v": "oce_yv",
            }
        )

        # Rename variables
        dataset = dataset.rename(variables_to_keep)

        dataset = xr.merge([self.geometry["oce"], dataset])

        dataset["oce_ssu"] = self.regridder["oce_u_to_r"](dataset.oce_ssu)
        dataset["oce_sustr"] = self.regridder["oce_u_to_r"](dataset.oce_sustr)
        dataset["oce_ssv"] = self.regridder["oce_v_to_r"](dataset.oce_ssv)
        dataset["oce_svstr"] = self.regridder["oce_v_to_r"](dataset.oce_svstr)

        dataset = dataset.drop_vars(
            [
                "oce_lonu",
                "oce_latu",
                "oce_lonv",
                "oce_latv",
                "oce_xu",
                "oce_yv",
            ]
        )
        return dataset

    def _load_atm(self, file_number_low, oce_time):
        """
        Interpolate atmospheric fields at times oce_time

        Args:
            file_number_low (int): the first atmosphere file to open
            oce_time (xarray.DataArray): the time at which atmospheric fields
                will be interpolated

        Returns:
            dataset (xarray.Dataset): the dataset containing the atmosphere fields
                layout
        """
        variables_to_keep = {
            "time": "atm_time",
            "SST": "atm_sst",  # ERA5 sea surface temperature
            "UT": "atm_u",  # u wind speed
            "VT": "atm_v",  # u wind speed
            "THT": "atm_t",  # air potential temperature
            "PABST": "atm_p",  # ait surface pressure
            "RVT": "atm_h",  # air huminidy
        }

        def _preprocess(dataset):
            # Remove duplicate coordinates (via rename to temporary name <>_)
            dataset = dataset.rename_dims(
                {
                    "ni": "_ni",
                    "ni_u": "_ni",
                    "ni_v": "_ni",
                    "nj": "_nj",
                    "nj_u": "_nj",
                    "nj_v": "_nj",
                    "level": "_level",
                    "level_w": "_level",
                }
            )

            for var in dataset.variables:
                if variables_to_keep.get(var) is None:
                    dataset = dataset.drop_vars(var)

            dataset = dataset.rename(variables_to_keep)
            dataset = dataset.rename_dims({"_ni": "atm_xr", "_nj": "atm_yr", "_level": "level"})

            dataset = xr.merge([dataset, self.geometry["atm"]])
            dataset = dataset.isel(level=0)
            return dataset

        # Open atmosphere data at time t and t+1. This is required in order to
        # interpolate temporally at ocean times
        dataset_t0 = xr.open_dataset(self.atm_ncfiles[file_number_low])
        dataset_t0 = _preprocess(dataset_t0)
        dataset_t1 = xr.open_dataset(self.atm_ncfiles[file_number_low + 1])
        dataset_t1 = _preprocess(dataset_t1)

        dataset = xr.combine_by_coords([dataset_t0, dataset_t1], combine_attrs="drop_conflicts")

        dataset_t0.close()
        dataset_t1.close()

        # ERA5 data over continent has to be ignored for oceanic flux
        # calculations. We place nan values in non-relevant regions. These
        # values will be ignored during the spatal interpolation (before flux
        # calculation)
        add_nan = xr.where(dataset.atm_maskr, 1, np.nan)
        for _, var in variables_to_keep.items():
            if var != "atm_time":
                dataset[var].values = dataset[var] * add_nan

        # Interpolate atmospheric variable at oceanic time
        dataset = dataset.interp(
            {"atm_time": oce_time},
            method="linear"
        )
        # Drop the atmospheric time coordinate
        dataset = dataset.drop_vars(["atm_time"])
        return dataset

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
        atm = self._load_atm(oce_file_number // 6, oce.oce_time)

        self.logger.info(
            "Loaded simulation snapshot: %s - %s",
            oce.oce_time.data[0],
            oce.oce_time.data[-1]
        )

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

        atm_u = self.regridder["atm_to_oce"](atm.atm_u)
        atm_t = self.regridder["atm_to_oce"](atm.atm_t)
        atm_v = self.regridder["atm_to_oce"](atm.atm_v)
        atm_p = self.regridder["atm_to_oce"](atm.atm_p)
        atm_h = self.regridder["atm_to_oce"](atm.atm_h)

        # Compute fluxes with the coare parametrization
        with np.errstate(invalid="ignore"):
            sustr, svstr, shflx_lat, shflx_sen = xr.apply_ufunc(
                coare_croco,
                atm_u.data - oce.oce_ssu.data,
                atm_v.data - oce.oce_ssv.data,
                atm_t.data,
                atm_h.data,
                atm_p.data,
                oce.oce_sst.data,
                3,
                output_core_dims=[[], [], [], []],
            )

        # Store the fluxes in the ocean dataset
        oce["atm_sustr"] = (("oce_time", "oce_yr", "oce_xr"), sustr)
        oce["atm_svstr"] = (("oce_time", "oce_yr", "oce_xr"), svstr)
        oce["atm_shflx_lat"] = (("oce_time", "oce_yr", "oce_xr"), shflx_lat)
        oce["atm_shflx_sen"] = (("oce_time", "oce_yr", "oce_xr"), shflx_sen)

        return oce

    def _save(self, oce, atm, save_dir):
        """
        Save the ocean and atmosphere datasets to the netcdf format

        Args:
            oce (xr.Dataset): the ocean dataset to save
            atm (xr.Dataset): the atmosphere dataset to save
            save_dir (str): the location to save netcdf files

        """
        time, dset = zip(*oce.groupby("oce_time"))
        path = [save_dir + f"/oce_{t}.nc" for t in time]
        xr.save_mfdataset(dset, path)

        time, dset = zip(*atm.groupby("oce_time"))
        path = [save_dir + f"/atm_{t}.nc" for t in time]
        xr.save_mfdataset(dset, path)

        self.logger.info("Saved processed snapshots.")

        oce.close()
        atm.close()

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

    # def set_ncg_lookup_tables(self):
    #    oce_rshape = self.geometry["oce"].oce_maskr.shape
    #    self.oce_to_atm_ngcIndex = [[],[]]
    #    curr_i = 0
    #    ngc_i = np.zeros(oce_rshape[0], dtype=int)
    #    for i in range(ngc_i.shape[0]):
    #        if self.atm_geometry.atm_latr[curr_i+1, 0] < self.geometry["oce"].oce_latr[i, 0]:
    #            curr_i += 1
    #        ngc_i[i] = curr_i
    #    self.oce_to_atm_ngcIndex[0] = ngc_i
    #
    #    curr_j = 0
    #    ngc_j = np.zeros(oce_rshape[1], dtype=int)
    #    for j in range(ngc_j.shape[0]):
    #        if self.atm_geometry.atm_lonr[0, curr_j+1] < self.geometry["oce"].oce_lonr[0,j]:
    #            curr_j += 1
    #        ngc_j[j] = curr_j
    #    self.oce_to_atm_ngcIndex[1] = ngc_j
    #
    # def get_era5_ngc(self, i, j):
    #    ngc_i = self.oce_to_atm_ngcIndex[0][i]
    #    ngc_j = self.oce_to_atm_ngcIndex[1][j]
    #    return ngc_i, ngc_j
