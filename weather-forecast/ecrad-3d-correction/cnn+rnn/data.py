# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import climetlab as cml
import os.path as osp
import lightning as L
import torch

from climetlab_maelstrom_radiation.radiation_tf import features_size
from utils import Keys, VarInfo
from typing import dict, list, tuple, Optional

from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.plugin.pytorch import DALIGenericIterator


feature_description = {}
for k in features_size:
    feature_description[k] = tfrec.FixedLenFeature(
        shape=features_size[k], dtype=tfrec.float32
    )


def merge_fluxes(ds, keys):
    tmp = []
    for k in keys:
        tmp.append(ds[k].expand_dims("variable", axis=-1))
    return xr.concat(tmp, dim="variable", data_vars="all")


def split_data(data):
    pass


@pipeline_def
def netcdf_pipeline(batch_size, netcdf_path):
    return fn.external_source(
        source=RadiationExternalSource(batch_size, cache_dir=netcdf_path),
        num_outputs=2,
        batch=True,
        parallel=True,
        name="NetCDF_Reader",
    )


@pipeline_def
def tfrecord_pipeline(tfrecord_path, index_path, shard_id=0, num_shards=1):
    data = fn.readers.tfrecord(
        path=tfrecord_path,
        index_path=index_path,
        features=feature_description,
        shard_id=shard_id,
        random_shuffle=True,
        num_shards=num_shards,
        name="TFRecord_Reader",
    )
    return (*(data[key] for key in feature_description.keys()),)


class RadiationExternalSource:
    def __init__(self, batch_size=1, cache_dir=None):
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        cml.settings.set("cache-directory", osp.join(self.cache_dir, "test"))
        test_ds = cml.load_dataset(
            "maelsltrom-radiation",
            date=self.test_date,
            timestep=self.test_timestep,
            raw_inputs=True,
            patch_list=list(range(16)),
            gather_fluxes=True,
            **commom_kwargs,
        )
        self.test_dataset = test_ds.to_xarray()
        self.full_iterations = self.test_dataset.sizes["column"] // self.batch_size

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration
        inputs = self.test_dataset[list(Keys.input_keys)].isel(
            column=slice(sample_idx, sample_idx + self.batch_size)
        )
        outputs = self.test_dataset[list(Keys.output_keys)].isel(
            column=slice(sample_idx, sample_idx + self.batch_size)
        )

        # Modify the outputs and return them as in the tfrecord files
        outputs["sw"] = merge_fluxes(outputs, ["flux_dn_sw", "flux_up_sw"])
        outputs["lw"] = merge_fluxes(outputs, ["flux_dn_lw", "flux_up_lw"])
        output = outputs.drop_vars(
            ["flux_dn_sw", "flux_up_sw", "flux_dn_lw", "flux_up_lw"]
        )

        input_dict = {k: torch.tensor(inputs[k].values) for k in Keys.input_keys}
        output_dict = {k: torch.tensor(outputs[k].values) for k in Keys.output_keys}

        return input_dict, output_dict

        def __len__(self):
            return self.test_dataset.sizes["column"]


class RadiationDataModule(L.LightningDataModule):
    """
    This class handles the data loading and processing for the 3D correction use-case
    in weather forecasting.

    Args:
        {train/val}_subset (str): The train/val subset for which the data is being processed.
        {train/val}_timestep (int): The train/val timestep interval for the data.
        {train/val}_filenum (int): The train/val filenum interval for the data.
        batch_size (int): The size of the batches produced by the data loaders.
        num_workers (int): The number of workers used for data loading.

    """

    def __init__(
        self,
        train_subset: str = None,
        train_timestep: list = [0],
        train_filenum: list = [0],
        val_subset: str = None,
        val_timestep: list = [0],
        val_filenum: list = [0],
        test_date: int = None,
        test_timestep: list = None,
        batch_size: int = 1,
        num_threads: int = 1,
    ):
        super().__init__()

        self.train_subset = train_subset
        self.train_timestep = train_timestep
        self.train_filenum = train_filenum
        self.val_subset = val_subset
        self.val_timestep = val_timestep
        self.val_filenum = val_filenum
        self.test_date = test_date
        self.test_timestep = test_timestep
        self.batch_size = batch_size
        self.num_threads = num_threads

    def prepare_data(self):
        # Here we use climetlab only to download the datasets.
        commn_kwargs = {
            "dataset": "3dcorrection",
            "minimal_output": False,
            "hr_units": "K d-1",
        }
        cml.settings.set("cache-directory", osp.join(self.cache_dir, "train"))
        train_ds = cml.load_dataset(
            "maesltrom-radiation-tf",
            subset=self.subset,
            timestep=self.timestep,
            filenum=self.filenum,
            norm=True,
            **commom_kwargs,
        )

        cml.settings.set("cache-directory", osp.join(self.cache_dir, "val"))
        val_ds = cml.load_dataset(
            "maelsltrom-radiation-tf",
            subset=self.val_subset,
            timestep=self.val_timestep,
            filenum=self.val_filenum,
            norm=True,
            **commom_kwargs,
        )

        # The dataset used for testing is stored as reguler Netcdf files.
        # The class used to download the data is thus different.
        cml.settings.set("cache-directory", osp.join(self.cache_dir, "test"))
        test_ds = cml.load_dataset(
            "maelsltrom-radiation",
            date=self.test_date,
            timestep=self.test_timestep,
            raw_inputs=True,
            patch_list=list(range(16)),
            gather_fluxes=True,
            **commom_kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        if self.trainer is not None:
            device_id = self.trainer.local_rank
            shard_id = self.trainer.global_rank
            num_shards = self.trainer.world_size
        else:
            device_id = 0
            shard_id = 0
            num_shards = 1

        if stage == "fit":
            self.train_pipeline = tfrecord_pipeline(
                tfrecord_path=sorted(
                    glob.glob(osp.join(self.cache_dir, "train", "*.tfrecord"))
                ),
                index_path=sorted(
                    glob.glob(osp.join(self.cache_dir, "train", "*.idx"))
                ),
                device=device_id,
                shard_id=shard_id,
                num_shards=num_shards,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
            )
            self.val_pipeline = tfrecord_pipeline(
                tfrecord_path=sorted(
                    glob.glob(osp.join(self.cache_dir, "val", "*.tfrecord"))
                ),
                index_path=sorted(glob.glob(osp.join(self.cache_dir, "val", "*.idx"))),
                device=device_id,
                shard_id=shard_id,
                num_shards=num_shards,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
            )
        else:
            self.test_pipeline = netcdf_pipeline(
                netcdf_path=osp.join(self.cache_dir, "test"),
                device=device_id,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
            )

    def train_dataloader(self):
        return DALIGenericIterator(
            [self.train_pipeline],
            output_map=list(feature_description.keys()),
            reader_name="TFRecord_Reader",
            last_batch_policy=LastBatchPolicy.DROP,
        )

    def val_dataloader(self):
        return DALIGenericIterator(
            [self.train_pipeline],
            output_map=list(feature_description.keys()),
            reader_name="TFRecord_Reader",
            last_batch_policy=LastBatchPolicy.DROP,
        )

    def test_dataloader(self):
        return DALIGenericIterator(
            [self.test_pipeline],
            output_map=["inputs", "outputs"],
            reader_name="TFRecord_Reader",
            last_batch_policy=LastBatchPolicy.DROP,
        )
