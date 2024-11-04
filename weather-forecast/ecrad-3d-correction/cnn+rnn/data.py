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


def split_data(data):
    pass


@pipeline_def
def tfrecord_pipeline(tfrecord_path, index_path, device, shard_id=0, num_shards=1):
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


def netcdf_pipeline():
    pass


class RadiationDataModule(L.LightningDataModule):
    """
    This class handles the data loading and processing for the 3D correction use-case
    in weather forecasting.

    Args:
        date (str): The date for which the data is being processed.
        timestep (int): The timestep interval for the data.
        patchstep (int): The patch step interval for the data.
        batch_size (int): The size of the batches produced by the data loaders.
        num_workers (int): The number of workers used for data loading.

    """

    def __init__(
        self,
        subset: str = None,
        timestep: list = [0],
        filenum: list = [0],
        batch_size: int = 1,
        num_threads: int = 1,
    ):
        super().__init__()

        self.subset = subset
        self.timestep = timestep
        self.filenum = filenum
        self.batch_size = batch_size
        self.num_threads = num_threads

    def prepare_data(self):
        # Here we use climetlab only to download the dataset.
        cml.settings.set("cache-directory", self.cache_dir)
        ds = cml.load_dataset(
            "maesltrom-radiation-tf",
            dataset="3dcorrection",
            subset=self.subset,
            timestep=self.timestep,
            filenum=self.filenum,
            minimal_output=False,
            hr_units="K d-1",
            norm=True,
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
                tfrecord_path=sorted(glob.glob(osp.join(self.cache_dir, "*.tfrecord"))),
                index_path=sorted(glob.glob(osp.join(self.cache_dir, "*.idx"))),
                device=device_id,
                shard_id=shard_id,
                num_shards=num_shards,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
            )
            self.val_pipeline = netcdf_pipeline()

    def train_dataloader(self):
        return DALIGenericIterator(
            [self.pipeline],
            output_map=list(feature_description.keys()),
            reader_name="TFRecord_Reader",
            last_batch_policy=LastBatchPolicy.DROP,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
