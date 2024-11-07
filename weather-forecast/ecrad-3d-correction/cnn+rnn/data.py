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
import glob
import os.path as osp
from subprocess import call
from tqdm import tqdm

import lightning as L
import torch
from torch.utils.data import Dataset

from climetlab_maelstrom_radiation.radiation_tf import features_size
from utils import Keys, VarInfo
from typing import List, Optional

from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy


feature_description = {}
for k in features_size:
    feature_description[k] = tfrec.FixedLenFeature(
        features_size[k], tfrec.float32, -999.0
    )


def split_dataset(data):
    results = {}
    # Sca variables
    sca_var = VarInfo().sca_variables
    for k in sca_var:
        idx = sca_var[k]["idx"]
        shape = sca_var[k]["shape"]
        if isinstance(idx, range):
            idx = slice(idx.start, idx.stop, idx.step)
        results.update(
            {
                k: fn.reshape(
                    data["sca_inputs"][idx],
                    shape=[*shape],
                )
            }
        )
    # Col variables
    col_var = VarInfo().col_variables
    for k in col_var:
        idx = col_var[k]["idx"]
        shape = col_var[k]["shape"]
        if isinstance(idx, range):
            idx = slice(idx.start, idx.stop, idx.step)
        results.update(
            {
                k: fn.reshape(
                    data["col_inputs"][:, idx],
                    shape=[*shape],
                )
            }
        )
    results["aerosol_mmr"] = fn.transpose(results["aerosol_mmr"], perm=[1, 0])
    # Hl variables
    hl_var = VarInfo().hl_variables
    for k in hl_var:
        idx = hl_var[k]["idx"]
        shape = hl_var[k]["shape"]
        if isinstance(idx, range):
            idx = slice(idx.start, idx.stop, idx.step)
        results.update(
            {
                k: fn.reshape(
                    data["hl_inputs"][:, idx],
                    shape=[*shape],
                )
            }
        )
    # Inter variables
    inter_var = VarInfo().inter_variables
    for k in inter_var:
        idx = inter_var[k]["idx"]
        shape = inter_var[k]["shape"]
        if isinstance(idx, range):
            idx = slice(idx.start, idx.stop, idx.step)
        results.update(
            {
                k: fn.reshape(
                    data["inter_inputs"][:, idx],
                    shape=[*shape],
                )
            }
        )
    # Output variables
    for k in Keys().output_keys:
        results.update({k: data[k]})

    return results


def create_tfrecord_indexes(tfrecord_path):
    tfrecord_files = glob.glob(osp.join(tfrecord_path, "*.tfrecord"))
    tfrecord_idxs = [filename + ".idx" for filename in tfrecord_files]

    for tfrecord, tfrecord_idx in tqdm(
        zip(tfrecord_files, tfrecord_idxs), total=len(tfrecord_files)
    ):
        if not osp.isfile(tfrecord_idx):
            call(["tfrecord2idx", tfrecord, tfrecord_idx])


@pipeline_def
def tfrecord_pipeline(
    tfrecord_path, index_path, shard_id=0, num_shards=1, device="cpu"
):
    data = fn.readers.tfrecord(
        path=tfrecord_path,
        index_path=index_path,
        features=feature_description,
        shard_id=shard_id,
        random_shuffle=False,
        num_shards=num_shards,
        device=device,
        name="TFRecord_Reader",
    )
    results = split_dataset(data)

    return (*(results[key] for key in results.keys()),)
    # return (*(data[key] for key in feature_description.keys()),)


class RadiationTestDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing radiation test dataset in NETCDF format.
    Args:
        cache_dir (str): The directory where cached data is stored.
        test_date (str): The date of the test data to load.
        test_timestep (int): The timestep of the test data to load.
    Methods:
        __getitem__(idx):
            Returns the input and output data for a given index.
        __len__():
            Returns the number of columns in the test dataset.
    """

    def __init__(self, cache_dir, test_date, test_timestep):
        super().__init__()
        self.cache_dir = cache_dir
        self.test_date = test_date
        self.test_timestep = test_timestep

        cml.settings.set("cache-directory", osp.join(self.cache_dir, "test"))
        test_ds = cml.load_dataset(
            "maelstrom-radiation",
            date=self.test_date,
            timestep=self.test_timestep,
            raw_inputs=True,
            patch=list(range(16)),
            gather_fluxes=True,
            dataset="3dcorrection",
            minimal_outputs=False,
            hr_units="K d-1",
        )
        self.test_dataset = test_ds.to_xarray()

    def __getitem__(self, idx):
        inputs = self.test_dataset[list(Keys.input_keys)].isel(column=idx)
        outputs = self.test_dataset[list(Keys.output_keys)].isel(column=idx)

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
        cache_dir: str = None,
        train_subset: str = None,
        train_timestep: List[int] = [0],
        train_filenum: List[int] = [0],
        val_subset: str = None,
        val_timestep: List[int] = [0],
        val_filenum: List[int] = [0],
        test_date: int = None,
        test_timestep: List[int] = None,
        batch_size: int = 1,
        num_threads: int = 1,
    ):
        super().__init__()

        self.cache_dir = cache_dir
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
        common_kwargs = {
            "dataset": "3dcorrection",
            "minimal_outputs": False,
            "hr_units": "K d-1",
        }
        cml.settings.set("cache-directory", osp.join(self.cache_dir, "train"))
        train_ds = cml.load_dataset(
            "maelstrom-radiation-tf",
            subset=self.train_subset,
            timestep=self.train_timestep,
            filenum=self.train_filenum,
            norm=True,
            **common_kwargs,
        )

        cml.settings.set("cache-directory", osp.join(self.cache_dir, "val"))
        val_ds = cml.load_dataset(
            "maelstrom-radiation-tf",
            subset=self.val_subset,
            timestep=self.val_timestep,
            filenum=self.val_filenum,
            norm=True,
            **common_kwargs,
        )

        for stage in ["train", "val"]:
            create_tfrecord_indexes(osp.join(self.cache_dir, stage))

        # The dataset used for testing is stored as reguler Netcdf files.
        # The class used to download the data is thus different.
        cml.settings.set("cache-directory", osp.join(self.cache_dir, "test"))
        test_ds = cml.load_dataset(
            "maelstrom-radiation",
            date=self.test_date,
            timestep=self.test_timestep,
            raw_inputs=True,
            patch=list(range(16)),
            gather_fluxes=True,
            **common_kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        if self.trainer is not None:
            device_id = self.trainer.local_rank
            shard_id = self.trainer.global_rank
            num_shards = self.trainer.world_size
        else:
            device_id = None
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
                device="cpu",
                device_id=device_id,
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
                device="cpu",
                device_id=device_id,
                shard_id=shard_id,
                num_shards=num_shards,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
            )
        else:
            self.test_dataset = RadiationTestDataset(
                self.cache_dir, self.test_date, self.test_timestep
            )

    def train_dataloader(self):
        return DALIGenericIterator(
            [self.train_pipeline],
            output_map=list(
                tuple(VarInfo().sca_variables.keys())
                + tuple(VarInfo().col_variables.keys())
                + tuple(VarInfo().hl_variables.keys())
                + tuple(VarInfo().inter_variables.keys())
                + Keys().output_keys
            ),
            reader_name="TFRecord_Reader",
            last_batch_policy=LastBatchPolicy.DROP,
        )

    def val_dataloader(self):
        return DALIGenericIterator(
            [self.train_pipeline],
            output_map=list(
                tuple(VarInfo().sca_variables.keys())
                + tuple(VarInfo().col_variables.keys())
                + tuple(VarInfo().hl_variables.keys())
                + tuple(VarInfo().inter_variables.keys())
                + Keys().output_keys
            ),
            reader_name="TFRecord_Reader",
            last_batch_policy=LastBatchPolicy.DROP,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_threads,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )


if __name__ == "__main__":
    datamodule = RadiationDataModule(
        cache_dir="/fs1/ECMWF/3dcorrection/data",
        train_subset=None,
        train_timestep=[2019013100],
        train_filenum=list(range(0, 52, 5)),
        val_subset=None,
        val_timestep=[2019013100],
        val_filenum=[0],
        test_date=[20190531, 20191028],
        test_timestep=list(range(0, 3501, 1000)),
        batch_size=8,
        num_threads=1,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    for batch in datamodule.train_dataloader():
        for k in batch[0].keys():
            print(k, batch[0][k].shape)
        break
