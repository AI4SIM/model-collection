import config
import h5py
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
import torch_optimizer as optim
import yaml

class CnfCombustionDataset(torch.utils.data.Dataset):

    def __init__(self, root):
        super().__init__()
        self.root = root
        for idx, filename in enumerate(self.raw_filenames):
            raw_path = os.path.join(self.raw_dir, filename)
            processed_path = os.path.join(self.processed_dir, self.processed_filenames[idx])
            if not os.path.isfile(processed_path):
                self.process(idx, raw_path)

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_filenames(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_filenames(self):
        return [f"{filename.split('.')[0]}.pt" for filename in self.raw_filenames]

    def __len__(self):
        return len(self.raw_filenames)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.processed_dir, self.processed_filenames[idx]))

    def process(self, idx, path):
        out_path = os.path.join(self.processed_dir, self.processed_filenames[idx])
        os.makedirs(self.processed_dir, exist_ok=True)
        with h5py.File(path, "r") as file:
            c = torch.tensor(file["/filt_8"][:])
            sigma = torch.tensor(file["/filt_grad_8"][:])

        # Preprocessing (adding dummy axis for channels and cropping)
        c = c[None, 1:, 1:, 1:]
        sigma = sigma[None, 1:, 1:, 1:]
        torch.save((c, sigma), out_path)


class RandomCrop3D(object):
    """Randomly crop a sub-block.

    Args:
        output_size (tuple or int): desired output size.
    """

    def __init__(self, out_size):
        assert isinstance(out_size, (int, tuple))
        if isinstance(out_size, int):
            self.out_size = (out_size, out_size, out_size)
        else:
            assert len(out_size) == 3
            self.out_size = out_size

    def __call__(self, sample):
        h, w, d = sample.shape[1:]
        bh, bw, bd = self.out_size
        tx = np.random.randint(0, x - bx)
        ty = np.random.randint(0, y - by)
        tz = np.random.randint(0, z - bz)
        return sample[tx:tx+bh, ty:ty+bw, tz:tz+bd]


@DATAMODULE_REGISTRY
class CnfCombustionDataModule(pl.LightningDataModule):
    """
    Providing 3D blocks of the CNF combustion dataset.

    Args:
        splitting_lengths (list): lengths of training, validation and testing sets.
        shuffling: whether to shuffle the trainset.
        subblock_size (int or tuple): data augmentation by randomly cropping sub-blocks.
    """

    def __init__(self, batch_size: int, num_workers: int, splitting_lengths: list, shuffling=False, subblock_size=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splitting_lengths = splitting_lengths
        self.shuffling = shuffling
        self.cropper = RandomCrop3D(subblock_size) if subblock_size is not None else None
        super().__init__()

    def prepare_data(self):
        CnfCombustionDataset(config.data_path)

    def setup(self, stage):
        dataset = CnfCombustionDataset(config.data_path)
        self.train, self.val, self.test = torch.utils.data.random_split(dataset, self.splitting_lengths)

        # Preprocessing on trainset.
        if self.shuffling: self.train = self.train.shuffle()
        if self.cropper is not None: self.train = self.cropper(self.train)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
