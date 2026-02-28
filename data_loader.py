import os, sys, glob
import time
from tqdm import tqdm
import numpy as np
import h5py
import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler
import MinkowskiEngine as ME
from data_utils import read_h5_geo, read_ply_ascii_geo, read_h5_geo_color, read_h5_scannet, read_ply_ascii_mpeg, read_ply_ascii_scannet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)



def collate_pointcloud_fn(list_data, device):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    coords, feats = list(zip(*list_data))

    coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)

    # data
    input_pc = ME.SparseTensor(features=feats_batch.float(), coordinates=coords_batch, device=device,
                               minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT)

    return input_pc


class PCDataset(torch.utils.data.Dataset):

    def __init__(self, files):
        self.files = []
        self.cache = {}
        self.last_cache_percent = 0
        self.files = files

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):
        filedir = self.files[idx]

        if idx in self.cache:
            coords, feats = self.cache[idx]
        else:
            # if filedir.endswith('.h5'): coords = read_h5_geo(filedir)
            # if filedir.endswith('.ply'): coords = read_ply_ascii_geo(filedir)
            # feats = np.expand_dims(np.ones(coords.shape[0]), 1).astype('int')
            # if filedir.endswith('.h5'): coords, feats = read_h5_geo_color(filedir)
            if filedir.endswith('.h5'): coords, feats = read_h5_scannet(filedir)
            if filedir.endswith('.ply'): coords, feats = read_ply_ascii_mpeg(filedir)
            # if filedir.endswith('.ply'): coords, feats = read_ply_ascii_scannet(filedir)

            # cache
            self.cache[idx] = (coords, feats)
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.last_cache_percent = cache_percent

        # test
        # coords = coords[:2,...]
        # feats = feats[:2,...]

        centroid = np.round(np.mean(coords, axis=0))
        coords = coords - centroid
        #
        feats = rgb2yuv(feats) / 255.
        # feats = feats / 255.


        return (coords, feats)


def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False,
                     collate_fn=collate_pointcloud_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader


def rgb2yuv(rgb):
    rgb_ = rgb.transpose()
    A = np.array([[0.212600, 0.715200, 0.072200],
                  [-0.114572, -0.385428, 0.5],
                  [0.5, -0.454153, -0.045847]])
    b = np.array([[0.],
                  [128.],
                  [128.]])
    yuv_ = np.matmul(A, rgb_) + b
    yuv = yuv_.transpose()
    yuv = np.clip(yuv, 0, 255)
    return yuv


def yuv2rgb(yuv):
    yuv_ = yuv.transpose()
    A = np.array([[1., 0., 1.57480],
                  [1., -0.18733, -0.46813],
                  [1., 1.88563, 0.]])
    b = np.array([[0.],
                  [-128.],
                  [-128.]])
    yuv1 = yuv_ + b
    rgb_ = np.matmul(A, yuv1)
    rgb = rgb_.transpose()
    rgb = np.round(np.clip(rgb, 0, 255))
    return rgb


if __name__ == "__main__":
    # filedirs = sorted(glob.glob('/home/ubuntu/HardDisk2/color_training_datasets/training_dataset/'+'*.h5'))
    filedirs = sorted(glob.glob(
        '/home/ubuntu/HardDisk1/point_cloud_testing_datasets/8i_voxeilzaed_full_bodies/8i/longdress/Ply/' + '*.ply'))
    test_dataset = PCDataset(filedirs[:10])
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=2, shuffle=True, num_workers=1, repeat=False,
                                       collate_fn=collate_pointcloud_fn)
    for idx, (coords, feats) in enumerate(tqdm(test_dataloader)):
        print("=" * 20, "check dataset", "=" * 20,
              "\ncoords:\n", coords, "\nfeat:\n", feats)

    test_iter = iter(test_dataloader)
    print(test_iter)
    for i in tqdm(range(10)):
        coords, feats = test_iter.next()
        print("=" * 20, "check dataset", "=" * 20,
              "\ncoords:\n", coords, "\nfeat:\n", feats)
