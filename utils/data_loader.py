import os
import sys
import json
import shutil
import random
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

TRAIN = 0.95

class Loader:

    def  __init__(self, workspace_dir, depth=False):

        self.depth = depth
        self.workspace_dir = workspace_dir
        self.dense_dir = os.path.join(self.workspace_dir, 'dense')
        self.pose_dir = os.path.join(self.workspace_dir, 'pose')
        self.rgb_dir = os.path.join(self.workspace_dir, 'rgb')

        if os.path.exists(self.pose_dir):
            shutil.rmtree(self.pose_dir)
        if os.path.exists(self.rgb_dir):
            shutil.rmtree(self.rgb_dir)

        os.mkdir(self.pose_dir)
        os.mkdir(self.rgb_dir)

    def load(self):
        data, train, test = self.load_cameras(
                os.path.join(workspace_dir, 'posed_images/cameras_normalized.json'))
        self.write_data(train, data, 'train')
        self.write_data(test, data, 'test')

    def read_array(self, path):
        with open(path, "rb") as fid:
            width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                    usecols=(0, 1, 2), dtype=int)
            fid.seek(0)
            num_delimiter = 0
            byte = fid.read(1)
            while True:
                if byte == b"&":
                    num_delimiter += 1
                    if num_delimiter >= 3:
                        break
                byte = fid.read(1)
            array = np.fromfile(fid, np.float32)
        array = array.reshape((width, height, channels), order="F")
        return np.transpose(array, (1, 0, 2)).squeeze()

    def load_cameras(self, jsonfile):
        with open(jsonfile) as f:
            data = json.load(f)

        # Generate train/test splits
        keys = list(data.keys())
        train, test = train_test_split(keys, test_size=(1-TRAIN))

        return data, train, test

    def write_data(self, keys, data, split):
        if split == 'train':
            intrinsics = np.array(data[list(keys)[0]]['K'])
            intrinsics = intrinsics.reshape(4,4)  # K is always 4x4 matrix
            np.savetxt(os.path.join(self.workspace_dir, 'intrinsics.txt'), intrinsics)

        for key in keys:
            if split == 'train':
                fname = '0_' + key.split('.')[0] + '.txt'
                os.symlink(os.path.abspath(os.path.join(self.dense_dir, f'images/{key}')), os.path.join(self.rgb_dir, '0_' + key))
            elif split == 'test':
                fname = '1_' + key.split('.')[0] + '.txt'
                os.symlink(os.path.abspath(os.path.join(self.dense_dir, f'images/{key}')), os.path.join(self.rgb_dir, '1_' + key))
            else:
                raise NotImplementedError
            if self.depth:
                depth_map = self.read_array(os.path.join(self.dense_dir, f'stereo/depth_maps/{key}.geometric.bin'))
                np.savez(os.path.join(self.pose_dir, key.split('.')[0] + '_depth'), depth_map)

            pose = np.array(data[key]['W2C'])
            np.savetxt(os.path.join(self.pose_dir, fname), pose.reshape(4,4))

if __name__ == '__main__':

    workspace_dir = sys.argv[1]
    loader = Loader(workspace_dir)
    loader.load()
