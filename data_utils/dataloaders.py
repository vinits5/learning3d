import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import h5py
import subprocess
import shlex
import json
import glob
from ops import transform_functions, se3

def download_modelnet40():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
		www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
		zipfile = os.path.basename(www)
		os.system('wget %s; unzip %s' % (www, zipfile))
		os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
		os.system('rm %s' % (zipfile))

def load_data(train):
	if train: partition = 'train'
	else: partition = 'test'
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	all_data = []
	all_label = []
	for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
		f = h5py.File(h5_name)
		data = f['data'][:].astype('float32')
		label = f['label'][:].astype('int64')
		f.close()
		all_data.append(data)
		all_label.append(label)
	all_data = np.concatenate(all_data, axis=0)
	all_label = np.concatenate(all_label, axis=0)
	return all_data, all_label

def deg_to_rad(deg):
    return np.pi / 180 * deg

def create_random_transform(dtype, max_rotation_deg, max_translation):
    max_rotation = deg_to_rad(max_rotation_deg)
    rot = np.random.uniform(-max_rotation, max_rotation, [1, 3])
    trans = np.random.uniform(-max_translation, max_translation, [1, 3])
    quat = transform_functions.euler_to_quaternion(rot, "xyz")

    vec = np.concatenate([quat, trans], axis=1)
    vec = torch.tensor(vec, dtype=dtype)
    return vec


class RandomTransformSE3:
    """ rigid motion """
    def __init__(self, mag=1, mag_randomly=False):
        self.mag = mag
        self.randomly = mag_randomly

        self.gt = None
        self.igt = None

    def generate_transform(self):
        # return: a twist-vector
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp

        return x # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        g = se3.exp(x).to(p0)   # [1, 4, 4]
        gt = se3.exp(-x).to(p0) # [1, 4, 4]

        p1 = se3.transform(g, p0)
        self.gt = gt.squeeze(0) #  gt: p1 -> p0
        self.igt = g.squeeze(0) # igt: p0 -> p1
        return p1

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)



class UnknownDataTypeError(Exception):
	def __init__(self, *args):
		if args: self.message = args[0]
		else: self.message = 'Datatype not understood for dataset.'

	def __str__(self):
		return self.message


class ModelNet40Data(Dataset):
	def __init__(
		self,
		train=True,
		num_points=1024,
		download=True,
		randomize_data=False
	):
		super(ModelNet40Data, self).__init__()
		if download: download_modelnet40()
		self.data, self.labels = load_data(train)
		if not train: self.shapes = self.read_classes_ModelNet40()
		self.num_points = num_points
		self.randomize_data = randomize_data

	def __getitem__(self, idx):
		if self.randomize_data: current_points = self.randomize(idx)
		else: current_points = self.data[idx].copy()

		current_points = torch.from_numpy(current_points).float()
		label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

		return current_points, label

	def __len__(self):
		return self.data.shape[0]

	def randomize(self, idx):
		pt_idxs = np.arange(0, self.num_points)
		np.random.shuffle(pt_idxs)
		return self.data[idx, pt_idxs].copy()

	def get_shape(self, label):
		return self.shapes[label]

	def read_classes_ModelNet40(self):
		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
		file = open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r')
		shape_names = file.read()
		shape_names = np.array(shape_names.split('\n')[:-1])
		return shape_names


class ClassificationData(Dataset):
	def __init__(self, data_class=ModelNet40Data()):
		super(ClassificationData, self).__init__()
		self.set_class(data_class)

	def __len__(self):
		return len(self.data_class)

	def set_class(self, data_class):
		self.data_class = data_class

	def get_shape(self, label):
		try:
			return self.data_class.get_shape(label)
		except:
			return -1

	def __getitem__(self, index):
		return self.data_class[index]


class RegistrationData(Dataset):
	def __init__(self, algorithm, data_class=ModelNet40Data()):
		super(RegistrationData, self).__init__()
		available_algorithms = ['PCRNet', 'PointNetLK', 'DCP', 'PRNet', 'iPCRNet']
		if algorithm in available_algorithms: self.algorithm = algorithm
		else: raise Exception("Algorithm not available for registration.")
		
		self.set_class(data_class)

		if self.algorithm == 'PCRNet' or self.algorithm == 'iPCRNet':
			self.transforms = [create_random_transform(torch.float32, 45, 1) for _ in range(len(self.data_class))]
		if self.algorithm == 'PointNetLK':
			self.transforms = RandomTransformSE3(0.8, True)

	def __len__(self):
		return len(self.data_class)

	def set_class(self, data_class):
		self.data_class = data_class

	def pcrnet_data_loader(self, index):
		template, label = self.data_class[index]
		igt = self.transforms[index]
		gt = transform_functions.create_pose_7d(igt)
		source = transform_functions.quaternion_rotate(template, gt)
		return template, source, igt

	def pointnetlk_data_loader(self, index):
		template, label = self.data_class[index]
		source = self.transforms(template)
		igt = self.transforms.igt
		return template, source, igt

	def __getitem__(self, index):
		if self.algorithm == 'PCRNet' or self.algorithm == 'iPCRNet':
			return self.pcrnet_data_loader(index)
		elif self.algorithm == 'PointNetLK':
			return self.pointnetlk_data_loader(index)


class SegmentationData(Dataset):
	def __init__(self):
		super(SegmentationData, self).__init__()

	def __len__(self):
		pass

	def __getitem__(self, index):
		pass


class FlowData(Dataset):
	def __init__(self):
		super(FlowData, self).__init__()
		self.pc1, self.pc2, self.flow = self.read_data()

	def __len__(self):
		if isinstance(self.pc1, np.ndarray):
			return self.pc1.shape[0]
		elif isinstance(self.pc1, list):
			return len(self.pc1)
		else:
			raise UnknownDataTypeError

	def read_data(self):
		pass

	def __getitem__(self, index):
		return self.pc1[index], self.pc2[index], self.flow[index]


if __name__ == '__main__':
	class Data():
		def __init__(self):
			super(Data, self).__init__()
			self.data, self.label = self.read_data()

		def read_data(self):
			return [4,5,6], [4,5,6]

		def __len__(self):
			return len(self.data)

		def __getitem__(self, idx):
			return self.data[idx], self.label[idx]

	cd = RegistrationData('abc')
	import ipdb; ipdb.set_trace()