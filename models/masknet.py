import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import PointNet
from .pooling import Pooling

class PointNetMask(nn.Module):
	def __init__(self, template_feature_size=1024, source_feature_size=1024, feature_model=PointNet()):
		super().__init__()
		self.feature_model = feature_model
		self.pooling = Pooling()

		input_size = template_feature_size + source_feature_size
		self.h3 = nn.Sequential(nn.Conv1d(input_size, 1024, 1), nn.ReLU(),
								nn.Conv1d(1024, 512, 1), nn.ReLU(),
								nn.Conv1d(512, 256, 1), nn.ReLU(),
								nn.Conv1d(256, 128, 1), nn.ReLU(),
								nn.Conv1d(128,   1, 1), nn.Sigmoid())

	def find_mask(self, x, t_out_h1):
		batch_size, _ , num_points = t_out_h1.size()
		x = x.unsqueeze(2)
		x = x.repeat(1,1,num_points)
		x = torch.cat([t_out_h1, x], dim=1)
		x = self.h3(x)
		return x.view(batch_size, -1)

	def forward(self, template, source):
		source_features = self.feature_model(source)				# [B x C x N]
		template_features = self.feature_model(template)			# [B x C x N]

		source_features = self.pooling(source_features)
		mask = self.find_mask(source_features, template_features)
		return mask


class MaskNet(nn.Module):
	def __init__(self, feature_model=PointNet(use_bn=True), is_training=True):
		super().__init__()
		self.maskNet = PointNetMask(feature_model=feature_model)
		self.is_training = is_training

	@staticmethod
	def index_points(points, idx):
		"""
		Input:
			points: input points data, [B, N, C]
			idx: sample index data, [B, S]
		Return:
			new_points:, indexed points data, [B, S, C]
		"""
		device = points.device
		B = points.shape[0]
		view_shape = list(idx.shape)
		view_shape[1:] = [1] * (len(view_shape) - 1)
		repeat_shape = list(idx.shape)
		repeat_shape[0] = 1
		batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
		new_points = points[batch_indices, idx, :]
		return new_points

	# This function is only useful for testing with a single pair of point clouds.
	@staticmethod
	def find_index(mask_val):
		mask_idx = torch.nonzero((mask_val[0]>0.5)*1.0)
		return mask_idx.view(1, -1)

	def forward(self, template, source, point_selection='threshold'):
		mask = self.maskNet(template, source)

		if point_selection == 'topk' or self.is_training:
			_, self.mask_idx = torch.topk(mask, source.shape[1], dim=1, sorted=False)
		elif point_selection == 'threshold':
			self.mask_idx = self.find_index(mask)

		template = self.index_points(template, self.mask_idx)
		return template, mask


if __name__ == '__main__':
	template, source = torch.rand(10,1024,3), torch.rand(10,1024,3)
	net = MaskNet()
	result = net(template, source)
	import ipdb; ipdb.set_trace()