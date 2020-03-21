import torch
import torch.nn as nn
import torch.nn.functional as F
from .pooling import Pooling


class PointNet(torch.nn.Module):
	def __init__(self, emb_dims=1024, input_shape="bnc", use_bn=False, global_feat=True):
		# emb_dims:			Embedding Dimensions for PointNet.
		# input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
		super(PointNet, self).__init__()
		if input_shape not in ["bcn", "bnc"]:
			raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' ")
		self.input_shape = input_shape
		self.emb_dims = emb_dims
		self.use_bn = use_bn
		self.global_feat = global_feat
		if not self.global_feat: self.pooling = Pooling('max')

		self.conv1 = torch.nn.Conv1d(3, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 64, 1)
		self.conv3 = torch.nn.Conv1d(64, 64, 1)
		self.conv4 = torch.nn.Conv1d(64, 128, 1)
		self.conv5 = torch.nn.Conv1d(128, emb_dims, 1)

		if use_bn:
			self.bn1 = torch.nn.BatchNorm1d(64)
			self.bn2 = torch.nn.BatchNorm1d(64)
			self.bn3 = torch.nn.BatchNorm1d(64)
			self.bn4 = torch.nn.BatchNorm1d(128)
			self.bn5 = torch.nn.BatchNorm1d(emb_dims)

	def forward(self, input_data):
		# input_data: 		Point Cloud having shape input_shape.
		# output:			PointNet features (Batch x emb_dims)
		if self.input_shape == "bnc":
			num_points = input_data.shape[1]
			input_data = input_data.permute(0, 2, 1)
		else:
			num_points = input_data.shape[2]
		if input_data.shape[1] != 3:
			raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

		if self.use_bn:
			output = F.relu(self.bn1(self.conv1(input_data)))
			point_feature = F.relu(self.bn2(self.conv2(output)))
			output = F.relu(self.bn3(self.conv3(point_feature)))
			output = F.relu(self.bn4(self.conv4(output)))
			output = F.relu(self.bn5(self.conv5(output)))
		else:
			output = F.relu(self.conv1(input_data))
			point_feature = F.relu(self.conv2(output))
			output = F.relu(self.conv3(point_feature))
			output = F.relu(self.conv4(output))
			output = F.relu(self.conv5(output))  # Batch x 1024 x NumInPoints

		if self.global_feat:
			return output
		else:
			output = self.pooling(output)
			output = output.view(-1, self.emb_dims, 1).repeat(1, 1, num_points)
			return torch.cat([output, point_feature], 1)


if __name__ == '__main__':
	# Test the code.
	x = torch.rand((10,1024,3))

	pn = PointNet()
	y = pn(x)
	print("Input Shape of PointNet: ", x.shape, "\nOutput Shape of PointNet: ", y.shape)