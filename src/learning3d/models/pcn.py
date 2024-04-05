# author: Vinit Sarode (vinitsarode5@gmail.com) 03/23/2020

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pooling import Pooling

class PCN(torch.nn.Module):
	def __init__(self, emb_dims=1024, input_shape="bnc", num_coarse=1024, grid_size=4, detailed_output=False):
		# emb_dims:			Embedding Dimensions for PCN.
		# input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
		super(PCN, self).__init__()
		if input_shape not in ["bcn", "bnc"]:
			raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' ")
		self.input_shape = input_shape
		self.emb_dims = emb_dims
		self.num_coarse = num_coarse
		self.detailed_output = detailed_output
		self.grid_size = grid_size
		self.num_fine = self.grid_size ** 2 * self.num_coarse
		self.pooling = Pooling('max')

		self.encoder()
		self.decoder_layers = self.decoder()
		if detailed_output: self.folding_layers = self.folding()

	def encoder_1(self):
		self.conv1 = torch.nn.Conv1d(3, 128, 1)
		self.conv2 = torch.nn.Conv1d(128, 256, 1)
		self.relu = torch.nn.ReLU()

		# self.bn1 = torch.nn.BatchNorm1d(128)
		# self.bn2 = torch.nn.BatchNorm1d(256)

		layers = [self.conv1, self.relu,
				  self.conv2]
		return layers

	def encoder_2(self):
		self.conv3 = torch.nn.Conv1d(2*256, 512, 1)
		self.conv4 = torch.nn.Conv1d(512, self.emb_dims, 1)

		# self.bn3 = torch.nn.BatchNorm1d(512)
		# self.bn4 = torch.nn.BatchNorm1d(self.emb_dims)
		self.relu = torch.nn.ReLU()

		layers = [self.conv3, self.relu,
				  self.conv4]
		return layers

	def encoder(self):
		self.encoder_layers1 = self.encoder_1()
		self.encoder_layers2 = self.encoder_2()

	def decoder(self):
		self.linear1 = torch.nn.Linear(self.emb_dims, 1024)
		self.linear2 = torch.nn.Linear(1024, 1024)
		self.linear3 = torch.nn.Linear(1024, self.num_coarse*3)

		# self.bn1 = torch.nn.BatchNorm1d(1024)
		# self.bn2 = torch.nn.BatchNorm1d(1024)
		# self.bn3 = torch.nn.BatchNorm1d(self.num_coarse*3)
		self.relu = torch.nn.ReLU()

		layers = [self.linear1, self.relu,
				  self.linear2, self.relu,
				  self.linear3]
		return layers

	def folding(self):
		self.conv5 = torch.nn.Conv1d(1029, 512, 1)
		self.conv6 = torch.nn.Conv1d(512, 512, 1)
		self.conv7 = torch.nn.Conv1d(512, 3, 1)

		# self.bn5 = torch.nn.BatchNorm1d(512)
		# self.bn6 = torch.nn.BatchNorm1d(512)
		self.relu = torch.nn.ReLU()

		layers = [self.conv5, self.relu,
				  self.conv6, self.relu,
				  self.conv7]
		return layers

	def fine_decoder(self):
		# Fine Output
		linspace = torch.linspace(-0.05, 0.05, steps=self.grid_size)
		grid = torch.meshgrid(linspace, linspace)
		grid = torch.reshape(torch.stack(grid, dim=2), (-1,2))								# 16x2
		grid = torch.unsqueeze(grid, dim=0)													# 1x16x2
		grid_feature = grid.repeat([self.coarse_output.shape[0], self.num_coarse, 1])		# Bx16384x2

		point_feature = torch.unsqueeze(self.coarse_output, dim=2)							# Bx1024x1x3
		point_feature = point_feature.repeat([1, 1, self.grid_size ** 2, 1])				# Bx1024x16x3
		point_feature = torch.reshape(point_feature, (-1, self.num_fine, 3))				# Bx16384x3

		global_feature = torch.unsqueeze(self.global_feature_v, dim=1)						# Bx1x1024
		global_feature = global_feature.repeat([1, self.num_fine, 1])						# Bx16384x1024

		feature = torch.cat([grid_feature, point_feature, global_feature], dim=2)			# Bx16384x1029

		center = torch.unsqueeze(self.coarse_output, dim=2)									# Bx1024x1x3
		center = center.repeat([1, 1, self.grid_size ** 2, 1])								# Bx1024x16x3
		center = torch.reshape(center, [-1, self.num_fine, 3])								# Bx16384x3

		output = feature.permute(0, 2, 1)
		for idx, layer in enumerate(self.folding_layers):
			output = layer(output)
		fine_output = output.permute(0, 2, 1) + center
		return fine_output

	def encode(self, input_data):
		output = input_data
		for idx, layer in enumerate(self.encoder_layers1):
			output = layer(output)

		global_feature_g = self.pooling(output)

		global_feature_g = global_feature_g.unsqueeze(2)
		global_feature_g = global_feature_g.repeat(1,1,self.num_points)
		output = torch.cat([output, global_feature_g], dim=1)

		for idx, layer in enumerate(self.encoder_layers2):
			output = layer(output)

		self.global_feature_v = self.pooling(output)

	def decode(self):
		output = self.global_feature_v
		for idx, layer in enumerate(self.decoder_layers):
			output = layer(output)		
		self.coarse_output = output.view(self.global_feature_v.shape[0], self.num_coarse, 3)

	def forward(self, input_data):
		# input_data: 		Point Cloud having shape input_shape.
		# output:			PointNet features (Batch x emb_dims)
		if self.input_shape == "bnc":
			self.num_points = input_data.shape[1]
			input_data = input_data.permute(0, 2, 1)
		else:
			self.num_points = input_data.shape[2]
		if input_data.shape[1] != 3:
			raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

		self.encode(input_data)
		self.decode()

		result = {'coarse_output': self.coarse_output}

		if self.detailed_output: 
			fine_output = self.fine_decoder()
			result['fine_output'] = fine_output

		return result

		
if __name__ == '__main__':
	# Test the code.
	x = torch.rand((10,1024,3))

	pcn = PCN()
	y = pcn(x)
	print("Network Architecture: ")
	print(pn)
	print("Input Shape of PCN: ", x.shape, "\nOutput Shape of PCN: ", y['coarse_output'].shape)