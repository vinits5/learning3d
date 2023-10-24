import torch
import torch.nn as nn
import torch.nn.functional as F
from .pooling import Pooling


# Mish Activation Function
class Mish(nn.Module):
	def __init__(self):
		super(Mish, self).__init__()

	def forward(self, x):
		return x * torch.tanh(F.softplus(x))


# Basic Convolution Block
class BasicConv1D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, active = True):
		super(BasicConv1D, self).__init__()
		self.active = active
		self.bn = nn.BatchNorm1d( out_channels)
		if self.active == True:
			self.activation = Mish()
		self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False)
		#self.dropout = nn.Dropout(0.5)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		if self.active == True:
			x = self.activation(x)
		return x


class Self_Attn(nn.Module):
	""" Self attention Layer"""
	def __init__(self, in_dim, out_dim):
		super(Self_Attn,self).__init__()

		self.in_dim = in_dim
		self.out_dim = out_dim
	
		# Query Convolution
		self.query_conv =BasicConv1D(in_dim, out_dim)  
	
		self.beta = nn.Parameter(torch.zeros(1))

		self.softmax  = nn.Softmax(dim=-1) #

	def forward(self,x):
		"""
			inputs :
				x : input feature maps( B X C X N)  32, 1024, 64
			returns :
				out : self attention value + input feature 
				attention: B X N X N (N is Width*Height)
		"""

		proj_query  = self.query_conv(x).permute(0,2,1)      # B, in_dim, N   ---> B, in_dim // 8, N   ---->  B, N, in_dim // 8
		proj_key =   proj_query.permute(0,2,1) #B, in_dim, N   ---> B, in_dim // 8, N
		
		energy =  torch.bmm(proj_query,proj_key) # transpose check    B, N, N

		attention = self.softmax(energy) # B , N,  N  

		out_x = torch.bmm(proj_key, attention.permute(0,2,1) )   #B, out_dim, N
	
		out =  self.beta * out_x + proj_key 
		
		return out

class PointNet(torch.nn.Module):
	def __init__(self, emb_dims=224, input_shape="bnc", use_bn=False, global_feat=True):
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

		self.conv1 = Self_Attn(3, 32)
		self.conv2 = Self_Attn(32, 64)
		self.conv3 = Self_Attn(64, 64)
		self.conv4 = Self_Attn(64, 128)
		self.conv5 = Self_Attn(128, self.emb_dims)


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

		output = input_data
		
		x1 = self.conv1(output)  #32
		x2 = self.conv2(x1)      #64
		x3 = self.conv3(x2)   #64
		x4 = self.conv4(x3+x2) #128
		x5 = self.conv5(x4)

		output  = torch.cat([ x1, x2,  x3, x4, x5], dim=1)  #256, x4 x0,
		point_feature = output

		if self.global_feat:
			return output
		else:
			output = self.pooling(output)
			output = output.view(-1, self.emb_dims, 1).repeat(1, 1, num_points)
			return torch.cat([output, point_feature], 1)


# self attention mechanism
class self_attention_fc(nn.Module):
	""" Self attention Layer""" 
	def __init__(self,in_dim, out_dim):     #1024
		super(self_attention_fc,self).__init__()
		
		self.in_dim = in_dim
		self.out_dim = out_dim

		self.query_conv = BasicConv1D(in_dim, out_dim)

		self.beta = nn.Parameter(torch.zeros(1))
		self.softmax  = nn.Softmax(dim=-1) #

	def forward(self,x, y):   #B, 1024 , 1
		"""
			inputs :
				x : input feature maps( B X C,1 )
			returns :
				out : self attention value + input feature 
				attention: B X N X N (N is Width*Height)
		"""
		proj_query_x  = self.query_conv(x)                      #[B, in_dim, 1]----->[B, out_dim1, 1]

		proj_key_y =   self.query_conv(y).permute(0,2,1)        #[B, 1, out_dim1]
		
		energy_xy =  torch.bmm(proj_query_x, proj_key_y)        #  xi Attention scores for all points in y [B, 64, 64]   

		attention_xy = self.softmax(energy_xy)
		attention_yx = self.softmax(energy_xy.permute(0,2,1))

		proj_value_x = proj_query_x                             # self.value_conv_x(x) # [B, out_dim, 64]
		proj_value_y = proj_key_y.permute(0,2,1)                # self.value_conv_x(y) # [B, out_dim, 64]

		out_x = torch.bmm(attention_xy, proj_value_x)           # [B, out_dim]
		out_x =  self.beta* out_x +  proj_value_x               # self.kama* 

		out_y = torch.bmm(attention_yx, proj_value_y )          # [B, out_dim]
		out_y =  self.beta*out_y +   proj_value_y               # self.kama *

		return out_x, out_y



class PointNetMask(nn.Module):
	def __init__(self, template_feature_size=1024, source_feature_size=1024, feature_model=PointNet()):
		super().__init__()
		self.feature_model = feature_model
		self.pooling_max = Pooling(pool_type='max')
		self.pooling_avg = Pooling(pool_type='avg')

		input_size = template_feature_size + source_feature_size

		self.global_feat_1 = self_attention_fc(1024, 512)
		self.global_feat_2 = self_attention_fc(512, 256)
		self.global_feat_3 = self_attention_fc(256, 512)

		self.h3 = nn.Sequential(BasicConv1D(1024, 512),
								BasicConv1D(512,  256),
								BasicConv1D(256,  128),
								nn.Conv1d(128,  1, 1), nn.Sigmoid())


	def find_mask(self, source_features, template_features):
		global_source_features_max = self.pooling_max(source_features)
		global_template_features_max = self.pooling_max(template_features)
		global_source_features_avg = self.pooling_avg(source_features)
		global_template_features_avg = self.pooling_avg(template_features)
		global_source_features = torch.cat([global_source_features_max, global_source_features_avg], dim=1)
		global_template_features = torch.cat([global_template_features_max, global_template_features_avg], dim=1)

		shared_feat_1,shared_feat_2  = self.global_feat_1(global_source_features.unsqueeze(2), global_template_features.unsqueeze(2))
		shared_feat_1,shared_feat_2  = self.global_feat_2(shared_feat_1, shared_feat_2) 
		shared_feat_1,shared_feat_2  = self.global_feat_3(shared_feat_1, shared_feat_2) 

		batch_size, _ , num_points = source_features.size()
		global_source_features = shared_feat_1
		global_source_features = global_source_features.repeat(1,1,num_points)
		x = torch.cat([template_features, global_source_features], dim=1)
		x = self.h3(x)

		batch_size, _ , num_points = template_features.size()
		global_template_features = shared_feat_2
		global_template_features = global_template_features.repeat(1,1,num_points)
		y = torch.cat([source_features, global_template_features], dim=1)
		y = self.h3(y)

		return x.view(batch_size, -1), y.view(batch_size, -1)

	def forward(self, template, source):
		source_features = self.feature_model(source)				# [B x C x N]
		template_features = self.feature_model(template)			# [B x C x N]

		template_mask, source_mask = self.find_mask(source_features, template_features)
		return template_mask, source_mask

class MaskNet2(nn.Module):
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

	def forward(self, template, source, point_selection='threshold', mask_threshold = 0.5):
		template_mask, source_mask = self.maskNet(template, source)   #B, N
		if not torch.cuda.is_available():
			device = 'cpu'
		device = torch.device(device)
		
		source_binary_mask = torch.where(source_mask > mask_threshold, torch.ones(source_mask.size()).to(device), torch.zeros(source_mask.size()).to(device))  
		template_binary_mask = torch.where(template_mask > mask_threshold, torch.ones(template_mask.size()).to(device), torch.zeros(template_mask.size()).to(device))

		masked_template = template[:, torch.tensor(template_binary_mask, dtype = torch.bool).squeeze(0), 0:3]
		masked_source = source[:, torch.tensor(source_binary_mask, dtype = torch.bool).squeeze(0), 0:3]

		return masked_template, masked_source, template_mask, source_mask


if __name__ == '__main__':
	template, source = torch.rand(10,1024,3), torch.rand(10,1024,3)
	net = MaskNet2()
	result = net(template, source)
	import ipdb; ipdb.set_trace()