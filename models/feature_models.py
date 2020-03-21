import torch
import torch.nn.functional as F

def knn(x, k):
	inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
	xx = torch.sum(x ** 2, dim=1, keepdim=True)
	pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

	idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
	return idx


def get_graph_feature(x, k=20):
	# x = x.squeeze()
	idx = knn(x, k=k)  # (batch_size, num_points, k)
	batch_size, num_points, _ = idx.size()
	
	if not torch.cuda.is_available():
		device = torch.device("cpu")
	else:
		device = torch.device('cuda')

	idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

	idx = idx + idx_base

	idx = idx.view(-1)

	_, num_dims, _ = x.size()

	# (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
	x = x.transpose(2, 1).contiguous()  

	feature = x.view(batch_size * num_points, -1)[idx, :]
	feature = feature.view(batch_size, num_points, k, num_dims)
	x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

	feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

	return feature

class Pooling(torch.nn.Module):
	def __init__(self, pool_type='max'):
		self.pool_type = pool_type
		super(Pooling, self).__init__()

	def forward(self, input):
		if self.pool_type == 'max':
			return torch.max(input, 2)[0].contiguous()
		elif self.pool_type == 'avg' or self.pool_type == 'average':
			return torch.mean(input, 2).contiguous()
	

class DGCNN(torch.nn.Module):
	def __init__(self, emb_dims=1024, input_shape="bnc"):
		super(DGCNN, self).__init__()
		if input_shape not in ["bcn", "bnc"]:
			raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' ")
		self.input_shape = input_shape
		self.emb_dims = emb_dims

		self.conv1 = torch.nn.Conv2d(6, 64, kernel_size=1, bias=False)
		self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=1, bias=False)
		self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=1, bias=False)
		self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=1, bias=False)
		self.conv5 = torch.nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
		self.bn1 = torch.nn.BatchNorm2d(64)
		self.bn2 = torch.nn.BatchNorm2d(64)
		self.bn3 = torch.nn.BatchNorm2d(128)
		self.bn4 = torch.nn.BatchNorm2d(256)
		self.bn5 = torch.nn.BatchNorm2d(emb_dims)

	def forward(self, input_data):
		if self.input_shape == "bnc":
			input_data = input_data.permute(0, 2, 1)
		if input_data.shape[1] != 3:
			raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

		batch_size, num_dims, num_points = input_data.size()
		output = get_graph_feature(input_data)

		output = F.relu(self.bn1(self.conv1(output)))
		output1 = output.max(dim=-1, keepdim=True)[0]

		output = F.relu(self.bn2(self.conv2(output)))
		output2 = output.max(dim=-1, keepdim=True)[0]

		output = F.relu(self.bn3(self.conv3(output)))
		output3 = output.max(dim=-1, keepdim=True)[0]

		output = F.relu(self.bn4(self.conv4(output)))
		output4 = output.max(dim=-1, keepdim=True)[0]

		output = torch.cat((output1, output2, output3, output4), dim=1)

		output = F.relu(self.bn5(self.conv5(output))).view(batch_size, -1, num_points)
		return output


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

	dgcnn = DGCNN()
	y = pn(x)
	print("\nInput Shape of DGCNN: ", x.shape, "\nOutput Shape of DGCNN: ", y.shape)