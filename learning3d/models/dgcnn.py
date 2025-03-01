import torch
import torch.nn.functional as F
from .. utils import knn, get_graph_feature


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


if __name__ == '__main__':
	# Test the code.
	x = torch.rand((10,1024,3))

	dgcnn = DGCNN()
	y = dgcnn(x)
	print("\nInput Shape of DGCNN: ", x.shape, "\nOutput Shape of DGCNN: ", y.shape)