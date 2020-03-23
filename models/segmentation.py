import torch
import torch.nn as nn
import torch.nn.functional as F


class Segmentation(nn.Module):
	def __init__(self, feature_model, num_classes=40):
		super(Segmentation, self).__init__()
		self.feature_model = feature_model
		self.num_classes = num_classes

		self.conv1 = torch.nn.Conv1d(self.feature_model.emb_dims+64, 512, 1)
		self.conv2 = torch.nn.Conv1d(512, 256, 1)
		self.conv3 = torch.nn.Conv1d(256, 128, 1)
		self.conv4 = torch.nn.Conv1d(128, self.num_classes, 1)
		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)
		self.bn3 = nn.BatchNorm1d(128)

	def forward(self, input_data):
		output = self.feature_model(input_data)
		output = F.relu(self.bn1(self.conv1(output)))
		output = F.relu(self.bn2(self.conv2(output)))
		output = F.relu(self.bn3(self.conv3(output)))
		output = self.conv4(output)
		output = output.permute(0, 2, 1)				# B x N x num_classes
		return output

if __name__ == '__main__':
	from pointnet import PointNet
	x = torch.rand(10,1024,3)

	pn = PointNet(global_feat=False)
	seg = Segmentation(pn)
	seg_result = seg(x)
	
	print('Input Shape: {}\n Segmentation Output Shape: {}'
		  .format(x.shape, seg_result.shape))