import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_models import Pooling

class Classifier(nn.Module):
	def __init__(self, feature_model, num_classes=40):
		super(Classifier, self).__init__()
		self.feature_model = feature_model
		self.num_classes = num_classes

		self.linear1 = torch.nn.Linear(self.feature_model.emb_dims, 512)
		self.bn1 = torch.nn.BatchNorm1d(512)
		self.dropout1 = torch.nn.Dropout(p=0.7)
		self.linear2 = torch.nn.Linear(512, 256)
		self.bn2 = torch.nn.BatchNorm1d(256)
		self.dropout2 = torch.nn.Dropout(p=0.7)
		self.linear3 = torch.nn.Linear(256, self.num_classes)

		self.pooling = Pooling('max')

	def forward(self, input_data):
		output = self.pooling(self.feature_model(input_data))
		output = F.relu(self.bn1(self.linear1(output)))
		output = self.dropout1(output)
		output = F.relu(self.bn2(self.linear2(output)))
		output = self.dropout2(output)
		output = self.linear3(output)
		return output


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
	from feature_models import PointNet, DGCNN, Pooling
	x = torch.rand(10,1024,3)

	pn = PointNet(global_feat=False)
	seg = Segmentation(pn)
	seg_result = seg(x)

	pn = PointNet()
	classifier = Classifier(pn)
	classes = classifier(x)
	
	print('Input Shape: {}\nClassification Output Shape: {}\nSegmentation Output Shape: {}'
		  .format(x.shape, seg_result.shape, classes.shape))