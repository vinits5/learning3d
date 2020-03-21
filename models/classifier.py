import torch
import torch.nn as nn
import torch.nn.functional as F
from .pooling import Pooling

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


if __name__ == '__main__':
	from pointnet import PointNet
	x = torch.rand(10,1024,3)

	pn = PointNet()
	classifier = Classifier(pn)
	classes = classifier(x)
	
	print('Input Shape: {}\nClassification Output Shape: {}'
		  .format(x.shape, classes.shape))