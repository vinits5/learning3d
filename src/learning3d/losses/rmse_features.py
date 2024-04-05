import torch
import torch.nn as nn
import torch.nn.functional as F

def rmseOnFeatures(feature_difference):
	# |feature_difference| should be 0
	gt = torch.zeros_like(feature_difference)
	return torch.nn.functional.mse_loss(feature_difference, gt, size_average=False)


class RMSEFeaturesLoss(nn.Module):
	def __init__(self):
		super(RMSEFeaturesLoss, self).__init__()

	def forward(self, feature_difference):
		return rmseOnFeatures(feature_difference)