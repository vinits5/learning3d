import torch
import torch.nn as nn
import torch.nn.functional as F

def frobeniusNormLoss(predicted, igt):
	""" |predicted*igt - I| (should be 0) """
	assert predicted.size(0) == igt.size(0)
	assert predicted.size(1) == igt.size(1) and predicted.size(1) == 4
	assert predicted.size(2) == igt.size(2) and predicted.size(2) == 4

	error = predicted.matmul(igt)
	I = torch.eye(4).to(error).view(1, 4, 4).expand(error.size(0), 4, 4)
	return torch.nn.functional.mse_loss(error, I, size_average=True) * 16

def rmseOnFeatures(feature_difference):
	# |feature_difference| should be 0
	gt = torch.zeros_like(feature_difference)
	return torch.nn.functional.mse_loss(feature_difference, gt, size_average=False)

def chamfer_distance(template: torch.Tensor, source: torch.Tensor):
	from ops.losses.chamfer_distance import ChamferDistance
	cost_p0_p1, cost_p1_p0 = ChamferDistance()(template, source)
	cost_p0_p1 = torch.mean(torch.sqrt(cost_p0_p1))
	cost_p1_p0 = torch.mean(torch.sqrt(cost_p1_p0))
	chamfer_loss = (cost_p0_p1 + cost_p1_p0)/2.0
	return chamfer_loss

def emd(template: torch.Tensor, source: torch.Tensor):
	from emd import EMDLoss
	emd_loss = torch.mean(self.emd(template, source))/(template.size()[1])
	return emd_loss

def classification_loss(prediction: torch.Tensor, target: torch.Tensor):
	return F.nll_loss(prediction, target)


class ClassificationLoss(nn.Module):
	def __init__(self):
		super(ClassificationLoss, self).__init__()

	def forward(self, prediction, target):
		return classification_loss(prediction, target)


class FrobeniusNormLoss(nn.Module):
	def __init__(self):
		super(FrobeniusNormLoss, self).__init__()

	def forward(self, predicted, igt):
		return frobeniusNormLoss(predicted, igt)


class RMSEFeaturesLoss(nn.Module):
	def __init__(self):
		super(RMSEFeatures, self).__init__()

	def forward(self, feature_difference):
		return rmseOnFeatures(feature_difference)


class ChamferDistanceLoss(nn.Module):
	def __init__(self):
		super(ChamferDistanceLoss, self).__init__()

	def forward(template, source):
		return chamfer_distance(template, source)


class EMDLoss(nn.Module):
	def __init__(self):
		super(EMDLoss, self).__init__()

	def forward(template, source):
		return emd(template, source)