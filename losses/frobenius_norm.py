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


class FrobeniusNormLoss(nn.Module):
	def __init__(self):
		super(FrobeniusNormLoss, self).__init__()

	def forward(self, predicted, igt):
		return frobeniusNormLoss(predicted, igt)