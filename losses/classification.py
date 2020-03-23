import torch
import torch.nn as nn
import torch.nn.functional as F

def classification_loss(prediction: torch.Tensor, target: torch.Tensor):
	return F.nll_loss(prediction, target)


class ClassificationLoss(nn.Module):
	def __init__(self):
		super(ClassificationLoss, self).__init__()

	def forward(self, prediction, target):
		return classification_loss(prediction, target)