import torch
import torch.nn as nn
import torch.nn.functional as F

def chamfer_distance(template: torch.Tensor, source: torch.Tensor):
	from .cuda.chamfer_distance import ChamferDistance
	cost_p0_p1, cost_p1_p0 = ChamferDistance()(template, source)
	cost_p0_p1 = torch.mean(torch.sqrt(cost_p0_p1))
	cost_p1_p0 = torch.mean(torch.sqrt(cost_p1_p0))
	chamfer_loss = (cost_p0_p1 + cost_p1_p0)/2.0
	return chamfer_loss


class ChamferDistanceLoss(nn.Module):
	def __init__(self):
		super(ChamferDistanceLoss, self).__init__()

	def forward(self, template, source):
		return chamfer_distance(template, source)