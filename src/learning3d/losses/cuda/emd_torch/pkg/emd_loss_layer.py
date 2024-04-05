import torch
import torch.nn as nn

import _emd_ext._emd as emd


class EMDFunction(torch.autograd.Function):
	@staticmethod
	def forward(self, xyz1, xyz2):
		cost, match = emd.emd_forward(xyz1, xyz2)
		self.save_for_backward(xyz1, xyz2, match)
		return cost


	@staticmethod
	def backward(self, grad_output):
		xyz1, xyz2, match = self.saved_tensors
		grad_xyz1, grad_xyz2 = emd.emd_backward(xyz1, xyz2, match)
		return grad_xyz1, grad_xyz2




class EMDLoss(nn.Module):
	'''
	Computes the (approximate) Earth Mover's Distance between two point sets. 

	IMPLEMENTATION LIMITATIONS:
	- Double tensors must have <=11 dimensions
	- Float tensors must have <=23 dimensions
	This is due to the use of CUDA shared memory in the computation. This shared memory is limited by the hardware to 48kB.
	'''

	def __init__(self):
		super(EMDLoss, self).__init__()

	def forward(self, xyz1, xyz2):

		assert xyz1.shape[-1] == xyz2.shape[-1], 'Both point sets must have the same dimensions!'
		assert xyz1.shape[1] == xyz2.shape[1], 'Both Point Clouds must have same number of points in it.'
		return EMDFunction.apply(xyz1, xyz2)