import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. utils import square_distance, angle_difference
from .. ops.transform_functions import convert2transformation
from .ppfnet import PPFNet
_EPS = 1e-5  # To prevent division by zero


class ParameterPredictionNet(nn.Module):
	def __init__(self, weights_dim):
		"""PointNet based Parameter prediction network

		Args:
			weights_dim: Number of weights to predict (excluding beta), should be something like
						 [3], or [64, 3], for 3 types of features
		"""

		super().__init__()

		self._logger = logging.getLogger(self.__class__.__name__)

		self.weights_dim = weights_dim

		# Pointnet
		self.prepool = nn.Sequential(
			nn.Conv1d(4, 64, 1),
			nn.GroupNorm(8, 64),
			nn.ReLU(),

			nn.Conv1d(64, 64, 1),
			nn.GroupNorm(8, 64),
			nn.ReLU(),

			nn.Conv1d(64, 64, 1),
			nn.GroupNorm(8, 64),
			nn.ReLU(),

			nn.Conv1d(64, 128, 1),
			nn.GroupNorm(8, 128),
			nn.ReLU(),

			nn.Conv1d(128, 1024, 1),
			nn.GroupNorm(16, 1024),
			nn.ReLU(),
		)
		self.pooling = nn.AdaptiveMaxPool1d(1)
		self.postpool = nn.Sequential(
			nn.Linear(1024, 512),
			nn.GroupNorm(16, 512),
			nn.ReLU(),

			nn.Linear(512, 256),
			nn.GroupNorm(16, 256),
			nn.ReLU(),

			nn.Linear(256, 2 + np.prod(weights_dim)),
		)

		self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

	def forward(self, x):
		""" Returns alpha, beta, and gating_weights (if needed)

		Args:
			x: List containing two point clouds, x[0] = src (B, J, 3), x[1] = ref (B, K, 3)

		Returns:
			beta, alpha, weightings
		"""

		src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
		ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
		concatenated = torch.cat([src_padded, ref_padded], dim=1)

		prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
		pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
		raw_weights = self.postpool(pooled)

		beta = F.softplus(raw_weights[:, 0])
		alpha = F.softplus(raw_weights[:, 1])

		return beta, alpha



def to_numpy(tensor):
	"""Wrapper around .detach().cpu().numpy() """
	if isinstance(tensor, torch.Tensor):
		return tensor.detach().cpu().numpy()
	elif isinstance(tensor, np.ndarray):
		return tensor
	else:
		raise NotImplementedError


def se3_transform(g, a, normals=None):
	""" Applies the SE3 transform

	Args:
		g: SE3 transformation matrix of size ([1,] 3/4, 4) or (B, 3/4, 4)
		a: Points to be transformed (N, 3) or (B, N, 3)
		normals: (Optional). If provided, normals will be transformed

	Returns:
		transformed points of size (N, 3) or (B, N, 3)

	"""
	R = g[..., :3, :3]  # (B, 3, 3)
	p = g[..., :3, 3]  # (B, 3)

	if len(g.size()) == len(a.size()):
		b = torch.matmul(a, R.transpose(-1, -2)) + p[..., None, :]
	else:
		raise NotImplementedError
		b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p  # No batch. Not checked

	if normals is not None:
		rotated_normals = normals @ R.transpose(-1, -2)
		return b, rotated_normals

	else:
		return b


def match_features(feat_src, feat_ref, metric='l2'):
	""" Compute pairwise distance between features

	Args:
		feat_src: (B, J, C)
		feat_ref: (B, K, C)
		metric: either 'angle' or 'l2' (squared euclidean)

	Returns:
		Matching matrix (B, J, K). i'th row describes how well the i'th point
		 in the src agrees with every point in the ref.
	"""
	assert feat_src.shape[-1] == feat_ref.shape[-1]

	if metric == 'l2':
		dist_matrix = square_distance(feat_src, feat_ref)
	elif metric == 'angle':
		feat_src_norm = feat_src / (torch.norm(feat_src, dim=-1, keepdim=True) + _EPS)
		feat_ref_norm = feat_ref / (torch.norm(feat_ref, dim=-1, keepdim=True) + _EPS)

		dist_matrix = angle_difference(feat_src_norm, feat_ref_norm)
	else:
		raise NotImplementedError

	return dist_matrix


def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
	""" Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

	Args:
		log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
		n_iters (int): Number of normalization iterations
		slack (bool): Whether to include slack row and column
		eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

	Returns:
		log(perm_matrix): Doubly stochastic matrix (B, J, K)

	Modified from original source taken from:
		Learning Latent Permutations with Gumbel-Sinkhorn Networks
		https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
	"""

	# Sinkhorn iterations
	prev_alpha = None
	if slack:
		zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
		log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

		log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

		for i in range(n_iters):
			# Row normalization
			log_alpha_padded = torch.cat((
					log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
					log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
				dim=1)

			# Column normalization
			log_alpha_padded = torch.cat((
					log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
					log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
				dim=2)

			if eps > 0:
				if prev_alpha is not None:
					abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
					if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
						break
				prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

		log_alpha = log_alpha_padded[:, :-1, :-1]
	else:
		for i in range(n_iters):
			# Row normalization (i.e. each row sum to 1)
			log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

			# Column normalization (i.e. each column sum to 1)
			log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

			if eps > 0:
				if prev_alpha is not None:
					abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
					if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
						break
				prev_alpha = torch.exp(log_alpha).clone()

	return log_alpha


def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
	"""Compute rigid transforms between two point sets

	Args:
		a (torch.Tensor): (B, M, 3) points
		b (torch.Tensor): (B, N, 3) points
		weights (torch.Tensor): (B, M)

	Returns:
		Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
	"""

	weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
	centroid_a = torch.sum(a * weights_normalized, dim=1)
	centroid_b = torch.sum(b * weights_normalized, dim=1)
	a_centered = a - centroid_a[:, None, :]
	b_centered = b - centroid_b[:, None, :]
	cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

	# Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
	# and choose based on determinant to avoid flips
	u, s, v = torch.svd(cov, some=False, compute_uv=True)
	rot_mat_pos = v @ u.transpose(-1, -2)
	v_neg = v.clone()
	v_neg[:, :, 2] *= -1
	rot_mat_neg = v_neg @ u.transpose(-1, -2)
	rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
	assert torch.all(torch.det(rot_mat) > 0)

	# Compute translation (uncenter centroid)
	translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

	transform = torch.cat((rot_mat, translation), dim=2)
	return transform


class RPMNet(nn.Module):
	def __init__(self, feature_model=PPFNet()):
		super().__init__()

		self.add_slack = True
		self.num_sk_iter = 5

		self.weights_net = ParameterPredictionNet(weights_dim=[0])
		self.feat_extractor = feature_model

	def compute_affinity(self, beta, feat_distance, alpha=0.5):
		"""Compute logarithm of Initial match matrix values, i.e. log(m_jk)"""
		if isinstance(alpha, float):
			hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha)
		else:
			hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha[:, None, None])
		return hybrid_affinity

	@staticmethod
	def split_normals(data):
		if data.shape[2] == 6:
			xyz, normals = data[:, :, :3], data[:, :, 3:6]
		elif data.shape[2] == 3:
			xyz, normals = data, torch.zeros(data.shape).to(data.device)
		return xyz, normals

	def spam(self, xyz_template, norm_template, xyz_source, norm_source):
		self.beta, self.alpha = self.weights_net([xyz_source, xyz_template])
		self.feat_source = self.feat_extractor(xyz_source, norm_source)
		self.feat_template = self.feat_extractor(xyz_template, norm_template)

		feat_distance = match_features(self.feat_source, self.feat_template)
		self.affinity = self.compute_affinity(self.beta, feat_distance, alpha=self.alpha)

		# Compute weighted coordinates
		log_perm_matrix = sinkhorn(self.affinity, n_iters=self.num_sk_iter, slack=self.add_slack)
		self.perm_matrix = torch.exp(log_perm_matrix)
		weighted_template = self.perm_matrix @ xyz_template / (torch.sum(self.perm_matrix, dim=2, keepdim=True) + _EPS)

		return weighted_template

	def forward(self, template, source, max_iterations: int = 1):
		"""Forward pass for RPMNet

		Args:
			data: Dict containing the following fields:
					'points_src': Source points (B, J, 6)
					'points_ref': Reference points (B, K, 6)
			num_iter (int): Number of iterations. Recommended to be 2 for training

		Returns:
			transform: Transform to apply to source points such that they align to reference
			src_transformed: Transformed source points
		"""

		xyz_template, norm_template = self.split_normals(template)
		xyz_source, norm_source = self.split_normals(source)

		xyz_source_t, norm_source_t = xyz_source, norm_source

		transforms = []
		all_gamma, all_perm_matrices, all_weighted_template = [], [], []
		all_beta, all_alpha = [], []

		for i in range(max_iterations):
			weighted_template = self.spam(xyz_template, norm_template, xyz_source_t, norm_source_t)			# Finding better correspondences after each iteration.
			
			# Compute transform and transform points
			transform = compute_rigid_transform(xyz_source, weighted_template, weights=torch.sum(self.perm_matrix, dim=2))
			xyz_source_t, norm_source_t = se3_transform(transform.detach(), xyz_source, norm_source)			# Apply transformation to original source.

			transforms.append(transform)
			all_gamma.append(torch.exp(self.affinity))
			all_perm_matrices.append(self.perm_matrix)
			all_weighted_template.append(weighted_template)
			all_beta.append(to_numpy(self.beta))
			all_alpha.append(to_numpy(self.alpha))

		est_T = convert2transformation(transforms[max_iterations-1][:, :3, :3], transforms[max_iterations-1][:, :3, 3])
		transformed_source = torch.bmm(est_T[:, :3, :3], source[:,:,:3].permute(0, 2, 1)).permute(0, 2, 1) + est_T[:, :3, 3].unsqueeze(1)

		result = {'est_R': est_T[:, :3, :3],				# source -> template
				  'est_t': est_T[:, :3,  3],				# source -> template
				  'est_T': est_T,			# source -> template
				  # 'r': self.feat_template - self.feat_source,
				  'transformed_source': transformed_source}
		
		result['perm_matrices_init'] = all_gamma
		result['perm_matrices'] = all_perm_matrices
		result['weighted_template'] = all_weighted_template
		result['beta'] = np.stack(all_beta, axis=0)
		result['alpha'] = np.stack(all_alpha, axis=0)
		result['transforms'] = transforms

		return result


if __name__ == '__main__':
	template, source = torch.rand(10,1024,6), torch.rand(10,1024,6)
	
	net = RPMNet()
	result = net(template, source)
	import ipdb; ipdb.set_trace()