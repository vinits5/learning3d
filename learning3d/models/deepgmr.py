'''
We thank the author of DeepGMR paper to open-source their code.
Modified by Vinit Sarode.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. ops import transform_functions as transform


def gmm_params(gamma, pts):
	'''
	Inputs:
		gamma: B x N x J
		pts: B x N x 3
	'''
	# pi: B x J
	pi = gamma.mean(dim=1)
	Npi = pi * gamma.shape[1]
	# mu: B x J x 3
	mu = gamma.transpose(1, 2) @ pts / Npi.unsqueeze(2)
	# diff: B x N x J x 3
	diff = pts.unsqueeze(2) - mu.unsqueeze(1)
	# sigma: B x J x 3 x 3
	eye = torch.eye(3).unsqueeze(0).unsqueeze(1).to(gamma.device)
	sigma = (
		((diff.unsqueeze(3) @ diff.unsqueeze(4)).squeeze() * gamma).sum(dim=1) / Npi
	).unsqueeze(2).unsqueeze(3) * eye
	return pi, mu, sigma


def gmm_register(pi_s, mu_s, mu_t, sigma_t):
	'''
	Inputs:
		pi: B x J
		mu: B x J x 3
		sigma: B x J x 3 x 3
	'''
	c_s = pi_s.unsqueeze(1) @ mu_s
	c_t = pi_s.unsqueeze(1) @ mu_t
	Ms = torch.sum((pi_s.unsqueeze(2) * (mu_s - c_s)).unsqueeze(3) @
				   (mu_t - c_t).unsqueeze(2) @ sigma_t.inverse(), dim=1)
	U, _, V = torch.svd(Ms.cpu())
	U = U.cuda() if torch.cuda.is_available() else U
	V = V.cuda() if torch.cuda.is_available() else V
	S = torch.eye(3).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
	S[:, 2, 2] = torch.det(V @ U.transpose(1, 2))
	R = V @ S @ U.transpose(1, 2)
	t = c_t.transpose(1, 2) - R @ c_s.transpose(1, 2)
	bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device)
	T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
	return T


class Conv1dBNReLU(nn.Sequential):
	def __init__(self, in_planes, out_planes):
		super(Conv1dBNReLU, self).__init__(
			nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False),
			nn.BatchNorm1d(out_planes),
			nn.ReLU(inplace=True))


class FCBNReLU(nn.Sequential):
	def __init__(self, in_planes, out_planes):
		super(FCBNReLU, self).__init__(
			nn.Linear(in_planes, out_planes, bias=False),
			nn.BatchNorm1d(out_planes),
			nn.ReLU(inplace=True))


class TNet(nn.Module):
	def __init__(self):
		super(TNet, self).__init__()
		self.encoder = nn.Sequential(
			Conv1dBNReLU(3, 64),
			Conv1dBNReLU(64, 128),
			Conv1dBNReLU(128, 256))
		self.decoder = nn.Sequential(
			FCBNReLU(256, 128),
			FCBNReLU(128, 64),
			nn.Linear(64, 6))

	@staticmethod
	def f2R(f):
		r1 = F.normalize(f[:, :3])
		proj = (r1.unsqueeze(1) @ f[:, 3:].unsqueeze(2)).squeeze(2)
		r2 = F.normalize(f[:, 3:] - proj * r1)
		r3 = r1.cross(r2)
		return torch.stack([r1, r2, r3], dim=2)

	def forward(self, pts):
		f = self.encoder(pts)
		f, _ = f.max(dim=2)
		f = self.decoder(f)
		R = self.f2R(f)
		return R @ pts


class PointNet(nn.Module):
	def __init__(self, use_rri, use_tnet=False, nearest_neighbors=20):
		super(PointNet, self).__init__()
		self.use_tnet = use_tnet
		self.tnet = TNet() if self.use_tnet else None
		d_input = nearest_neighbors * 4 if use_rri else 3
		self.encoder = nn.Sequential(
			Conv1dBNReLU(d_input, 64),
			Conv1dBNReLU(64, 128),
			Conv1dBNReLU(128, 256),
			Conv1dBNReLU(256, args.d_model))
		self.decoder = nn.Sequential(
			Conv1dBNReLU(args.d_model * 2, 512),
			Conv1dBNReLU(512, 256),
			Conv1dBNReLU(256, 128),
			nn.Conv1d(128, args.n_clusters, kernel_size=1))

	def forward(self, pts):
		pts = self.tnet(pts) if self.use_tnet else pts
		f_loc = self.encoder(pts)
		f_glob, _ = f_loc.max(dim=2)
		f_glob = f_glob.unsqueeze(2).expand_as(f_loc)
		y = self.decoder(torch.cat([f_loc, f_glob], dim=1))
		return y.transpose(1, 2)


class DeepGMR(nn.Module):
	def __init__(self, use_rri=True, feature_model=None, nearest_neighbors=20):
		super(DeepGMR, self).__init__()
		self.backbone = feature_model if not None else PointNet(use_rri=use_rri, nearest_neighbors=nearest_neighbors)
		self.use_rri = use_rri

	def forward(self, template, source):
		if self.use_rri:
			self.template = template[..., :3]
			self.source = source[..., :3]
			template_features = template[..., 3:].transpose(1, 2)
			source_features = source[..., 3:].transpose(1, 2)
		else:
			self.template = template
			self.source = source
			template_features = (template - template.mean(dim=2, keepdim=True)).transpose(1, 2)
			source_features = (source - source.mean(dim=2, keepdim=True)).transpose(1, 2)

		self.template_gamma = F.softmax(self.backbone(template_features), dim=2)
		self.template_pi, self.template_mu, self.template_sigma = gmm_params(self.template_gamma, self.template)
		self.source_gamma = F.softmax(self.backbone(source_features), dim=2)
		self.source_pi, self.source_mu, self.source_sigma = gmm_params(self.source_gamma, self.source)

		self.est_T_inverse = gmm_register(self.template_pi, self.template_mu, self.source_mu, self.source_sigma)
		self.est_T = gmm_register(self.source_pi, self.source_mu, self.template_mu, self.template_sigma) # [template = source * est_T]
		self.igt = igt				# [source = template * igt]

		transformed_source = transform.transform_point_cloud(source, est_T[:, :3, :3], est_T[:, :3, 3])

		result = {'est_R': est_T[:, :3, :3],
				  'est_t': est_T[:, :3, 3],
				  'est_R_inverse': est_T_inverse[:, :3, :3],
				  'est_t_inverese': est_T_inverse[:, :3, 3],
				  'est_T': est_T,
				  'est_T_inverse': est_T_inverse,
				  'r': template_features - source_features,
				  'transformed_source': transformed_source}

		return result
