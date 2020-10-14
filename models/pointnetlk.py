import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import PointNet
from .pooling import Pooling
from .. ops import data_utils
from .. ops import se3, so3, invmat


class PointNetLK(nn.Module):
	def __init__(self, feature_model=PointNet(), delta=1.0e-2, learn_delta=False, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True, pooling='max'):
		super().__init__()
		self.feature_model = feature_model
		self.pooling = Pooling(pooling)
		self.inverse = invmat.InvMatrix.apply
		self.exp = se3.Exp # [B, 6] -> [B, 4, 4]
		self.transform = se3.transform # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

		w1, w2, w3, v1, v2, v3 = delta, delta, delta, delta, delta, delta
		twist = torch.Tensor([w1, w2, w3, v1, v2, v3])
		self.dt = torch.nn.Parameter(twist.view(1, 6), requires_grad=learn_delta)

		# results
		self.last_err = None
		self.g_series = None # for debug purpose
		self.prev_r = None
		self.g = None # estimation result
		self.itr = 0
		self.xtol = xtol
		self.p0_zero_mean = p0_zero_mean
		self.p1_zero_mean = p1_zero_mean

	def forward(self, template, source, maxiter=10):
		template, source, template_mean, source_mean = data_utils.mean_shift(template, source, 
																			 self.p0_zero_mean, self.p1_zero_mean)

		result = self.iclk(template, source, maxiter)
		result = data_utils.postprocess_data(result, template, source, template_mean, source_mean, 
											 self.p0_zero_mean, self.p1_zero_mean)
		return result

	def iclk(self, template, source, maxiter):
		batch_size = template.size(0)

		est_T0 = torch.eye(4).to(template).view(1, 4, 4).expand(template.size(0), 4, 4).contiguous()
		est_T = est_T0
		self.est_T_series = torch.zeros(maxiter+1, *est_T0.size(), dtype=est_T0.dtype)
		self.est_T_series[0] = est_T0.clone()

		training = self.handle_batchNorm(template, source)

		# re-calc. with current modules
		template_features = self.pooling(self.feature_model(template)) # [B, N, 3] -> [B, K]

		# approx. J by finite difference
		dt = self.dt.to(template).expand(batch_size, 6)
		J = self.approx_Jic(template, template_features, dt)

		self.last_err = None
		pinv = self.compute_inverse_jacobian(J, template_features, source)
		if pinv == {}:
			result = {'est_R': est_T[:,0:3,0:3],
					  'est_t': est_T[:,0:3,3],
					  'est_T': est_T,
					  'r': None,
					  'transformed_source': self.transform(est_T.unsqueeze(1), source),
					  'itr': 1,
					  'est_T_series': self.est_T_series}
			return result

		itr = 0
		r = None
		for itr in range(maxiter):
			self.prev_r = r
			transformed_source = self.transform(est_T.unsqueeze(1), source) # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
			source_features = self.pooling(self.feature_model(transformed_source)) # [B, N, 3] -> [B, K]
			r = source_features - template_features

			pose = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

			check = pose.norm(p=2, dim=1, keepdim=True).max()
			if float(check) < self.xtol:
				if itr == 0:
					self.last_err = 0 # no update.
				break

			est_T = self.update(est_T, pose)
			self.est_T_series[itr+1] = est_T.clone()

		rep = len(range(itr, maxiter))
		self.est_T_series[(itr+1):] = est_T.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

		self.feature_model.train(training)
		self.est_T = est_T

		result = {'est_R': est_T[:,0:3,0:3],
				  'est_t': est_T[:,0:3,3],
				  'est_T': est_T,
				  'r': r,
				  'transformed_source': self.transform(est_T.unsqueeze(1), source),
				  'itr': itr+1,
				  'est_T_series': self.est_T_series}
		
		return result

	def update(self, g, dx):
		# [B, 4, 4] x [B, 6] -> [B, 4, 4]
		dg = self.exp(dx)
		return dg.matmul(g)

	def approx_Jic(self, template, template_features, dt):
		# p0: [B, N, 3], Variable
		# f0: [B, K], corresponding feature vector
		# dt: [B, 6], Variable
		# Jk = (feature_model(p(-delta[k], p0)) - f0) / delta[k]

		batch_size = template.size(0)
		num_points = template.size(1)

		# compute transforms
		transf = torch.zeros(batch_size, 6, 4, 4).to(template)
		for b in range(template.size(0)):
			d = torch.diag(dt[b, :]) # [6, 6]
			D = self.exp(-d) # [6, 4, 4]
			transf[b, :, :, :] = D[:, :, :]
		transf = transf.unsqueeze(2).contiguous()  #   [B, 6, 1, 4, 4]
		p = self.transform(transf, template.unsqueeze(1)) # x [B, 1, N, 3] -> [B, 6, N, 3]

		#f0 = self.feature_model(p0).unsqueeze(-1) # [B, K, 1]
		template_features = template_features.unsqueeze(-1) # [B, K, 1]
		f = self.pooling(self.feature_model(p.view(-1, num_points, 3))).view(batch_size, 6, -1).transpose(1, 2) # [B, K, 6]

		df = template_features - f # [B, K, 6]
		J = df / dt.unsqueeze(1)

		return J

	def compute_inverse_jacobian(self, J, template_features, source):
		# compute pinv(J) to solve J*x = -r
		try:
			Jt = J.transpose(1, 2) # [B, 6, K]
			H = Jt.bmm(J) # [B, 6, 6]
			B = self.inverse(H)
			pinv = B.bmm(Jt) # [B, 6, K]
			return pinv
		except RuntimeError as err:
			# singular...?
			self.last_err = err
			g = torch.eye(4).to(source).view(1, 4, 4).expand(source.size(0), 4, 4).contiguous()
			#print(err)
			# Perhaps we can use MP-inverse, but,...
			# probably, self.dt is way too small...
			source_features = self.pooling(self.feature_model(source)) # [B, N, 3] -> [B, K]
			r = source_features - template_features
			self.feature_model.train(self.feature_model.training)
			return {}

	def handle_batchNorm(self, template, source):
		training = self.feature_model.training
		if training:
			# first, update BatchNorm modules
			template_features, source_features = self.pooling(self.feature_model(template)), self.pooling(self.feature_model(source))
		self.feature_model.eval()	# and fix them.
		return training


if __name__ == '__main__':
	template, source = torch.rand(10,1024,3), torch.rand(10,1024,3)
	pn = PointNet()

	net = PointNetLK(pn)
	result = net(template, source)
	import ipdb; ipdb.set_trace()