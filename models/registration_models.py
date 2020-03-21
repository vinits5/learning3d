import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_models import DGCNN, PointNet, Pooling
from ops import transform_functions as transform
from ops import data_utils
from ops import se3, so3, invmat
from utils import transformer, svd

class DCP(nn.Module):
	def __init__(self, feature_model=PointNet(), cycle=False, pointer_='transformer', head='svd'):
		super(DCP, self).__init__()
		self.cycle = cycle
		self.emb_nn = feature_model

		if pointer_ == 'identity':
			self.pointer = transformer.Identity()
		elif pointer_ == 'transformer':
			self.pointer = transformer.Transformer(self.emb_nn.emb_dims, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)
		else:
			raise Exception("Not implemented")

		if head == 'mlp':
			self.head = MLPHead(self.emb_nn.emb_dims)
		elif head == 'svd':
			self.head = svd.SVDHead(self.emb_nn.emb_dims)
		else:
			raise Exception('Not implemented')

	def forward(self, template, source):
		source_features = self.emb_nn(source)
		template_features = self.emb_nn(template)

		source_features_p, template_features_p = self.pointer(source_features, template_features)

		source_features = source_features + source_features_p
		template_features = template_features + template_features_p

		rotation_ab, translation_ab = self.head(source_features, template_features, source, template)
		if self.cycle:
			rotation_ba, translation_ba = self.head(template_features, source_features, template, source)
		else:
			rotation_ba = rotation_ab.transpose(2, 1).contiguous()
			translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)

		transformed_source = transform_point_cloud(src, rotation_ab, translation_ab)

		result = {'est_R': rotation_ab,
				  'est_t': translation_ab,
				  'est_T': transform.convert2transformation(rotation_ab, translation_ab),
				  'r': template_features - source_features,
				  'transformed_source': transformed_source}
		return result


class iPCRNet(nn.Module):
	def __init__(self, feature_model=PointNet(), droput=0.0, pooling='max'):
		super().__init__()
		self.feature_model = feature_model
		self.pooling = Pooling(pooling)

		self.linear = [nn.Linear(self.feature_model.emb_dims * 2, 1024), nn.ReLU(),
				   	   nn.Linear(1024, 1024), nn.ReLU(),
				   	   nn.Linear(1024, 512), nn.ReLU(),
				   	   nn.Linear(512, 512), nn.ReLU(),
				   	   nn.Linear(512, 256), nn.ReLU()]

		if droput>0.0:
			self.linear.append(nn.Dropout(droput))
		self.linear.append(nn.Linear(256,7))

		self.linear = nn.Sequential(*self.linear)

	def forward(self, template, source, max_itr=8):
		est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()         # (Bx3x3)
		est_t = torch.zeros(1,3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()     # (Bx1x3)

		template_features = self.pooling(self.feature_model(template))

		# Iterations
		for i in range(max_itr):
			source_features = self.pooling(self.feature_model(source))
			y = torch.cat([template_features, source_features], dim=1)

			pose_7d = self.linear(y)
			pose_7d = transform.create_pose_7d(pose_7d)

			source = transform.quaternion_transform(source, pose_7d)      # Ps' = est_R*Ps + est_t
			est_t = transform.quaternion_rotate(est_t, pose_7d) + transform.get_translation(pose_7d).view(-1, 1, 3)
			est_R = transform.quaternion_rotate(est_R, pose_7d)

		result = {'est_R': est_R,
				  'est_t': est_t,
				  'est_T': transform.convert2transformation(est_R, est_t),
				  'r': template_features - source_features,
				  'transformed_source': source}
		
		return result


class MLPHead(nn.Module):
    def __init__(self, emb_dims):
        super(MLPHead, self).__init__()
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


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
			g = torch.eye(4).to(p0).view(1, 4, 4).expand(p0.size(0), 4, 4).contiguous()
			#print(err)
			# Perhaps we can use MP-inverse, but,...
			# probably, self.dt is way too small...
			source_features = self.pooling(self.feature_model(source)) # [B, N, 3] -> [B, K]
			r = source_features - template_features
			self.feature_model.train(training)
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
	
	net = iPCRNet(pn)
	result = net(template, source)

	# Not Tested Yet.
	# net = DCP(pn)
	# result = net(template, source)

	net = PointNetLK(pn)
	result = net(template, source)
	import ipdb; ipdb.set_trace()