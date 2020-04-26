import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import PointNet
from .pooling import Pooling
from .. ops.transform_functions import PCRNetTransform as transform


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

	# Single Pass Alignment Module (SPAM)
	def spam(self, template_features, source, est_R, est_t):
		batch_size = source.size(0)

		self.source_features = self.pooling(self.feature_model(source))
		y = torch.cat([template_features, self.source_features], dim=1)
		pose_7d = self.linear(y)
		pose_7d = transform.create_pose_7d(pose_7d)

		# Find current rotation and translation.
		identity = torch.eye(3).to(source).view(1,3,3).expand(batch_size, 3, 3).contiguous()
		est_R_temp = transform.quaternion_rotate(identity, pose_7d).permute(0, 2, 1)
		est_t_temp = transform.get_translation(pose_7d).view(-1, 1, 3)

		# update translation matrix.
		est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
		# update rotation matrix.
		est_R = torch.bmm(est_R_temp, est_R)
		
		source = transform.quaternion_transform(source, pose_7d)      # Ps' = est_R*Ps + est_t
		return est_R, est_t, source

	def forward(self, template, source, max_iteration=8):
		est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()         # (Bx3x3)
		est_t = torch.zeros(1,3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()     # (Bx1x3)
		template_features = self.pooling(self.feature_model(template))

		if max_iteration == 1:
			est_R, est_t, source = self.spam(template_features, source, est_R, est_t)
		else:
			for i in range(max_iteration):
				est_R, est_t, source = self.spam(template_features, source, est_R, est_t)

		result = {'est_R': est_R,				# source -> template
				  'est_t': est_t,				# source -> template
				  'est_T': transform.convert2transformation(est_R, est_t),			# source -> template
				  'r': template_features - self.source_features,
				  'transformed_source': source}
		return result


if __name__ == '__main__':
	template, source = torch.rand(10,1024,3), torch.rand(10,1024,3)
	pn = PointNet()
	
	net = iPCRNet(pn)
	result = net(template, source)
	import ipdb; ipdb.set_trace()