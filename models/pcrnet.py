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


if __name__ == '__main__':
	template, source = torch.rand(10,1024,3), torch.rand(10,1024,3)
	pn = PointNet()
	
	net = iPCRNet(pn)
	result = net(template, source)
	import ipdb; ipdb.set_trace()