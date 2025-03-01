import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgcnn import DGCNN
from .pointnet import PointNet
from .. ops import transform_functions as transform
from .. utils import Transformer, SVDHead, Identity


class DCP(nn.Module):
	def __init__(self, feature_model=DGCNN(), cycle=False, pointer_='transformer', head='svd'):
		super(DCP, self).__init__()
		self.cycle = cycle
		self.emb_nn = feature_model

		if pointer_ == 'identity':
			self.pointer = Identity()
		elif pointer_ == 'transformer':
			self.pointer = Transformer(self.emb_nn.emb_dims, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)
		else:
			raise Exception("Not implemented")

		if head == 'mlp':
			self.head = MLPHead(self.emb_nn.emb_dims)
		elif head == 'svd':
			self.head = SVDHead(self.emb_nn.emb_dims)
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

		transformed_source = transform.transform_point_cloud(source, rotation_ab, translation_ab)

		result = {'est_R': rotation_ab,
				  'est_t': translation_ab,
				  'est_R_': rotation_ba,
				  'est_t_': translation_ba,
				  'est_T': transform.convert2transformation(rotation_ab, translation_ab),
				  'r': template_features - source_features,
				  'transformed_source': transformed_source}
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


if __name__ == '__main__':
	template, source = torch.rand(10,1024,3), torch.rand(10,1024,3)
	pn = PointNet()

	# Not Tested Yet.
	net = DCP(pn)
	result = net(template, source)
	import ipdb; ipdb.set_trace()