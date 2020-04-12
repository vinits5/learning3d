import open3d as o3d
import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))
	
from learning3d.models import DGCNN, DCP
from learning3d.data_utils import RegistrationData, ModelNet40Data

def get_transformations(igt):
	R_ba = igt[:, 0:3, 0:3]                             # Ps = R_ba * Pt
	translation_ba = igt[:, 0:3, 3].unsqueeze(2)        # Ps = Pt + t_ba
	R_ab = R_ba.permute(0, 2, 1)                        # Pt = R_ab * Ps
	translation_ab = -torch.bmm(R_ab, translation_ba)   # Pt = Ps + t_ab
	return R_ab, translation_ab, R_ba, translation_ba

def display_open3d(template, source, transformed_source):
	template_ = o3d.geometry.PointCloud()
	source_ = o3d.geometry.PointCloud()
	transformed_source_ = o3d.geometry.PointCloud()
	template_.points = o3d.utility.Vector3dVector(template)
	source_.points = o3d.utility.Vector3dVector(source + np.array([0,0,0]))
	transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
	template_.paint_uniform_color([1, 0, 0])
	source_.paint_uniform_color([0, 1, 0])
	transformed_source_.paint_uniform_color([0, 0, 1])
	o3d.visualization.draw_geometries([template_, source_, transformed_source_])

def test_one_epoch(device, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(test_loader)):
		template, source, igt = data
		transformations = get_transformations(igt)
		transformations = [t.to(device) for t in transformations]
		R_ab, translation_ab, R_ba, translation_ba = transformations

		template = template.to(device)
		source = source.to(device)
		igt = igt.to(device)

		output = model(template, source)
		display_open3d(template.detach().cpu().numpy()[0], source.detach().cpu().numpy()[0], output['transformed_source'].detach().cpu().numpy()[0])

		identity = torch.eye(3).cuda().unsqueeze(0).repeat(template.shape[0], 1, 1)
		loss_val = torch.nn.functional.mse_loss(torch.matmul(output['est_R'].transpose(2, 1), R_ab), identity) \
			   + torch.nn.functional.mse_loss(output['est_t'], translation_ab[:,:,0])

		cycle_loss = torch.nn.functional.mse_loss(torch.matmul(output['est_R_'].transpose(2, 1), R_ba), identity) \
			   + torch.nn.functional.mse_loss(output['est_t_'], translation_ba[:,:,0])
		loss_val = loss_val + cycle_loss * 0.1

		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count
	return test_loss

def test(args, model, test_loader):
	test_loss, test_accuracy = test_one_epoch(args.device, model, test_loader)

def options():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')
	parser.add_argument('--exp_name', type=str, default='exp_ipcrnet', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--dataset_path', type=str, default='ModelNet40',
						metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

	# settings for input data
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')

	# settings for PointNet
	parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
						help='train pointnet (default: tune)')
	parser.add_argument('--emb_dims', default=512, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
						help='symmetric function (default: max)')

	# settings for on training
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=2, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--pretrained', default='learning3d/pretrained/exp_dcp/models/best_model.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args

def main():
	args = options()
	torch.backends.cudnn.deterministic = True
	
	trainset = RegistrationData('DCP', ModelNet40Data(train=True))
	testset = RegistrationData('DCP', ModelNet40Data(train=False))
	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
	test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Create PointNet Model.
	dgcnn = DGCNN(emb_dims=args.emb_dims)
	model = DCP(feature_model=dgcnn, cycle=True)
	model = model.to(args.device)

	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained), strict=False)
	model.to(args.device)

	test(args, model, test_loader)

if __name__ == '__main__':
	main()