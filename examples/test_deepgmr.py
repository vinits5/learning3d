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
	
from learning3d.models import DeepGMR
from learning3d.data_utils import RegistrationData, ModelNet40Data

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

def rotation_error(R, R_gt):
	cos_theta = (torch.einsum('bij,bij->b', R, R_gt) - 1) / 2
	cos_theta = torch.clamp(cos_theta, -1, 1)
	return torch.acos(cos_theta) * 180 / math.pi

def translation_error(t, t_gt):
	return torch.norm(t - t_gt, dim=1)

def rmse(pts, T, T_gt):
	pts_pred = pts @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3].unsqueeze(1)
	pts_gt = pts @ T_gt[:, :3, :3].transpose(1, 2) + T_gt[:, :3, 3].unsqueeze(1)
	return torch.norm(pts_pred - pts_gt, dim=2).mean(dim=1)

def test_one_epoch(device, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	rotation_errors, translation_errors, rmses = [], [], []

	for i, data in enumerate(tqdm(test_loader)):
		template, source, igt = data

		template = template.to(device)
		source = source.to(device)
		igt = igt.to(device)

		output = model(template, source)
		display_open3d(template.detach().cpu().numpy()[0, :, :3], source.detach().cpu().numpy()[0, :, :3], output['transformed_source'].detach().cpu().numpy()[0])

		eye = torch.eye(4).expand_as(igt).to(igt.device)
		mse1 = F.mse_loss(output['est_T_inverse'] @ torch.inverse(igt), eye)
		mse2 = F.mse_loss(output['est_T'] @ igt, eye)
		loss = mse1 + mse2

		r_err = rotation_error(est_T_inverse[:, :3, :3], igt[:, :3, :3])
		t_err = translation_error(est_T_inverse[:, :3, 3], igt[:, :3, 3])
		rmse_val = rmse(template[:, :100], est_T_inverse, igt)
		rotation_errors.append(r_err)
		translation_errors.append(t_err)
		rmses.append(rmse_val)

		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count
	print("Mean rotation error: {}, Mean translation error: {} and Mean RMSE: {}".format(np.mean(rotation_errors), np.mean(translation_errors), np.mean(rmses)))
	return test_loss

def test(args, model, test_loader):
	test_loss = test_one_epoch(args.device, model, test_loader)

def options():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')
	parser.add_argument('--exp_name', type=str, default='exp_deepgmr', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--dataset_path', type=str, default='ModelNet40',
						metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

	# settings for input data
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')

	parser.add_argument('--nearest_neighbors', default=20, type=int,
						metavar='K', help='No of nearest neighbors to be estimated.')
	parser.add_argument('--use_rri', default=True, type=bool,
						help='Find nearest neighbors to estimate features from PointNet.')

	# settings for on training
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=2, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--pretrained', default='learning3d/pretrained/exp_deepgmr/models/best_model.pth', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args

def main():
	args = options()
	torch.backends.cudnn.deterministic = True
	
	trainset = RegistrationData('DeepGMR', ModelNet40Data(train=True))
	testset = RegistrationData('DeepGMR', ModelNet40Data(train=False))
	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
	test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	model = DeepGMR(use_rri=args.use_rri, nearest_neighbors=args.nearest_neighbors)
	model = model.to(args.device)

	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained), strict=False)
	model.to(args.device)

	test(args, model, test_loader)

if __name__ == '__main__':
	main()