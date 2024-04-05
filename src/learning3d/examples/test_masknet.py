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
	
from learning3d.models import MaskNet
from learning3d.data_utils import RegistrationData, ModelNet40Data

def pc2open3d(data):
	if torch.is_tensor(data): data = data.detach().cpu().numpy()
	if len(data.shape) == 2:
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(data)
		return pc
	else:
		print("Error in the shape of data given to Open3D!, Shape is ", data.shape)

def display_results(template, source, masked_template):
	template = pc2open3d(template)
	source = pc2open3d(source)
	masked_template = pc2open3d(masked_template)
	
	template.paint_uniform_color([1, 0, 0])
	source.paint_uniform_color([0, 1, 0])
	masked_template.paint_uniform_color([0, 0, 1])

	o3d.visualization.draw_geometries([template, source])
	o3d.visualization.draw_geometries([masked_template, source])

def evaluate_metrics(TP, FP, FN, TN, gt_mask):
	# TP, FP, FN, TN: 		True +ve, False +ve, False -ve, True -ve
	# gt_mask:				Ground Truth mask [Nt, 1]
	
	accuracy = (TP + TN)/gt_mask.shape[1]
	misclassification_rate = (FN + FP)/gt_mask.shape[1]
	# Precision: (What portion of positive identifications are actually correct?)
	precision = TP / (TP + FP)
	# Recall: (What portion of actual positives are identified correctly?)
	recall = TP / (TP + FN)

	fscore = (2*precision*recall) / (precision + recall)
	return accuracy, precision, recall, fscore

# Function used to evaluate the predicted mask with ground truth mask.
def evaluate_mask(gt_mask, predicted_mask, predicted_mask_idx):
	# gt_mask:					Ground Truth Mask [Nt, 1]
	# predicted_mask:			Mask predicted by network [Nt, 1]
	# predicted_mask_idx:		Point indices chosen by network [Ns, 1]

	if torch.is_tensor(gt_mask): gt_mask = gt_mask.detach().cpu().numpy()
	if torch.is_tensor(gt_mask): predicted_mask = predicted_mask.detach().cpu().numpy()
	if torch.is_tensor(predicted_mask_idx): predicted_mask_idx = predicted_mask_idx.detach().cpu().numpy()
	gt_mask, predicted_mask, predicted_mask_idx = gt_mask.reshape(1,-1), predicted_mask.reshape(1,-1), predicted_mask_idx.reshape(1,-1)
	
	gt_idx = np.where(gt_mask == 1)[1].reshape(1,-1) 				# Find indices of points which are actually in source.

	# TP + FP = number of source points.
	TP = np.intersect1d(predicted_mask_idx[0], gt_idx[0]).shape[0]			# is inliner and predicted as inlier (True Positive) 		(Find common indices in predicted_mask_idx, gt_idx)
	FP = len([x for x in predicted_mask_idx[0] if x not in gt_idx])			# isn't inlier but predicted as inlier (False Positive)
	FN = FP															# is inlier but predicted as outlier (False Negative) (due to binary classification)
	TN = gt_mask.shape[1] - gt_idx.shape[1] - FN 					# is outlier and predicted as outlier (True Negative)
	return evaluate_metrics(TP, FP, FN, TN, gt_mask)

def test_one_epoch(args, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	precision_list = []

	for i, data in enumerate(tqdm(test_loader)):
		template, source, igt, gt_mask = data

		template = template.to(args.device)
		source = source.to(args.device)
		igt = igt.to(args.device)						# [source] = [igt]*[template]
		gt_mask = gt_mask.to(args.device)

		masked_template, predicted_mask = model(template, source)
		
		# Evaluate mask based on classification metrics.
		accuracy, precision, recall, fscore = evaluate_mask(gt_mask, predicted_mask, predicted_mask_idx = model.mask_idx)
		precision_list.append(precision)
		
		# Different ways to visualize results.
		display_results(template.detach().cpu().numpy()[0], source.detach().cpu().numpy()[0], masked_template.detach().cpu().numpy()[0])

	print("Mean Precision: ", np.mean(precision_list))

def test(args, model, test_loader):
	test_one_epoch(args, model, test_loader)

def options():
	parser = argparse.ArgumentParser(description='MaskNet: A Fully-Convolutional Network For Inlier Estimation (Testing)')

	# settings for input data
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')
	parser.add_argument('--partial_source', default=True, type=bool,
						help='create partial source point cloud in dataset.')
	parser.add_argument('--noise', default=False, type=bool,
						help='Add noise in source point clouds.')
	parser.add_argument('--outliers', default=False, type=bool,
						help='Add outliers to template point cloud.')

	# settings for on testing
	parser.add_argument('-j', '--workers', default=1, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--test_batch_size', default=1, type=int,
						metavar='N', help='test-mini-batch size (default: 1)')
	parser.add_argument('--pretrained', default='learning3d/pretrained/exp_masknet/models/best_model.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')
	parser.add_argument('--unseen', default=False, type=bool,
						help='Use first 20 categories for training and last 20 for testing')

	args = parser.parse_args()
	return args

def main():
	args = options()
	torch.backends.cudnn.deterministic = True

	testset = RegistrationData('PointNetLK', ModelNet40Data(train=False, num_points=args.num_points),
									partial_source=args.partial_source, noise=args.noise,
									additional_params={'use_masknet': True})
	test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Load Pretrained MaskNet.
	model = MaskNet()
	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
	model = model.to(args.device)

	test(args, model, test_loader)

if __name__ == '__main__':
	main()