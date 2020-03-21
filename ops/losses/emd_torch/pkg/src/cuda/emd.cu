#include <ATen/ATen.h>

#include <vector>

#include "cuda/emd.cuh"


std::vector<at::Tensor> emd_forward_cuda(
	at::Tensor xyz1, // B x N1 x D
	at::Tensor xyz2) // B x N2 x D
{
	// Some useful values
	const int batch_size = xyz1.size(0);
	const int num_pts_1 = xyz1.size(1);
	const int num_pts_2 = xyz2.size(1);

	// Allocate necessary data structures
	at::Tensor match = at::zeros({batch_size, num_pts_1, num_pts_2}, 
		xyz1.options());
	at::Tensor cost = at::zeros({batch_size}, xyz1.options());
	at::Tensor temp = at::zeros({batch_size, 2 * (num_pts_1 + num_pts_2)}, 
		xyz1.options());

	// Find the approximate matching
	approxmatchLauncher(
		batch_size, num_pts_1, num_pts_2,
		xyz1,
		xyz2, 
		match,
		temp
	);

	// Compute the matching cost
	matchcostLauncher(
		batch_size, num_pts_1, num_pts_2, 
		xyz1,
		xyz2, 
		match,
		cost
	);

	return {cost, match};
}

std::vector<at::Tensor> emd_backward_cuda(
	at::Tensor xyz1, 
	at::Tensor xyz2, 
	at::Tensor match)
{
	// Some useful values
	const int batch_size = xyz1.size(0);
	const int num_pts_1 = xyz1.size(1);
	const int num_pts_2 = xyz2.size(1);

	// Allocate necessary data structures
	at::Tensor grad_xyz1 = at::zeros_like(xyz1);
	at::Tensor grad_xyz2 = at::zeros_like(xyz2);

	// Compute the gradient with respect to the two inputs (xyz1 and xyz2)
	matchcostgradLauncher(
		batch_size, num_pts_1, num_pts_2,
		xyz1,
		xyz2,
		match,
		grad_xyz1,
		grad_xyz2
	);

	return {grad_xyz1, grad_xyz2};
}