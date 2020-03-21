#ifndef EMD_H_
#define EMD_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"


std::vector<at::Tensor> emd_forward_cuda(
	at::Tensor xyz1,
	at::Tensor xyz2);

std::vector<at::Tensor> emd_backward_cuda(
	at::Tensor xyz1, 
	at::Tensor xyz2, 
	at::Tensor match);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

std::vector<at::Tensor> emd_forward(
	at::Tensor xyz1, 
	at::Tensor xyz2)
{
	CHECK_INPUT(xyz1);
	CHECK_INPUT(xyz2);

	return emd_forward_cuda(xyz1, xyz2);
}

std::vector<at::Tensor> emd_backward(
	at::Tensor xyz1, 
	at::Tensor xyz2,
	at::Tensor match)
{
	CHECK_INPUT(xyz1);
	CHECK_INPUT(xyz2);
	CHECK_INPUT(match);

	return emd_backward_cuda(xyz1, xyz2, match);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("emd_forward", &emd_forward, "Compute Earth Mover's Distance");
	m.def("emd_backward", &emd_backward, "Compute Gradients for Earth Mover's Distance");
}



#endif