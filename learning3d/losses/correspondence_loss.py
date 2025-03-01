import torch

class CorrespondenceLoss(torch.nn.Module):
	def forward(self, template, source, corr_mat_pred, corr_mat):
		# corr_mat:			batch_size x num_template x num_source (ground truth correspondence matrix)
		# corr_mat_pred:	batch_size x num_source x num_template (predicted correspondence matrix)
		batch_size, _, num_points_template = template.shape
		_, _, num_points = source.shape
		return torch.nn.functional.cross_entropy(corr_mat_pred.view(batch_size*num_points, num_points_template), 
				torch.argmax(corr_mat.transpose(1,2).reshape(-1, num_points_template), axis=1))