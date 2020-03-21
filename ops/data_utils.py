import torch

def mean_shift(template, source, p0_zero_mean, p1_zero_mean):
	template_mean = torch.eye(3).view(1, 3, 3).expand(template.size(0), 3, 3).to(template) 		# [B, 3, 3]
	source_mean = torch.eye(3).view(1, 3, 3).expand(source.size(0), 3, 3).to(source) 			# [B, 3, 3]
	
	if p0_zero_mean:
		p0_m = template.mean(dim=1) # [B, N, 3] -> [B, 3]
		template_mean = torch.cat([template_mean, p0_m.unsqueeze(-1)], dim=2)
		one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(template_mean.shape[0], 1, 1).to(template_mean)    # (Bx1x4)
		template_mean = torch.cat([template_mean, one_], dim=1)
		template = template - p0_m.unsqueeze(1)
	# else:
		# q0 = template

	if p1_zero_mean:
		#print(numpy.any(numpy.isnan(p1.numpy())))
		p1_m = source.mean(dim=1) # [B, N, 3] -> [B, 3]
		source_mean = torch.cat([source_mean, -p0_m.unsqueeze(-1)], dim=2)
		one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(source_mean.shape[0], 1, 1).to(source_mean)    # (Bx1x4)
		source_mean = torch.cat([source_mean, one_], dim=1)
		source = source - p1_m.unsqueeze(1)
	# else:
		# q1 = source
	return template, source, template_mean, source_mean

def postprocess_data(result, p0, p1, a0, a1, p0_zero_mean, p1_zero_mean):
	#output' = trans(p0_m) * output * trans(-p1_m)
	#        = [I, p0_m;] * [R, t;] * [I, -p1_m;]
	#          [0, 1    ]   [0, 1 ]   [0,  1    ]
	est_g = result['est_T']
	if p0_zero_mean:
		est_g = a0.to(est_g).bmm(est_g)
	if p1_zero_mean:
		est_g = est_g.bmm(a1.to(est_g))
	result['est_T'] = est_g

	est_gs = result['est_T_series'] # [M, B, 4, 4]
	if p0_zero_mean:
		est_gs = a0.unsqueeze(0).contiguous().to(est_gs).matmul(est_gs)
	if p1_zero_mean:
		est_gs = est_gs.matmul(a1.unsqueeze(0).contiguous().to(est_gs))
	result['est_T_series'] = est_gs

	return result
