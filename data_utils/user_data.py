import os
import numpy as np
import torch

class ClassificationData:
	def __init__(self, data_dict):
		self.data_dict = data_dict
		self.pcs = self.find_attribute('pcs')
		self.labels = self.find_attribute('labels')
		self.check_data()

	def find_attribute(self, attribute):
		try:
			attribute_data = self.data_dict[attribute]
		except:
			print("Given data directory has no key attribute \"{}\"".format(attribute))
		return attribute_data

	def check_data(self):
		assert 1 < len(self.pcs.shape) < 4, "Error in dimension of point clouds! Given data dimension: {}".format(self.pcs.shape)
		assert 0 < len(self.labels.shape) < 3, "Error in dimension of labels! Given data dimension: {}".format(self.labels.shape)
		
		if len(self.pcs.shape)==2: self.pcs = self.pcs.reshape(1, -1, 3)
		if len(self.labels.shape) == 1: self.labels = self.labels.reshape(1, -1)

		assert self.pcs.shape[0] == self.labels.shape[0], "Inconsistency in the number of point clouds and number of ground truth labels!"


	def __len__(self):
		return self.pcs.shape[0]

	def __getitem__(self, index):
		return torch.tensor(self.pcs[index]).float(), torch.from_numpy(self.labels[idx]).type(torch.LongTensor)


class RegistrationData:
	def __init__(self, data_dict):
		self.data_dict = data_dict
		self.template = self.find_attribute('template')
		self.source = self.find_attribute('source')
		self.transformation = self.find_attribute('transformation')
		self.check_data()

	def find_attribute(self, attribute):
		try:
			attribute_data = self.data[attribute]
		except:
			print("Given data directory has no key attribute \"{}\"".format(attribute))
		return attribute_data

	def check_data(self):
		assert 1 < len(self.template.shape) < 4, "Error in dimension of point clouds! Given data dimension: {}".format(self.template.shape)
		assert 1 < len(self.source.shape) < 4, "Error in dimension of point clouds! Given data dimension: {}".format(self.source.shape)
		assert 1 < len(self.transformation.shape) < 4, "Error in dimension of transformations! Given data dimension: {}".format(self.transformation.shape)

		if len(self.template.shape)==2: self.template = self.template.reshape(1, -1, 3)
		if len(self.source.shape)==2: self.source = self.source.reshape(1, -1, 3)
		if len(self.transformation.shape) == 2: self.transformation = self.transformation.reshape(1, 4, 4)

		assert self.template.shape[0] == self.source.shape[0], "Inconsistency in the number of template and source point clouds!"
		assert self.source.shape[0] == self.transformation.shape[0], "Inconsistency in the number of transformation and source point clouds!"

	def __len__(self):
		return self.template.shape[0]

	def __getitem__(self, index):
		return torch.tensor(self.template[index]).float(), torch.tensor(self.source[index]).float(), torch.tensor(self.transformation[index]).float()


class FlowData:
	def __init__(self, data_dict):
		self.data_dict = data_dict
		self.frame1 = self.find_attribute('frame1')
		self.frame2 = self.find_attribute('frame2')
		self.flow = self.find_attribute('flow')
		self.check_data()

	def find_attribute(self, attribute):
		try:
			attribute_data = self.data[attribute]
		except:
			print("Given data directory has no key attribute \"{}\"".format(attribute))
		return attribute_data

	def check_data(self):
		assert 1 < len(self.frame1.shape) < 4, "Error in dimension of point clouds! Given data dimension: {}".format(self.frame1.shape)
		assert 1 < len(self.frame2.shape) < 4, "Error in dimension of point clouds! Given data dimension: {}".format(self.frame2.shape)
		assert 1 < len(self.flow.shape) < 4, "Error in dimension of flow! Given data dimension: {}".format(self.flow.shape)

		if len(self.frame1.shape)==2: self.frame1 = self.frame1.reshape(1, -1, 3)
		if len(self.frame2.shape)==2: self.frame2 = self.frame2.reshape(1, -1, 3)
		if len(self.flow.shape) == 2: self.flow = self.flow.reshape(1, -1, 3)

		assert self.frame1.shape[0] == self.frame2.shape[0], "Inconsistency in the number of frame1 and frame2 point clouds!"
		assert self.frame2.shape[0] == self.flow.shape[0], "Inconsistency in the number of flow and frame2 point clouds!"

	def __len__(self):
		return self.frame1.shape[0]

	def __getitem__(self, index):
		return torch.tensor(self.frame1[index]).float(), torch.tensor(self.frame2[index]).float(), torch.tensor(self.flow[index]).float()


class UserData:
	def __init__(self, application, data_dict):
		self.application = application

		if self.application == 'classification':
			self.data_class = ClassificationData(data_dict)
		elif self.application == 'registration':
			self.data_class = RegistrationData(data_dict)
		elif self.application == 'flow_estimation':
			self.data_class = FlowData(data_dict)

	def __len__(self):
		return len(self.data_class)

	def __getitem__(self, index):
		return self.data_class[index]
