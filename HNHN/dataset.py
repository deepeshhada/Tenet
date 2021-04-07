from collections import defaultdict
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GraphDataset(Dataset):
	def __init__(self, args):
		self.node_X = args.v
		self.v_weight = args.v_weight
		self.positive_vidx = args.vidx
		self.negatives = args.train_negatives
		self.num_ng = args.num_ng
		self.v_reg_wt = args.v_weight ** args.alpha_v
		self.embed_dim = args.embed_dim
		self.args = args

	def __len__(self):
		return self.args.train_len

	def __getitem__(self, index):
		vidx = self.positive_vidx[index*3:(index*3)+3]
		negatives = self.get_negative_instances(index)
		vidx = torch.cat((vidx, negatives))
		edge_x = self.node_X[vidx].view(-1, 3, self.embed_dim).sum(dim=1)
		v_reg_weight = self.v_reg_wt[vidx]

		return vidx, v_reg_weight, edge_x

	def get_negative_instances(self, index):
		return self.negatives[index*self.num_ng:(index*self.num_ng)+self.num_ng].view(-1)


class Collate:
	def __init__(self, args):
		self.args = args
		self.num_ng = args.num_ng
		self.v_reg_sum = args.v_reg_sum
		self.e_reg_wt = (1 / 3) ** args.alpha_e  # shape = 1

	def __call__(self, batch):
		batch_size = len(batch) * (self.num_ng + 1)
		eidx = torch.from_numpy(np.arange(batch_size)).repeat_interleave(3).long()
		e_reg_weight = (torch.ones(batch_size*3) * self.e_reg_wt).unsqueeze(-1)
		labels = torch.zeros(self.num_ng + 1)
		labels[0] = 1
		labels = labels.repeat(len(batch))

		vidx = batch[0][0]
		v_reg_weight = batch[0][1]
		edge_x = batch[0][2]
		for idx in range(1, len(batch)):
			vidx = torch.cat((vidx, batch[idx][0]))
			v_reg_weight = torch.cat((v_reg_weight, batch[idx][1]))
			edge_x = torch.cat((edge_x, batch[idx][2]))

		e_reg_sum = v_reg_weight.reshape((-1, 3)).sum(dim=1).unsqueeze(-1)

		return {
			"e": edge_x,
			"vidx": vidx.long(),
			"eidx": eidx,
			"v_reg_weight": v_reg_weight,
			"e_reg_weight": e_reg_weight,
			"v_reg_sum": self.v_reg_sum,
			"e_reg_sum": e_reg_sum,
			"labels": labels
		}


class GraphTestDataset(Dataset):
	def __init__(self, args, mode):
		data_path = os.path.join('HNHN', args.dataset_name + "_raw", mode + '_set.csv')
		test_points = []

		with open(data_path, 'r') as f:
			lines = f.readlines()
			for i, line in enumerate(lines):
				line = line.split('\t')
				test_points.append((int(line[0]), int(line[1]), int(line[2])))

		self.data_len = len(test_points)
		self.vidx = torch.tensor(test_points).view(-1)
		self.node_X = args.v
		self.v_weight = args.v_weight
		self.v_reg_wt = args.v_weight ** args.alpha_v
		self.embed_dim = args.embed_dim
		self.args = args

	def __len__(self):
		return self.data_len

	def __getitem__(self, index):
		vidx = self.vidx[index*3:(index*3)+3]
		v_reg_weight = self.v_reg_wt[vidx]
		edge_x = self.node_X[vidx].view(-1, 3, self.embed_dim).sum(dim=1)

		return vidx, v_reg_weight, edge_x


class CollateTest:
	def __init__(self, args):
		self.args = args
		self.num_ng = args.num_ng
		self.v_reg_sum = args.v_reg_sum
		self.e_reg_wt = (1 / 3) ** args.alpha_e

	def __call__(self, batch):
		batch_size = len(batch)
		eidx = torch.from_numpy(np.arange(batch_size)).repeat_interleave(3).long()
		e_reg_weight = (torch.ones(batch_size*3) * self.e_reg_wt).unsqueeze(-1)

		vidx = batch[0][0]
		v_reg_weight = batch[0][1]
		edge_x = batch[0][2]
		for idx in range(1, len(batch)):
			vidx = torch.cat((vidx, batch[idx][0]))
			v_reg_weight = torch.cat((v_reg_weight, batch[idx][1]))
			edge_x = torch.cat((edge_x, batch[idx][2]))

		e_reg_sum = v_reg_weight.reshape((-1, 3)).sum(dim=1).unsqueeze(-1)

		return {
			"e": edge_x,
			"vidx": vidx.long(),
			"eidx": eidx,
			"v_reg_weight": v_reg_weight,
			"e_reg_weight": e_reg_weight,
			"v_reg_sum": self.v_reg_sum,
			"e_reg_sum": e_reg_sum
		}
