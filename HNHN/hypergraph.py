import os
import sys
import time
from collections import defaultdict
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset

from HNHN import hnhn_args_utils as utils, hnhn_data_utils as data_utils
from HNHN.dataset import GraphDataset, Collate, GraphTestDataset, CollateTest

device = utils.device


class HyperMod(nn.Module):
	def __init__(self, args, is_last=False):
		super(HyperMod, self).__init__()
		self.args = args
		self.v_weight = args.v_weight.to(device)

		self.W_v2e = Parameter(torch.randn(args.n_hidden, args.n_hidden))
		self.W_e2v = Parameter(torch.randn(args.n_hidden, args.n_hidden))
		self.b_v = Parameter(torch.zeros(args.n_hidden))
		self.b_e = Parameter(torch.zeros(args.n_hidden))
		self.is_last_mod = is_last

	def forward(self, v, e, vidx, eidx, v_reg_weight, e_reg_weight, v_reg_sum, e_reg_sum):
		ve = F.relu(torch.matmul(v, self.W_v2e) + self.b_v)
		v_fac = 4
		v = v * self.v_weight * v_fac

		expanded_eidx = eidx.unsqueeze(-1).expand(-1, self.args.n_hidden)

		e = e.clone()
		ve = (ve * self.v_weight)[vidx]
		ve *= v_reg_weight
		e = e.scatter_add(src=ve, index=expanded_eidx, dim=0)
		e /= e_reg_sum
		ev = F.relu(torch.matmul(e, self.W_e2v) + self.b_e)

		expanded_vidx = vidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
		ev_vtx = (ev/3)[eidx]
		ev_vtx *= e_reg_weight
		v = v.scatter_add(src=ev_vtx, index=expanded_vidx, dim=0)
		v /= v_reg_sum
		if not self.is_last_mod:
			v = F.dropout(v, args.dropout_p)

		return v, e


class Hypergraph(nn.Module):
	"""
	Hypergraph class, uses weights for vertex-edge and edge-vertex incidence matrix.
	One large graph.
	"""

	def __init__(self, args):
		"""
		vidx: idx tensor of elements to select, shape (ne, max_n),
		shifted by 1 to account for 0th elem (which is 0)
		eidx has shape (nv, max n)..
		"""
		super(Hypergraph, self).__init__()
		self.args = args
		self.hypermods = []

		for i in range(args.n_layers):
			is_last = True if i == args.n_layers - 1 else False
			self.hypermods.append(
				HyperMod(args, is_last=is_last))

		self.edge_lin = torch.nn.Linear(args.input_dim, args.n_hidden)

		self.vtx_lin = torch.nn.Linear(args.input_dim, args.n_hidden)
		self.affine_output = nn.Linear(args.n_hidden, 1)
		self.logistic = nn.Sigmoid()

	def to_device(self, device):
		self.to(device)
		for mod in self.hypermods:
			mod.to(device)
		return self

	def all_params(self):
		params = []
		for mod in self.hypermods:
			params.extend(mod.parameters())
		return params

	def forward(self, v, e, vidx, eidx, v_reg_weight, e_reg_weight, v_reg_sum, e_reg_sum):
		"""
			Take initial embeddings from the select labeled data.
			Return predicted cls.
		"""
		v = self.vtx_lin(v)
		e = self.edge_lin(e)
		for mod in self.hypermods:
			v, e = mod(v, e, vidx, eidx, v_reg_weight, e_reg_weight, v_reg_sum, e_reg_sum)

		logits = self.affine_output(e)
		pred = self.logistic(logits)  # pred is the implicit rating in (0, 1)
		return v, e, pred


class Hypertrain:
	def __init__(self, args):
		self.loss_fn = nn.BCELoss()  # consider logits

		self.hypergraph = Hypergraph(args)
		self.optim = optim.Adam(self.hypergraph.all_params(), lr=args.learning_rate)

		milestones = [100 * i for i in range(1, 4)]  # [100, 200, 300]
		self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=milestones, gamma=0.51)
		self.args = args

	def train_one_epoch(self, train_loader):
		self.hypergraph = self.hypergraph.to_device(device)
		v_init = self.args.v.to(device)
		print("="*75)
		# print(f"Epoch {epoch}/{self.args.n_epoch}")
		epoch_losses = []
		for data in tqdm(train_loader, position=0, leave=False):
			data = {key: val.to(device) for key, val in data.items()}
			v, e, pred = self.hypergraph(
				v_init, data['e'], data['vidx'], data['eidx'],
				data['v_reg_weight'], data['e_reg_weight'], data['v_reg_sum'], data['e_reg_sum']
			)
			loss = self.loss_fn(pred.squeeze(), data['labels'].float())
			epoch_losses.append(loss.item())

			loss.backward()
			self.optim.step()
			self.scheduler.step()
		epoch_losses = np.array(epoch_losses)
		print(f'train loss: {np.mean(epoch_losses)}')

		# test_err = self.eval(v_init, graph_test_loader)
		# print("test loss:", test_err)
		print("="*75)
		# if test_err < best_err:
		# 	best_err = test_err
		# return pred_all, loss, best_err

	def eval(self, v_init, data_loader):
		with torch.no_grad():
			preds, tgt = [], []
			for data in tqdm(data_loader, position=0, leave=False):
				data = {key: val.to(device) for key, val in data.items()}
				v, e, pred = self.hypergraph(
					v_init, data['e'], data['vidx'], data['eidx'],
					data['v_reg_weight'], data['e_reg_weight'], data['v_reg_sum'], data['e_reg_sum']
				)
				preds.extend(pred.cpu().detach().tolist())

			preds = torch.Tensor(preds).squeeze().to(device)
		return preds


def train(args):
	"""
		args.vidx, args.eidx, args.nv, args.ne, args = s
		args.e_weight = s
		args.v_weight = s
		label_idx, labels = s
	"""
	hypertrain = Hypertrain(args)
	hypertrain.train()

	return test_err


def gen_data(args, data_dict):
	"""
		Retrieve and process data, can be used generically for any dataset with predefined data format, eg cora, citeseer, etc.
		flip_edge_node: whether to flip edge and node in case of relation prediction.
	"""
	paper_author = torch.LongTensor(data_dict['paper_author'])
	n_author = data_dict['n_author']  # num users
	n_paper = data_dict['n_paper']  # num items
	classes = data_dict['classes']  # [0, 1]
	paper_X = data_dict['paper_X']  # item features

	train_len = data_dict['train_len']
	val_len = data_dict['val_len']
	test_len = data_dict['test_len']

	if args.predict_edge:
		# edge representations
		# author_X = data_dict['author_X']
		author_classes = data_dict['author_classes']

	paperwt = data_dict['paperwt']
	authorwt = data_dict['authorwt']
	cls_l = list(set(classes))

	args.edge_classes = torch.LongTensor(author_classes)

	args.input_dim = paper_X.shape[-1]  # 300 if args.dataset_name == 'citeseer' else 300
	args.n_hidden = 800 if args.predict_edge else 400
	args.final_edge_dim = 100
	args.ne = n_author
	args.nv = n_paper
	ne = args.ne
	nv = args.nv
	args.n_cls = len(cls_l)

	n_labels = ne
	args.all_labels = torch.LongTensor(args.edge_classes)
	args.label_idx = torch.from_numpy(np.arange(n_labels)).to(torch.int64)

	args.train_negatives = data_dict['train_negatives']

	print('\ngetting validation indices...')
	# val_idx = torch.from_numpy(np.arange(start=train_len, stop=train_len+test_len))
	val_idx = torch.from_numpy(np.arange(start=0, stop=train_len))
	args.val_idx = args.label_idx[val_idx.long()]
	args.val_labels = args.all_labels[args.val_idx]

	ones = torch.ones(len(args.label_idx))
	ones[args.val_idx] = -1

	args.label_idx = args.label_idx[ones > -1]
	args.labels = args.all_labels[args.label_idx]
	args.all_labels = args.all_labels

	if isinstance(paper_X, np.ndarray):
		args.v = torch.from_numpy(paper_X.astype(np.float32))
	else:
		args.v = torch.from_numpy(np.array(paper_X.astype(np.float32).todense()))

	args.vidx = paper_author[:, 0]
	args.eidx = paper_author[:, 1]
	args.paper_author = paper_author
	args.v_weight = torch.Tensor([(1 / w if w > 0 else 1) for w in paperwt]).unsqueeze(-1)  # torch.ones((nv, 1)) / 2 #####
	args.e_weight = torch.Tensor([(1 / w if w > 0 else 1) for w in authorwt]).unsqueeze(-1)  # 1)) / 2 #####torch.ones(ne, 1) / 3
	assert len(args.v_weight) == nv and len(args.e_weight) == ne

	paper2sum = defaultdict(list)
	e_reg_wt = args.e_weight[0] ** args.alpha_e
	# for i, (paper_idx, author_idx) in enumerate(paper_author.tolist()):
	# 	paper2sum[paper_idx].append(e_reg_wt)

	v_reg_sum = torch.zeros(nv)
	# for paper_idx, wt_l in paper2sum.items():
	# 	v_reg_sum[paper_idx] = sum(wt_l)
	#
	# v_reg_sum[v_reg_sum == 0] = 1
	# args.v_reg_sum = torch.Tensor(v_reg_sum).unsqueeze(-1)
	args.v_reg_sum = torch.rand((21035, 1))
	print('dataset processed into tensors')
	return args


def build_hypergraph():
	args = utils.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	root_path = os.path.join('HNHN/processed', args.dataset_name)
	data_dict = data_utils.load_data_dict(root_path)

	args = gen_data(args=args, data_dict=data_dict)
	hnhn = Hypertrain(args)

	train_dataset = GraphDataset(args=args)
	train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=Collate(args))

	valid_batch_size = args.valid_batch_size * args.valid_dim
	val_dataset = GraphTestDataset(args=args, mode="val")
	val_loader = DataLoader(dataset=val_dataset, batch_size=valid_batch_size, shuffle=False, collate_fn=CollateTest(args))
	test_dataset = GraphTestDataset(args=args, mode="test")
	test_loader = DataLoader(dataset=test_dataset, batch_size=valid_batch_size, shuffle=False, collate_fn=CollateTest(args))

	return args, hnhn, train_dataset, train_loader, val_loader, test_loader
