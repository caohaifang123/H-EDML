import manifolds
import models.encoders as encoders
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.decoders import model2decoder
from layers.layers import FermiDiracDecoder
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.eval_utils import acc_f1


class BaseModel_e(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel_e, self).__init__()
        # self.manifold_name = args.manifold
        self.manifold_name = 'Euclidean'
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model_e)(self.c, args)

    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class BaseModel_h(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel_h, self).__init__()
        self.manifold_name = 'PoincareBall'
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model_h)(self.c, args)


    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel_e(BaseModel_e):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel_e, self).__init__(args)
        self.decoder = model2decoder[args.model_e](self.c, args)
        self.feat_dim = args.feat_dim
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        probs = F.log_softmax(output[idx], dim=1)
        output1 = output[idx]
        return probs, output1, output

    def compute_metrics(self, embeddings, data, split, args):
        idx = data[f'idx_{split}']
        probs, output, output1 = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(probs, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(probs, data['labels'][idx], average=self.f1_average)

        metrics = {'loss': loss, 'acc': acc, 'f1': f1, 'output': output}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class NCModel_h(BaseModel_h):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel_h, self).__init__(args)
        self.decoder = model2decoder[args.model_h](self.c, args)
        self.feat_dim = args.feat_dim
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        probs = F.log_softmax(output[idx], dim=1)
        output1 = output[idx]
        return probs, output1, output

    def compute_metrics(self, embeddings, data, split, args):
        idx = data[f'idx_{split}']
        probs, output, output1 = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(probs, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(probs, data['labels'][idx], average=self.f1_average)

        metrics = {'loss': loss, 'acc': acc, 'f1': f1, 'output': output}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]

