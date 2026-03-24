import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.01, 'learning rate'),
        'dropout': (0.4, 'dropout probability'),
        'cuda': (2, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (2000, 'maximum number of epochs to train'),
        # 'attention_epochs': (200, 'maximum number of epochs to train attention module'),
        'weight-decay': (0.001, 'l2 regularization strength'),
        'input-channel': (16, 'the input channel for training attention'),
        'optimizer-one': ('Adam', 'which optimizer to use for model_one, can be any of [Adam, RiemannianAdam]'),
        'optimizer-two': ('Adam', 'which optimizer to use for model_two, can be any of [Adam, RiemannianAdam]'),
        'optimizer-attention': ('Adam', 'which optimizer to use for attention, can be any of [Adam, RiemannianAdam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (200, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'temp':(1, 'temperature'),
        'log-freq': (50, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (1, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'log-dir': ('logs', ''),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs')
    },
    'model_config': {
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc]'),
        'model-e': ('GCN', 'which encoder to use for model_e, can be any of [Shallow, MLP, GCN, GAT]'),
        'model-h': ('HGAT', 'which encoder to use for model_h, can be any of [HNN, HGCN, HGNN, HGAT]'),
        'alpha-D': (0.5, 'The weight of Decision Mutual Learning loss'),
        'alpha-T': (0.1, 'The weight of Topology Mutual Learning loss'),
        'reduction-D': ('sum', 'The reduction of Decision Mutual Learning loss, can be any of [sum, batchmean]'),
        'reduction-T': ('sum', 'The reduction of Topology Mutual Learning loss, can be any of [sum, batchmean]'),
        'dim': (16, 'embedding dimension'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, PoincareBall]'),
        'input-type': ('eucl', '[eucl, hyper]'),
        'c': (1.0, 'hyperbolic radius'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n-heads': (8, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'concat': (1, 'concat of heads'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'),
        'drop-h': (0.9, 'dropout rate in hyperbolic'),
        'drop-e': (0., 'dropout rate in Euclidean'),
        'dist': (1, 'whether to use distance attention'),
    },
    'data_config': {
        'dataset': ('airport', 'which dataset to use'),
        # 'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        # 'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)