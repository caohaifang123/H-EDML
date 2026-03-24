from __future__ import division
from __future__ import print_function
import datetime
import json
import logging
import os
import pickle
import time
import networkx as nx
import math
import random

import numpy as np
import optimizers
import geoopt.optim as optim
import torch
import torch.nn.functional as F
from config import parser

from models.base_models import NCModel_e, NCModel_h
from utils.data_utils import load_data, sparse_mx_to_torch_sparse_tensor
from utils.train_utils import get_dir_name, format_metrics
import geoopt
import geoopt.manifolds.poincare.math as pmath

from utils.eval_utils import acc_f1
from layers.layers import Attention
from scipy.stats import wasserstein_distance as wd


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train(args):
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    temp = args.temp

    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            # save_dir = os.path.join(args.log_dir, args.task, args.dataset)
            save_dir = os.path.join(args.log_dir, args.task, args.dataset, f"seed_{args.seed}")
        else:
            save_dir = args.save_dir
        
        os.makedirs(save_dir, exist_ok=True)

        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    data = load_data(args, os.path.join('./data', args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape

    if args.task == 'nc':
        Model_one = NCModel_h
        Model_two = NCModel_e
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model_one = Model_one(args)
    model_two = Model_two(args)
    logging.info(str(model_one))
    logging.info(str(model_two))
    optimizer_one = getattr(optimizers, args.optimizer_one)(params=model_one.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    optimizer_two = getattr(optimizers, args.optimizer_two)(params=model_two.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler_one = torch.optim.lr_scheduler.StepLR(
        optimizer_one,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    lr_scheduler_two = torch.optim.lr_scheduler.StepLR(
        optimizer_two,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model_one.parameters()]) + sum([np.prod(p.size()) for p in model_two.parameters()])

    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model_one = model_one.to(args.device)
        model_two = model_two.to(args.device)
        attention = Attention(args.input_channel).to(args.device)

        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    # Train model
    t_total = time.time()
    counter1 = 0
    counter2 = 0

    best_val_metrics_one = model_one.init_metric_dict()
    best_val_metrics_two = model_two.init_metric_dict()

    best_test_metrics_one = None
    best_test_metrics_two = None

    best_emb_one = None
    best_emb_two = None

    for epoch in range(args.epochs):
        t = time.time()
        model_one.train()
        model_two.train()

        optimizer_one.zero_grad()
        optimizer_two.zero_grad()

        embeddings_one = model_one.encode(data['features'], data['adj_train_norm'])        #feature
        embeddings_one_eucl = pmath.logmap0(embeddings_one, c=args.c)
        embeddings_two = model_two.encode(data['features'], data['adj_train_norm'])        #feature 
        embeddings_two_eucl = embeddings_two

        train_metrics_one = model_one.compute_metrics(embeddings_one, data, 'train', args)
        train_metrics_two = model_two.compute_metrics(embeddings_two, data, 'train', args)

        output_one = train_metrics_one['output']        #logits 
        output_two = train_metrics_two['output']        #logits 

        similarity_graph_one = torch.pow(torch.mm(embeddings_one_eucl, embeddings_one_eucl.t()), 2)
        similarity_graph_two = torch.pow(torch.mm(embeddings_two_eucl, embeddings_two_eucl.t()), 2)

        idx_train = data['idx_train']
        if args.dataset == 'pubmed':
            topology_one_train = similarity_graph_one.index_select(0, torch.tensor(idx_train).to(args.device))               #pubmed
            topology_two_train = similarity_graph_two.index_select(0, torch.tensor(idx_train).to(args.device))               #pubmed
        else:
            topology_one_train = similarity_graph_one
            topology_two_train = similarity_graph_two

        loss_one_decision = F.kl_div((output_one/temp).softmax(dim=-1).log(), (output_two/temp).softmax(dim=-1), reduction=args.reduction_D) #sum or batchmean
        loss_two_decision = F.kl_div((output_two/temp).softmax(dim=-1).log(), (output_one/temp).softmax(dim=-1), reduction=args.reduction_D) #sum or batchmean

        loss_one_topology = F.kl_div(topology_one_train.softmax(dim=-1).log(), topology_two_train.softmax(dim=-1), reduction=args.reduction_T) #sum or batchmean
        loss_two_topology = F.kl_div(topology_two_train.softmax(dim=-1).log(), topology_one_train.softmax(dim=-1), reduction=args.reduction_T) #sum or batchmean

        loss_one_super = train_metrics_one['loss']
        loss_two_super = train_metrics_two['loss']


        loss_one = args.alpha_D * loss_one_decision + loss_one_super + args.alpha_T * loss_one_topology
        loss_two = args.alpha_D * loss_two_decision + loss_two_super + args.alpha_T * loss_two_topology

        loss_one.backward(retain_graph=True)
        loss_two.backward()


        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            torch.nn.utils.clip_grad_norm_(model_one.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(model_two.parameters(), max_norm)
        
        optimizer_one.step()
        optimizer_two.step()

        lr_scheduler_one.step()
        lr_scheduler_two.step()

        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                'lr: {}'.format(lr_scheduler_one.get_last_lr()[0]),
                                format_metrics(train_metrics_one, 'train'),
                                'time: {:.4f}s'.format(time.time() - t)
                                ]))
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                'lr: {}'.format(lr_scheduler_two.get_last_lr()[0]),
                                format_metrics(train_metrics_two, 'train'),
                                'time: {:.4f}s'.format(time.time() - t)
                                ]))

        if (epoch + 1) % args.eval_freq == 0:
            model_one.eval()
            model_two.eval()

            embeddings_one = model_one.encode(data['features'], data['adj_train_norm'])       
            embeddings_two = model_two.encode(data['features'], data['adj_train_norm'])        

            val_metrics_one = model_one.compute_metrics(embeddings_one, data, 'val', args)
            val_metrics_two = model_two.compute_metrics(embeddings_two, data, 'val', args)

            val_output_one = val_metrics_one['output']
            val_output_two = val_metrics_two['output']

            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics_one, 'val')]))
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics_two, 'val')]))

            if model_one.has_improved(best_val_metrics_one, val_metrics_one):
                best_test_metrics_one = model_one.compute_metrics(embeddings_one, data, 'test', args)
                best_output_one = best_test_metrics_one['output']
                best_emb_one = embeddings_one.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings_one.npy'), best_emb_one.detach().numpy())
                    torch.save(model_one.state_dict(), os.path.join(save_dir, 'model_one.pth'))
                best_val_metrics_one = val_metrics_one
                counter1 = 0
            else:
                counter1 += 1

            if model_two.has_improved(best_val_metrics_two, val_metrics_two):
                best_test_metrics_two = model_two.compute_metrics(embeddings_two, data, 'test', args)
                best_output_two = best_test_metrics_two['output']
                best_emb_two = embeddings_two.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings_two.npy'), best_emb_two.detach().numpy())
                    torch.save(model_two.state_dict(), os.path.join(save_dir, 'model_two.pth'))
                best_val_metrics_two = val_metrics_two
                counter2 = 0
            else:
                counter2 += 1

            if counter1 >= args.patience and counter2 >= args.patience and epoch > args.min_epochs:
                logging.info("Early stopping")
                break

    logging.info("model_one and model_two Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    if not best_test_metrics_one:
        model_one.eval()
        best_emb_one = model_one.encode(data['features'], data['adj_train_norm'])
        best_test_metrics_one = model_one.compute_metrics(best_emb_one, data, 'test', args)
        best_output_one = best_test_metrics_one['output']

    if not best_test_metrics_two:
        model_two.eval()
        best_emb_two = model_two.encode(data['features'], data['adj_train_norm'])
        best_test_metrics_two = model_two.compute_metrics(best_emb_two, data, 'test', args)
        best_output_two = best_test_metrics_two['output']


def train_attention(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    temp = args.temp

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    data = load_data(args, os.path.join('./data', args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape

    if args.task == 'nc':
        Model_one = NCModel_h
        Model_two = NCModel_e
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs
    

    ckpt_dir = os.path.join(args.log_dir, args.task, args.dataset, f"seed_{args.seed}")

    model_one = Model_one(args)
    model_one.load_state_dict(torch.load(os.path.join(ckpt_dir, 'model_one.pth')))
    model_two = Model_two(args)
    model_two.load_state_dict(torch.load(os.path.join(ckpt_dir, 'model_two.pth')))
    
    for K,V in model_one.named_parameters():
        V.requires_grad = False

    for K,V in model_two.named_parameters():
        V.requires_grad = False

    attention = Attention(args.input_channel)

    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model_one = model_one.to(args.device)
        model_two = model_two.to(args.device)
        attention = attention.to(args.device)

        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    model_one.eval()
    model_two.eval()
    logging.info(str(attention))

    optimizer_attention = getattr(optimizers, args.optimizer_attention)(params=attention.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
   
    lr_scheduler_attention = torch.optim.lr_scheduler.StepLR(
        optimizer_attention,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )

    tot_params = sum([np.prod(p.size()) for p in attention.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    
    # Train attention model
    t_total = time.time()
    counter = 0

    best_val_acc = 0
    best_val_f1 = 0

    best_test_acc = 0
    best_test_f1 = 0

    best_prob = None

    for epoch in range(200):
        t = time.time()
        attention.train()
        optimizer_attention.zero_grad()

        embeddings_one = model_one.encode(data['features'], data['adj_train_norm'])        #feature
        embeddings_one_eucl = pmath.logmap0(embeddings_one, c=args.c)
        embeddings_two = model_two.encode(data['features'], data['adj_train_norm'])        #feature 
        embeddings_two_eucl = embeddings_two

        train_metrics_one = model_one.compute_metrics(embeddings_one, data, 'train', args)
        train_metrics_two = model_two.compute_metrics(embeddings_two, data, 'train', args)

        output_one = train_metrics_one['output']        #logits 
        output_two = train_metrics_two['output']        #logits 

        idx_train = data['idx_train']
        embeddings_one_train = embeddings_one.index_select(0, torch.tensor(idx_train).to(args.device))
        embeddings_one_eucl_train = embeddings_one_eucl.index_select(0, torch.tensor(idx_train).to(args.device))
        embeddings_two_train = embeddings_two.index_select(0, torch.tensor(idx_train).to(args.device))
        embeddings_two_eucl_train = embeddings_two_eucl.index_select(0, torch.tensor(idx_train).to(args.device))
        attention_matrix = attention(embeddings_one_eucl_train, embeddings_two_eucl_train)
        probs = torch.stack([output_one.softmax(dim=1), output_two.softmax(dim=1)], dim=1)
        probs = probs * attention_matrix[:, :, None]
        probs = torch.sum(probs, dim=1)
        probs_train = torch.log(probs)
        
        idx_train = data['idx_train']
        loss = F.nll_loss(probs_train, data['labels'][idx_train])
        loss.backward()
        optimizer_attention.step()
        lr_scheduler_attention.step()


        if (epoch + 1) % args.log_freq == 0:
            idx = data['idx_train']
            if args.n_classes > 2:
                acc, f1 = acc_f1(probs, data['labels'][idx_train], average='micro')
            else:
                acc, f1 = acc_f1(probs, data['labels'][idx_train], average='binary')
            print('train_acc_total:', acc)
            print('train_f1_total:', f1)

        if (epoch + 1) % args.eval_freq == 0:
            attention.eval()
            embeddings_one = model_one.encode(data['features'], data['adj_train_norm'])       
            embeddings_one_eucl = pmath.logmap0(embeddings_one, c=args.c)
            embeddings_two = model_two.encode(data['features'], data['adj_train_norm'])        
            embeddings_two_eucl = embeddings_two

            val_metrics_one = model_one.compute_metrics(embeddings_one, data, 'val', args)
            val_metrics_two = model_two.compute_metrics(embeddings_two, data, 'val', args)

            val_output_one = val_metrics_one['output']
            val_output_two = val_metrics_two['output']

            idx_val = data['idx_val']
            embeddings_one_val = embeddings_one.index_select(0, torch.tensor(idx_val).to(args.device))
            embeddings_one_eucl_val = embeddings_one_eucl.index_select(0, torch.tensor(idx_val).to(args.device))
            embeddings_two_val = embeddings_two.index_select(0, torch.tensor(idx_val).to(args.device))
            embeddings_two_eucl_val = embeddings_two_eucl.index_select(0, torch.tensor(idx_val).to(args.device))
            attention_matrix_val = attention(embeddings_one_eucl_val, embeddings_two_eucl_val)
            probs = torch.stack([val_output_one.softmax(dim=1), val_output_two.softmax(dim=1)], dim=1)
            probs = probs * attention_matrix_val[:, :, None]
            probs = torch.sum(probs, dim=1)
            probs_val = torch.log(probs)

            if args.n_classes > 2:
                acc_val, f1_val = acc_f1(probs_val, data['labels'][idx_val], average='micro')
            else:
                acc_val, f1_val = acc_f1(probs_val, data['labels'][idx_val], average='binary')

            if (epoch + 1) % args.log_freq == 0:
                print('val_acc_total:', acc_val)
                print('val_f1_total:', f1_val)
            if best_val_acc <= acc_val:
                best_val_acc = acc_val
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break

    logging.info("attention Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    embeddings_one = model_one.encode(data['features'], data['adj_train_norm'])       
    embeddings_one_eucl = pmath.logmap0(embeddings_one, c=args.c)
    embeddings_two = model_two.encode(data['features'], data['adj_train_norm'])        
    embeddings_two_eucl = embeddings_two

    test_metrics_one = model_one.compute_metrics(embeddings_one, data, 'test', args)
    test_metrics_two = model_two.compute_metrics(embeddings_two, data, 'test', args)

    test_output_one = test_metrics_one['output']
    test_output_two = test_metrics_two['output']

    idx_test = data['idx_test']
    embeddings_one_test = embeddings_one.index_select(0, torch.tensor(idx_test).to(args.device))
    embeddings_one_eucl_test = embeddings_one_eucl.index_select(0, torch.tensor(idx_test).to(args.device))
    embeddings_two_test = embeddings_two.index_select(0, torch.tensor(idx_test).to(args.device))
    embeddings_two_eucl_test = embeddings_two_eucl.index_select(0, torch.tensor(idx_test).to(args.device))
    attention_matrix_test = attention(embeddings_one_eucl_test, embeddings_two_eucl_test)
    probs = torch.stack([test_output_one.softmax(dim=1), test_output_two.softmax(dim=1)], dim=1)
    probs = probs * attention_matrix_test[:, :, None]
    probs = torch.sum(probs, dim=1)
    probs_test = torch.log(probs)

    if args.n_classes > 2:
        acc_test, f1_test = acc_f1(probs_test, data['labels'][idx_test], average='micro')
    else:
        acc_test, f1_test = acc_f1(probs_test, data['labels'][idx_test], average='binary')
    acc, f1 = acc_f1(probs_test, data['labels'][idx_test], average='micro')
    print('test_acc_total:', acc_test)
    print('test_f1_total:', f1_test)

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
    train_attention(args)
    print(args)
    import sys

    sys.exit(0)
