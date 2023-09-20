# -*- coding: utf-8 -*-
"""GRU_sequence+attention.ipynb
# Classifying OUV using GRU sequence model + Attention

## Imports
"""

import sys
sys.executable
import os

import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp

from torch_geometric.data import (
    HeteroData,
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)

from argparse import Namespace
from collections import Counter
import json
import re
import string

import pandas as pd

import random

import torch
from torch.nn import DataParallel

from torch_geometric.transforms import RandomLinkSplit, ToUndirected
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear, to_hetero
from torch_geometric.utils import to_undirected

import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
import torch.optim as optim
import pickle

from torch_geometric.nn import MLP

from torch_geometric.loader import NeighborLoader

print("PyTorch version {}".format(torch.__version__))
print("GPU-enabled installation? {}".format(torch.cuda.is_available()))

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

args = Namespace(
    # Data and Path information
    path = 'dataset/Venice',
    save_dir='model_storage/MLP/',
    model_state_file='model.pth',
    
    # Model hyper parameters
    hidden_channels = 128,
    num_layers = 3,
    k=3,
    
    # Training hyper parameters
    sample_nodes = 25,
    batch_size=32,
    early_stopping_criteria=30,
    learning_rate=0.0005,
    l2=2e-4,
    dropout_p=0.2,
    num_epochs=300,
    seed=42,
    
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=True,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
)

class VEN_Homo(InMemoryDataset):
    r"""A subset of Flickr post collected in Venice annotated with Heritage 
    Values and Attributes, as collected in the `"Heri-Graphs: A Workflow of 
    Creating Datasets for Multi-modal Machine Learning on Graphs of Heritage 
    Values and Attributes with Social Media" <https://arxiv.org/abs/2205.07545>`
    paper.
    VEN_Homo is a homogeneous graph containing 2951 nodes and 1,071,977 links.
    Vis_only nodes are represented with 982-dimensional visual features and are
    divided into 9 heritage attribute categories 
    ('architectural elements', 'form', 'gastronomy', 'interior',
    'landscape scenery and natural features', 'monuments', 'people', 'product', 
    'urban scenery').
    Vis_text nodes are represented with 1753-dimensional visual and textual 
    features and are divided into 9 heritage attribute categories plus 11 
    heritage value categories ('criterion i-x', 'other').
    Both types of nodes are also merged into a single type of node 'all' with 
    1753-dimensional features and 20-dimensional label categories.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    
    Stats:
            * - #nodes
              - #edges
              - #features
              - #classes
            * - 2,951
              - 1,071,977
              - 1753
              - 20
    """

    url = 'https://drive.google.com/uc?export=download&id=1sxcKiZr1YGDv06wr03nsk5HVZledgzi9'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> List[str]:
        return [
            'A_simp.npz', 'A_SOC.npz', 'A_SPA.npz', 'A_TEM.npz', 'labels.npz',
            'node_types.npy', 'Textual_Features.npy', 'train_val_test_idx.npz',
            'Visual_Features.npy'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        data = Data()

        link_types = ['simp']

        vis = np.load(osp.join(self.raw_dir, 'Visual_Features.npy'),allow_pickle=True)[:,2:].astype(float)
        tex = np.load(osp.join(self.raw_dir, 'Textual_Features.npy'),allow_pickle=True)[:,5:].astype(float)

        x = np.hstack([vis,np.nan_to_num(tex)])

        node_type_idx = np.load(osp.join(self.raw_dir, 'node_types.npy'))
        node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)

        data.num_nodes = len(node_type_idx)

        data.x = torch.from_numpy(x).to(torch.float)


        y_s = np.load(osp.join(self.raw_dir, 'labels.npz'), allow_pickle=True)
        att_lab = y_s['ATT_LAB'][:,1:10].astype(float)
        val_lab = np.nan_to_num(y_s['VAL_LAB'][:,2:13].astype(float))
        ys = np.hstack([att_lab, val_lab])

        data.y = torch.from_numpy(ys).to(torch.float)

        data.node_type = node_type_idx

        split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        for name in ['train', 'val', 'test']:
            idx = split[f'{name}_idx']
            idx = torch.from_numpy(idx).to(torch.long)
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            data[f'{name}_mask'] = mask
                    
        s = {}
        
        for link in link_types:
            A_sub = sp.load_npz(osp.join(self.raw_dir, f'A_{link}.npz')).tocoo()
            if A_sub.nnz>0:
                row = torch.from_numpy(A_sub.row).to(torch.long)
                col = torch.from_numpy(A_sub.col).to(torch.long)
                data.edge_index = torch.stack([row, col], dim=0)
                data.edge_attr = torch.from_numpy(A_sub.data).to(torch.long)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'



"""## The Model: MLP Model"""

"""## Training Routine"""

def train_Homo(model, optimizer, train_loader):
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = args.batch_size
        out = model(batch.x)[:batch_size]
        out_att = out[:,:9]
        out_val = out[:,9:]
        y = batch.y
        y_att = y[:,:9]
        y_val = y[:,9:]
        
        loss = F.cross_entropy(out_att, y_att[:batch_size]) + F.cross_entropy(out_val, y_val[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples

@torch.no_grad()
def test_Homo(model, loader):
    model.eval()

    total_examples = 0
    running_loss_1 = running_loss_2 = 0.
    running_1_acc = 0.
    running_k_acc = 0.
    running_k_jac = 0.
    
    for batch in tqdm(loader):
        batch = batch.to(device)
        batch_size = batch.batch_size
        out = model(batch.x)[:batch_size]
        out_att = out[:,:9]
        out_val = out[:,9:]
        
        #pred_att = out_att.argmax(dim=-1)
        #pred_val = out_val.argmax(dim=-1)
        
        y = batch.y
        y_att = y[:,:9]
        y_val = y[:,9:]
        
        loss_1 = F.cross_entropy(out_att, y_att[:batch_size])
        loss_2 = F.cross_entropy(out_val, y_val[:batch_size])
        #loss_3 = loss_1 + loss_2
        
        acc_1_t = compute_1_accuracy(y_att[:batch_size], out_att)
        acc_k_t = compute_k_accuracy(y_val[:batch_size], out_val, args.k)
        jac_k_t = compute_jaccard_index(y_val[:batch_size], out_val, args.k)
        
        total_examples += batch_size
        #total_correct_att += int((pred_att == y_att[:batch_size]).sum())
        #total_correct_val += int((pred_val == y_val[:batch_size]).sum())
        
        running_loss_1 += float(loss_1) * batch_size
        running_loss_2 += float(loss_2) * batch_size
        running_1_acc += float(acc_1_t) * batch_size
        running_k_acc += float(acc_k_t) * batch_size
        running_k_jac += float(jac_k_t) * batch_size

    return running_loss_1/total_examples, running_loss_2/total_examples, running_1_acc/ total_examples, running_k_acc/ total_examples, running_k_jac/ total_examples, total_examples

"""### Helper Functions
"""

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_ATT_acc_val': 0,
            'early_stopping_best_VAL_acc_val': 0,
            'early_stopping_best_ATT_acc_val_2': 0,
            'early_stopping_lowest_loss': 1000,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_ATT_loss': [],
            'train_VAL_loss':[],
            'train_ATT_acc': [],
            'train_VAL_acc': [],
            'train_VAL_jac': [],
            'val_loss': [],
            'val_ATT_loss': [],
            'val_VAL_loss':[],
            'val_ATT_acc': [],
            'val_ATT_acc_2': [],
            'val_VAL_acc': [],
            'val_VAL_jac': [],
            'test_loss': -1,
            'test_ATT_loss': -1,
            'test_VAL_loss':-1,
            'test_ATT_acc': -1,
            'test_ATT_acc_2': -1,
            'test_VAL_acc': -1,
            'test_VAL_jac': -1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        ATT_acc_tm1, ATT_acc_t = train_state['val_ATT_acc'][-2:]
        #ATT_acc_2_tm1, ATT_acc_2_t = train_state['val_ATT_acc_2'][-2:]
        VAL_acc_tm1, VAL_acc_t = train_state['val_VAL_acc'][-2:]
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If accuracy worsened
        #if loss_t >= train_state['early_stopping_lowest_loss']:
        #    train_state['early_stopping_step'] += 1
        
        if ATT_acc_t <= train_state['early_stopping_best_ATT_acc_val'] and VAL_acc_t <= train_state['early_stopping_best_VAL_acc_val']:# and ATT_acc_2_t <= train_state['early_stopping_best_ATT_acc_val_2']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model from sklearn
            if VAL_acc_t > train_state['early_stopping_best_VAL_acc_val']:
                train_state['early_stopping_best_VAL_acc_val'] = VAL_acc_t
                
            if ATT_acc_t > train_state['early_stopping_best_ATT_acc_val']:
                train_state['early_stopping_best_ATT_acc_val'] = ATT_acc_t
            
            #if ATT_acc_2_t > train_state['early_stopping_best_ATT_acc_val_2']:
            #    train_state['early_stopping_best_ATT_acc_val_2'] = ATT_acc_2_t
                
            if loss_t < train_state['early_stopping_lowest_loss']:
                train_state['early_stopping_lowest_loss'] = loss_t
                torch.save(model.state_dict(), train_state['model_filename'])
                
            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

"""### Evaluation Metrics"""

def compute_cross_entropy(y_pred, y_target):
    y_target = y_target.cpu().float()
    y_pred = y_pred.cpu().float()
    criterion = nn.BCEWithLogitsLoss()
    return criterion(y_target, y_pred)

def compute_1_accuracy(y_pred, y_target):
    y_target_indices = y_target.max(dim=1)[1]
    y_pred_indices = y_pred.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target_indices).sum().item()
    return n_correct / len(y_pred_indices) * 100

def compute_k_accuracy(y_pred, y_target, k=3):
    y_pred_indices = y_pred.topk(k, dim=1)[1]
    y_target_indices = y_target.max(dim=1)[1]
    n_correct = torch.tensor([y_pred_indices[i] in y_target_indices[i] for i in range(len(y_pred))]).sum().item()
    return n_correct / len(y_pred_indices) * 100

def compute_k_jaccard_index(y_pred, y_target, k=3):
    y_target_indices = y_target.topk(k, dim=1)[1]
    y_pred_indices = y_pred.max(dim=1)[1]
    jaccard = torch.tensor([len(np.intersect1d(y_target_indices[i], y_pred_indices[i]))/
                            len(np.union1d(y_target_indices[i], y_pred_indices[i]))
                            for i in range(len(y_pred))]).sum().item()
    return jaccard / len(y_pred_indices)

def compute_jaccard_index(y_pred, y_target, k=3, multilabel=False):
    
    threshold = 1.0/(k+1)
    threshold_2 = 0.5
    
    if multilabel:
        y_pred_indices = y_pred.gt(threshold_2)
    else:
        y_pred_indices = y_pred.gt(threshold)
    
    y_target_indices = y_target.gt(threshold)
        
    jaccard = ((y_target_indices*y_pred_indices).sum(axis=1)/((y_target_indices+y_pred_indices).sum(axis=1)+1e-8)).sum().item()
    return jaccard / len(y_pred_indices)

# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = args.device
    return torch.from_numpy(df.values).float().to(device)

def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

"""### General Utilities"""

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if cuda:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def initialization():
    set_seed_everywhere(args.seed, args.cuda)
    #transform = T.Compose([T.ToSparseTensor()])
    dataset = VEN_Homo('dataset/Venice_homo')
    data = dataset[0].to(device)
    
    train_loader = NeighborLoader(
        data,
        # Sample 25 neighbors for each node and edge type for 2 iterations
        num_neighbors=[3*args.sample_nodes] * 2,
        # Use a batch size of 32 for sampling training nodes
        batch_size=args.batch_size,
        input_nodes=data.train_mask,
    )
    val_loader = NeighborLoader(
        data,
        # Sample 25 neighbors for each node and edge type for 2 iterations
        num_neighbors=[3*args.sample_nodes] * 2,
        # Use a batch size of 32 for sampling validating nodes
        batch_size=args.batch_size,
        input_nodes=data.val_mask,
    )
    test_loader = NeighborLoader(
        data,
        # Sample 25 neighbors for each node and edge type for 2 iterations
        num_neighbors=[3*args.sample_nodes] * 2,
        # Use a batch size of 32 for sampling testing nodes
        batch_size=args.batch_size,
        input_nodes=data.test_mask,
    )
 
    model = MLP(in_channels=data.x.shape[-1], hidden_channels = args.hidden_channels, 
            out_channels = data.y.shape[-1], dropout = args.dropout_p, num_layers=args.num_layers).to(device)
    return data, model, train_loader, val_loader, test_loader

def training_loop(verbose=False):
    
    _, model, train_loader, val_loader, test_loader = initialization()
    if torch.cuda.device_count() > 1:
        print("Use {} GPUs !".format(torch.cuda.device_count()))
        model = DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                           mode='min', factor=0.5,
    #                                           patience=1)

    train_state = make_train_state(args)

    try:
        for epoch in range(args.num_epochs):
            train_state['epoch_index'] = epoch
            
            loss = train_Homo(model, optimizer, train_loader)
            train_loss_att, train_loss_val, train_att_acc, train_val_acc, train_val_jac, _ = test_Homo(model, train_loader)
            val_loss_att, val_loss_val, val_att_acc, val_val_acc, val_val_jac, num_val_1 = test_Homo(model, val_loader)
            if verbose:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train_ATT: {train_att_acc:.4f}, Train_VAL: {train_val_acc:.4f}, Val_vis_tex_ATT: {val_att_acc:.4f}, Val_vis_tex_VAL: {val_val_acc:.4f}')
            
            train_state['train_loss'].append(loss)
            train_state['train_ATT_loss'].append(train_loss_att)
            train_state['train_VAL_loss'].append(train_loss_val)
            train_state['train_ATT_acc'].append(train_att_acc)
            train_state['train_VAL_acc'].append(train_val_acc)
            train_state['train_VAL_jac'].append(train_val_jac)
            
            train_state['val_ATT_loss'].append(val_loss_att)
            train_state['val_VAL_loss'].append(val_loss_val)
            train_state['val_loss'].append(val_loss_att + 3*val_loss_val)
            train_state['val_ATT_acc'].append(val_att_acc)
            train_state['val_VAL_acc'].append(val_val_acc)
            train_state['val_VAL_jac'].append(val_val_jac)
            
            train_state = update_train_state(args=args, model=model,
                                                train_state=train_state)
            if train_state['stop_early']:
                break

    except KeyboardInterrupt:
        print("Exiting loop")
        pass
    
    return train_state

def update_best_config(current_best,train_state, key):
    loss = train_state['early_stopping_lowest_loss']
    val_VAL_acc = train_state['early_stopping_best_VAL_acc_val']
    val_ATT_acc = train_state['early_stopping_best_ATT_acc_val']
    if loss < current_best['loss']:
        current_best['loss'] = loss
        current_best['ATT'] = val_ATT_acc
        current_best['VAL'] = val_VAL_acc
        current_best['args'] = str(args)
        current_best['key'] = key

def Hypersearch(hyperdict, current_best, verbose):
    '''
    Perform a hyperparameter search using grid search and save into the hyperdict
    '''
    s_dropout = [0.1, 0.2, 0.5]
    s_num_layers = [2, 3, 5]
    s_hidden_channels = [32, 64, 128, 256, 512]
    #s_l2 = [0, 1e-5, 2e-4]
    #s_batch_size = [32]
    s_lr = [0.01, 0.001, 0.0005]
    
    search_bar = tqdm(desc='hyper_searching', 
                              total=len(s_dropout)*len(s_num_layers)*len(s_hidden_channels)*len(s_lr))
    
    i=0
    
    for dp in s_dropout:
        for lay in s_num_layers:
            for hc in s_hidden_channels:
                for lr in s_lr:
                        
                    args.dropout_p = dp
                    args.num_layers = lay
                    args.hidden_channels = hc
                    args.learning_rate = lr

                    key = (dp, lay, hc, lr)

                    if not key in hyperdict:
                        train_state = training_loop(verbose=verbose)
                        hyperdict[key] = train_state
                        update_best_config(current_best,train_state, key)

                    search_bar.set_postfix(best_ATT_acc = current_best['ATT'],
                                            best_VAL_acc = current_best['VAL'],
                                            config = current_best['key'])
                    search_bar.update()
                    
                    #if i%5==0:
                    #    best_df = pd.DataFrame(current_best)
                    #    best_df.to_csv(args.save_dir+'best_config.csv')

                    with open(args.save_dir+'hyperdict.p', 'wb') as fp:
                        pickle.dump(hyperdict,fp, protocol=pickle.HIGHEST_PROTOCOL)
                    i+=1
                                
    #best_df = pd.DataFrame(current_best)
    #best_df.to_csv(args.save_dir+'best_config.csv')

def main():
    
    if args.expand_filepaths_to_save_dir:
        args.model_state_file = os.path.join(args.save_dir,
                                            args.model_state_file)
        
        print("Expanded filepaths: ")
        print("\t{}".format(args.model_state_file))
        
    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False
    else:
        torch.backends.cudnn.benchmark = True
        print('Using cudnn.benchmark.')

    print("Using CUDA: {}".format(args.cuda))

    args.device = torch.device("cuda" if args.cuda else "cpu")

    #s_seed = [0,1,2,42,100,233,1024,1337,2333,4399]
    s_seed = [42]

    for sd in s_seed:
        args.seed = sd
        args.save_dir = args.save_dir + '{}/'.format(sd)

        # handle dirs
        handle_dirs(args.save_dir)

        if 'hyperdict.p' in [files for root, dirs, files in os.walk(args.save_dir)][0]:
                with open(args.save_dir+'hyperdict.p', 'rb') as fp:
                    hyperdict = pickle.load(fp)
        else:
            hyperdict = {}

        # Train Model with Hyperparameter Search
        current_best = {}
        current_best['loss'] = 1e10
        current_best['ATT'] = 0
        current_best['VAL'] = 0
        current_best['args'] = None
        #current_best['state'] = None
        current_best['key'] = None

        Hypersearch(hyperdict, current_best,verbose=True)

        with open(args.save_dir+'hyperdict.p', 'wb') as fp:
            pickle.dump(hyperdict,fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        #dp, l2, bs, lr = current_best['key']
        #args.dropout_p = dp
        #args.l2 = l2
        #args.batch_size = bs
        #args.learning_rate = lr

        
if __name__ == "__main__":
    main()
"""## END"""