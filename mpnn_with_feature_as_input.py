import torch.utils.data
import torch
import torch.nn as nn
import dgl
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import numpy as np
import json
import torch.optim as optim
import pickle
import time
from tqdm import tqdm



DATASET_NAME = 'ZINC'
# data_dir = '/home/hbr-ubuntu/Downloads/LGP-GNN-main/data'
MODEL_NAME = 'GatedGCN'
random_seed = 42
out_dir = 'out/molecules_graph_regression/'




def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE


def set_all_params(dataset, MODEL_NAME, random_seed):
    
    use_gpu = True
    gpu_id = 0
    device = torch.device("cpu")
    n_heads = -1
    edge_feat = True
    pseudo_dim_MoNet = -1
    kernel = -1
    gnn_per_block = -1
    embedding_dim = -1
    pool_ratio = -1
    n_mlp_GIN = -1
    gated = False
    self_loop = False
    #self_loop = True
    max_time = 12
    pos_enc = False
    pos_enc_dim = 8
           

    if MODEL_NAME == 'GatedGCN':
        seed=random_seed; epochs=1000; batch_size=5; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr =0.00001; weight_decay=0
        L=16; hidden_dim=70; out_dim=hidden_dim; dropout=0.0; readout='mean'; max_time=24;

    

        
        
    # generic new_params
    net_params = {}
    net_params['device'] = device
    net_params['num_atom_type'] = dataset.num_atom_type
    net_params['num_bond_type'] = dataset.num_bond_type
    net_params['residual'] = True
    net_params['hidden_dim'] = hidden_dim
    net_params['out_dim'] = out_dim
    
    net_params['n_heads'] = n_heads
    net_params['L'] = L  # min L should be 2
    net_params['readout'] = "sum"
    net_params['layer_norm'] = True
    net_params['batch_norm'] = True
    net_params['in_feat_dropout'] = 0.0
    net_params['dropout'] = 0.0
    net_params['edge_feat'] = edge_feat
    net_params['self_loop'] = self_loop

    # for MLPNet 
    net_params['gated'] = gated  
    
    # specific for MoNet
    net_params['pseudo_dim_MoNet'] = pseudo_dim_MoNet
    net_params['kernel'] = kernel
    
    # specific for GIN
    net_params['n_mlp_GIN'] = n_mlp_GIN
    net_params['learn_eps_GIN'] = True
    net_params['neighbor_aggr_GIN'] = 'sum'
    
    # specific for graphsage
    net_params['sage_aggregator'] = 'meanpool'    



    
    
    # specific for pos_enc_dim
    net_params['pos_enc'] = pos_enc
    net_params['pos_enc_dim'] = pos_enc_dim
    
    params = {}
    params['seed'] = seed
    params['epochs'] = epochs
    params['batch_size'] = batch_size
    params['init_lr'] = init_lr
    params['lr_reduce_factor'] = lr_reduce_factor 
    params['lr_schedule_patience'] = lr_schedule_patience
    params['min_lr'] = min_lr
    params['weight_decay'] = weight_decay
    params['print_epoch_interval'] = 5
    params['max_time'] = max_time
    
    
    
    return params, net_params



class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/molecules/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels,dtype='float32')).unsqueeze(1)
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = torch.cat(tab_snorm_n).sqrt()  
        #tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        #tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        #snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs) 
    
        
        return batched_graph, labels

    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

import dgl.function as fn

class GatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, use_lapeig_loss=False, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.use_lapeig_loss = use_lapeig_loss
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A1 = nn.Linear(input_dim, output_dim, bias=True)
        self.A2 = nn.Linear(input_dim*2, output_dim, bias=True)
        self.B1 = nn.Linear(input_dim, output_dim, bias=True)
        self.B2 = nn.Linear(input_dim, output_dim, bias=True)
        self.B3 = nn.Linear(input_dim, output_dim, bias=True)
        self.C1 = nn.Linear(input_dim, output_dim, bias=True)
        self.C2 = nn.Linear(input_dim, output_dim, bias=True)
        
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        # self.bn_node_p = nn.BatchNorm1d(output_dim)

    def message_func_for_vij(self, edges):
        hj = edges.src['h'] # h_j
        pj = edges.src['p'] # p_j
        vij = self.A2(torch.cat((hj, pj), -1))
        return {'v_ij': vij} 
    
    def message_func_for_pj(self, edges):
        pj = edges.src['p'] # p_j
        return {'C2_pj': self.C2(pj)}
       
    def compute_normalized_eta(self, edges):
        return {'eta_ij': edges.data['sigma_hat_eta'] / (edges.dst['sum_sigma_hat_eta'] + 1e-6)} # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'
      
    def forward(self, g, h, e, p):   

        with g.local_scope():
        
            # for residual connection
            h_in = h 
            p_in = p 
            e_in = e 

            

            # print(h.size(),'h.size()---------------------------------------')
            # print(p.size(),'p.size()------------------------------------------')
            # print(p,'p------------------------------------------')
            # print(h,'h------------------------------------------')



            # For the h's
            g.ndata['h']  = h 
            # g.ndata['A1_h'] = self.A1(torch.cat((h, p), -1)) 
            g.ndata['A1_h'] = self.A1(h) 

            # self.A2 being used in message_func_for_vij() function
            g.ndata['B1_h'] = self.B1(h)
            g.ndata['B2_h'] = self.B2(h) 

            # For the p's
            g.ndata['p'] = p
            g.ndata['C1_p'] = self.C1(p)
            # self.C2 being used in message_func_for_pj() function

            # For the e's
            g.edata['e']  = e 
            g.edata['B3_e'] = self.B3(e) 

            #--------------------------------------------------------------------------------------#
            # Calculation of h
            g.apply_edges(fn.u_add_v('B1_h', 'B2_h', 'B1_B2_h'))
            g.edata['hat_eta'] = g.edata['B1_B2_h'] + g.edata['B3_e']
            g.edata['sigma_hat_eta'] = torch.sigmoid(g.edata['hat_eta'])
            g.update_all(fn.copy_e('sigma_hat_eta', 'm'), fn.sum('m', 'sum_sigma_hat_eta')) # sum_j' sigma_hat_eta_ij'
            g.apply_edges(self.compute_normalized_eta) # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'
            g.apply_edges(self.message_func_for_vij) # v_ij
            g.edata['eta_mul_v'] = g.edata['eta_ij'] * g.edata['v_ij'] # eta_ij * v_ij
            g.update_all(fn.copy_e('eta_mul_v', 'm'), fn.sum('m', 'sum_eta_v')) # sum_j eta_ij * v_ij
            g.ndata['h'] = g.ndata['A1_h'] + g.ndata['sum_eta_v']
            # g.ndata['h'] = g.ndata['sum_eta_v']


            # Calculation of p
            g.apply_edges(self.message_func_for_pj) # p_j
            g.edata['eta_mul_p'] = g.edata['eta_ij'] * g.edata['C2_pj'] # eta_ij * C2_pj
            g.update_all(fn.copy_e('eta_mul_p', 'm'), fn.sum('m', 'sum_eta_p')) # sum_j eta_ij * C2_pj
            g.ndata['p'] = g.ndata['C1_p'] + g.ndata['sum_eta_p']

            #--------------------------------------------------------------------------------------#

            # passing towards output
            h = g.ndata['h'] 
            p = g.ndata['p']
            e = g.edata['hat_eta'] 

            # GN from benchmarking-gnns-v1
            # h = h * snorm_n
            
            # batch normalization  
            if self.batch_norm:
                h = self.bn_node_h(h)
                e = self.bn_node_e(e)
                # No BN for p

            # non-linear activation
            h = F.relu(h) 
            e = F.relu(e) 
            p = torch.tanh(p)

            # residual connection
            if self.residual:
                h = h_in + h 
                p = p_in + p
                e = e_in + e 

            # dropout
            h = F.dropout(h, self.dropout, training=self.training)
            p = F.dropout(p, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)

            return h, e, p
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)


class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_dim = net_params['in_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        self.hidden_dim = hidden_dim
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        
        # self.embedding_h = nn.Linear(in_dim, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                    self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        
        
    def forward(self, g, h, e, f):

        # input embedding
        # print(h,'--------------------------------------------------------------------------')
        # h = self.embedding_h(h)
        h = nn.Embedding(len(h),self.hidden_dim)(h)
        h = self.in_feat_dropout(h)

        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)   

        f = nn.Linear(len(f),len(f))(f)

        f = f.long()

        for j in range(len(f)):
            if f[j]<0:
                f[j] = f[j]*-1

        f = nn.Embedding(len(h),self.hidden_dim)(f)
        
        # convnets
        for conv in self.layers:
            h, e, f = conv(g, h, e, f)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)
        
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss


def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_f = batch_graphs.ndata['features'].to(device)
        batch_targets = batch_targets.to(device)
        # print(batch_f.dtype,'batch_f*****************************************************')
        # print(batch_f,'batch_f*****************************************************')
        # print(batch_f.size(),'batch_f*****************************************************')
        # print(batch_x.size(),'batch_x size*****************************************************')



        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_f)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_f = batch_graphs.ndata['features'].to(device)
            batch_targets = batch_targets.to(device)
            

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_f)
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae




dataset = MoleculeDataset(DATASET_NAME)
dataset_helper = MoleculeDataset('ZINC_3Cy')
# print(dataset.num_atom_type,'dataset.num_atom_type*******************************************************')
trainset, valset, testset = dataset.train, dataset.val, dataset.test
# print(trainset[10][0].ndata['feat'], 'trainset[1][0].ndata//////////////////////////////////////////////////')

trainset_helper, valset_helper, testset_helper = dataset_helper.train, dataset_helper.val, dataset_helper.test

i = 0
for g in trainset_helper.graph_lists:
    features = []
    for node in range(len(g.ndata['feat'])):
        features.append(g.ndata['feat'][node][-1])
    # print(features, type(features), '8888888888888888888888888888888888888888888888')
    trainset[i][0].ndata['features'] = torch.Tensor(features)
    i = i + 1

i = 0
for g in valset_helper.graph_lists:
    features = []
    for node in range(len(g.ndata['feat'])):
        features.append(g.ndata['feat'][node][-1])
    # print(features, type(features), '8888888888888888888888888888888888888888888888')
    valset[i][0].ndata['features'] = torch.Tensor(features)
    i = i + 1

i = 0
for g in testset_helper.graph_lists:
    features = []
    for node in range(len(g.ndata['feat'])):
        features.append(g.ndata['feat'][node][-1])
    # print(features, type(features), '8888888888888888888888888888888888888888888888')
    testset[i][0].ndata['features'] = torch.Tensor(features)
    i = i + 1


# print(trainset[10][0].ndata['features'], 'trainset[i][0].ndatafeatures---------------------------------------------------')
params, net_params = set_all_params(dataset, MODEL_NAME, random_seed)
# net_params['in_dim'] = net_params['num_atom_type']
net_params['in_dim'] = dataset.num_atom_type
# print(net_params,'net_params********************************')

train_loader = DataLoader(trainset, num_workers=4,batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
val_loader = DataLoader(valset, num_workers=4,batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
test_loader = DataLoader(testset, num_workers=4,batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

device = net_params['device']
# print(net_params)

random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
if device.type == 'cuda':
    torch.cuda.manual_seed(params['seed'])
    torch.cuda.manual_seed_all(params['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model = GatedGCNNet(net_params)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=params['lr_reduce_factor'],
                                                    patience=params['lr_schedule_patience'],
                                                    verbose=True)

for epoch in tqdm(range(1)):
    epoch_train_loss, epoch_train_mae, optimizer = train_epoch_sparse(model, optimizer, device, train_loader, epoch)
    epoch_val_loss, epoch_val_mae = evaluate_network_sparse(model, device, val_loader, epoch)
    epoch_test_loss, epoch_test_mae = evaluate_network_sparse(model, device, test_loader, epoch)

    

    scheduler.step(epoch_val_loss)


    print(epoch_train_mae,'epoch_train_mae*****************************')
    print(epoch_val_mae,'epoch_val_mae*****************************')
    print(epoch_test_mae,'epoch_test_mae*****************************')

    



