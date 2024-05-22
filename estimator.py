# USAGE EXAPMLES:
# 
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python cvae.py --distill_range train --cvae_type nh-3
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python cvae.py --distill_range train --cvae_type nh-3 --dataset citeseer
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python cvae.py --distill_range train --cvae_type nh-3 --dataset pubmed
# 
# always change the .m output file name to x_hop1_h[-3].m for latter use
# 

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
# import matplotlib.pyplot as plt

import pickle as pkl
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import MSELoss
import os
import gc
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import argparse
import random
from tqdm import tqdm
import math
from utils import _get_uniques_,_conv_to_our_format_,get_alternative_graph_repr, parse_args
from gcn_tools import rotate, softmax, get_param, ccorr
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add, scatter_mean
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from stare_models.models_statements import StarE_Transformer
from sklearn.mixture import GaussianMixture
import pickle,json,copy,gc, math

def make_print_to_file( model,path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    # import config_file as cfg_file
    import sys
    import datetime
  
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
  
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
  
        def flush(self):
            pass
  
  
  
  
    fileName = datetime.datetime.now().strftime('day'+'%Y_%m_%d')
    # if config['USE_TEST'] == True:
    #     sys.stdout = Logger(fileName + '_'+model.replace('.pth','')+'eval.log', path=path)
    # else:
    sys.stdout = Logger(fileName + '_'+model.replace('.pth','')+'.log', path=path)
  
    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60,'*'))


class TransEncoder(nn.Module):
    def __init__(self, hidden_size=1600, num_heads=32, num_layers=2):
        super(TransEncoder, self).__init__()
        

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads
            ),
            num_layers=num_layers
        )

    def forward(self, x):
        # x = self.embedding(x)
        x = x.unsqueeze(0)  # 在第一个维度上添加一个维度

        # 对于 Transformer 模型，需要提供 sequence_length x embedding_dim 的输入
        # 这里假设 sequence_length 为 1
        # x = x.transpose(0, 1)  # 交换第一个维度和第二个维度
        # print(x.shape)
        # 输入 Transformer 编码器
        x = self.transformer_encoder(x)
        return x

class Estimator(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, mapping,edge_mapping):
        super(Estimator, self).__init__()
        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/25/model.torch')
        pretrained1 = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/26/model.torch')
        self.device = torch.device('cuda')
        self.transformer_encoder = TransEncoder(
            hidden_size=input_size,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(self.device)
        encoder_layers = TransformerEncoderLayer(400, 4, 512, 0.1)
        self.encoder = TransformerEncoder(encoder_layers,2).to(self.device)
        new_state_dict = OrderedDict()
        for k, v in pretrained1.items():
            if k.startswith('encoder.'):
                new_state_dict[k[8:]] = v
        self.encoder.load_state_dict(new_state_dict)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.position_embeddings = nn.Embedding(14, 400).to(self.device)
        new_state_dict = OrderedDict()
        
        new_state_dict['weight'] = pretrained1['position_embeddings.weight']
        self.position_embeddings.load_state_dict(new_state_dict)
        self.position_embeddings.eval()
        for param in self.position_embeddings.parameters():
            param.requires_grad = False
        
        self.s_extractor = nn.Sequential(nn.Linear(input_size, 400),nn.ReLU()).to(self.device)
        self.p_extractor = nn.Sequential(nn.Linear(input_size, 400),nn.ReLU()).to(self.device)
        self.o_extractor = nn.Sequential(nn.Linear(input_size, 400),nn.ReLU()).to(self.device)
        self.qr_extractor = nn.Sequential(nn.Linear(input_size, 400),nn.ReLU()).to(self.device)
        self.qe_extractor = nn.Sequential(nn.Linear(input_size, 400),nn.ReLU()).to(self.device)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 200),  # Output a single number,
            nn.ReLU(),
            nn.Linear(200, 1) 
        ).to(self.device)
        self.embedding_size = 400
        self.init_ent = []
        self.init_rel = []
        for idx,embed in enumerate(self.pretrained['init_embed']):
            
            self.init_ent.append(self.pretrained['init_embed'][idx].to(self.device))
        
        for idx,embed in enumerate(self.pretrained['init_rel']):
                # if idx != 0 and idx != 532:
                    # if 
            if idx >= 532:
                self.init_rel.append(self.pretrained['init_rel'][idx-532])
            else:
                self.init_rel.append(self.pretrained['init_rel'][idx])
        self.pretrained = None
        pretrained1 = None
        import gc
        gc.collect()
        self.nodem = mapping
        self.edgem = edge_mapping

    def concat(self,e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, 400)
        rel_embed = rel_embed.view(-1, 1, 400)
        qual_rel_embed = qual_rel_embed.expand(6, -1)
        qual_obj_embed = qual_obj_embed.expand(6, -1)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be 
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 1).view(1, 2 * qual_rel_embed.shape[0],
                                                                    qual_rel_embed.shape[1])
        stack_inp = torch.cat([e1_embed, rel_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp

    def LP(self,sub_emb, rel_emb, qual_rel_emb, qual_obj_emb):
        with torch.no_grad():
            stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)

            
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0).to(self.device)
            stk_inp = stk_inp + pos_embeddings
            mask = torch.tensor([[False,False,False,False,True,True,True,True,True,True,True,True,True,True]]).bool().to(self.device)
            x = self.encoder(stk_inp, src_key_padding_mask=mask)

        return torch.mean(x, dim=0).squeeze()
        # x = 

        # x = self.fc(x)

        # # x = torch.mm(x, all_ent.transpose(1, 0))

        # score = torch.sigmoid(x)

    def forward(self,x):
        x_items = x.split('-')
        x_embed = []
        # print(x_items)
        fact_vector = []
        fact_vector1 = []
        for i,item in enumerate(x_items):
            if item.startswith('?'):
                if i < 3:
                    x_embed.append(torch.cat((torch.tensor([i],device=self.device),torch.tensor(np.zeros(self.embedding_size-1),device=self.device))))
                    # fact_vector.append(i)
                else:
                    x_embed.append(torch.cat((torch.tensor([i],device=self.device),torch.tensor(np.ones(self.embedding_size-1),device=self.device))))
                # if i <= 3:
                fact_vector.append(i)
                fact_vector1.append(i)
            else:
                if item.startswith('P'):
                    x_embed.append(self.init_rel[self.edgem[item]])
                    fact_vector.append(self.init_rel[self.edgem[item]].view(1,400))
                    
                else:
                    x_embed.append(self.init_ent[self.nodem[item]])
                    fact_vector.append(self.init_ent[self.nodem[item]].view(1,400))
                fact_vector1.append(i)
        # print(fact_vector)
        x_rotate_embed = [x_embed[0],x_embed[1],rotate(x_embed[4],x_embed[3]),x_embed[2]] 
        x_embed = torch.cat(x_rotate_embed,dim=0).float().to(self.device)#.view(-1,1)
        # print(x_embed.shape)
        transformer_output = self.transformer_encoder(x_embed)
        
        for idx in range(len(fact_vector)):
            if type(fact_vector[idx]) == int:
                if idx == 0:
                    fact_vector[idx] = self.s_extractor(transformer_output)
                elif idx == 1:
                    fact_vector[idx] = self.p_extractor(transformer_output)
                elif idx == 2:
                    fact_vector[idx] = self.o_extractor(transformer_output)
                elif idx == 3:
                    fact_vector[idx] = self.qr_extractor(transformer_output)
                else:
                    fact_vector[idx] = self.qe_extractor(transformer_output)
            else:
                if idx == 0:
                    fact_vector1[idx] = self.s_extractor(transformer_output)
                elif idx == 1:
                    fact_vector1[idx] = self.p_extractor(transformer_output)
                elif idx == 2:
                    fact_vector1[idx] = self.o_extractor(transformer_output)
                elif idx == 3:
                    fact_vector1[idx] = self.qr_extractor(transformer_output)
                else:
                    fact_vector1[idx] = self.qe_extractor(transformer_output)
        
        # print(x)
        # for item in fact_vector:
        #     print(item.shape)
        output = [0,0,0,0,0]
        LP_output = self.LP(fact_vector[0],fact_vector[1],fact_vector[3],fact_vector[4])
        # print(LP_output.shape)
        # print(fact_vector[2].shape)
        cos = F.cosine_similarity(LP_output, fact_vector[2].squeeze(), dim=0)
        bounded_cos = 0
        for idx,item in enumerate(fact_vector1):
            if type(item) != int:
                if type(bounded_cos) != int:
                    bounded_cos = torch.cat((bounded_cos,F.cosine_similarity(fact_vector1[idx].squeeze(), fact_vector[idx].squeeze(), dim=0).unsqueeze(0)),0)
                else:
                    bounded_cos = F.cosine_similarity(fact_vector1[idx].squeeze(), fact_vector[idx].squeeze(), dim=0).unsqueeze(0)
                output[idx] = fact_vector1[idx]
                
                # print(output.shape)
            else:
                output[idx] = fact_vector[idx]
        output = [output[0],output[1],rotate(output[4],output[3]),output[2]] 
        output = torch.cat(output,dim=1).float().to(self.device)
        output = self.mlp(output)
        # print(bounded_cos)
        # print()
        return output,cos,bounded_cos
# 示例用法
# input_size = 10
# d_model = 64
# nhead = 4
# num_layers = 2

# # 创建模型
# model = TransformerModel(input_size, d_model, nhead, num_layers)

# # 随机生成输入
# input_vector = torch.rand(1, input_size)

# # 前向传播
# output_vector = model(input_vector)

# print("Input shape:", input_vector.shape)
# print("Output shape:", output_vector.shape)

class VAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,dataset,mapping,edge_mapping,
                 aggregate,conditional=False, conditional_size=0):
        super().__init__()
        
        if conditional:
            assert conditional_size > 0
        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        self.device = torch.device('cuda')
        if aggregate == 'cat':
            if dataset == 'wd50k' or dataset == 'wd50k_nary':
                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/33/model.torch')
            elif dataset == 'jf17k':
                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/jf17k/stare_transformer/5/model.torch')
            elif dataset == 'wikipeople':
                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wikipeople/stare_transformer/2/model.torch')
        else:
            if dataset == 'wd50k' or dataset == 'wd50k_nary':
                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/31/model.torch')
            elif dataset == 'jf17k':
                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/jf17k/stare_transformer/4/model.torch')
            elif dataset == 'wikipeople':
                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wikipeople/stare_transformer/0/model.torch')
        self.init_ent = []
        self.init_rel = []
        inv_mapping = {}
        inv_edge_mapping = {}
        if mapping is not None:
            for k,v in mapping.items():
                
                inv_mapping[v] = k
            for k,v in edge_mapping.items():
                
                inv_edge_mapping[v] = k
            
            self.inv_mapping = inv_mapping
            self.inv_edge_mapping = inv_edge_mapping
            
            self.nodem = mapping
            self.edgem = edge_mapping
        for idx,embed in enumerate(self.pretrained['init_embed']):
            
            self.init_ent.append(self.pretrained['init_embed'][idx].to(self.device))
            # with open(dataset+'/features/ent/'+inv_mapping[idx]+'.pkl','wb') as f:
            #     pickle.dump(self.pretrained['init_embed'][idx].to('cpu'),f)
        # print(self.pretrained['init_rel'].shape[0]//2)
        for idx,embed in enumerate(self.pretrained['init_rel']):
                # if idx != 0 and idx != 532:
                    # if 
            if idx >= self.pretrained['init_rel'].shape[0] // 2:
                self.init_rel.append(self.pretrained['init_rel'][idx-self.pretrained['init_rel'].shape[0]//2])
            else:
                self.init_rel.append(self.pretrained['init_rel'][idx])
                # with open(dataset+'/features/rel/'+inv_edge_mapping[idx]+'.pkl','wb') as f:
                #     pickle.dump(self.pretrained['init_rel'][idx].to('cpu'),f)
        self.pretrained = None
        import gc
        gc.collect()
        self.latent_size = latent_size
        self.fact_mapping = {}
        self.dataset = dataset
        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, conditional_size).to(self.device)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, conditional_size).to(self.device)

    def forward(self, x, c=None):
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)
        
        return recon_x, means, log_var, z
    
    # def return_z(self, x, c=None):
    #     means, log_var = self.encoder(x, c)
    #     z = self.reparameterize(means, log_var)
    #     recon_x = self.decoder(z, c)

    def reparameterize(self, means, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means + eps * std
        
    def inference(self, z, c=None):
        recon_x = self.decoder(z, c)
        
        return recon_x

    def load_data(self,fact):
        x_items = fact
        x_embed = []
        qual_embed = []
        # print(x_items)
        # self.fact_mapping = {}
        # cnt = 0 
        for i,item in enumerate(x_items):
            if '-'.join(x_items[3:]) not in self.fact_mapping:
                self.fact_mapping['-'.join(x_items[3:])] = len(self.fact_mapping)
                
            if item.startswith('?'):
                if i < 3:
                    x_embed.append(torch.cat((torch.tensor([i],device=self.device),torch.tensor(np.zeros(199),device=self.device))))
                    # fact_vector.append(i)
                else:
                    x_embed.append(torch.cat((torch.tensor([i],device=self.device),torch.tensor(np.ones(199),device=self.device))))
                # if i <= 3:
                # fact_vector.append(i)
                # fact_vector1.append(i)
            else:
                if i < 3:
                    if item in self.edgem:
                        x_embed.append(self.init_rel[self.edgem[item]])
                        
                    else:
                        x_embed.append(self.init_ent[self.nodem[item]])
                else:
                    if item in self.edgem:
                        qual_embed.append(self.init_rel[self.edgem[item]])
                        # with open(self.dataset+'/features/qual_item/'+item+'.pkl','wb') as f:
                        #     pickle.dump(self.init_rel[self.edgem[item]].to('cpu'),f)
                        
                    else:
                        qual_embed.append(self.init_ent[self.nodem[item]])
                        # with open(self.dataset+'/features/qual_item/'+self.inv_mapping[self.nodem[item]]+'.pkl','wb') as f:
                        #     pickle.dump(self.init_ent[self.nodem[item]].to('cpu'),f)
                        # with open(self.dataset+'/features/qual_agg/'+x_items[i-1]+'-'+item+'.pkl','wb') as f:
                        #     pickle.dump(rotate(self.init_rel[self.edgem[x_items[i-1]]],self.init_ent[self.nodem[item]]).to('cpu'),f)
        if qual_embed != []:
            pairs = list(zip(qual_embed[0::2], qual_embed[1::2]))
            sum_result = torch.zeros_like(qual_embed[0])
            for pair in pairs:
                sum_result += rotate(pair[0],pair[1])
            # with open(self.dataset+'_vae/qual_rotate_reducetriple/'+str(self.fact_mapping['-'.join(x_items[3:])])+'.pkl','wb') as f:
            #     pickle.dump(sum_result.to('cpu'),f)
        else:
            sum_result = torch.zeros_like(x_embed[0])
        # print(fact_vector)
        x_rotate_embed = [x_embed[0],x_embed[1],x_embed[2]] 
        x = torch.cat(x_rotate_embed,dim=0).float().to(self.device)
        return x,sum_result.float(),self.fact_mapping['-'.join(x_items[3:])]

    def load_data2(self,fact):
        x_items = fact['x']
        x_embed = []
        qual_embed = []
        y_embed = []
        # print(x_items)
        # self.fact_mapping = {}
        # cnt = 0 
        for i,item in enumerate(x_items):
            # if '-'.join(x_items[3:]) not in self.fact_mapping:
            #     self.fact_mapping['-'.join(x_items[3:])] = len(self.fact_mapping)
                
            if item.startswith('?'):
                if i < 3:
                    x_embed.append(torch.cat((torch.tensor([i],device=self.device),torch.tensor(np.zeros(199),device=self.device))))
                    # fact_vector.append(i)
                else:
                    qual_embed.append(torch.cat((torch.tensor([i],device=self.device),torch.tensor(np.ones(199),device=self.device))))
                # if i <= 3:
                # fact_vector.append(i)
                # fact_vector1.append(i)
            else:
                if i < 3:
                    if item in self.edgem:
                        x_embed.append(self.init_rel[self.edgem[item]])
                        
                    else:
                        x_embed.append(self.init_ent[self.nodem[item]])
                else:
                    if item in self.edgem:
                        qual_embed.append(self.init_rel[self.edgem[item]])
                        # with open(self.dataset+'/features/qual_item/'+item+'.pkl','wb') as f:
                        #     pickle.dump(self.init_rel[self.edgem[item]].to('cpu'),f)
                        
                    else:
                        qual_embed.append(self.init_ent[self.nodem[item]])
                        # with open(self.dataset+'/features/qual_item/'+self.inv_mapping[self.nodem[item]]+'.pkl','wb') as f:
                        #     pickle.dump(self.init_ent[self.nodem[item]].to('cpu'),f)
                        # with open(self.dataset+'/features/qual_agg/'+x_items[i-1]+'-'+item+'.pkl','wb') as f:
                        #     pickle.dump(rotate(self.init_rel[self.edgem[x_items[i-1]]],self.init_ent[self.nodem[item]]).to('cpu'),f)
        if qual_embed != []:
            pairs = list(zip(qual_embed[0::2], qual_embed[1::2]))
            rotate_result = torch.zeros_like(qual_embed[0])
            for pair in pairs:
                rotate_result += rotate(pair[0],pair[1])
            # with open(self.dataset+'_vae/qual_rotate_reducetriple/'+str(self.fact_mapping['-'.join(x_items[3:])])+'.pkl','wb') as f:
            #     pickle.dump(sum_result.to('cpu'),f)
        else:
            rotate_result = torch.zeros_like(x_embed[0])
        for j,item in enumerate(fact['y']):
            
                # if i <= 3:
                # fact_vector.append(i)
                # fact_vector1.append(i)
            
        
            if item in self.edgem:
                y_embed.append(self.init_rel[self.edgem[item]])
                # with open(self.dataset+'/features/qual_item/'+item+'.pkl','wb') as f:
                #     pickle.dump(self.init_rel[self.edgem[item]].to('cpu'),f)
                
            else:
                y_embed.append(self.init_ent[self.nodem[item]])
        if y_embed != []:
            
            sum_result = rotate(y_embed[0],y_embed[1])
            # with open(self.dataset+'_vae/qual_rotate_reducetriple/'+str(self.fact_mapping['-'.join(x_items[3:])])+'.pkl','wb') as f:
            #     pickle.dump(sum_result.to('cpu'),f)
        else:
            sum_result = torch.zeros_like(x_embed[0])
        # print(fact_vector)
        x_rotate_embed = [x_embed[0],x_embed[1],x_embed[2],rotate_result] 
        x = torch.cat(x_rotate_embed,dim=0).float().to(self.device)
        return x,sum_result.float()
    
class Encoder(nn.Module):
    
    def __init__(self, layer_sizes, latent_size, conditional, conditional_size):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += conditional_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, conditional_size,
                out_type = None):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + conditional_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            elif out_type=='sigmoid':
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
            else:
                pass

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x

def transform_data(typ,dic):
    dataset = []
    typ = typ.split('_')
    
    for k in dic:
        data = {}
        data['gt'] = len(dic[k])
        if '7' in typ[0]:
            if typ[1] == 'e':
                data['fact'] = '?s-?p-?o-?qr-'+k
            elif typ[1] == 'p':
                data['fact'] = '?s-?p-?o-'+k+'-?qr'
            else:
                data['fact'] = '?s-?p-?o-'+k[0]+'-'+k[1]
        elif '6' in typ[0]:
            if len(typ) == 1:
                data['fact'] = '?s-?p-'+k+'-?qr-?qe'
            elif typ[1] == 'e':
                data['fact'] = '?s-?p-'+k[0]+'-?qr-'+k[1]
            elif typ[1] == 'p':
                data['fact'] = '?s-?p-'+k[0]+'-'+k[1]+'-?qe'
            else:
                data['fact'] = '?s-?p-'+k[0]+'-'+k[1]+'-'+k[2]
        elif '5' in typ[0]:
            if len(typ) == 1:
                
                data['fact'] = k+'-?p-?o-?qr-?qe'
            elif typ[1] == 'e':
                data['fact'] = k[0]+'-?p-?o-?qr-'+k[1]
            elif typ[1] == 'p':
                data['fact'] = k[0]+'-?p-?o-'+k[1]+'-?qe'
            else:
                data['fact'] = k[0]+'-?p-?o-'+k[1]+'-'+k[2]
        elif '4' in typ[0]:
            if len(typ) == 1:
                data['fact'] = '?s-'+k+'-?o-?qr-?qe'
            elif typ[1] == 'e':
                data['fact'] = '?s-'+k[0]+'-?o-?qr-'+k[1]
            elif typ[1] == 'p':
                data['fact'] = '?s-'+k[0]+'-?o-'+k[1]+'-?qe'
            else:
                data['fact'] = '?s-'+k[0]+'-?o-'+k[1]+'-'+k[2]
        elif '3' in typ[0]:
            if len(typ) == 1:
                data['fact'] = '?s-'+k[0]+'-'+k[1]+'-?qr-?qe'
            elif typ[1] == 'e':
                data['fact'] = '?s-'+k[0]+'-'+k[1]+'-?qr-'+k[2]
            elif typ[1] == 'p':
                data['fact'] = '?s-'+k[0]+'-'+k[1]+'-'+k[2]+'-?qe'
            else:
                data['fact'] = '?s-'+k[0]+'-'+k[1]+'-'+k[2]+'-'+k[3]
        elif '2' in typ[0]:
            if len(typ) == 1:
                data['fact'] = k[0]+'-?p-'+k[1]+'-?qr-?qe'
            elif typ[1] == 'e':
                data['fact'] = k[0]+'-?p-'+k[1]+'-?qr-'+k[2]
            elif typ[1] == 'p':
                data['fact'] = k[0]+'-?p-'+k[1]+'-'+k[2]+'-?qe'
            else:
                data['fact'] = k[0]+'-?p-'+k[1]+'-'+k[2]+'-'+k[3]
        elif '1' in typ[0]:
            if len(typ) == 1:
                data['fact'] = k[0]+'-'+k[1]+'-?o-?qr-?qe'
            elif typ[1] == 'e':
                data['fact'] = k[0]+'-'+k[1]+'-?o-?qr-'+k[2]
            elif typ[1] == 'p':
                data['fact'] = k[0]+'-'+k[1]+'-?o-'+k[2]+'-?qe'
            else:
                data['fact'] = k[0]+'-'+k[1]+'-?o-'+k[2]+'-'+k[3]
        else:
            if len(typ) == 1:
                data['fact'] = k[0]+'-'+k[1]+'-'+k[2]+'-?qr-?qe'
            elif typ[1] == 'e':
                data['fact'] = k[0]+'-'+k[1]+'-'+k[2]+'-?qr-'+k[3]
            elif typ[1] == 'p':
                data['fact'] = k[0]+'-'+k[1]+'-'+k[2]+'-'+k[3]+'-?qe'
            else:
                data['fact'] = k[0]+'-'+k[1]+'-'+k[2]+'-'+k[3]+'-'+k[4]
        dataset.append(data)
    return dataset

def loss_fn(recon_x, x, mean, log_var):
#     BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
#     BCE = F.l1_loss(recon_x,x,reduction='sum')
    BCE = F.mse_loss(recon_x,x,reduction='sum')
#     BCE = 0
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#     KLD = 0
    return (BCE + KLD) #/ x.size(0)

def BCE_loss(recon_x, x):
#     BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
#     BCE = F.l1_loss(recon_x,x,reduction='sum')
    BCE = F.mse_loss(recon_x,x,reduction='sum')
#     KLD = 0
    return BCE 
def split_data(sublist):
    # 随机打乱顺序
    random.shuffle(sublist)
    # 计算分割点
    # split_point = int(0.35 * len(sublist))
    # split_point2 = int(0.5 * len(sublist))
    split_point = int(0.7 * len(sublist))
    split_point2 = len(sublist)
    # 分割成训练集和测试集
    train_set = sublist[:split_point]
    test_set = sublist[split_point:split_point2]
    return train_set, test_set

def train():
    docs = os.listdir('pattern')
    data = [[],[],[],[],[],[],[],[]]
    if args.LP:
        if args.Sim:
            lp_task = '_lp_sim'
        else:
            lp_task = '_lp'
    else:
        if args.Sim:
            lp_task = '_orgin_sim'
        else:
            lp_task = '_orgin'
    model_name = 'transformer_total1_rotate_estimator'+str(args.learning_rate)+'_'+str(args.nhead)+lp_task+'_1_.pth'
    model_name_mae = 'transformer_total1_rotate_estimator_mae'+str(args.learning_rate)+'_'+str(args.nhead)+lp_task+'_1_.pth'
    for doc in docs:
        with open('pattern/'+doc,'rb') as f:
            dic = pkl.load(f)
        idx = int(doc.replace('.pkl','').split('_')[0].replace('pattern','')) if doc.replace('.pkl','').split('_')[0].replace('pattern','') != '' else 0
        data[idx].extend(transform_data(doc.replace('.pkl',''),dic))
    train_data = []
    test_data = []
    loss = MSELoss()
    for item in data:
        train_set, test_set = split_data(item)
        train_data.extend(train_set)
        test_data.extend(test_set)
    with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/train.txt', 'r') as f:
        raw_trn = []
        triple_trn = []
        for line in f.readlines():
            raw_trn.append(line.strip("\n").split(","))
    with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/test.txt', 'r') as f:
        raw_tst = []
        triple_tst = []
        for line in f.readlines():
            raw_tst.append(line.strip("\n").split(","))
    with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/valid.txt', 'r') as f:
        raw_val = []
        triple_val = []
        for line in f.readlines():
            raw_val.append(line.strip("\n").split(","))
    statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
                                                             test_data=raw_tst,
                                                             valid_data=raw_val)
    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates
    mapping = {pred:i for i, pred in enumerate(st_entities)}
    edge_mapping = {pred:i for i, pred in enumerate(st_predicates)}
    model = Estimator(args.input_size, args.nhead, args.num_layers,mapping,edge_mapping)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    make_print_to_file(model_name, path='/export/data/kb_group_shares/GNCE/GNCE/training_logs/')
    min_q_error = 9999999
    min_mae = 99999999
    for epoch in tqdm(range(args.epochs)):
        points_processed = 0
        model.train()
        input_data = train_data
        random.Random(random.randint(0,args.epochs)).shuffle(input_data)
        q_errors = []
        abs_errors = []
        l2_errors = []
        l3_errors = []
        for d in tqdm(input_data):
            out,sim,bounded_sim = model(d['fact'])
            # print(sim)
            if args.LP:
                if args.Sim:
                    l1 = loss(out, torch.tensor(math.log(d['gt'])).to(model.device).float())
                    l2 = 10*loss(sim, torch.tensor([1]).to(model.device).float())
                    l3 = 10*(1/bounded_sim.shape[0])*loss(bounded_sim, torch.tensor([1]*bounded_sim.shape[0]).to(model.device).float())
                    l = l1 + l2 + l3
                else:
                    l = loss(out, torch.tensor(math.log(d['gt'])).to(model.device).float()) + 10*loss(sim, torch.tensor([1]).to(model.device).float())
            else:
                if args.Sim:
                    l = loss(out, torch.tensor(math.log(d['gt'])).to(model.device).float()) +  10*(1/bounded_sim.shape[0])*loss(bounded_sim, torch.tensor([1]*bounded_sim.shape[0]).to(model.device).float())
                else:
                    l = loss(out, torch.tensor(math.log(d['gt'])).to(model.device).float())
            pred = out.detach().cpu().numpy()
            # print(f'{pred} {y}')
            y = d["gt"]
            pred = np.exp(pred)

            abs_errors.append(l.detach().cpu().numpy())
            l2_errors.append(l2.detach().cpu().numpy())
            l3_errors.append(l3.detach().cpu().numpy())
            q_errors.append(np.max([np.abs(pred) / y, y / np.abs(pred)]))
            # Gradient Accumulation
            l.backward()
            points_processed += 1
            if points_processed > 32:
                optimizer.step()
                optimizer.zero_grad()
                # print(model.s_extractor[0].weight)
                points_processed = 0
        print('Train MAE: ', np.mean(abs_errors))
        print('Train L2: ', np.mean(l2_errors))
        print('Train L3: ', np.mean(l3_errors))
        print('Train Qerror: ', np.mean(q_errors))
        model.eval()
        q_errors = []
        abs_errors = []
        l2_errors = []
        l3_errors = []
        for d in test_data:
            out,sim,bounded_sim = model(d['fact'])
                
            y = d["gt"]
            pred = out.detach().cpu().numpy()
            # print(f'{pred} {y}')
            pred = np.exp(pred)
            # print(sim.detach().cpu().numpy())
            if args.LP:
                if args.Sim:
                    abs_errors.append(np.abs(pred - y) + np.abs(10*(sim.detach().cpu().numpy()-1)) + 10*torch.mean(torch.abs((bounded_sim-torch.tensor([1]*bounded_sim.shape[0]).to(model.device).float()))).detach().cpu().numpy())
                    l2_errors.append(np.abs(10*(sim.detach().cpu().numpy()-1)))
                    l3_errors.append( 10*torch.mean(torch.abs((bounded_sim-torch.tensor([1]*bounded_sim.shape[0]).to(model.device).float()))).detach().cpu().numpy())
                else:
                    abs_errors.append(np.abs(pred - y) + np.abs(10*(sim.detach().cpu().numpy()-1)) )
            else:
                abs_errors.append(np.abs(pred - y))
            q_errors.append(np.max([np.abs(pred) / y, y / np.abs(pred)]))
        # print(gts)
        # print(preds)
        print('MAE: ', np.mean(abs_errors))
        print('Test L2: ', np.mean(l2_errors))
        print('Test L3: ', np.mean(l3_errors))
        print('Qerror: ', np.mean(q_errors))
        if (np.mean(q_errors) < min_q_error) and (np.mean(q_errors) < 8 or epoch > 45):
            torch.save(model.state_dict(), "models_extend/"+model_name)
            min_q_error = np.mean(q_errors)
        if (np.mean(abs_errors) < min_mae) and (np.mean(q_errors) < 8 or epoch > 45):
            torch.save(model.state_dict(), "models_extend/"+model_name_mae)
            min_mae = np.mean(abs_errors)
def prepare_data(dataset):
    import copy
    
    if dataset == 'wd50k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/valid.txt', 'r') as f:
            raw_val = []
            triple_val = []
            for line in f.readlines():
                raw_val.append(line.strip("\n").split(","))
    elif dataset == 'jf17k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))
        raw_val = []
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))

    elif dataset == 'wikipeople':
        import json
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_train.json', 'r') as f:
            raw_trn = []
            for line in f.readlines():
                raw_trn.append(json.loads(line))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_test.json', 'r') as f:
            raw_tst = []
            for line in f.readlines():
                raw_tst.append(json.loads(line))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_valid.json', 'r') as f:
            raw_val = []
            for line in f.readlines():
                raw_val.append(json.loads(line))

        # raw_trn[:-10], raw_tst[:10], raw_val[:10]
        # Conv data to our format
        conv_trn, conv_tst, conv_val = _conv_to_our_format_(raw_trn, filter_literals=True), \
                                    _conv_to_our_format_(raw_tst, filter_literals=True), \
                                    _conv_to_our_format_(raw_val, filter_literals=True)
        raw_trn, raw_tst, raw_val = conv_trn, conv_tst, conv_val 
    augmented_train = []
    
    for d in raw_trn:
        augmented_train.append(d)
        # if random.uniform(0,1) <= 0.3:
        d1 = copy.deepcopy(d)
        for i in range(0,3):
            if i >= len(d1):
                break
            d1[i] = '?'+str(i)
            augmented_train.append(d1)
        for i in range(0,3):
            d1 = copy.deepcopy(d)
            if i ==0:
                d1[0] = '?'+str(i)
                d1[2] = '?'+str(i)
            elif i == 1:
                d1[1] = '?'+str(i)
                d1[2] = '?'+str(i)
            elif i == 2:
                d1[0] = '?'+str(i)
                d1[1] = '?'+str(i)
            augmented_train.append(d1)
        if len(d) > 3:
            temp = copy.deepcopy(d[0:3])
            extra_st = []
            for i, uri in enumerate(d[3:]):
                if i % 2 == 0:
                    # if random.uniform(0,1) < 0.4:
                    #     temp.append('?qr')
                    # else:
                    temp.append(uri)
                else:
                    # if temp[-1]!= '?qr' and random.uniform(0,1) < 0.4:
                    #     temp.append('?qe')
                    # else:
                    temp.append(uri)
                    extra_st.append(temp)
            augmented_train.extend(extra_st)
    augmented_valid = []
    for d in raw_val:
        augmented_valid.append(d)
        # if random.uniform(0,1) <= 0.3:
        
        for i in range(0,3):
            d1 = copy.deepcopy(d)
            if i >= len(d1):
                break
            d1[i] = '?'+str(i)
            augmented_valid.append(d1)
        
        for i in range(0,3):
            d1 = copy.deepcopy(d)
            if i ==0:
                d1[0] = '?'+str(i)
                d1[2] = '?'+str(i)
            elif i == 1:
                d1[1] = '?'+str(i)
                d1[2] = '?'+str(i)
            elif i == 2:
                d1[0] = '?'+str(i)
                d1[1] = '?'+str(i)
            augmented_valid.append(d1)
        if len(d) > 3:
            temp = copy.deepcopy(d[0:3])
            extra_st = []
            for i, uri in enumerate(d[3:]):
                if i % 2 == 0:
                    # if random.uniform(0,1) < 0.4:
                    #     temp.append('?qr')
                    # else:
                    temp.append(uri)
                else:
                    # if temp[-1]!= '?qr' and random.uniform(0,1) < 0.4:
                    #     temp.append('?qe')
                    # else:
                    temp.append(uri)
                    extra_st.append(temp)
            augmented_valid.extend(extra_st)
    augmented_test = []
    for d in raw_tst:
        augmented_test.append(d)
        # if random.uniform(0,1) <= 0.3:
        
        for i in range(0,3):
            d1 = copy.deepcopy(d)
            if i >= len(d1):
                break
            d1[i] = '?'+str(i)
            augmented_test.append(d1)
        for i in range(0,3):
            d1 = copy.deepcopy(d)
            if i ==0:
                d1[0] = '?'+str(i)
                d1[2] = '?'+str(i)
            elif i == 1:
                d1[1] = '?'+str(i)
                d1[2] = '?'+str(i)
            elif i == 2:
                d1[0] = '?'+str(i)
                d1[1] = '?'+str(i)
            augmented_test.append(d1)
        if len(d) > 3:
            temp = copy.deepcopy(d[0:3])
            extra_st = []
            for i, uri in enumerate(d[3:]):
                if i % 2 == 0:
                    # if random.uniform(0,1) < 0.4:
                    #     temp.append('?qr')
                    # else:
                    temp.append(uri)
                else:
                    # if temp[-1]!= '?qr' and random.uniform(0,1) < 0.4:
                    #     temp.append('?qe')
                    # else:
                    temp.append(uri)
                    extra_st.append(temp)
            augmented_test.extend(extra_st)
    
    # return raw_trn, raw_val, raw_tst
    return augmented_train, augmented_valid, augmented_test

def prepare_data1(dataset):
    import copy
    
    if dataset == 'wd50k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/valid.txt', 'r') as f:
            raw_val = []
            triple_val = []
            for line in f.readlines():
                raw_val.append(line.strip("\n").split(","))
    elif dataset == 'jf17k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))
        raw_val = []
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))

    elif dataset == 'wikipeople':
        import json
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_train.json', 'r') as f:
            raw_trn = []
            for line in f.readlines():
                raw_trn.append(json.loads(line))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_test.json', 'r') as f:
            raw_tst = []
            for line in f.readlines():
                raw_tst.append(json.loads(line))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_valid.json', 'r') as f:
            raw_val = []
            for line in f.readlines():
                raw_val.append(json.loads(line))

        # raw_trn[:-10], raw_tst[:10], raw_val[:10]
        # Conv data to our format
        conv_trn, conv_tst, conv_val = _conv_to_our_format_(raw_trn, filter_literals=True), \
                                    _conv_to_our_format_(raw_tst, filter_literals=True), \
                                    _conv_to_our_format_(raw_val, filter_literals=True)
        raw_trn, raw_tst, raw_val = conv_trn, conv_tst, conv_val 
    augmented_train = []
    triple_train = []
    qual_facts = []
    for d in raw_trn:
        if len(d) > 3:
            augmented_train.append(d)
            # qual_facts.append(d)
            # extra_st = []
            # for j, uri in enumerate(d[3:]):
            #     temp = []
            #     for i in range(0,3):
            #         d1 = copy.deepcopy(d[:3])
            #         if i ==0:
            #             d1[0] = '?'+str(i)
            #             d1[2] = '?'+str(i)
            #         elif i == 1:
            #             d1[1] = '?'+str(i)
            #             d1[2] = '?'+str(i)
            #         elif i == 2:
            #             d1[0] = '?'+str(i)
            #             d1[1] = '?'+str(i)
            #         temp.append(d1)
            #     # print(temp)
                
            #     if j % 2 == 0:
            #         qual = []
            #         qual.append(uri)
            #     else:
            #         qual.append(uri)
            #         for item in temp:
            #             item.extend(qual)
            #             # print(qual)
            #             extra_st.append(item)
            # augmented_train.extend(extra_st)
        else:
            triple_train.append(d)
            # for i in range(0,3):
            #     d1 = copy.deepcopy(d)
            #     if i ==0:
            #         d1[0] = '?'+str(i)
            #         d1[2] = '?'+str(i)
            #     elif i == 1:
            #         d1[1] = '?'+str(i)
            #         d1[2] = '?'+str(i)
            #     elif i == 2:
            #         d1[0] = '?'+str(i)
            #         d1[1] = '?'+str(i)
            #     triple_train.append(d1)
    augmented_valid = []
    triple_valid = []
    for d in raw_val:
        if len(d) > 3:
            augmented_valid.append(d)
            # qual_facts.append(d)
            # extra_st = []
            
            # for j, uri in enumerate(d[3:]):
            #     temp = []
            #     for i in range(0,3):
            #         d1 = copy.deepcopy(d[:3])
            #         if i ==0:
            #             d1[0] = '?'+str(i)
            #             d1[2] = '?'+str(i)
            #         elif i == 1:
            #             d1[1] = '?'+str(i)
            #             d1[2] = '?'+str(i)
            #         elif i == 2:
            #             d1[0] = '?'+str(i)
            #             d1[1] = '?'+str(i)
            #         temp.append(d1)
            #     # print(temp)
                
            #     if j % 2 == 0:
            #         qual = []
            #         qual.append(uri)
            #     else:
            #         qual.append(uri)
            #         for item in temp:
            #             item.extend(qual)
            #             # print(qual)
            #             extra_st.append(item)
            # # print(extra_st)
            # augmented_valid.extend(extra_st)
        else:
            triple_valid.append(d)
            # for i in range(0,3):
            #     d1 = copy.deepcopy(d)
            #     if i ==0:
            #         d1[0] = '?'+str(i)
            #         d1[2] = '?'+str(i)
            #     elif i == 1:
            #         d1[1] = '?'+str(i)
            #         d1[2] = '?'+str(i)
            #     elif i == 2:
            #         d1[0] = '?'+str(i)
            #         d1[1] = '?'+str(i)
            #     triple_valid.append(d1)
    augmented_test = []
    triple_tst = []
    for d in raw_tst:
        if len(d) > 3:
            augmented_test.append(d)
            # qual_facts.append(d)
            # extra_st = []
            # for j, uri in enumerate(d[3:]):
            #     temp = []
            #     for i in range(0,3):
            #         d1 = copy.deepcopy(d[:3])
            #         if i ==0:
            #             d1[0] = '?'+str(i)
            #             d1[2] = '?'+str(i)
            #         elif i == 1:
            #             d1[1] = '?'+str(i)
            #             d1[2] = '?'+str(i)
            #         elif i == 2:
            #             d1[0] = '?'+str(i)
            #             d1[1] = '?'+str(i)
            #         temp.append(d1)
            #     # print(temp)
                
            #     if j % 2 == 0:
            #         qual = []
            #         qual.append(uri)
            #     else:
            #         qual.append(uri)
            #         for item in temp:
            #             item.extend(qual)
            #             # print(qual)
            #             extra_st.append(item)
            # augmented_test.extend(extra_st)
        else:
            triple_tst.append(d)
            # for i in range(0,3):
            #     d1 = copy.deepcopy(d)
            #     if i ==0:
            #         d1[0] = '?'+str(i)
            #         d1[2] = '?'+str(i)
            #     elif i == 1:
            #         d1[1] = '?'+str(i)
            #         d1[2] = '?'+str(i)
            #     elif i == 2:
            #         d1[0] = '?'+str(i)
            #         d1[1] = '?'+str(i)
            #     triple_tst.append(d1)
    
    # return raw_trn, raw_val, raw_tst
    #加一个sample取triple
    print(f'{len(augmented_train)} {len(augmented_valid)} {len(augmented_test)} {len(qual_facts)}')
    # augmented_train.extend(random.sample(triple_train,int(0.15*len(augmented_train))))
    # augmented_valid.extend(random.sample(triple_valid,int(0.15*len(augmented_valid))))
    # augmented_test.extend(random.sample(triple_tst,int(0.15*len(augmented_test))))
    print(f'{len(augmented_train)} {len(augmented_valid)} {len(augmented_test)} {len(qual_facts)}')
    return augmented_train, augmented_valid, augmented_test

def prepare_data2(dataset):
    import copy
    train = {}
    
    if dataset == 'wd50k':
        with open(dataset+'/cvae_training.json','r') as f:
            dic = json.load(f)
        # cnt = len(dic)
        with open(dataset+'/cvae_training_subgraph.json','r') as f:
            subgraph = json.load(f)
        for k in dic:
            if int(k) == 56147:
                continue
            # train[k] = random.sample(dic[k],1)[0]
            temp = sorted(dic[k],key=lambda x:x[1])[0][0]
            # if len(temp) < 5:
                
            #     print(temp)
            # cnt 
            train[k] = temp
    return train,subgraph

def prepare_data3(dataset,arg):
    import copy
    train = []
    
    if dataset == 'wd50k':
        with open(dataset+'/cvae_training.json','r') as f:
            dic = json.load(f)
        # cnt = len(dic)
        if arg == 'degree':
            with open(dataset+'/degree.json','r') as f:
                degree = json.load(f)
            with open(dataset+'/dense_items.json','r') as f:
                edges = set(json.load(f)['rel'])
        elif arg == 'pattern':
            with open('/export/data/kb_group_shares/wd50k/WD50K/wd50k/statements/test.txt','r') as f:
                line2 = f.readlines()
            
            with open('/export/data/kb_group_shares/wd50k/WD50K/wd50k/statements/valid.txt','r') as f:
                line3 = f.readlines()
            
            with open('/export/data/kb_group_shares/wd50k/WD50K/wd50k/statements/train.txt','r') as f:
                line1 = f.readlines()
            type_fact = {}
            # print('t')
            for lines in line1+line2+line3:
                
                line = lines.strip().split(',')

                # print(f'{line[0]} {line[1]}')
                if (line[0],line[1],line[2]) not in type_fact:
                    type_fact[(line[0],line[1],line[2])] = set()
                if len(line) == 3:
                    type_fact[(line[0],line[1],line[2])].add('empty')
                else:
                    temp = ''
                    for idx, it in enumerate(line[3:]):
                        if idx % 2 == 0:
                            temp += it + '-'
                        else:
                            temp += it
                            type_fact[(line[0],line[1],line[2])].add(temp)
                            temp = ''
                # if 'Q12174,P2293,Q18052481,P459,Q1098876,P459,Q23190853' in lines:
                #     print(f'{line[0]} {line[1]} {line[2]}')
                #     print(type_fact[(line[0],line[1],line[2])])
            type_num = {}
            for k in type_fact:
                type_num[k] = len(type_fact[k]) 
            # print(type_num.keys())
            # print(type_num.keys() == type_fact.keys())
            with open(dataset+'/qual_main_pattern.json','r') as f:
                pattern = set(json.load(f))
        for k in dic:
            if int(k) == 56147:
                continue
            # train[k] = random.sample(dic[k],1)[0]
            temp = sorted(dic[k],key=lambda x:x[1])[0][0]
            # if len(temp) < 5:
                
            #     print(temp)
            # cnt 
            line = copy.deepcopy(temp)
            train.append(temp)
            if arg == 'degree':
                if degree[temp[0]] >= 80:
                    temp_0 = copy.deepcopy(temp)
                    temp_0[0] = '?v'
                    # temp_0[1] = '?p'
                    train.append(temp_0)
                if degree[temp[2]] >= 80:
                    temp_2 = copy.deepcopy(temp)
                    temp_2[2] = '?v'
                    # temp_2[1] = '?p'
                    train.append(temp_2)
                if temp[1] in edges:
                    temp_2 = copy.deepcopy(temp)
                    temp_2[1] = '?p'
                    train.append(temp_2)
                if degree[temp[2]] >= 80 and degree[temp[0]] >= 80:
                    temp_2 = copy.deepcopy(temp)
                    temp_2[0] = '?v'
                    temp_2[2] = '?v'
                    train.append(temp_2)
                    # temp_2 = copy.deepcopy(temp)
                if degree[temp[2]] >= 80 and temp[1] in edges:
                    temp_2 = copy.deepcopy(temp)
                    temp_2[2] = '?v'
                    temp_2[1] = '?p'
                    train.append(temp_2)
                if degree[temp[0]] >= 80 and temp[1] in edges:
                    temp_2 = copy.deepcopy(temp)
                    temp_2[0] = '?v'
                    temp_2[1] = '?p'
                    train.append(temp_2)
                    # temp_2 = copy.deepcopy(temp)
                    # temp_2[0] = '?v'
                    # temp_2[2] = '?v'
                    # temp_2[2] = '?p'
                    # train.append(temp_2)
            elif arg == 'pattern':
                if type_num[(line[0],line[1],line[2])] > 20:
                    temp_2 = copy.deepcopy(line)
                    temp_2[0] = '?v'
                    temp_2[2] = '?v'
                    train.append(temp_2)
                    temp_3 = copy.deepcopy(line)
                    temp_3[2] = '?v'
                    temp_3[1] = '?p'
                    train.append(temp_3)
                    temp_4 = copy.deepcopy(line)
                    temp_4[0] = '?v'
                    temp_4[1] = '?p'
                    train.append(temp_4)
                
                else:
                    f = 0
                    for qual in type_fact[(line[0],line[1],line[2])]:
                        if qual in pattern:
                            f = 1
                            break
                    if f == 1:
                        temp_2 = copy.deepcopy(line)
                        temp_2[0] = '?v'
                        temp_2[2] = '?v'
                        train.append(temp_2)
                        temp_3 = copy.deepcopy(line)
                        temp_3[2] = '?v'
                        temp_3[1] = '?p'
                        train.append(temp_3)
                        temp_4 = copy.deepcopy(line)
                        temp_4[0] = '?v'
                        temp_4[1] = '?p'
                        train.append(temp_4)
                    
    return train

def prepare_data4(dataset):
    import copy
    train = {}
    
    if dataset == 'wd50k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/valid.txt', 'r') as f:
            raw_val = []
            triple_val = []
            for line in f.readlines():
                raw_val.append(line.strip("\n").split(","))
    elif dataset == 'jf17k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))
        raw_val = []
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))

    elif dataset == 'wikipeople':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/wikipeople/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/wikipeople/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/wikipeople/valid.txt', 'r') as f:
            raw_val = []
            triple_val = []
            for line in f.readlines():
                raw_val.append(line.strip("\n").split(","))

    dup_qual = {}

    for line in raw_trn+raw_tst+raw_val:
        
        if line[1] not in dup_qual:
            dup_qual[line[1]] = {}
        if len(line) > 3:
            temp = ''
            for idx,item in enumerate(line[3:]):
                if idx % 2 == 0:
                    temp += item + ' '
                else:
                    temp += item
                    if temp not in dup_qual[line[1]]:
                        dup_qual[line[1]][temp] = 0
                    dup_qual[line[1]][temp] += 1
                    temp = ''
    
    # with open(dataset+'/duplicate_qualifiers.json','w') as f:
    #     json.dump(dup_qual,f,indent=2)
    for k in dup_qual:
        max_value = 0
        max_qual = ''
        if dup_qual[k] == {}:
            continue
        if dup_qual[k] != {} and len(dup_qual[k].keys()) >= 2:
            a = sorted(dup_qual[k].items(),key=lambda x:x[1],reverse=True)
            
            dup_qual[k]['max'] = [[a[0][0],a[0][1]],[a[1][0],a[1][1]]]
        else:
            a = sorted(dup_qual[k].items(),key=lambda x:x[1],reverse=True)
            
            dup_qual[k]['max'] = [[a[0][0],a[0][1]]]
    train = []
    
    for line in raw_trn+raw_tst+raw_val:
        
        if line[1] in dup_qual and len(line) == 3:
            if random.uniform(0,1) < 0.3:
                train.append({'x':line,'y':[]})
        else:
            quals = []
            temp = ''
            for j,item in enumerate(line[3:]):
                if j % 2 == 0:
                    temp += item + ' '
                else:
                    temp += item
                    quals.append(temp)

            temp_2 = copy.deepcopy(line[:3])
            temp_2[0] = '?v'
            temp_2[2] = '?v'
            train.append({'x':temp_2,'y':dup_qual[line[1]]['max'][0][0].split(' ')})

            temp_3 = copy.deepcopy(line[:3])
            temp_3[2] = '?v'
            temp_3[1] = '?p'
            if dup_qual[line[1]]['max'][0][0] in quals:
                train.append({'x':temp_3,'y':dup_qual[line[1]]['max'][0][0].split(' ')})
            temp_4 = copy.deepcopy(line[:3])
            temp_4[0] = '?v'
            temp_4[1] = '?p'
            if dup_qual[line[1]]['max'][0][0] in quals:
                train.append({'x':temp_4,'y':dup_qual[line[1]]['max'][0][0].split(' ')})
            if len(dup_qual[line[1]].keys()) >= 3:
                temp_2 = copy.deepcopy(line[:3])
                temp_2[0] = '?v'
                temp_2[2] = '?v'
                
                train.append({'x':temp_2,'y':dup_qual[line[1]]['max'][1][0].split(' ')})
                temp_3 = copy.deepcopy(line[:3])
                temp_3[2] = '?v'
                temp_3[1] = '?p'
                if dup_qual[line[1]]['max'][0][0] in quals:
                    train.append({'x':temp_3,'y':dup_qual[line[1]]['max'][1][0].split(' ')})
                temp_4 = copy.deepcopy(line[:3])
                temp_4[0] = '?v'
                temp_4[1] = '?p'
                if dup_qual[line[1]]['max'][0][0] in quals:
                    train.append({'x':temp_4,'y':dup_qual[line[1]]['max'][1][0].split(' ')})
                for value in dup_qual[line[1]]:
                    # if dup_qual[line[1]][value] <= 2:
                    if value not in quals:
                        continue
                    if value == 'max' or value in dup_qual[line[1]]['max'] or dup_qual[line[1]][value] > 2:
                        continue
                    temp_2 = copy.deepcopy(line[:3]+value.split(' '))
                    temp_2[0] = '?v'
                    temp_2[2] = '?v'
                    if dup_qual[line[1]]['max'][1][1] > 1:
                        train.append({'x':temp_2,'y':dup_qual[line[1]]['max'][1][0].split(' ')})
                    if dup_qual[line[1]]['max'][0][1] > 1:
                        train.append({'x':temp_2,'y':dup_qual[line[1]]['max'][0][0].split(' ')})
                    temp_3 = copy.deepcopy(line[:3]+value.split(' '))
                    temp_3[2] = '?v'
                    temp_3[1] = '?p'
                    if dup_qual[line[1]]['max'][1][1] > 1:
                        train.append({'x':temp_3,'y':dup_qual[line[1]]['max'][1][0].split(' ')})
                    if dup_qual[line[1]]['max'][0][1] > 1:
                        train.append({'x':temp_3,'y':dup_qual[line[1]]['max'][0][0].split(' ')})
                    temp_4 = copy.deepcopy(line[:3]+value.split(' '))
                    temp_4[0] = '?v'
                    temp_4[1] = '?p'
                    if dup_qual[line[1]]['max'][1][1] > 1:
                        train.append({'x':temp_4,'y':dup_qual[line[1]]['max'][1][0].split(' ')})
                    if dup_qual[line[1]]['max'][0][1] > 1:
                        train.append({'x':temp_4,'y':dup_qual[line[1]]['max'][0][0].split(' ')})
                aggregated_value = []
                for value in dup_qual[line[1]]:
                    # if dup_qual[line[1]][value] <= 2:
                    if value not in quals:
                        continue
                    if value == 'max' or value in dup_qual[line[1]]['max'] or dup_qual[line[1]][value] > 1:
                        continue
                    aggregated_value.extend(value.split(' '))

                temp_2 = copy.deepcopy(line[:3]+aggregated_value)
                temp_2[0] = '?v'
                temp_2[2] = '?v'
                if dup_qual[line[1]]['max'][1][1] > 1:
                    train.append({'x':temp_2,'y':dup_qual[line[1]]['max'][1][0].split(' ')})
                if dup_qual[line[1]]['max'][0][1] > 1:
                    train.append({'x':temp_2,'y':dup_qual[line[1]]['max'][0][0].split(' ')})
                temp_3 = copy.deepcopy(line[:3]+aggregated_value)
                temp_3[2] = '?v'
                temp_3[1] = '?p'
                if dup_qual[line[1]]['max'][1][1] > 1:
                    train.append({'x':temp_3,'y':dup_qual[line[1]]['max'][1][0].split(' ')})
                if dup_qual[line[1]]['max'][0][1] > 1:
                    train.append({'x':temp_3,'y':dup_qual[line[1]]['max'][0][0].split(' ')})
                temp_4 = copy.deepcopy(line[:3]+aggregated_value)
                temp_4[0] = '?v'
                temp_4[1] = '?p'
                if dup_qual[line[1]]['max'][1][1] > 1:
                    train.append({'x':temp_4,'y':dup_qual[line[1]]['max'][1][0].split(' ')})
                if dup_qual[line[1]]['max'][0][1] > 1:
                    train.append({'x':temp_4,'y':dup_qual[line[1]]['max'][0][0].split(' ')})
    return train

def train_vae():
    if args.degree:
        degree = 'degree'
    else:
        degree = ''
    if args.pattern:
        pattern = 'pattern'
    else:
        pattern = ''
    if args.rule:
        rule = 'rule'
    else:
        rule = ''
    model_name = str(args.dataset)+'_vae_half_rotate_'+args.subgraph+'_'+degree+'_'+pattern+'_'+rule+'_'+args.aggregate+'_estimator'+str(args.learning_rate)+'_'+str(args.nhead)+'_1_.pth'
    if str(args.dataset) == 'wd50k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/valid.txt', 'r') as f:
            raw_val = []
            triple_val = []
            for line in f.readlines():
                raw_val.append(line.strip("\n").split(","))
        statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
                                                                test_data=raw_tst,
                                                                valid_data=raw_val)
    elif str(args.dataset) == 'jf17k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))

            
            
            
            
        # Get uniques
        statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
                                                                test_data=raw_tst,
                                                                valid_data=[])
    elif str(args.dataset) == 'wikipeople':
        import json
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_train.json', 'r') as f:
            raw_trn = []
            for line in f.readlines():
                raw_trn.append(json.loads(line))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_test.json', 'r') as f:
            raw_tst = []
            for line in f.readlines():
                raw_tst.append(json.loads(line))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_valid.json', 'r') as f:
            raw_val = []
            for line in f.readlines():
                raw_val.append(json.loads(line))

        # raw_trn[:-10], raw_tst[:10], raw_val[:10]
        # Conv data to our format
        conv_trn, conv_tst, conv_val = _conv_to_our_format_(raw_trn, filter_literals=True), \
                                    _conv_to_our_format_(raw_tst, filter_literals=True), \
                                    _conv_to_our_format_(raw_val, filter_literals=True)
        
        
        
        
        # Get uniques
        statement_entities, statement_predicates = _get_uniques_(train_data=conv_trn,
                                                                test_data=conv_tst,
                                                                valid_data=conv_val)
    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates
    mapping = {pred:i for i, pred in enumerate(st_entities)}
    edge_mapping = {pred:i for i, pred in enumerate(st_predicates)}
    
    print(args.dataset)
    if args.rule:
        model = VAE(encoder_layer_sizes = [200,256],
            latent_size = 200,
            #   decoder_layer_sizes = [256, h.shape[1]],
            decoder_layer_sizes = [512,200],
            dataset=args.dataset,
            mapping=mapping,
            edge_mapping=edge_mapping,
            aggregate=args.aggregate,
            conditional=True,
            conditional_size=800)
    else:
        model = VAE(encoder_layer_sizes = [200,256],
            latent_size = 200,
            #   decoder_layer_sizes = [256, h.shape[1]],
            decoder_layer_sizes = [512,200],
            dataset=args.dataset,
            mapping=mapping,
            edge_mapping=edge_mapping,
            aggregate=args.aggregate,
            conditional=True,
            conditional_size=600)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    make_print_to_file(model_name, path='/export/data/kb_group_shares/GNCE/GNCE/training_logs/')
    min_q_error = 9999999
    min_mae = 99999999
    import json
    try:
        if args.rule:
            with open(args.dataset+'/sampled_train_rule.json','r') as f:
                # json.dump([train_data,subgraph],f)
                d = json.load(f)
                train_data = d
        elif args.degree:
            with open(args.dataset+'/sampled_train_degree.json','r') as f:
                # json.dump([train_data,subgraph],f)
                d = json.load(f)
                train_data = d
        elif args.pattern:
            with open(args.dataset+'/sampled_train_pattern.json','r') as f:
                # json.dump([train_data,subgraph],f)
                d = json.load(f)
                train_data = d
        else:
            with open(args.dataset+'/sampled_train.json','r') as f:
                # json.dump([train_data,subgraph],f)
                train_data = json.load(f)
                
    except:
        print('no file found')
        if args.rule:
            train_data = prepare_data4(args.dataset)
            with open(args.dataset+'/sampled_train_rule.json','w') as f:
                json.dump(train_data,f)
        elif args.degree:
            train_data = prepare_data3(args.dataset,'degree')
            with open(args.dataset+'/sampled_train_degree.json','w') as f:
                json.dump(train_data,f)
        elif args.pattern:
            train_data = prepare_data3(args.dataset,'pattern')
            with open(args.dataset+'/sampled_train_pattern.json','w') as f:
                json.dump(train_data,f)
        else:
            train,valid,test = prepare_data1(args.dataset)
            with open(args.dataset+'/sampled_train.json','w') as f:
                json.dump(train+valid+test,f)
            train_data = train+valid+test
            
    print(len(train_data))
    input_data = []
    # in_, out_, data, nodem, edgem = neighbor(args.dataset)
    qual_rotate_list = {}
    if type(train_data) == list:
        for idx,d in tqdm(enumerate(train_data)):
            # d = train_data[idx]
            # print(d)
            if args.rule:
                x,qual = model.load_data2(d)
            else:
                x,qual,id = model.load_data(d)
            
                
            # item_name = '-'.join(d[:3])+'_'+str(id)
            # qual_rotate_list[item_name] = {'freq':len(d[3:]) // 2 if len(d) > 3 else 0,'norm':float(torch.norm(qual)),'fact':d}
            input_data.append({'idx':idx,'x':x,'qual':qual,'data':d})
    else:
        for idx in tqdm(list(train_data.keys())):
            d = train_data[idx]
            # print(d)
            x,qual,id = model.load_data(d)
            if args.subgraph == 'subgraph':
                node_list = [d[0],d[2]]
                edge_list = [d[1]]
                edge_index, edge_type, quals = adjlist1(idx,subgraph,nodem,edgem)
                sub = torch.tensor([nodem[item] for item in node_list]).to('cuda')
                rel = torch.tensor([edgem[item] for item in edge_list]).to('cuda')
                ents, rels = gcn(sub, rel, edge_index, edge_type, quals)
                ent_1 = ents[0].view(1,-1)  # 第一行
                ent_2 = ents[1].view(1,-1)
                qual = qual.cuda()
                # 拼接A_1, A_2和B
                x = torch.cat((ent_1, rels), dim=1)
                x= torch.cat((x, ent_2), dim=1).squeeze()
                
            item_name = '-'.join(d[:3])+'_'+str(id)
            qual_rotate_list[item_name] = {'freq':len(d[3:]) // 2 if len(d) > 3 else 0,'norm':float(torch.norm(qual)),'fact':d}
            input_data.append({'idx':idx,'x':x,'qual':qual,'data':d})
    # input_data = train_data[0] + train_data[1] + train_data[2]
    # input_data = random.sample(input_data,int(0.5*len(input_data)))
    # node_list = ['Q515632','Q3739104']
    # edge_list = ['P1196']
    gc.collect()
    # with open(args.dataset+'/qual_rotate_addqual_'+args.subgraph+'_'+degree+'.json','w') as f:
    #     json.dump(qual_rotate_list,f)
    # print(data['quals'].shape)
    word_list = {}
    
    min_loss = 300
    for epoch in tqdm(range(args.epochs)):
        points_processed = 0
        model.train()
        # input_data = train_data[0] + train_data[1] + train_data[2]
        random.Random(random.randint(0,args.epochs)).shuffle(input_data)
        cvae_epoch_loss = []
        for d in tqdm(input_data):
            
            # _,_, id = model.load_data(d['data'])
            # print(x.shape)
            x = d['x']
            qual = d['qual']
            # print(x.shape)
            # item_name = '-'.join(d['data'][:3])+'_'+str(id)
            recon_x, mean, log_var, _ = model(qual,x)
            
            # print(sim)
            cvae_loss = loss_fn(recon_x, qual, mean, log_var)
            optimizer.zero_grad()
            cvae_loss.backward()
            optimizer.step()
            cvae_epoch_loss.append(cvae_loss.item())
            # word_list[item_name] = {'freq':len(d['data'][3:]) // 2 if len(d) > 3 else 0,'norm':float(torch.norm(recon_x)),'qual':0}
            # qual_rotate_list[item_name] = {'freq':len(d[3:]) // 2 if len(d) > 3 else 0,'norm':float(torch.norm(qual)),'qual':0}
        
       
        print(np.mean(cvae_epoch_loss))
        if min_loss > np.mean(cvae_epoch_loss):
            torch.save(model.state_dict(), "models_extend/"+model_name)
            min_loss = np.mean(cvae_epoch_loss)
            
            # try:
            #     with open(args.dataset+'/recon_x_notriple_'+args.subgraph+'_'+degree+'.json','w') as f:
            #         json.dump(word_list,f)
            # except:
            #     print(word_list.keys())
            
            # if args.dataset != 'wd50k':
            # model.load_state_dict(torch.load('models_extend/'+args.dataset+'_vae_half_rotate_addqual_estimator0.001_32_1_.pth'))
    # else:
    # model.load_state_dict(torch.load('models_extend/'+args.dataset+'_vae_half_rotate_dropzero_estimator0.001_32_1_.pth'))
    # input_data = train_data[0] + train_data[1] + train_data[2]
    # model.eval()
    # bce_epoch_loss = []
    # kld_epoch_loss = []
    # word_list = {}
    # qual_rotate_list = {}
    
    # for d in tqdm(input_data):
    #     # if len(d) == 3:
    #     #     continue
    #     # if len(d) <= 7:
    #     #     continue
    #     with torch.no_grad():
    #         x,qual,id = model.load_data(d)
    #         item_name = '-'.join(d[:3])+'_'+str(id)
    #         z = torch.randn(1,200).to('cuda').squeeze(0)
            
    #         recon_x = model.inference(z,x)
    #     if abs(float(torch.norm(recon_x)) - float(torch.norm(qual))) > 2:
    #         word_list[item_name] = {'freq':len(d[3:]) // 2 if len(d) > 3 else 0,'norm':float(torch.norm(recon_x)),'fact':d}
    #         qual_rotate_list[item_name] = {'freq':len(d[3:]) // 2 if len(d) > 3 else 0,'norm':float(torch.norm(qual)),'fact':d}
    #         # qual = qual.cuda()
    #         # recon_x, mean, log_var, z = model(qual,x)
    #         # cvae_loss = loss_fn(recon_x, qual, mean, log_var)
    #         # bce_loss = BCE_loss(recon_x, qual)
    #         # bce_epoch_loss.append(bce_loss.item())
    #         # kld_epoch_loss.append(cvae_loss.item()-bce_loss.item())
    #         # if torch.norm(recon_x) == 0.03769657388329506:
    #         # print(f'{d} {bce_loss} {torch.norm(recon_x)}')
    #         # with open(args.dataset+'_vae/mean_reducetriple/'+'-'.join(d[:3])+'_'+str(id)+'.pkl','wb') as f:
    #         #     pickle.dump(mean,f)
    #         # with open(args.dataset+'_vae/var_reducetriple/'+'-'.join(d[:3])+'_'+str(id)+'.pkl','wb') as f:
    #         #     pickle.dump(log_var,f)
    #         # with open(args.dataset+'_vae/z_reducetriple/'+'-'.join(d[:3])+'_'+str(id)+'.pkl','wb') as f:
    #         #     pickle.dump(z,f)
    #         # with open(args.dataset+'_vae/recon_all_reducetriple/'+'-'.join(d[:3])+'_'+str(id)+'.pkl','wb') as f:
    #         #     pickle.dump(recon_x,f)
    # with open(args.dataset+'/qual_rotate_error.json','w') as f:
    #     json.dump(qual_rotate_list,f)
    # with open(args.dataset+'/recon_x_error.json','w') as f:
    #     json.dump(word_list,f)
    # print(np.mean(bce_epoch_loss))
    # print(np.mean(kld_epoch_loss))

    

def train_num_vae():
    model_name = str(args.dataset)+'_vae_half_num_rotate_estimator'+str(args.learning_rate)+'_'+str(args.nhead)+'_1_.pth'
    if str(args.dataset) == 'wd50k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/valid.txt', 'r') as f:
            raw_val = []
            triple_val = []
            for line in f.readlines():
                raw_val.append(line.strip("\n").split(","))
    elif str(args.dataset) == 'jf17k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))

        # Get uniques
        statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
                                                                test_data=raw_tst,
                                                                valid_data=[])
    elif str(args.dataset) == 'wikipeople':
        import json
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_train.json', 'r') as f:
            raw_trn = []
            for line in f.readlines():
                raw_trn.append(json.loads(line))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_test.json', 'r') as f:
            raw_tst = []
            for line in f.readlines():
                raw_tst.append(json.loads(line))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_valid.json', 'r') as f:
            raw_val = []
            for line in f.readlines():
                raw_val.append(json.loads(line))

        # raw_trn[:-10], raw_tst[:10], raw_val[:10]
        # Conv data to our format
        conv_trn, conv_tst, conv_val = _conv_to_our_format_(raw_trn, filter_literals=True), \
                                    _conv_to_our_format_(raw_tst, filter_literals=True), \
                                    _conv_to_our_format_(raw_val, filter_literals=True)
        
        
        
        
        # Get uniques
        statement_entities, statement_predicates = _get_uniques_(train_data=conv_trn,
                                                                test_data=conv_tst,
                                                                valid_data=conv_val)
    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates
    mapping = {pred:i for i, pred in enumerate(st_entities)}
    edge_mapping = {pred:i for i, pred in enumerate(st_predicates)}
    print(args.dataset)
    model = VAE(encoder_layer_sizes = [200,256],
          latent_size = 200,
        #   decoder_layer_sizes = [256, h.shape[1]],
          decoder_layer_sizes = [512,200],
          dataset=args.dataset,
          mapping=mapping,
          edge_mapping=edge_mapping,
          conditional=True,
          conditional_size=600)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    make_print_to_file(model_name, path='/export/data/kb_group_shares/GNCE/GNCE/training_logs/')
    min_q_error = 9999999
    min_mae = 99999999
    train_data = prepare_data(args.dataset)
    input_data = train_data[0] + train_data[1] + train_data[2]
    input_data = random.sample(input_data,int(0.5*len(input_data)))
    # for epoch in tqdm(range(args.epochs)):
    #     points_processed = 0
    #     model.train()
    #     # input_data = train_data[0] + train_data[1] + train_data[2]
    #     random.Random(random.randint(0,args.epochs)).shuffle(input_data)
    #     cvae_epoch_loss = []
    #     for d in tqdm(input_data):
            
    #         x,qual = model.load_data(d)
    #         qual = qual.cuda()
    #         recon_x, mean, log_var, _ = model(qual,x)
    #         # print(sim)
    #         cvae_loss = loss_fn(recon_x, qual, mean, log_var)
    #         optimizer.zero_grad()
    #         cvae_loss.backward()
    #         optimizer.step()
    #         cvae_epoch_loss.append(cvae_loss.item())
    #     print(np.mean(cvae_epoch_loss))
        # torch.save(model.state_dict(), "models_extend/"+model_name)

    model.load_state_dict(torch.load('models_extend/wikipeople_vae_half_rotate_estimator0.001_32_1_.pth'))
    input_data = train_data[0] + train_data[1] + train_data[2]
    model.eval()
    with torch.no_grad():
        for d in tqdm(input_data):
            x,qual = model.load_data(d)
            qual = qual.cuda()
            recon_x, mean, log_var, _ = model(qual,x)
        

def generate_data(num_samples, num_features_x, num_features_y, num_labels, num_clusters):
    np.random.seed(42)
    # X = np.random.rand(num_samples, num_features_x)
    # true_labels = np.random.randint(0, num_labels, size=num_samples)
    # true_means = np.random.rand(num_clusters, num_features_y)
    # true_covariances = [np.identity(num_features_y) for _ in range(num_clusters)]
    
    # # Generate data based on true parameters
    # D = []
    # for i in range(num_samples):
    #     cluster_label = true_labels[i]
    #     x = X[i]
    #     y = np.random.multivariate_normal(true_means[cluster_label], true_covariances[cluster_label])
    #     D.append((x, y))
    
    # return D
    X = np.random.rand(num_samples, num_features_x)
    true_labels = np.random.randint(0, num_labels, size=num_samples)
    true_means = np.random.rand(num_clusters, num_features_y)
    true_covariances = [np.identity(num_features_y) for _ in range(num_clusters)]
    
    # Generate data based on true parameters
    D = []
    for i in range(num_samples):
        cluster_label = true_labels[i]
        x = X[i]
        y = np.random.multivariate_normal(true_means[cluster_label], true_covariances[cluster_label])
        concatenated_data = np.concatenate((x, y))
        D.append(concatenated_data)
    
    return D

def em_algorithm(D, num_clusters):
    # X = np.array([item[0] for item in D])
    Y = np.array([np.concatenate((item[0].numpy(),item[1].numpy())) for item in D])
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    gmm.fit(Y)
    
    return gmm

def predict_y(gmm, x, num_features_y):
    # Given x, predict y using the fitted Gaussian Mixture Model
    n = np.random.rand(num_features_y)
    # print(n.shape)
    # print(x.shape)
    responsibilities = gmm.predict_proba([np.concatenate((x, n))])[0]
    means = gmm.means_
    
    # Weighted sum of means based on responsibilities
    predicted_y = np.sum(responsibilities.reshape(-1, 1) * means, axis=0)
    
    return predicted_y

def load_data(dataset,facts,edgem,nodem):
    x_data = []
    init_ent = []
    init_rel = []
    # pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/33/model.torch',map_location=torch.device('cpu'))
    # print('pretrained loaded')
    # for idx,embed in enumerate(pretrained['init_embed']):
        
    #     init_ent.append(pretrained['init_embed'][idx])
    
    # for idx,embed in enumerate(pretrained['init_rel']):
    #         # if idx != 0 and idx != 532:
    #             # if 
    #     if idx >= 532:
    #         init_rel.append(pretrained['init_rel'][idx-532])
    #     else:
    #         init_rel.append(pretrained['init_rel'][idx])
    if dataset == 'wd50k':
        pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/33/model.torch',map_location=torch.device('cpu'))
    elif dataset == 'jf17k':
        pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/jf17k/stare_transformer/5/model.torch',map_location=torch.device('cpu'))
    elif dataset == 'wikipeople':
        pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wikipeople/stare_transformer/2/model.torch',map_location=torch.device('cpu'))
    init_ent = []
    init_rel = []
    for idx,embed in enumerate(pretrained['init_embed']):
        
        init_ent.append(pretrained['init_embed'][idx].to('cpu'))
    print(pretrained['init_rel'].shape[0]//2)
    for idx,embed in enumerate(pretrained['init_rel']):
            # if idx != 0 and idx != 532:
                # if 
        if idx >= pretrained['init_rel'].shape[0] // 2:
            init_rel.append(pretrained['init_rel'][idx-pretrained['init_rel'].shape[0]//2])
        else:
            init_rel.append(pretrained['init_rel'][idx])
    import gc
    pretrained = None
    gc.collect()
    for fact in facts:
        x_items = fact
        x_embed = []
        qual_embed = []
        # print(x_items)
        for i,item in enumerate(x_items):
            if item.startswith('?'):
                if i < 3:
                    x_embed.append(torch.cat((torch.tensor([i]),torch.tensor(np.zeros(199)))))
                    # fact_vector.append(i)
                else:
                    x_embed.append(torch.cat((torch.tensor([i]),torch.tensor(np.ones(199)))))
                # if i <= 3:
                # fact_vector.append(i)
                # fact_vector1.append(i)
            else:
                if i < 3:
                    if item in edgem:
                        x_embed.append(init_rel[edgem[item]])
                        
                    else:
                        x_embed.append(init_ent[nodem[item]])
                else:
                    if item in edgem:
                        qual_embed.append(init_rel[edgem[item]])
                        
                    else:
                        qual_embed.append(init_ent[nodem[item]])
        if qual_embed != []:
            pairs = list(zip(qual_embed[0::2], qual_embed[1::2]))
            sum_result = torch.zeros_like(qual_embed[0])
            for pair in pairs:
                sum_result += rotate(pair[0],pair[1])
        else:
            sum_result = torch.zeros_like(x_embed[0])
        # print(fact_vector)
        x_rotate_embed = [x_embed[0],x_embed[1],x_embed[2]] 
        x = torch.cat(x_rotate_embed,dim=0)
        x_data.append([x,sum_result])
    return x_data
class LinearEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearEstimator, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x
def train_Gaussian():
    # num_samples = 1000
    num_features_x = 600
    num_features_y = 200
    # num_labels = 2
    num_clusters = 15
    model_name = 'gaussian_'+str(args.dataset)+'_'+str(num_clusters)+'_rotate_estimator.pth'
    linear_model_name = 'gaussian_'+str(args.dataset)+'_'+str(num_clusters)+'_rotate_estimator_linear.pth'
    if str(args.dataset) == 'wd50k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/valid.txt', 'r') as f:
            raw_val = []
            triple_val = []
            for line in f.readlines():
                raw_val.append(line.strip("\n").split(","))
    elif str(args.dataset) == 'jf17k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))

            
            
            
            
        # Get uniques
        statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
                                                                test_data=raw_tst,
                                                                valid_data=[])
    elif str(args.dataset) == 'wikipeople':
        import json
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_train.json', 'r') as f:
            raw_trn = []
            for line in f.readlines():
                raw_trn.append(json.loads(line))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_test.json', 'r') as f:
            raw_tst = []
            for line in f.readlines():
                raw_tst.append(json.loads(line))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/raw_data/wikipeople/n-ary_valid.json', 'r') as f:
            raw_val = []
            for line in f.readlines():
                raw_val.append(json.loads(line))

        # raw_trn[:-10], raw_tst[:10], raw_val[:10]
        # Conv data to our format
        conv_trn, conv_tst, conv_val = _conv_to_our_format_(raw_trn, filter_literals=True), \
                                    _conv_to_our_format_(raw_tst, filter_literals=True), \
                                    _conv_to_our_format_(raw_val, filter_literals=True)
        
        
        
        
        # Get uniques
        statement_entities, statement_predicates = _get_uniques_(train_data=conv_trn,
                                                                test_data=conv_tst,
                                                                valid_data=conv_val)
    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates
    mapping = {pred:i for i, pred in enumerate(st_entities)}
    edge_mapping = {pred:i for i, pred in enumerate(st_predicates)}
    # with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/train.txt', 'r') as f:
    #     raw_trn = []
    #     triple_trn = []
    #     for line in f.readlines():
    #         raw_trn.append(line.strip("\n").split(","))
    # with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/test.txt', 'r') as f:
    #     raw_tst = []
    #     triple_tst = []
    #     for line in f.readlines():
    #         raw_tst.append(line.strip("\n").split(","))
    # with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/valid.txt', 'r') as f:
    #     raw_val = []
    #     triple_val = []
    #     for line in f.readlines():
    #         raw_val.append(line.strip("\n").split(","))
    # statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
    #                                                          test_data=raw_tst,
    #                                                          valid_data=raw_val)
    # st_entities = ['__na__'] + statement_entities
    # st_predicates = ['__na__'] + statement_predicates
    # mapping = {pred:i for i, pred in enumerate(st_entities)}
    # edge_mapping = {pred:i for i, pred in enumerate(st_predicates)}
    train_data = prepare_data(args.dataset)
    input_data = train_data[0] + train_data[1] + train_data[2]
    print('prepared')
    # Generate synthetic data
    D = load_data(args.dataset, input_data, edgem=edge_mapping,nodem=mapping)
    print('loaded')
    es = LinearEstimator(800,200)
    # Run EM algorithm
    gmm = em_algorithm(D[:int(0.7*len(D))], num_clusters)
    optimizer = torch.optim.Adam(es.parameters(), lr=args.learning_rate)
    with open('models_extend/'+model_name, 'wb') as f:
        pickle.dump(gmm,f)
    # Example prediction
    es.train()
    cvae_epoch_loss = []
    print('EM end')
    for ep in range(3):
        for d in D[int(0.7*len(D)):]:
            main_triple = d[0]  # Replace with your input
            qual = predict_y(gmm, main_triple,num_features_y)
            qual_tensor = torch.tensor(qual, dtype=torch.float32)
            recon_qual = es(qual_tensor)
            gaussian_loss = BCE_loss(recon_qual.float(),d[1].float()).float()
            optimizer.zero_grad()
            gaussian_loss.backward()
            optimizer.step()
            cvae_epoch_loss.append(gaussian_loss.item())
        print(np.mean(cvae_epoch_loss))
    torch.save(es.state_dict(), "models_extend/"+linear_model_name)

def neighbor(dataset):
    if dataset == 'wd50k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/valid.txt', 'r') as f:
            raw_val = []
            triple_val = []
            for line in f.readlines():
                raw_val.append(line.strip("\n").split(","))
    elif dataset == 'jf17k':
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/train.txt', 'r') as f:
            raw_trn = []
            triple_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))
        raw_val = []
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/test.txt', 'r') as f:
            raw_tst = []
            triple_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))

    elif dataset == 'wikipeople':
        import json
        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/wikipeople/train.txt', 'r') as f:
            raw_trn = []
            for line in f.readlines():
                raw_trn.append(line.strip("\n").split(","))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/wikipeople/valid.txt', 'r') as f:
            raw_tst = []
            for line in f.readlines():
                raw_tst.append(line.strip("\n").split(","))

        with open('/export/data/kb_group_shares/wd50k/StarE-master/data/wikipeople/test.txt', 'r') as f:
            raw_val = []
            for line in f.readlines():
                raw_val.append(line.strip("\n").split(","))

        # raw_trn[:-10], raw_tst[:10], raw_val[:10]
        # Conv data to our format
    statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
                                                                test_data=raw_tst,
                                                                valid_data=raw_val)
    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates
    mapping = {pred:i for i, pred in enumerate(st_entities)}
    edge_mapping = {pred:i for i, pred in enumerate(st_predicates)}
    

    in_neighbor = {}
    out_neighbor = {}
    for line in raw_trn+raw_tst+raw_val:
        
        if line[0] not in in_neighbor:
            in_neighbor[line[0]] = {}
        if line[2] not in out_neighbor:
            out_neighbor[line[2]] = {} 
        if line[2] not in in_neighbor[line[0]]:
            in_neighbor[line[0]][line[2]] = []
        if line[0] not in out_neighbor[line[2]]:
            out_neighbor[line[2]][line[0]] = []
        
        in_neighbor[line[0]][line[2]].append({'p':line[1],'qual':'-'.join(line[3:]) if len(line) > 3 else ''})
        out_neighbor[line[2]][line[0]].append({'p':line[1],'qual':'-'.join(line[3:]) if len(line) > 3 else ''})
    data_gcn,nodem,edgem = get_alternative_graph_repr(raw_trn+raw_tst+raw_val, mapping,edge_mapping,has_qualifiers=True)
    # print(edgem)
    # print(edge_mapping)
    # print(edgem==edge_mapping)
    return in_neighbor,out_neighbor, data_gcn,mapping,edge_mapping

def adjlist(nodes,in_neighbor,out_neighbor,nodem,edgem):
    edge_index = [[],[]]
    edge_type = []
    qual = [[],[],[]]
    duplicate = set()
    for n in nodes:
        if n in in_neighbor:
            for n1 in in_neighbor[n]:
                for edge in in_neighbor[n][n1]:
                    if n+'-'+edge['p']+'-'+n1+'-'+edge['qual'] not in duplicate and n1+'-'+edge['p']+'-'+n+'-'+edge['qual'] not in duplicate:
                        edge_index[0].append(nodem[n])
                        edge_index[1].append(nodem[n1])
                        edge_type.append(edgem[edge['p']])
                        if edge['qual'] == '':
                            continue
                        for idx in range(1,len(edge['qual'].split('-')),2):
                            qual[0].append(edgem[edge['qual'].split('-')[idx-1]])
                            qual[1].append(nodem[edge['qual'].split('-')[idx]])
                            qual[2].append(len(edge_index[0])-1)
                        duplicate.add(n+'-'+edge['p']+'-'+n1+'-'+edge['qual'])
                        duplicate.add(n1+'-'+edge['p']+'-'+n+'-'+edge['qual'])
        if n in out_neighbor:
            for n1 in out_neighbor[n]:
                for edge in out_neighbor[n][n1]:
                    if n+'-'+edge['p']+'-'+n1+'-'+edge['qual'] not in duplicate and n1+'-'+edge['p']+'-'+n+'-'+edge['qual'] not in duplicate:
                        edge_index[0].append(nodem[n1])
                        edge_index[1].append(nodem[n1])
                        edge_type.append(edgem[edge['p']])
                        if edge['qual'] == '':
                            continue
                        for idx in range(1,len(edge['qual'].split('-')),2):
                            qual[0].append(edgem[edge['qual'].split('-')[idx-1]])
                            qual[1].append(nodem[edge['qual'].split('-')[idx]])
                            qual[2].append(len(edge_index[0])-1)
                        duplicate.add(n+'-'+edge['p']+'-'+n1+'-'+edge['qual'])
                        duplicate.add(n1+'-'+edge['p']+'-'+n+'-'+edge['qual'])

    # print(selected_nodes)
    num_nodes = 3
    temp = copy.deepcopy(edge_index[0])
    edge_index[0].extend(edge_index[1])
    edge_index[1].extend(temp)
    temp = copy.deepcopy(edge_type)
    for item in edge_type:
        temp.append(item+len(edgem)+1)
    edge_type = temp
    temp = copy.deepcopy(qual[0])
    qual[0].extend(temp)
    temp = copy.deepcopy(qual[1])
    qual[1].extend(temp)
    temp = copy.deepcopy(qual[2])
    qual[2].extend(temp)
    quals = torch.tensor(qual, dtype=torch.long).to('cuda')
    edge_type = torch.tensor(edge_type, dtype=torch.long).to('cuda')
    edge_index = torch.tensor(edge_index, dtype=torch.long).to('cuda')
    # edge_index[1, len(raw):] = edge_index[0, :len(raw)]
    # edge_index[0, len(raw):] = edge_index[1, :len(raw)]
    # edge_type[len(raw):] = edge_type[:len(raw)] + edge_cnt 
    # print(edge_index)
    # print(edge_index.shape)
    # print(edge_type.shape)
    # print(quals.shape)
    # 生成子图的边索引
    # zero_rows = torch.all(edge_index==0,dim=1)
    # print(index)
    # sub_edge_index = torch.index_select(edge_index,1,index)
    # sub_edge_type = torch.index_select(edge_type,0,index)

    # selected_data = []
    # cnt = 0
    # prev = 0
    # for i in range(len(qual[0])): 
    #     if qual[2][i] in index: 
            
    #         # print(f'{qual[0][i]} {qual[1][i]} {qual[2][i]}')
    #         selected_data.append([int(qual[0][i]),int(qual[1][i]),prev])
    # selected_data = torch.tensor(selected_data).T.to('cuda')
    # print(selected_data1.shape)
    # print(selected_data.shape)
    return edge_index,edge_type,quals

def adjlist1(idx,subgraph,nodem,edgem):
    edge_index = [[],[]]
    edge_type = []
    qual = [[],[],[]]
    duplicate = set()
    for item in subgraph[idx]:
        if '-'.join(item) in duplicate:
            continue
        edge_index[0].append(nodem[item[0]])
        edge_index[1].append(nodem[item[2]])
        edge_type.append(edgem[item[1]])
        
        for i in range(1,len(item[3:]),2):
            qual[0].append(edgem[item[3:][i-1]])
            qual[1].append(nodem[item[3:][i]])
            qual[2].append(len(edge_index[0])-1)
        duplicate.add('-'.join(item))
       

    # print(selected_nodes)
    num_nodes = 3
    temp = copy.deepcopy(edge_index[0])
    edge_index[0].extend(edge_index[1])
    edge_index[1].extend(temp)
    temp = copy.deepcopy(edge_type)
    for item in edge_type:
        temp.append(item+len(edgem)+1)
    edge_type = temp
    temp = copy.deepcopy(qual[0])
    qual[0].extend(temp)
    temp = copy.deepcopy(qual[1])
    qual[1].extend(temp)
    temp = copy.deepcopy(qual[2])
    qual[2].extend(temp)
    quals = torch.tensor(qual, dtype=torch.long).to('cuda')
    edge_type = torch.tensor(edge_type, dtype=torch.long).to('cuda')
    edge_index = torch.tensor(edge_index, dtype=torch.long).to('cuda')
    return edge_index,edge_type,quals

def gcn(sub, rel, edge_index, edge_type, quals):
    # model.setGraph({'edge_index':edge_index,'edge_type':edge_type,'quals':quals})
    if args.dataset == 'wd50k':
        state_dict = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/33/model.torch')
        n_entities = 47156
        n_relations = 532
    elif args.dataset == 'jf17k':
        state_dict = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/jf17k/stare_transformer/5/model.torch')
        n_entities = 28645
        n_relations = 322
    elif args.dataset == 'wikipeople':
        state_dict = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wikipeople/stare_transformer/2/model.torch')
        n_entities = 34826
        n_relations = 179

    import json
    with open('config.json','r') as f:
        config = json.load(f)
    config['DATASET'] = args.dataset
    config['NUM_ENTITIES'] = n_entities
    config['NUM_RELATIONS'] = n_relations
    model = StarE_Transformer({'edge_index':edge_index,'edge_type':edge_type,'quals':quals}, config).to(config['DEVICE'])
    model.load_state_dict(state_dict)
    with torch.no_grad():
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent, all_rel, mask = \
                model.forward_base(sub, rel, model.hidden_drop, model.feature_drop, quals, True, True)
    

    model = None
    
    return sub_emb, rel_emb
    
def save_dict(dataset):
    DEFAULT_CONFIG = {
    'BATCH_SIZE': 128,
    'DATASET': 'wikipeople',
    'DEVICE': 'cuda',
    'EMBEDDING_DIM': 200,
    'ENT_POS_FILTERED': True,
    'EPOCHS': 401,
    'EVAL_EVERY': 5,
    'LEARNING_RATE': 0.0001,
    'TEST':False,
    'MAX_QPAIRS': 15,
    'MODEL_NAME': 'stare_transformer',
    'CORRUPTION_POSITIONS': [0, 2],
    'SUBTYPE': 'statements',
    'QUAL_TEST':False,
    # # not used for now
    # 'MARGIN_LOSS': 5,
    # 'NARY_EVAL': False,
    # 'NEGATIVE_SAMPLING_PROBS': [0.3, 0.0, 0.2, 0.5],
    # 'NEGATIVE_SAMPLING_TIMES': 10,
    # 'NORM_FOR_NORMALIZATION_OF_ENTITIES': 2,
    # 'NORM_FOR_NORMALIZATION_OF_RELATIONS': 2,
    # 'NUM_FILTER': 5,
    # 'PROJECT_QUALIFIERS': False,
    # 'PRETRAINED_DIRNUM': '',
    # 'RUN_TESTBENCH_ON_TRAIN': False,
    # 'SAVE': False,
    # 'SELF_ATTENTION': 0,
    # 'SCORING_FUNCTION_NORM': 1,

    # important args
    'SAVE': False,
    'STATEMENT_LEN': -1,
    'USE_TEST': True,
    'WANDB': False,
    'LABEL_SMOOTHING': 0.1,
    'SAMPLER_W_QUALIFIERS': True,
    'OPTIMIZER': 'adam',
    'CLEANED_DATASET': False,  # should be false for WikiPeople and JF17K for their original data

    'GRAD_CLIPPING': True,
    'LR_SCHEDULER': True
}

    STAREARGS = {
        'LAYERS': 2,
        'N_BASES': 0,
        'GCN_DIM': 200,
        'GCN_DROP': 0.1,
        'HID_DROP': 0.3,
        'BIAS': False,
        'OPN': 'cat',
        'TRIPLE_QUAL_WEIGHT': 0.8,
        'QUAL_AGGREGATE': 'cat',  # or concat or mul
        'QUAL_OPN': 'rotate',
        'QUAL_N': 'sum',  # or mean
        'SUBBATCH': 0,
        'QUAL_REPR': 'sparse',  # sparse or full. Warning: full is 10x slower
        'ATTENTION': False,
        'ATTENTION_HEADS': 4,
        'ATTENTION_SLOPE': 0.2,
        'ATTENTION_DROP': 0.1,
        'HID_DROP2': 0.1,

        # For ConvE Only
        'FEAT_DROP': 0.3,
        'N_FILTERS': 200,
        'KERNEL_SZ': 7,
        'K_W': 10,
        'K_H': 20,

        # For Transformer
        'T_LAYERS': 2,
        'T_N_HEADS': 4,
        'T_HIDDEN': 512,
        'POSITIONAL': True,
        'POS_OPTION': 'default',
        'TIME': False,
        'POOLING': 'avg'

    }
    import sys
    DEFAULT_CONFIG['STAREARGS'] = STAREARGS
    config = DEFAULT_CONFIG.copy()
    gcnconfig = STAREARGS.copy()
    # parsed_args = parse_args(sys.argv[1:])
    # print(parsed_args)

    # # Superimpose this on default config
    # for k, v in parsed_args.items():
    #     # If its a generic arg
    #     if k in config.keys():
    #         default_val = config[k.upper()]
    #         if default_val is not None:
    #             needed_type = type(default_val)
    #             config[k.upper()] = needed_type(v)
    #         else:
    #             config[k.upper()] = v
    #     # If its a starearg
    #     elif k.lower().startswith('gcn_') and k[4:] in gcnconfig:
    #         default_val = gcnconfig[k[4:].upper()]
    #         if default_val is not None:
    #             needed_type = type(default_val)
    #             gcnconfig[k[4:].upper()] = needed_type(v)
    #         else:
    #             gcnconfig[k[4:].upper()] = v

    #     else:
    #         config[k.upper()] = v

    config['STAREARGS'] = gcnconfig

    if config['DATASET'] == 'jf17k' or config['DATASET'] == 'wikipeople':
        config['ENT_POS_FILTERED'] = False
    with open(dataset+'_config.json','w') as f:
        json.dump(config,f)

if __name__ == '__main__':
    seed = 200
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default = 1600, help=' ')
    parser.add_argument('--epochs', type=int, default=10, help=' ')
    parser.add_argument('--nhead', type=int, default=32, help=' ')
    parser.add_argument('--num_layers', type=int, default=1, help=' ')
    parser.add_argument('--learning_rate', type=float, default=0.001, help=' ')
    parser.add_argument('--LP', type=bool, default=False, help=' ')
    parser.add_argument('--Sim', type=bool, default=False, help=' ')
    parser.add_argument('--dataset', type=str, default='wd50k', help=' ')
    parser.add_argument('--subgraph', type=str, default='base', help=' ')
    parser.add_argument('--degree', type=bool, default=False, help=' ')
    parser.add_argument('--pattern', type=bool, default=False, help=' ')
    parser.add_argument('--rule', type=bool, default=False, help=' ')
    parser.add_argument('--aggregate', type=str, default='cat', help=' ')
    args = parser.parse_args()
    train_vae()
    # prepare_data4(args.dataset)
    # train_Gaussian()
    # save_dict()
    # node_list = ['Q515632','Q3739104']
    # edge_list = ['P1196']
    # in_neighbor, out_neighbor, data, nodem, edgem = neighbor(args.dataset)
    # print(len(edgem))
    # # print(data['quals'].shape)
    # edge_index, edge_type, quals = adjlist(node_list,in_neighbor, out_neighbor,nodem,edgem)
    # edge_index, edge_type, quals = adjlist1(node_list,data['edge_index'],data['edge_type'],data['quals'],in_neighbor, out_neighbor,nodem,edgem)
    # print(edge_index)
    # sub = torch.tensor([nodem[item] for item in node_list]).to('cuda')
    # rel = torch.tensor([edgem[item] for item in edge_list]).to('cuda')
    # ents, rels = gcn(args.dataset, sub, rel, edge_index, edge_type, quals)
    # print(ents.shape)
    # print(rels.shape)
    # print(edge_index.shape)
    # print(edge_type.shape)
    # print(quals.shape)
    # print(quals)
    # save_dict(args.dataset)