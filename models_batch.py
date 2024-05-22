import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import Linear
from torch.nn import Sigmoid
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool,GATConv
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Optional, Union
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_scatter import scatter_add, scatter_mean
from torch import Tensor
from torch_geometric.nn.inits import reset
import math
import numpy as np
import os, json, random
from gcn_tools import get_param, get_init_embed, ccorr, rotate, softmax
from utils import StatisticsLoader,clean_literals,_conv_to_our_format_
from estimator import VAE,LinearEstimator,predict_y
import copy
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
import pickle

class NaryConv(MessagePassing):
    r"""
    Message passing function for directional message passing
    based on the GINE Conv operator
    Equation:
        .. math::
             x_i^{(k)} = h_\theta^{(k)}  \biggl( x_i^{(k-1)} \ +& \sum_{j \in \mathcal{N}^+(i)}
            \mathrm{ReLU}(x_i^{(k-1)}||e^{j,i}||x_j^{(k-1)}) \ +\\
            & \sum_{j \in \mathcal{N}^-(i)}
            \mathrm{ReLU}(x_j^{(k-1)}||e^{i,j}||x_i^{(k-1)}) \biggr)


    The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, idx, nn: torch.nn.Module, w_q,  eps: float = 0., 
                 train_eps: bool = False, edge_dim: Optional[int] = None,estimator=None,gussian=None, config=None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn.float()
        self.initial_eps = eps
        self.device = config['DEVICE']
        self.config = config
        if estimator is None:
            self.estimator = None
        else:
            self.estimator = estimator
        if gussian is None:
            self.GBRFE = None
            self.sigmoid = Sigmoid()
            self.GBRFE_lin = None
            if config['ESTIMATE_GATE'] != 'none':
                self.vae_weight = get_param((1, 200))
        else:
            self.GBRFE = gussian['model']
            self.sigmoid = Sigmoid()
            self.GBRFE_lin = Linear(3 * edge_dim, edge_dim)
            self.vae_weight = get_param((1, 200))
        
        # self.bn = torch.nn.BatchNorm1d(out_channels)
        self.w_q = w_q
        if config['HID_DIM'] > 202:
            if config['OCCURENCES'] != 'no':
                self.w_out = Linear(config['EMBEDDING_DIM'],config['HID_DIM']) # (100,200)
            else:
                if idx == 1:
                    self.w_out = Linear(config['EMBEDDING_DIM'],config['HID_DIM'])
                else:
                    self.w_out = Linear(config['HID_DIM'],config['HID_DIM'])
        else:
            if config['OCCURENCES'] != 'no':
                self.w_in = get_param((202, 202))  # (100,200)
                # self.w_out = Linear(202,202) # (100,200)
                self.w_out = get_param((202, 202))
            else:
                
                self.w_in = get_param((200, 200))
                # self.w_in = get_param((400, 400))  # (100,200)
                # self.w_out = Linear(config['HID_DIM'],config['HID_DIM'])
                self.w_out = torch.nn.Sequential(
                    torch.nn.Linear(200, 200),
                    # torch.nn.Dropout(p=0.2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(200, 200),
                    # torch.nn.Dropout(p=0.2)
                ).to(self.config['DEVICE'])
        self.p = config
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            in_channels = 101
            
            if config['REMOVE_SELF']:
                self.lin_in = Linear(2 * edge_dim, edge_dim)
                self.lin_out = Linear(2 * edge_dim, edge_dim)
                self.lin = None
            elif config['STAREARGS']['QUAL_AGGREGATE'] == 'cat' and config['REMOVE_QUAL'] is False:
                self.lin = Linear(4 * edge_dim, edge_dim)
                self.lin_in = None
                self.lin_out = None
            else:
                self.lin = Linear(3 * edge_dim, edge_dim)
                self.lin_in = None
                self.lin_out = None
        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()
        if self.lin_in is not None:
            self.lin_in.reset_parameters()
        if self.lin_out is not None:
            self.lin_out.reset_parameters()
        if self.config['HID_DIM'] > 202:
            if self.w_out is not None:
                self.w_out.reset_parameters()
        if self.GBRFE_lin is not None:
            self.GBRFE_lin.reset_parameters()
            
    def forward(self, x, edge_index, edge_type, rel_embed,
                qualifier_ent=None, qualifier_rel=None, quals=None,triple_index=None):

        """

        See end of doc string for explaining.

        :param x: all entities*dim_of_entities (for jf17k -> 28646*200)
        :param edge_index: COO matrix (2 list each having nodes with index
        [1,2,3,4,5]
        [3,4,2,5,4]

        Here node 1 and node 3 are connected with edge.
        And the type of edge can be found using edge_type.

        Note that there are twice the number of edges as each edge is also reversed.
        )
        :param edge_type: The type of edge connecting the COO matrix
        :param rel_embed: 2 Times Total relation * emb_dim (200 in our case and 2 Times because of inverse relations)
        :param qualifier_ent:
        :param qualifier_rel:
        :param quals: Another sparse matrix

        where
            quals[0] --> qualifier relations type
            quals[1] --> qualifier entity
            quals[2] --> index of the original COO matrix that states for which edge this qualifier exists ()


        For argument sake if a knowledge graph has following statements

        [e1,p1,e4,qr1,qe1,qr2,qe2]
        [e1,p1,e2,qr1,qe1,qr2,qe3]
        [e1,p2,e3,qr3,qe3,qr2,qe2]
        [e1,p2,e5,qr1,qe1]
        [e2,p1,e4]
        [e4,p3,e3,qr4,qe1,qr2,qe4]
        [e1,p1,e5]
                                                 (incoming)         (outgoing)
                                            <----(regular)------><---(inverse)------->
        Edge index would be             :   [e1,e1,e1,e1,e2,e4,e1,e4,e2,e3,e5,e4,e3,e5]
                                            [e4,e2,e3,e5,e4,e3,e5,e1,e1,e1,e1,e2,e4,e1]

        Edge Type would be              :   [p1,p1,p2,p2,p1,p3,p1,p1_inv,p1_inv,p2_inv,p2_inv,p1_inv,p3_inv,p1_inv]

                                            <-------on incoming-----------------><---------on outgoing-------------->
        quals would be                  :   [qr1,qr2,qr1,qr2,qr3,qr2,qr1,qr4,qr2,qr1,qr2,qr1,qr2,qr3,qr2,qr1,qr4,qr2]
                                            [qe1,qe2,qe1,qe3,qe3,qe2,qe1,qe1,qe4,qe1,qe2,qe1,qe3,qe3,qe2,qe1,qe1,qe4]
                                            [0,0,1,1,2,2,3,5,5,0,0,1,1,2,2,3,5,5]
                                            <--on incoming---><--outgoing------->

        Note that qr1,qr2... and qe1, qe2, ... all belong to the same space
        :return:
        """
        if self.device is None:
            self.device = edge_index.device
        if isinstance(x, Tensor):
            x_copy = x.clone().detach()
        # rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        # print(edge_index.shape)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)
        # print(quals)
        # print(edge_index)
        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        if quals is not None:
            num_quals = quals.size(1) // 2
            self.in_index_qual_ent, self.out_index_qual_ent = quals[1, :num_quals], quals[1, num_quals:]
            self.in_index_qual_rel, self.out_index_qual_rel = quals[0, :num_quals], quals[0, num_quals:]
            self.quals_index_in, self.quals_index_out = quals[2, :num_quals], quals[2, num_quals:]
        # print(quals)
        '''
            Adding self loop by creating a COO matrix. Thus \
             loop index [1,2,3,4,5]
                        [1,2,3,4,5]
             loop type [10,10,10,10,10] --> assuming there are 9 relations


        '''
        if quals is not None:
            in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type,
                                    rel_embed=rel_embed, mode='in',
                                    ent_embed=x, qualifier_ent=self.in_index_qual_ent,
                                    qualifier_rel=self.in_index_qual_rel,
                                    qual_index=self.quals_index_in,
                                    var_index=self.in_index,triple_index=triple_index)

            out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type,
                                        rel_embed=rel_embed, mode='out',
                                        ent_embed=x, qualifier_ent=self.out_index_qual_ent,
                                        qualifier_rel=self.out_index_qual_rel,
                                        qual_index=self.quals_index_out,
                                        var_index=self.out_index,triple_index=triple_index)
        else:
            in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type,
                                    rel_embed=rel_embed, mode='in',
                                    ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                    qual_index=None, source_index=None)

            out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type,
                                     rel_embed=rel_embed, mode='out',
                                     ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                     qual_index=None, source_index=None)


        out = in_res * (0.5) +out_res * (0.5)
        
        if x_copy is not None:
            out = out + (1 + self.eps) * x_copy
        # print(f'{out.shape} {rel_embed.shape} {x_copy.shape} {x[0].shape}')
        if self.config['HID_DIM'] > 202:
            return self.nn(out.float()), self.w_out(rel_embed.float())#rel_embed.float()
        elif self.config['RELATION_TRANS']:
            return F.relu(self.nn(out.float())), F.relu(self.w_out(rel_embed.float()))
        else:
            return F.relu(self.nn(out.float())), rel_embed

    def update_rel_emb_with_qualifier(self, ent_embed, rel_embed,qualifier_ent, qualifier_rel, edge_type, qual_index=None,edge_index=None,triple_index=None):
        """
        The update_rel_emb_with_qualifier method performs following functions:

        Input is the secondary COO matrix (QE (qualifier entity), QR (qualifier relation), edge index (Connection to the primary COO))

        Step1 : Embed all the input
            Step1a : Embed the qualifier entity via ent_embed (So QE shape is 33k,1 -> 33k,200)
            Step1b : Embed the qualifier relation via rel_embed (So QR shape is 33k,1 -> 33k,200)
            Step1c : Embed the main statement edge_type via rel_embed (So edge_type shape is 61k,1 -> 61k,200)

        Step2 : Combine qualifier entity emb and qualifier relation emb to create qualifier emb (See self.qual_transform).
            This is generally just summing up. But can be more any pair-wise function that returns one vector for a (qe,qr) vector

        Step3 : Update the edge_type embedding with qualifier information. This uses scatter_add/scatter_mean.


        before:
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type          :   [q,w,e',r,t,y,u,i,o,p, .....]        (here q,w,e' .. are of 200 dim each)

        After:
            edge_type          :   [q+(a+b+d),w+(c+e+g),e'+f,......]        (here each element in the list is of 200 dim)


        :param ent_embed: essentially x (28k*200 in case of Jf17k)
        :param rel_embed: essentially relation embedding matrix

        For secondary COO matrix (QE, QR, edge index)
        :param qualifier_ent:  QE
        :param qualifier_rel: QR
        edge_type:
        :return:

        index select from embedding
        phi operation between qual_ent, qual_rel
        """
        # print(edge_type)
        # Step 1: embedding
        # print('test2')
        # print(f'qual_rel: {qualifier_rel}')
        qualifier_emb_rel = rel_embed[qualifier_rel]
        
        qualifier_emb_ent = ent_embed[qualifier_ent]
        # print('test1')
        # print(edge_type)
        # print(rel_embed.shape)
        # print(re)
        # print(edge_type)
        # print(rel_embed.shape)
        rel_part_emb = rel_embed[edge_type]
        # print('test')
        # print(edge_type)
        # Step 2: pass it through qual_transform
        qualifier_emb = self.qual_transform(qualifier_ent=qualifier_emb_ent,
                                            qualifier_rel=qualifier_emb_rel)

        # Pass it through a aggregate layer
        if self.config['PRINT_VECTOR']:
            result = self.qualifier_aggregate(qualifier_emb, rel_part_emb, alpha=self.p['STAREARGS']['TRIPLE_QUAL_WEIGHT'],
                                            qual_index=qual_index,edge_index=edge_index,ent_embed=ent_embed,triple_index=triple_index)
            return result[0],result[1],result[2]
        else:
            return self.qualifier_aggregate(qualifier_emb, rel_part_emb, alpha=self.p['STAREARGS']['TRIPLE_QUAL_WEIGHT'],
                                            qual_index=qual_index,edge_index=edge_index,ent_embed=ent_embed,triple_index=triple_index)

    def rel_transform(self, ent_embed, rel_embed):
        # if self.p['STAREARGS']['OPN'] == 'corr':
        #     trans_embed = ccorr(ent_embed, rel_embed)
        # elif self.p['STAREARGS']['OPN'] == 'sub':
        #     trans_embed = ent_embed - rel_embed
        # elif self.p['STAREARGS']['OPN'] == 'mult':
        #     trans_embed = ent_embed * rel_embed
        # elif self.p['STAREARGS']['OPN'] == 'rotate':
        #     trans_embed = rotate(ent_embed, rel_embed)
        # else:
        #     raise NotImplementedError

        trans_embed = torch.cat((ent_embed,rel_embed),dim=0)

        return trans_embed

    def qual_transform(self, qualifier_ent, qualifier_rel):
        """

        :return:
        """
        # if self.p['STAREARGS']['QUAL_OPN'] == 'corr':
        #     trans_embed = ccorr(qualifier_ent, qualifier_rel)
        # elif self.p['STAREARGS']['QUAL_OPN'] == 'sub':
        #     trans_embed = qualifier_ent - qualifier_rel
        # print(qualifier_ent.shape)
        # print(qualifier_rel.shape)
        if self.p['STAREARGS']['QUAL_OPN'] == 'mult':
            trans_embed = qualifier_ent * qualifier_rel
        elif self.p['STAREARGS']['QUAL_OPN'] == 'rotate':
            trans_embed = rotate(qualifier_ent, qualifier_rel)
        else:
            raise NotImplementedError

        return trans_embed

    def qualifier_aggregate(self, qualifier_emb, rel_part_emb, alpha=0.5, qual_index=None,edge_index=None,ent_embed=None,triple_index=None):
        """
            In qualifier_aggregate method following steps are performed

            qualifier_emb looks like -
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            rel_part_emb       :   [qq,ww,ee,rr,tt, .....]                      (here qq, ww, ee .. are of 200 dim)

            Note that rel_part_emb for jf17k would be around 61k*200

            Step1 : Pass the qualifier_emb to self.coalesce_quals and multiply the returned output with a weight.
            qualifier_emb   : [aa,bb,cc,dd,ee, ...... ]                 (here aa, bb, cc are of 200 dim each)
            Note that now qualifier_emb has the same shape as rel_part_emb around 61k*200

            Step2 : Combine the updated qualifier_emb (see Step1) with rel_part_emb based on defined aggregation strategy.



            Aggregates the qualifier matrix (3, edge_index, emb_dim)
        :param qualifier_emb:
        :param rel_part_emb:
        :param type:
        :param alpha
        :return:

        self.coalesce_quals    returns   :  [q+a+b+d,w+c+e+g,e'+f,......]        (here each element in the list is of 200 dim)

        """
        # print(qualifier_emb)
        # print(qualifier_emb.shape)
        # print(qual_index)
        if self.config['ESTIMATE'] != '' and self.estimator is not None:
            with torch.no_grad():
                x_i = ent_embed[edge_index[0]] 
                x_j = ent_embed[edge_index[1]]
                combined = torch.cat([x_i, rel_part_emb, x_j], dim=1).to(self.device)
                input_blocks = torch.chunk(combined, chunks=x_i.size(0), dim=0)
                if self.config['ESTIMATE'] == 'vae' or self.config['ESTIMATE'] == 'combine':
                    
                    if self.config['ESTIMATE_GATE'] == 'freq':
                        samples = sum(triple_index) / len(triple_index)
                        if samples > 12000:
                            samples = 5
                        elif samples <= 12000 and samples > 10000:
                            samples = 4
                        elif samples <= 10000 and samples > 8000:
                            samples = 3
                        elif samples <= 8000 and samples > 6000:
                            samples = 2
                        else:
                            samples = 1
                        outputs = []
                        noises = []
                        for sample_iter in range(samples):
                            noises.append(torch.randn(1,self.config['EMBEDDING_DIM']).to(self.config['DEVICE']))
                        for idx, block in enumerate(input_blocks):
                            output = []
                            for noise in noises:
                                output.append(self.estimator.inference(noise,block))
                            output = torch.stack([t for t in output]).to(self.config['DEVICE'])
                            # print(output.shape)
                            outputs.append(torch.mean(output,dim=0).to(self.config['DEVICE']))

                        
                    else:
                        n1 = torch.randn(1,self.config['EMBEDDING_DIM']).to(self.config['DEVICE'])
                        n2 = torch.randn(1,self.config['EMBEDDING_DIM']).to(self.config['DEVICE'])
                        outputs = [self.estimator.inference(n1,block) for block in input_blocks]
                    
                    estimated_result = torch.cat(outputs, dim=0)
                    # print(estimated_result.shape)
                
                    
                elif self.config['ESTIMATE'] == 'gaussian':
                    outputs = []
                    for block in input_blocks:
                          
                        outputs.append(self.estimator['linear'](self.estimator['gaussian'](block,torch.randn(1,self.config['EMBEDDING_DIM']).to(self.config['DEVICE']))).view(1,-1))
                    estimated_result = torch.cat(outputs, dim=0)
                else:
                    pass
                if 'gaussian' in self.config['ESTIMATE_GATE']:
                    gate = self.sigmoid(self.GBRFE.mu.weight+torch.randn(1,600).to(self.config['DEVICE'])*self.GBRFE.sigma.weight)
            if 'gaussian' in self.config['ESTIMATE_GATE']:
                estimated_result = self.GBRFE_lin(gate)*estimated_result
            elif self.config['ESTIMATE_GATE'] == 'vaen1':
                
                estimated_result = self.sigmoid(n1)*estimated_result
            elif self.config['ESTIMATE_GATE'] == 'vaen2':
                estimated_result = self.sigmoid(n2)*estimated_result
            elif self.config['ESTIMATE_GATE'] == 'linear':
                estimated_result = self.sigmoid(self.vae_weight)*estimated_result
            # print(estimated_result.shape)
        if self.p['STAREARGS']['QUAL_AGGREGATE'] == 'sum':
            # print(self.w_q.shape)
            # print(self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0]).float().shape)
            qualifier_emb = torch.einsum('ij,jk -> ik',
                                         self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0]).float(),
                                         self.w_q)
            # print(qualifier_emb.shape)
            # print(rel_part_emb.shape)
            # print(f'shape{(alpha * rel_part_emb + (1 - alpha) * qualifier_emb).shape}')
            # print(f'qualifier aggregate:{qualifier_emb}')
            # print(f'rel qual agg:{alpha * rel_part_emb + (1 - alpha) * qualifier_emb} {(alpha * rel_part_emb + (1 - alpha) * qualifier_emb).shape}')
            if self.config['ESTIMATE'] != '' and self.estimator is not None:
                # qual_matrix = self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0])
                zero_rows = torch.all(qualifier_emb==0,dim=1)
                # print(zero_rows.shape)
                # print(qualifier_emb.shape)
                # qualifier_emb[zero_rows] = torch.ones(qualifier_emb.size(1)).to(self.p['DEVICE'])
                # if zero_rows
                # if zero_rows.shape[0] == qualifier_emb.shape[0]: #控制是否只对纯triple query做estimation
                
                qualifier_emb[zero_rows] = estimated_result[zero_rows]
            #     if self.config['ESTIMATE'] == 'combine': #or self.config['ESTIMATE'] == 'vae':
            #         agg_rel = torch.cat((rel_part_emb, estimated_result), dim=1)
            #     else:
            #         agg_rel = torch.cat((rel_part_emb, qualifier_emb), dim=1)
            # else:
            #     agg_rel = torch.cat((rel_part_emb, qualifier_emb), dim=1)
            return alpha * rel_part_emb + (1 - alpha) * qualifier_emb      # [N_EDGES / 2 x EMB_DIM]
        elif self.p['STAREARGS']['QUAL_AGGREGATE'] == 'concat':
            # print(qualifier_emb.shape)
            qualifier_emb = self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0]).float()
            # print(qualifier_emb.shape)
            # print(rel_part_emb.shape)
            agg_rel = torch.cat((rel_part_emb, qualifier_emb), dim=1)  # [N_EDGES / 2 x 2 * EMB_DIM]
            # print(torch.mm(agg_rel.float(), self.w_q.float()).shape)
            return torch.mm(agg_rel.float(), self.w_q.float())                         # [N_EDGES / 2 x EMB_DIM]

        elif self.p['STAREARGS']['QUAL_AGGREGATE'] == 'mul':
            qualifier_emb = torch.mm(self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0], fill=1).float(), self.w_q)
            return rel_part_emb * qualifier_emb
        elif self.p['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
            qualifier_emb = torch.mm(self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0]),self.w_q)
            # print(f'qualifier aggregate:{qualifier_emb.shape}')
            # print(qualifier_emb)
            if self.config['ESTIMATE'] != '' and self.estimator is not None:
                # qual_matrix = self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0])
                if self.config['ESTIMATE'] == 'rulevae':
                    with torch.no_grad():
                        combined = torch.cat([x_i, rel_part_emb, x_j,qualifier_emb], dim=1).to(self.device)
                        input_blocks = torch.chunk(combined, chunks=x_i.size(0), dim=0)
                        n1 = torch.randn(1,self.config['EMBEDDING_DIM']).to(self.config['DEVICE'])
                        # print(input_blocks[0].shape)
                        outputs = [self.estimator.inference(n1,block) for block in input_blocks]
                        
                        estimated_result = torch.cat(outputs, dim=0)
                        agg_rel = torch.cat((rel_part_emb, qualifier_emb+estimated_result), dim=1)
                else:
                    zero_rows = torch.all(qualifier_emb==0,dim=1)
                    # print(zero_rows.shape)
                    # print(qualifier_emb.shape)
                    # qualifier_emb[zero_rows] = torch.ones(qualifier_emb.size(1)).to(self.p['DEVICE'])
                    # if zero_rows
                    # if zero_rows.shape[0] == qualifier_emb.shape[0]: #控制是否只对纯triple query做estimation
                    
                    qualifier_emb[zero_rows] = estimated_result[zero_rows]
                    if self.config['ESTIMATE'] == 'combine': #or self.config['ESTIMATE'] == 'vae':
                        agg_rel = torch.cat((rel_part_emb, estimated_result), dim=1)
                    else:
                        agg_rel = torch.cat((rel_part_emb, qualifier_emb), dim=1)
            else:
                agg_rel = torch.cat((rel_part_emb, qualifier_emb), dim=1)
            # print(f'rel qual agg:{agg_rel} {agg_rel.shape}')
            if self.config['PRINT_VECTOR']:
                return agg_rel,qualifier_emb,zero_rows
            else:
                return agg_rel
        else:
            raise NotImplementedError

    def message(self, x_j, x_i, edge_type, rel_embed,  mode, ent_embed=None, qualifier_ent=None,
                qualifier_rel=None, qual_index=None, var_index=None,triple_index=None):

        """

        The message method performs following functions

        Step1 : get updated relation representation (rel_embed) [edge_type] by aggregating qualifier information (self.update_rel_emb_with_qualifier).
        Step2 : Obtain edge message by transforming the node embedding with updated relation embedding (self.rel_transform).
        Step3 : Multiply edge embeddings (transform) by weight
        Step4 : Return the messages. They will be sent to subjects (1st line in the edge index COO)
        Over here the node embedding [the first list in COO matrix] is representing the message which will be sent on each edge


        More information about updating relation representation please refer to self.update_rel_emb_with_qualifier

        :param x_j: objects of the statements (2nd line in the COO)
        :param x_i: subjects of the statements (1st line in the COO)
        :param edge_type: relation types
        :param rel_embed: embedding matrix of all relations
        :param edge_norm:
        :param mode: in (direct) / out (inverse) / loop
        :param ent_embed: embedding matrix of all entities
        :param qualifier_ent:
        :param qualifier_rel:
        :param qual_index:
        :param source_index:
        :return:
        """
        # weight = getattr(self, 'w_{}'.format(mode))
        # print(len(qualifier_ent) == 0)
        # print(qualifier_ent)
        if qualifier_ent is not None and self.config['REMOVE_QUAL'] is False:
        # if len(qualifier_ent) != 0:
            # print(ent_embed.shape)
            # print(rel_embed.shape)
            # print(qualifier_ent)
            if self.config['PRINT_VECTOR']:
                rel_emb, qual_emb, qual_idx = self.update_rel_emb_with_qualifier(ent_embed, rel_embed, qualifier_ent,
                                                                        qualifier_rel, edge_type, qual_index,edge_index=var_index,triple_index=triple_index)
                self.qual_emb = qual_emb
                self.qual_idx = qual_idx
            else:
                rel_emb = self.update_rel_emb_with_qualifier(ent_embed, rel_embed, qualifier_ent,
                                                                        qualifier_rel, edge_type, qual_index,edge_index=var_index,triple_index=triple_index)
        else:
            rel_emb = torch.index_select(rel_embed, 0, edge_type)
        
        if not self.config['REMOVE_SELF']:
            # print(torch.cat((x_i.float(), rel_emb.float(), x_j.float()), 1).shape)
            return self.lin(torch.cat((x_i.float(), rel_emb.float(), x_j.float()), 1)).relu()
        else:
            if mode == 'in':
                return self.lin_in(torch.cat((rel_emb.float(), x_j.float()), 1)).relu()
            else:
                return self.lin_out(torch.cat((rel_emb.float(), x_j.float()), 1)).relu()

    def return_qual(self):
        return self.qual_emb, self.qual_idx

    @staticmethod
    def compute_norm(edge_index, num_ent):
        """
        Re-normalization trick used by GCN-based architectures without attention.

        Yet another torch scatter functionality. See coalesce_quals for a rough idea.

        row         :      [1,1,2,3,3,4,4,4,4, .....]        (about 61k for Jf17k)
        edge_weight :      [1,1,1,1,1,1,1,1,1,  ....] (same as row. So about 61k for Jf17k)
        deg         :      [2,1,2,4,.....]            (same as num_ent about 28k in case of Jf17k)

        :param edge_index:
        :param num_ent:
        :return:
        """
        row, col = edge_index
        edge_weight = torch.ones_like(
            row).float()  # Identity matrix where we know all entities are there
        deg = scatter_add(edge_weight, row, dim=0,
                          dim_size=num_ent)  # Summing number of weights of
        # the edges, D = A + I
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0  # for numerical stability
        norm = deg_inv[row] * edge_weight * deg_inv[
            col]  # Norm parameter D^{-0.5} *

        return norm

    def coalesce_quals(self, qual_embeddings, qual_index, num_edges, fill=0):
        """

        before:
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type          :   [0,0,0,0,0,0,0, .....]               (empty array of size num_edges)

        After:
            edge_type          :   [a+b+d,c+e+g,f ......]        (here each element in the list is of 200 dim)

        :param qual_embeddings: shape of [1, N_QUALS]
        :param qual_index: shape of [1, N_QUALS] which states which quals belong to which main relation from the index,
            that is, all qual_embeddings that have the same index have to be summed up
        :param num_edges: num_edges to return the appropriate tensor
        :param fill: fill value for the output matrix - should be 0 for sum/concat and 1 for mul qual aggregation strat
        :return: [1, N_EDGES]
        """
        # print(qual_index)
        # print(qual_embeddings.shape)
        # print(num_edges)
        if self.p['STAREARGS']['QUAL_N'] == 'sum':
            output = scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges)
        elif self.p['STAREARGS']['QUAL_N'] == 'mean':
            output = scatter_mean(qual_embeddings, qual_index, dim=0, dim_size=num_edges)

        if fill != 0:
            # by default scatter_ functions assign zeros to the output, so we assign them 1's for correct mult
            mask = output.sum(dim=-1) == 0
            output[mask] = fill

        return output

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

class NaryModel(torch.nn.Module):
    """
    GNN model to predict cardinality of a query,
    given the query graph and embeddings of nodes
    and edges.

    Args:


    """
    def __init__(self,config,ent,rel,ocr,freq,return_type,triple_ent=None,triple_rel=None,nodem=None,edgem=None,qual_ratio=None):
        super().__init__()
        torch.manual_seed(12345)
        self.device = config['DEVICE']
        self.nodem = nodem
        self.edgem = edgem
        if config['DATASET'] == 'wd50k' or config['DATASET'] == 'wd50k_nary':
            if config['HAS_QUAL'] is False and config['INIT_EMBED'] == 'stare':
                r_num = 487
            elif config['SUBTYPE'] == 'cleaned_statements_removeloop':
                r_num = 519
            elif config['SUBTYPE'] == 'cleaned_statements_removeloopqual':
                r_num = 508
            else:
                r_num = 532
        elif config['DATASET'] == 'jf17k':
            r_num = 502
        elif config['DATASET'] == 'wikipeople':
            r_num = 179
        self.r_num = r_num
        self.return_type = return_type
        if config['DATASET'] == 'wd50k' or config['DATASET'] == 'wd50k_nary':
            if config['INIT_EMBED'] == 'stare':

                if config['EMBEDDING_DIM'] >= 400:
                    if config['DISTILL']:
                        self.pretrained1 = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/26/model.torch')
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/25/model.torch')
                    else:
                        if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
                            if config['SPLIT'] is False:
                                print('21')
                                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/21/model.torch')
                            else:
                                print('25')
                                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/25/model.torch')
                        elif config['STAREARGS']['QUAL_AGGREGATE'] == 'concat':
                                print('23')
                                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/400_dim_400_epoch/23/model.torch')
                        else:
                            if config['HAS_QUAL'] is False:
                                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/2/model.torch')
                            elif config['SUBTYPE'] == 'cleaned_statements_removeloop':
                                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/7/model.torch')
                            elif config['SUBTYPE'] == 'cleaned_statements_removeloopqual':
                                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/11/model.torch')
                            else:
                                if config['SPLIT'] is False:
                                    self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/1/model.torch')
                                else:
                                    print('26')
                                    self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/26/model.torch')
                    
                else:
                    if config['DISTILL']:
                        self.pretrained1 = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/25/model.torch')
                    if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
                        if config['SPLIT'] is False:
                            print('0')
                            self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/trained/model.torch')
                        else:
                            print('33')
                            self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/33/model.torch')
                    
                    else:
                    
                        if config['SPLIT'] is False:
                            self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/trained/model.torch')
                        else:
                            print('31')
                            self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/31/model.torch')
                    if config['SUBTYPE'] == 'cleaned_statements_removeloop':
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/6/model.torch')
                    elif config['SUBTYPE'] == 'cleaned_statements_removeloopqual':
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/12/model.torch')
                    elif config['USE_ATTENTION'] == True:
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/4/model.torch')
                    else:
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/trained/model.torch')
                    # self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/2/model.torch')
                # self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/400_dim/model.torch')
                self.pretrained['init_embed'].to(self.device)
                self.ent_map = None
                self.rel_map = None
            elif config['INIT_EMBED'] == 'starecat':
                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/25/model.torch')
            elif config['INIT_EMBED'] == 'staresum':
                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/26/model.torch')
            elif config['INIT_EMBED'] == 'starecatvar':
                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/35/model.torch')    
            elif config['INIT_EMBED'] == 'shrinke':
                self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/ShrinkE-main/models/wd50k/stare_shrinking_fc/0/model.torch')
                self.pretrained['init_embed'] = self.pretrained['entities'].to(self.device)
                self.pretrained['init_rel'] = self.pretrained['relations'].to(self.device)
                self.ent_map = None
                self.rel_map = None
            elif config['INIT_EMBED'] == 'rdf2vec':
                self.pretrained = {}
                self.triple_embed = {}
                if config['EMBEDDING_DIM'] == 200:
                    if config['TRIPLE']:
                        statistics = StatisticsLoader(os.path.join("/export/data/kb_group_shares/GNCE/wd50k/statistics_200"),200)
                        self.pretrained['init_embed'] = [[0 for j in range(0,200)] for i in range(0,47156)]
                        self.pretrained['init_rel'] = [[0 for j in range(0,200)] for i in range(0,1064)]
                    else:
                        statistics = StatisticsLoader(os.path.join("/export/data/kb_group_shares/GNCE/wd50k/statistics_qual_200"),200)
                        self.pretrained['init_embed'] = [[0 for j in range(0,200)] for i in range(0,47156)]
                        self.pretrained['init_rel'] = [[0 for j in range(0,200)] for i in range(0,1064)]
                else:
                    if config['MULTI_DIM'] is False:

                        statistics = StatisticsLoader(os.path.join("/export/data/kb_group_shares/GNCE/wd50k/statistics_400"),400)
                        
                        self.pretrained['init_embed'] = [[0 for j in range(0,400)] for i in range(0,47156)]
                        self.pretrained['init_rel'] = [[0 for j in range(0,400)] for i in range(0,1064)]
                    else:
                        statistics = StatisticsLoader(os.path.join("/export/data/kb_group_shares/GNCE/wd50k/statistics_merge"),400)
                        
                        self.pretrained['init_embed'] = [[[0 for j in range(0,400)]]*config['MULTI_DIM_SIZE'] for i in range(0,47156)]
                        self.pretrained['init_rel'] = [[[0 for j in range(0,400)]]*config['MULTI_DIM_SIZE'] for i in range(0,1064)]
                # with open('/export/data/kb_group_shares/GNCE/wd50k/data_graph/id_to_id_mapping_predicate.json', 'r') as f:
                #     pred_map = json.load(f)
                # with open('/export/data/kb_group_shares/GNCE/wd50k/data_graph/id_to_id_mapping.json', 'r') as f:
                #     ent_map = json.load(f)
                # self.ent_map = ent_map
                # self.rel_map = pred_map
                
                for idx in ent:

                    
                    if ent[idx] in ocr:
                        # if ent[idx] in ent_map:
                        self.pretrained['init_embed'][idx] = statistics[str(ent[idx])]["embedding"]
                        
                
                for idx in rel:
                    if idx < r_num:
                        if rel[idx] in ocr:
                            # if rel[idx] in pred_map:
                            self.pretrained['init_rel'][idx] = torch.tensor(statistics[str(rel[idx])]["embedding"])
                        
                    else:
                        # if rel[idx-r_num] in pred_map:
                        self.pretrained['init_rel'][idx] = torch.tensor(statistics[str(rel[idx-r_num])]["embedding"])
                self.pretrained['init_rel'] = torch.tensor(self.pretrained['init_rel']).to(self.device)
                self.pretrained['init_embed'] = torch.tensor(self.pretrained['init_embed']).to(self.device)
            else:
                self.pretrained = {}
                self.pretrained['init_rel'] = get_init_embed((r_num * 2, 200))
                self.pretrained['init_rel'].to(self.device)
                self.pretrained['init_embed'] = get_init_embed((47156, 200))
                self.pretrained['init_embed'].to(self.device)
                self.ent_map = None
                self.rel_map = None
        elif config['DATASET'] == 'jf17k':
            if config['INIT_EMBED'] == 'stare':
                # if config['DISTILL']:
                #     self.pretrained1 = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/25/model.torch')
                if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
                    if config['SPLIT'] is False:
                        print('2')
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/jf17k/stare_transformer/2/model.torch')
                    else:
                        print('5')
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/jf17k/stare_transformer/5/model.torch')
                
                else:
                
                    if config['SPLIT'] is False:
                        print('3')
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/jf17k/stare_transformer/3/model.torch')
                    else:
                        print('4')
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/jf17k/stare_transformer/4/model.torch')
                
                # self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/2/model.torch')
            # self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/400_dim/model.torch')
                self.pretrained['init_embed'].to(self.device)
                self.ent_map = None
                self.rel_map = None
        elif config['DATASET'] == 'wikipeople':
            if config['INIT_EMBED'] == 'stare':
                # if config['DISTILL']:
                #     self.pretrained1 = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wd50k/stare_transformer/25/model.torch')
                if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
                    if config['SPLIT'] is False:
                        print('6')
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wikipeople/stare_transformer/6/model.torch')
                    else:
                        print('2')
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wikipeople/stare_transformer/2/model.torch')
                
                else:
                
                    if config['SPLIT'] is False:
                        print('5')
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wikipeople/stare_transformer/5/model.torch')
                    else:
                        print('0')
                        self.pretrained = torch.load('/export/data/kb_group_shares/wd50k/StarE-master/models/wikipeople/stare_transformer/0/model.torch')
        else:
            raise NotImplementedError
        if config['ESTIMATE'] == 'vae' or config['ESTIMATE'] == 'combine':
            if config['DATASET'] == 'wd50k'  or config['DATASET'] == 'wd50k_nary':
                # estimator_dic = torch.load('models_extend/vae_half_rotate_estimator0.001_32_1_.pth')
                # estimator_dic = torch.load('models_extend/wd50k_vae_half_rotate_baseline_estimator0.001_32_1_.pth')
                # estimator_dic = torch.load('models_extend/wd50k_vae_half_rotate_subgraph_update_estimator0.001_32_1_.pth')
                # estimator_dic = torch.load('models_extend/wd50k_vae_half_rotate_base_degree_estimator0.001_32_1_.pth')
                if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
                    estimator_dic = torch.load('models_extend/wd50k_vae_half_rotate_base__pattern_estimator0.001_32_1_.pth')
                else:
                    estimator_dic = torch.load('models_extend/wd50k_vae_half_rotate_base____sum_estimator0.001_32_1_.pth')
                # estimator_dic = torch.load('models_extend/transformer_rotate_estimator5e-05_32_lp_.pth')
                # estimator_dic = torch.load('models_extend/transformer_rotate_estimator0.0001_32_orgin_.pth')
            elif config['DATASET'] == 'jf17k':
                if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
                    estimator_dic = torch.load('models_extend/jf17k_vae_half_rotate_estimator0.001_32_1_.pth')
                else:
                    estimator_dic = torch.load('models_extend/jf17k_vae_half_rotate_base____sum_estimator0.001_32_1_.pth')
            elif config['DATASET'] == 'wikipeople':
                if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
                    estimator_dic = torch.load('models_extend/wikipeople_vae_half_rotate_estimator0.001_32_1_.pth')
                else:
                    estimator_dic = torch.load('models_extend/wikipeople_vae_half_rotate_base____sum_estimator0.001_32_1_.pth')
                
            self.estimator = VAE(encoder_layer_sizes = [200,256],
          latent_size = 200,
        #   decoder_layer_sizes = [256, h.shape[1]],
          decoder_layer_sizes = [512,200],
          dataset=config['DATASET'],
          mapping=None,
          edge_mapping=None,
          aggregate=config['STAREARGS']['QUAL_AGGREGATE'],
          conditional=True,
          conditional_size=600)
            self.estimator.to(config['DEVICE'])
            self.estimator.load_state_dict(estimator_dic)
            for param in self.estimator.parameters():
                param.requires_grad = False
        elif config['ESTIMATE'] == 'rulevae':
            if config['DATASET'] == 'wd50k' or config['DATASET'] == 'wd50k_nary':
                # estimator_dic = torch.load('models_extend/vae_half_rotate_estimator0.001_32_1_.pth')
                # estimator_dic = torch.load('models_extend/wd50k_vae_half_rotate_baseline_estimator0.001_32_1_.pth')
                # estimator_dic = torch.load('models_extend/wd50k_vae_half_rotate_subgraph_update_estimator0.001_32_1_.pth')
                # estimator_dic = torch.load('models_extend/wd50k_vae_half_rotate_base_degree_estimator0.001_32_1_.pth')
                if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
                    estimator_dic = torch.load('models_extend/wd50k_vae_half_rotate_base___rule_cat_estimator0.001_32_1_.pth')
                    # print('loaded')
                else:
                    estimator_dic = torch.load('models_extend/wd50k_vae_half_rotate_base____sum_estimator0.001_32_1_.pth')
                # estimator_dic = torch.load('models_extend/transformer_rotate_estimator5e-05_32_lp_.pth')
                # estimator_dic = torch.load('models_extend/transformer_rotate_estimator0.0001_32_orgin_.pth')
            elif config['DATASET'] == 'jf17k':
                if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
                    estimator_dic = torch.load('models_extend/jf17k_vae_half_rotate_estimator0.001_32_1_.pth')
                else:
                    estimator_dic = torch.load('models_extend/jf17k_vae_half_rotate_base____sum_estimator0.001_32_1_.pth')
            elif config['DATASET'] == 'wikipeople':
                if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
                    estimator_dic = torch.load('models_extend/wikipeople_vae_half_rotate_estimator0.001_32_1_.pth')
                else:
                    estimator_dic = torch.load('models_extend/wikipeople_vae_half_rotate_base____sum_estimator0.001_32_1_.pth')
            self.estimator = VAE(encoder_layer_sizes = [200,256],
          latent_size = 200,
        #   decoder_layer_sizes = [256, h.shape[1]],
          decoder_layer_sizes = [512,200],
          dataset=config['DATASET'],
          mapping=None,
          edge_mapping=None,
          aggregate=config['STAREARGS']['QUAL_AGGREGATE'],
          conditional=True,
          conditional_size=800)
            self.estimator.to(config['DEVICE'])
            self.estimator.load_state_dict(estimator_dic)
            for param in self.estimator.parameters():
                param.requires_grad = False
        elif config['ESTIMATE'] == 'gaussian':
            if config['DATASET'] == 'wd50k'  or config['DATASET'] == 'wd50k_nary':
                with open("models_extend/gaussian_15_rotate_estimator.pth", 'rb') as fr:
                    gaussian = pickle.load(fr)
            elif config['DATASET'] == 'jf17k':
                with open("models_extend/gaussian_jf17k_15_rotate_estimator.pth", 'rb') as fr:
                    gaussian = pickle.load(fr)
            elif config['DATASET'] == 'wikipeople':
                with open("models_extend/gaussian_wikipeople_15_rotate_estimator.pth", 'rb') as fr:
                    gaussian = pickle.load(fr)
            gmm_means = torch.tensor(gaussian.means_, dtype=torch.float64).to(config['DEVICE'])
            gmm_covariances = torch.tensor(gaussian.covariances_, dtype=torch.float64).to(config['DEVICE'])
            gmm_weights = torch.tensor(gaussian.weights_, dtype=torch.float64).to(config['DEVICE'])
            gmm = PredictYModel(device=config['DEVICE'],num_components=gmm_weights.shape[0], num_features=800).to(config['DEVICE'])

            # Set the PyTorch model parameters with the GMM parameters
            with torch.no_grad():
                gmm.means.copy_(gmm_means.clone())
                gmm.covariances.copy_(gmm_covariances.clone())
                gmm.weights.copy_(gmm_weights.clone())
                gmm.compute_dist()
            linear_es = LinearEstimator(800,200)
            estimator_dic = torch.load('models_extend/gaussian_15_rotate_estimator_linear.pth')
            linear_es.load_state_dict(estimator_dic)
            linear_es.to(config['DEVICE'])
            self.estimator = {'gaussian':gmm,'linear':linear_es}
            for param in self.estimator['linear'].parameters():
                param.requires_grad = False
            for param in self.estimator['gaussian'].parameters():
                param.requires_grad = False
        else:
            self.estimator= None
        self.config = config
        # init_ent = torch.cat((self.pretrained['init_embed'],torch.tensor(np.ones(200), device=self.device).view(1,200)), dim = 0)
        # if config['USE_VAR']:
        #     init_rel = self.pretrained['init_rel'].to(self.device)
        # else:
        #     if config['var'] == 'zero':
        #         if config['MULTI_DIM'] is False:
        #             init_rel = torch.cat((torch.tensor(np.zeros(config['EMBEDDING_DIM']), device=self.device).view(1,config['EMBEDDING_DIM']),self.pretrained['init_rel'][1:532].to(self.device),torch.tensor(np.zeros(config['EMBEDDING_DIM']), device=self.device).view(1,config['EMBEDDING_DIM']),self.pretrained['init_rel'][533:].to(self.device)),dim = 0)
        #         else:
                    
        #             init_rel = torch.cat((torch.tensor(np.zeros(config['EMBEDDING_DIM']*config['MULTI_DIM_SIZE']), device=self.device).view(config['MULTI_DIM_SIZE'],config['EMBEDDING_DIM']),self.pretrained['init_rel'][1:532].to(self.device),torch.tensor(np.zeros(config['EMBEDDING_DIM']*config['MULTI_DIM_SIZE']), device=self.device).view(config['MULTI_DIM_SIZE'],config['EMBEDDING_DIM']),self.pretrained['init_rel'][533:].to(self.device)),dim = 0)
        #     elif config['var'] == 'one':
        #         if config['MULTI_DIM'] is False:
        #             init_rel = torch.cat((torch.tensor(np.ones(config['EMBEDDING_DIM']), device=self.device).view(1,config['EMBEDDING_DIM']),self.pretrained['init_rel'][1:532].to(self.device),torch.tensor(np.ones(config['EMBEDDING_DIM']), device=self.device).view(1,config['EMBEDDING_DIM']),self.pretrained['init_rel'][533:].to(self.device)),dim = 0)
        #         else:
        #             print((torch.tensor(np.zeros(config['EMBEDDING_DIM']*config['MULTI_DIM_SIZE']), device=self.device).view(config['MULTI_DIM_SIZE'],config['EMBEDDING_DIM'])).shape)
        #             print((self.pretrained['init_rel'][1:532].to(self.device)).shape)
        #             init_rel = torch.cat((torch.tensor(np.ones(config['EMBEDDING_DIM']*config['MULTI_DIM_SIZE']), device=self.device).view(config['MULTI_DIM_SIZE'],config['EMBEDDING_DIM']),self.pretrained['init_rel'][1:532].to(self.device),torch.tensor(np.ones(config['EMBEDDING_DIM']*config['MULTI_DIM_SIZE']), device=self.device).view(config['MULTI_DIM_SIZE'],config['EMBEDDING_DIM']),self.pretrained['init_rel'][533:].to(self.device)),dim = 0)
        init_rel = self.pretrained['init_rel'].to(self.device)
        if config['DISTILL']:
            init_rel1 = self.pretrained1['init_rel'].to(self.device)
            # if config['TRIPLE']:
            #     if config['var'] == 'zero':
            #         init_triple_rel = torch.cat((torch.tensor(np.zeros(config['EMBEDDING_DIM']), device=self.device).view(1,config['EMBEDDING_DIM']),self.triple_embed['init_rel'][1:triple_r_num].to(self.device),torch.tensor(np.zeros(config['EMBEDDING_DIM']), device=self.device).view(1,config['EMBEDDING_DIM']),self.triple_embed['init_rel'][triple_r_num+1:].to(self.device)),dim = 0)
            #     elif config['var'] == 'one':
            #         init_triple_rel = torch.cat((torch.tensor(np.ones(config['EMBEDDING_DIM']), device=self.device).view(1,config['EMBEDDING_DIM']),self.triple_embed['init_rel'][1:triple_r_num].to(self.device),torch.tensor(np.ones(config['EMBEDDING_DIM']), device=self.device).view(1,config['EMBEDDING_DIM']),self.triple_embed['init_rel'][triple_r_num+1:].to(self.device)),dim = 0)
        self.init_rel = [] #self.init_rel
        # print(f'{ents} {self.init_ent.shape}')
        self.init_ent = []
        self.init_triple_ent = []
        self.init_triple_rel = []
        self.init_rel1 = [] #self.init_rel
        # print(f'{ents} {self.init_ent.shape}')
        self.init_ent1 = []
        # for ent in ents:
        #     x.append(self.init_ent[ent])
        # print(self.pretrained['init_embed'].shape)
        
        for idx,embed in enumerate(self.pretrained['init_embed']):
            if idx == 0:
                if config['USE_VAR']:
                    self.init_ent.append(self.pretrained['init_embed'][idx].to(self.device))
                else:
                    if config['MULTI_DIM']:
                        if config['var'] == 'zero':
                            self.init_ent.append(torch.tensor(np.zeros(config['EMBEDDING_DIM']*config['MULTI_DIM_SIZE']),device=self.device).view(config['MULTI_DIM_SIZE'],config['EMBEDDING_DIM']))
                        elif config['var'] == 'one':
                            self.init_ent.append(torch.tensor(np.ones(config['EMBEDDING_DIM']*config['MULTI_DIM_SIZE']),device=self.device).view(config['MULTI_DIM_SIZE'],config['EMBEDDING_DIM']))
                    else:
                        if config['var'] == 'zero':
                            self.init_ent.append(torch.tensor(np.zeros(config['EMBEDDING_DIM']),device=self.device))
                        elif config['var'] == 'one':
                            self.init_ent.append(torch.tensor(np.ones(config['EMBEDDING_DIM']),device=self.device))
            else:
                self.init_ent.append(self.pretrained['init_embed'][idx].to(self.device))
        if config['DISTILL']:
            for idx,embed in enumerate(self.pretrained1['init_embed']):
                if idx == 0:
                    if config['USE_VAR']:
                        self.init_ent1.append(self.pretrained1['init_embed'][idx].to(self.device))
                    else:
                        if config['MULTI_DIM']:
                            if config['var'] == 'zero':
                                self.init_ent1.append(torch.tensor(np.zeros(config['EMBEDDING_DIM']*config['MULTI_DIM_SIZE']),device=self.device).view(config['MULTI_DIM_SIZE'],config['EMBEDDING_DIM']))
                            elif config['var'] == 'one':
                                self.init_ent1.append(torch.tensor(np.ones(config['EMBEDDING_DIM']*config['MULTI_DIM_SIZE']),device=self.device).view(config['MULTI_DIM_SIZE'],config['EMBEDDING_DIM']))
                        else:
                            if config['var'] == 'zero':
                                self.init_ent1.append(torch.tensor(np.zeros(config['EMBEDDING_DIM']),device=self.device))
                            elif config['var'] == 'one':
                                self.init_ent1.append(torch.tensor(np.ones(config['EMBEDDING_DIM']),device=self.device))
                else:
                    self.init_ent1.append(self.pretrained1['init_embed'][idx].to(self.device))
        
        
                    
        for idx,embed in enumerate(init_rel):
            # if idx != 0 and idx != 532:
                # if 
            if idx >= r_num:
                if config['REVERSE']:
                    self.init_rel.append(init_rel[idx])
                else:
                    self.init_rel.append(init_rel[idx-r_num])
            else:
                self.init_rel.append(init_rel[idx])
        if self.config['DISTILL']:
            for idx,embed in enumerate(init_rel1):
                # if idx != 0 and idx != 532:
                    # if 
                if idx >= r_num:
                    if config['REVERSE']:
                        self.init_rel1.append(init_rel1[idx])
                    else:
                        self.init_rel1.append(init_rel1[idx-r_num])
                else:
                    self.init_rel1.append(init_rel1[idx])
        if config['TRIPLE']:
            self.init_triple_rel = torch.stack([t for t in self.init_triple_rel])
        if config['STAREARGS']['QUAL_AGGREGATE'] == 'sum' or config['STAREARGS']['QUAL_AGGREGATE'] == 'mul' or config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
            self.conv_w_q= []
            self.conv_w_q_1 = get_param((config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
            self.conv_w_q_2 = get_param((config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
            if config['LAYERS'] >= 3:
                self.conv_w_q_3 = get_param((config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
            if config['LAYERS'] >= 5:
                self.conv_w_q_4 = get_param((config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
                self.conv_w_q_5 = get_param((config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
            if config['LAYERS'] >= 7:
                self.conv_w_q_6 = get_param((config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
                self.conv_w_q_7 = get_param((config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
            if config['LAYERS'] >= 10:
                self.conv_w_q_8 = get_param((config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
                self.conv_w_q_9 = get_param((config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
                self.conv_w_q_10 = get_param((config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
        else:
            self.conv_w_q_1 = get_param((2*config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
            self.conv_w_q_2 = get_param((2*config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
            if config['LAYERS'] >= 3:
                self.conv_w_q_3 = get_param((2*config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
            if config['LAYERS'] >= 5:
                self.conv_w_q_3 = get_param((2*config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
                self.conv_w_q_4 = get_param((2*config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
                self.conv_w_q_5 = get_param((2*config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
            if config['LAYERS'] >= 7:
                self.conv_w_q_6 = get_param((2*config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
                self.conv_w_q_7 = get_param((2*config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
            if config['LAYERS'] >= 10:
                self.conv_w_q_8 = get_param((2*config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
                self.conv_w_q_9 = get_param((2*config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
                self.conv_w_q_10 = get_param((2*config['EMBEDDING_DIM'],config['EMBEDDING_DIM'])).to(config['DEVICE'])
         
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(config['EMBEDDING_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(config['HID_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2)
        ).to(config['DEVICE'])
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(config['EMBEDDING_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(config['HID_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2)
        ).to(config['DEVICE'])
        if config['LAYERS'] >= 3:
            self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(config['EMBEDDING_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(config['HID_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2)
        ).to(config['DEVICE'])
            
        if config['LAYERS'] >= 5:
            
            self.mlp4 = torch.nn.Sequential(
            torch.nn.Linear(config['EMBEDDING_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(config['HID_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2)
        ).to(config['DEVICE'])
            self.mlp5 = torch.nn.Sequential(
            torch.nn.Linear(config['EMBEDDING_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(config['HID_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2)
        ).to(config['DEVICE'])
        if config['LAYERS'] >= 7:
            self.mlp6 = torch.nn.Sequential(
            torch.nn.Linear(config['EMBEDDING_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(config['HID_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2)
        ).to(config['DEVICE'])
            self.mlp7 = torch.nn.Sequential(
            torch.nn.Linear(config['EMBEDDING_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(config['HID_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2)
        ).to(config['DEVICE'])
            
        if config['LAYERS'] >= 10:
            self.mlp8 = torch.nn.Sequential(
            torch.nn.Linear(config['EMBEDDING_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(config['HID_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2)
        ).to(config['DEVICE'])
            self.mlp9 = torch.nn.Sequential(
            torch.nn.Linear(config['EMBEDDING_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(config['HID_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2)
        ).to(config['DEVICE'])
            self.mlp10 = torch.nn.Sequential(
            torch.nn.Linear(config['EMBEDDING_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(config['HID_DIM'], config['HID_DIM']),
            # torch.nn.Dropout(p=0.2)
        ).to(config['DEVICE'])
        
        if config['ESTIMATE_GATE'] != 'none':
            self.G_BRFE = Gbrfe(600).to(self.device)
            if config['DATASET'] == 'wd50k'  or config['DATASET'] == 'wd50k_nary':
                with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/train.txt', 'r') as f:
                    raw_trn = []
                    for line in f.readlines():
                        raw_trn.append(line.strip("\n").split(",")[:3])
                with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/test.txt', 'r') as f:
                    raw_tst = []
                    for line in f.readlines():
                        raw_tst.append(line.strip("\n").split(",")[:3])
                with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/valid.txt', 'r') as f:
                    raw_val = []
                    for line in f.readlines():
                        raw_val.append(line.strip("\n").split(",")[:3])
            elif config['DATASET'] == 'jf17k':
                with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/train.txt', 'r') as f:
                    raw_trn = []
                    triple_trn = []
                    for line in f.readlines():
                        raw_trn.append(line.strip("\n").split(",")[:3])
                raw_val = []
                with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/test.txt', 'r') as f:
                    raw_tst = []
                    triple_tst = []
                    for line in f.readlines():
                        raw_tst.append(line.strip("\n").split(",")[:3])
            elif config['DATASET'] == 'wikipeople':
                # Load data from disk

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
                raw_trn, raw_tst, raw_val = _conv_to_our_format_(raw_trn, filter_literals=True), \
                                            _conv_to_our_format_(raw_tst, filter_literals=True), \
                                            _conv_to_our_format_(raw_val, filter_literals=True)
                
            augmented_train = []
            triple_dic = {}
            for d in raw_trn:
                augmented_train.append(d)
                if config['ESTIMATE_GATE'] == 'novargaussian':
                    continue
                
                if (d[0],d[1],d[2]) not in triple_dic:
                    triple_dic[(d[0],d[1],d[2])] = 0
                triple_dic[(d[0],d[1],d[2])] += 1
                for i in range(0,3):
                    d1 = copy.deepcopy(d)
                    if i >= len(d1):
                        break
                    d1[i] = '?'+str(i)
                    if (d1[0],d1[1],d1[2]) not in triple_dic:
                        triple_dic[(d1[0],d1[1],d1[2])] = 0
                    triple_dic[(d1[0],d1[1],d1[2])] += 1
                    if config['ESTIMATE_GATE'] == 'qrgaussian' and random.uniform(0,1) <= qual_ratio[config['DATASET']] or config['ESTIMATE_GATE'] == 'gaussian':
                        augmented_train.append(d1)
                for i in range(0,3):
                    d1 = copy.deepcopy(d)
                    if i ==0:
                        d1[0] = '?'+str(i)
                        d1[2] = '?'+str(i)
                    elif i == 1:
                        d1[1] = '?'+str(i)
                        d1[2] = '?'+str(i)
                    elif i == 1:
                        d1[0] = '?'+str(i)
                        d1[1] = '?'+str(i)
                    if (d1[0],d1[1],d1[2]) not in triple_dic:
                        triple_dic[(d1[0],d1[1],d1[2])] = 0
                    triple_dic[(d1[0],d1[1],d1[2])] += 1
                    if config['ESTIMATE_GATE'] == 'qrgaussian' and random.uniform(0,1) <= qual_ratio[config['DATASET']] or config['ESTIMATE_GATE'] == 'gaussian':
                        augmented_train.append(d1)
                
            augmented_valid = []
            for d in raw_val:
                augmented_valid.append(d)
                # if random.uniform(0,1) <= 0.3:
                if config['ESTIMATE_GATE'] == 'novargaussian':
                    continue
                for i in range(0,3):
                    d1 = copy.deepcopy(d)
                    if i >= len(d1):
                        break
                    
                    d1[i] = '?'+str(i)
                    if (d1[0],d1[1],d1[2]) not in triple_dic:
                        triple_dic[(d1[0],d1[1],d1[2])] = 0
                    triple_dic[(d1[0],d1[1],d1[2])] += 1
                    if config['ESTIMATE_GATE'] == 'qrgaussian' and random.uniform(0,1) <= qual_ratio[config['DATASET']] or config['ESTIMATE_GATE'] == 'gaussian':
                        augmented_valid.append(d1)
                
                for i in range(0,3):
                    d1 = copy.deepcopy(d)
                    
                    if i ==0:
                        d1[0] = '?'+str(i)
                        d1[2] = '?'+str(i)
                    elif i == 1:
                        d1[1] = '?'+str(i)
                        d1[2] = '?'+str(i)
                    elif i == 1:
                        d1[0] = '?'+str(i)
                        d1[1] = '?'+str(i)
                    if (d1[0],d1[1],d1[2]) not in triple_dic:
                        triple_dic[(d1[0],d1[1],d1[2])] = 0
                    triple_dic[(d1[0],d1[1],d1[2])] += 1
                    if config['ESTIMATE_GATE'] == 'qrgaussian' and random.uniform(0,1) <= qual_ratio[config['DATASET']] or config['ESTIMATE_GATE'] == 'gaussian':
                        augmented_valid.append(d1)
            
            augmented_test = []
            for d in raw_tst:
                augmented_test.append(d)
                # if random.uniform(0,1) <= 0.3:
                if config['ESTIMATE_GATE'] == 'novargaussian':
                    continue
                for i in range(0,3):
                    d1 = copy.deepcopy(d)
                    if i >= len(d1):
                        break
                    d1[i] = '?'+str(i)
                    if (d1[0],d1[1],d1[2]) not in triple_dic:
                        triple_dic[(d1[0],d1[1],d1[2])] = 0
                    triple_dic[(d1[0],d1[1],d1[2])] += 1
                    if config['ESTIMATE_GATE'] == 'qrgaussian' and random.uniform(0,1) <= qual_ratio[config['DATASET']] or config['ESTIMATE_GATE'] == 'gaussian':
                        augmented_test.append(d1)
                for i in range(0,3):
                    d1 = copy.deepcopy(d)
                    if i ==0:
                        d1[0] = '?'+str(i)
                        d1[2] = '?'+str(i)
                    elif i == 1:
                        d1[1] = '?'+str(i)
                        d1[2] = '?'+str(i)
                    elif i == 1:
                        d1[0] = '?'+str(i)
                        d1[1] = '?'+str(i)
                    if (d1[0],d1[1],d1[2]) not in triple_dic:
                        triple_dic[(d1[0],d1[1],d1[2])] = 0
                    triple_dic[(d1[0],d1[1],d1[2])] += 1
                    if config['ESTIMATE_GATE'] == 'qrgaussian' and random.uniform(0,1) <= qual_ratio[config['DATASET']] or config['ESTIMATE_GATE'] == 'gaussian':
                        augmented_test.append(d1)
            self.triple_dic = triple_dic 
            data = augmented_train + augmented_valid + augmented_test
            h = []
            for d in data:
                x_embed = []
                # print(x_items)
                for i,item in enumerate(d):
                    if item.startswith('?'):
                        x_embed.append(torch.cat((torch.tensor([i],device=self.device),torch.tensor(np.zeros(199),device=self.device))))
                        # fact_vector.append(i)
                        
                    else:
                        
                        if item in self.edgem:
                            x_embed.append(self.init_rel[self.edgem[item]])
                            
                        else:
                            x_embed.append(self.init_ent[self.nodem[item]])
                x = torch.cat([x_embed[0],x_embed[1],x_embed[2]] ,dim=0).float().to(self.device)
                h.append(x)
            h = torch.stack([t for t in h]).to(self.device)
            self.mu = torch.mean(h,dim=0)
            self.sigma = torch.var(h,dim=0)
            self.G_BRFE.mu.weight.data = self.mu.clone()
            self.G_BRFE.sigma.weight.data = self.sigma.clone()
            self.gussian = {'model':self.G_BRFE,'mu':self.mu,'sigma':self.sigma,'triples':triple_dic}
            if config['ESTIMATE_GATE'] == 'linear':
                self.mu = 0
                self.sigma = 1
                self.gussian = None
        
        if config['ESTIMATE']:
            if config['ESTIMATE_GATE'] != 'none':
                self.conv_1 = NaryConv(idx=1,nn=self.mlp1,w_q=self.conv_w_q_1,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,gussian=self.gussian,config=config).to(config['DEVICE'])
                self.conv_2 = NaryConv(idx=2,nn=self.mlp2,w_q=self.conv_w_q_2,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,gussian=self.gussian,config=config).to(config['DEVICE'])
                if config["LAYERS"] >= 3:
                    self.conv_3 = NaryConv(idx=3,nn=self.mlp3,w_q=self.conv_w_q_3,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,gussian=self.gussian,config=config).to(config['DEVICE'])
                if config["LAYERS"] >= 5:
                    self.conv_4 = NaryConv(idx=4,nn=self.mlp4,w_q=self.conv_w_q_4,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,gussian=self.gussian,config=config).to(config['DEVICE'])
                    self.conv_5 = NaryConv(idx=5,nn=self.mlp5,w_q=self.conv_w_q_5,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,gussian=self.gussian,config=config).to(config['DEVICE'])
                if config["LAYERS"] >= 7:
                    self.conv_6 = NaryConv(idx=6,nn=self.mlp6,w_q=self.conv_w_q_6,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,gussian=self.gussian,config=config).to(config['DEVICE'])
                    self.conv_7 = NaryConv(idx=7,nn=self.mlp7,w_q=self.conv_w_q_7,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,gussian=self.gussian,config=config).to(config['DEVICE'])
            else:
                self.conv_1 = NaryConv(idx=1,nn=self.mlp1,w_q=self.conv_w_q_1,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,config=config).to(config['DEVICE'])
                self.conv_2 = NaryConv(idx=2,nn=self.mlp2,w_q=self.conv_w_q_2,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,config=config).to(config['DEVICE'])
                if config["LAYERS"] >= 3:
                    self.conv_3 = NaryConv(idx=3,nn=self.mlp3,w_q=self.conv_w_q_3,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,config=config).to(config['DEVICE'])
                if config["LAYERS"] >= 5:
                    self.conv_4 = NaryConv(idx=4,nn=self.mlp4,w_q=self.conv_w_q_4,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,config=config).to(config['DEVICE'])
                    self.conv_5 = NaryConv(idx=5,nn=self.mlp5,w_q=self.conv_w_q_5,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,config=config).to(config['DEVICE'])
                if config["LAYERS"] >= 7:
                    self.conv_6 = NaryConv(idx=6,nn=self.mlp6,w_q=self.conv_w_q_6,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,config=config).to(config['DEVICE'])
                    self.conv_7 = NaryConv(idx=7,nn=self.mlp7,w_q=self.conv_w_q_7,edge_dim=config['EMBEDDING_DIM'],estimator=self.estimator,config=config).to(config['DEVICE'])
        else:
            self.conv_1 = NaryConv(idx=1,nn=self.mlp1,w_q=self.conv_w_q_1,edge_dim=config['EMBEDDING_DIM'],estimator=None,config=config).to(config['DEVICE'])
            self.conv_2 = NaryConv(idx=2,nn=self.mlp2,w_q=self.conv_w_q_2,edge_dim=config['EMBEDDING_DIM'],estimator=None,config=config).to(config['DEVICE'])
            if config["LAYERS"] >= 3:
                self.conv_3 = NaryConv(idx=3,nn=self.mlp3,w_q=self.conv_w_q_3,edge_dim=config['EMBEDDING_DIM'],estimator=None,config=config).to(config['DEVICE'])
            if config["LAYERS"] >= 5:
                self.conv_4 = NaryConv(idx=4,nn=self.mlp4,w_q=self.conv_w_q_4,edge_dim=config['EMBEDDING_DIM'],estimator=None,config=config).to(config['DEVICE'])
                self.conv_5 = NaryConv(idx=5,nn=self.mlp5,w_q=self.conv_w_q_5,edge_dim=config['EMBEDDING_DIM'],estimator=None,config=config).to(config['DEVICE'])
            if config["LAYERS"] >= 7:
                self.conv_6 = NaryConv(idx=6,nn=self.mlp6,w_q=self.conv_w_q_6,edge_dim=config['EMBEDDING_DIM'],estimator=None,config=config).to(config['DEVICE'])
                self.conv_7 = NaryConv(idx=7,nn=self.mlp7,w_q=self.conv_w_q_7,edge_dim=config['EMBEDDING_DIM'],estimator=None,config=config).to(config['DEVICE'])
            # self.lin0 = torch.nn.Linear(2*config['EMBEDDING_DIM'], config['EMBEDDING_DIM'])

        self.n_layer = config["LAYERS"]
        self.proj = Linear(200,1)
        # self.lin = Linear(200, 50)
        # self.lin2 = Linear(50, 1)
        if config['HID_DIM'] >= 202:
            self.lin = Linear(config['HID_DIM'], 200)
            self.lin3 = Linear(200, 50)
        else:
            self.lin = Linear(200, 50)
            self.lin3 = None
            self.lin5 = None
        self.lin2 = Linear(50, 1)
        # self.cla = Linear(50, config['CLASSES'])
        self.ent = ent
        
        # self.dropout = torch.nn.Dropout(p=0.2)
    
    def transform(self,ents,rels,nodem,edgem,graph_repr):
        
        x = [[] for i in range(len(nodem.keys()))]
        r = [[] for i in range(len(edgem.keys()))] + [[] for i in range(len(edgem.keys()))]
        x_dis = [[] for i in range(len(nodem.keys()))]
        r_dis = [[] for i in range(len(edgem.keys()))] + [[] for i in range(len(edgem.keys()))]
        if graph_repr['quals'] is not None:
            quals = torch.tensor(graph_repr['quals'], dtype=torch.long, device=self.device)
        else:
            quals = None
        quals_ent = set()
        quals_rel = set()
        if quals is not None:
            for item in quals[0]:
                quals_ent.add(item)
            for item in quals[1]:
                quals_rel.add(item)
        for idx,embed in enumerate(self.init_ent):
            if idx in ents:
                
                for it in ents[idx]:
                    # if self.ent_map is not None and it not in self.ent_map:
                    #     print(f'node: {it}')
                    if idx == 0 or nodem[it] in quals_ent:
                        pass
                    else:
                        # x[nodem[it]] = torch.matmul(self.ent_trans_matrix,embed)
                        if self.config['DISTILL_AGGREGATION'] == 'part':
                            x[nodem[it]] = torch.matmul(embed,self.ent_trans_matrix1)
                            x_dis[nodem[it]] = self.init_ent1[idx]
                        else:
                            x[nodem[it]] = embed
                            x_dis[nodem[it]] = torch.matmul(self.init_ent1[idx],self.ent_trans_matrix1)
                    # if idx == 0:
                    #     x[nodem[it]][0] = var[it]
                    #     print(x[nodem[it]])
        for idx,embed in enumerate(self.init_rel):
            if idx in rels:
                for it in rels[idx]:
                    # if self.rel_map is not None and it not in self.rel_map:
                    #     print(f'pred: {it}')
                    if idx == 0 or edgem[it] in quals_rel:
                        pass
                    else:
                        if self.config['DISTILL_AGGREGATION'] == 'part':
                            r[edgem[it]] = torch.matmul(embed,self.rel_trans_matrix1)
                            r[edgem[it]+len(r)//2] = torch.matmul(self.init_rel[idx+self.r_num],self.rel_trans_matrix1)
                            r_dis[edgem[it]] = self.init_rel1[idx]
                            r_dis[edgem[it]+len(r)//2] = self.init_rel1[idx+self.r_num]
                        else:
                            r[edgem[it]] = embed
                            r[edgem[it]+len(r)//2] = self.init_rel[idx+self.r_num]
                            r_dis[edgem[it]] = torch.matmul(self.init_rel1[idx],self.rel_trans_matrix1)
                            r_dis[edgem[it]+len(r)//2] = torch.matmul(self.init_rel1[idx+self.r_num],self.rel_trans_matrix1)
        temp = []
        for item in x+r:
            if item != []:
                temp.append(item)
        # print(x)
        x = torch.stack([i for i in temp])
        temp = []
        for item in x_dis+r_dis:
            if item != []:
                temp.append(item)
        x_dis = torch.stack([i for i in temp])
        return x,x_dis

    def load_queries(self,ents,rels,nodem,edgem,var,graph_repr):
        if graph_repr['quals'] is not None:
            quals = torch.tensor(graph_repr['quals'], dtype=torch.long, device=self.device)
        else:
            quals = None
        quals_ent = set()
        quals_rel = set()
        # print()
        edge_index = torch.tensor(graph_repr['edge_index'], dtype=torch.long, device=self.device)
        edge_type = torch.tensor(graph_repr['edge_type'], dtype=torch.long, device=self.device)
        # print(edge_index)
        # print(edge_type)
        qual_l = quals.tolist()
        if quals is not None:
            for item in quals[0]:
                quals_ent.add(item)
            for item in quals[1]:
                quals_rel.add(item)
        x = [[] for i in range(len(nodem.keys()))]
        r = [[] for i in range(len(edgem.keys()))] + [[] for i in range(len(edgem.keys()))]
        # print(nodem)
        # print(ents)
        # print(quals)
        # print(var)
        # print(rels)
        # print(edgem)
        # print(len(self.init_ent))
        cnt = 0
        # if triple_f:
        for idx in ents:
            for it in ents[idx]:
                if idx == 0 and self.config['USE_VAR'] is False:
                    if self.config['var'] == 'zero':
                        
                        if self.config['ESTIMATE']:
                            if nodem[it] in qual_l[1]:
                                embed1 = torch.cat((torch.tensor([var[it]],device=self.device),torch.tensor(np.ones(self.config['EMBEDDING_DIM']-1),device=self.device)))
                            else:
                                embed1 = torch.cat((torch.tensor([var[it]],device=self.device),torch.tensor(np.zeros(self.config['EMBEDDING_DIM']-1),device=self.device)))
                        else:
                            embed1 = torch.cat((torch.tensor([var[it]],device=self.device),torch.tensor(np.zeros(self.config['EMBEDDING_DIM']-1),device=self.device)))

                    elif self.config['var'] == 'one':
                        embed1  = torch.cat((torch.tensor([var[it]],device=self.device),torch.tensor(np.ones(self.config['EMBEDDING_DIM']-1),device=self.device)))
                    
                    x[nodem[it]] = embed1.view(-1,self.config['EMBEDDING_DIM'])
                else:
                    x[nodem[it]] = self.init_ent[idx].view(-1,self.config['EMBEDDING_DIM'])
        for idx in rels:
            for it in rels[idx]:
                if idx == 0 and self.config['USE_VAR'] is False: 
                    if self.config['var'] == 'zero':
                        if self.config['OCCURENCES'] == 'position':
                            embed1 = torch.cat((torch.tensor([var[it]],device=self.device),torch.tensor(np.zeros(self.config['EMBEDDING_DIM']-3),device=self.device),torch.tensor(np.ones(2),device=self.device)))
                        else:
                            
                            if self.config['ESTIMATE']:
                                if edgem[it] in qual_l[0] or edgem[it]+len(r)//2 in qual_l[0]:
                                    embed1 = torch.cat((torch.tensor([var[it]],device=self.device),torch.tensor(np.ones(self.config['EMBEDDING_DIM']-1),device=self.device)))
                                else:
                                    embed1 = torch.cat((torch.tensor([var[it]],device=self.device),torch.tensor(np.zeros(self.config['EMBEDDING_DIM']-1),device=self.device)))
                            else:
                                embed1 = torch.cat((torch.tensor([var[it]],device=self.device),torch.tensor(np.zeros(self.config['EMBEDDING_DIM']-1),device=self.device)))
                        
                    elif self.config['var'] == 'one':
                        if self.config['DISTILL'] and self.config['DISTILL_AGGREGATION'] == 'cat':
                            embed1 = torch.cat((torch.tensor([var[it]],device=self.device),torch.tensor(np.ones(2*self.config['EMBEDDING_DIM']-1),device=self.device)))
                        else:                           
                            embed1 = torch.cat((torch.tensor([var[it]],device=self.device),torch.tensor(np.ones(self.config['EMBEDDING_DIM']-1),device=self.device)))  
                    r[edgem[it]] = embed1.view(-1,self.config['EMBEDDING_DIM'])
                    r[edgem[it]+len(r)//2] = embed1.view(-1,self.config['EMBEDDING_DIM'])
                else:
                    
                    r[edgem[it]] = self.init_rel[idx].view(-1,self.config['EMBEDDING_DIM'])
                    r[edgem[it]+len(r)//2] = self.init_rel[idx+self.r_num].view(-1,self.config['EMBEDDING_DIM'])
        # print(x)
        if self.config['INIT_EMBED'] == 'stare':
            if self.config['HAS_QUAL'] is False:
                temp_x = []
                for item in x:
                    if item == []:
                        temp_x.append(torch.tensor(np.ones(self.config['EMBEDDING_DIM']),device=self.device))
                    else:
                        temp_x.append(item)
                x = temp_x
        return x, r, edge_index, edge_type, quals
    #ents: {0:[?v0,?v1]} nodem:{?v0:0,?v1:0}
    def forward(self, x, r, edge_index, edge_type, quals, batch=None):
        freq = None
        # x = torch.stack([t for t in x]).float()
        # r = torch.stack([t for t in r]).float()
        return_emb = {}
        x2 = x.clone()
        r1 = r.clone()
        # x = x.to(self.config['DEVICE'])
        # r = r.to(self.config['DEVICE'])
        node_embedding = []
        x, r = self.conv_1(x=x, edge_index=edge_index,
                                edge_type=edge_type, rel_embed=r,
                                qualifier_ent=None,
                                qualifier_rel=None,
                                quals=quals,triple_index=freq)
        # print(x.shape)
        # node_embedding.append(x)
        
        x, r = self.conv_2(x=x, edge_index=edge_index,
                        edge_type=edge_type, rel_embed=r,
                        qualifier_ent=None,
                        qualifier_rel=None,
                        quals=quals,triple_index=freq) 
        node_embedding.append(x)

        if self.config['LAYERS'] >= 3:
            
            x, r = self.conv_3(x=x, edge_index=edge_index,
                        edge_type=edge_type, rel_embed=r,
                        qualifier_ent=None,
                        qualifier_rel=None,
                        quals=quals,triple_index=freq) 
            node_embedding.append(x)
        if self.config['LAYERS'] >= 5:
            x, r = self.conv_4(x=x, edge_index=edge_index,
                        edge_type=edge_type, rel_embed=r,
                        qualifier_ent=None,
                        qualifier_rel=None,
                        quals=quals,triple_index=freq) 
            node_embedding.append(x)
            x, r = self.conv_5(x=x, edge_index=edge_index,
                        edge_type=edge_type, rel_embed=r,
                        qualifier_ent=None,
                        qualifier_rel=None,
                        quals=quals,triple_index=freq) 
            node_embedding.append(x)
        if self.config['LAYERS'] >= 7:
            x, r = self.conv_6(x=x, edge_index=edge_index,
                        edge_type=edge_type, rel_embed=r,
                        qualifier_ent=None,
                        qualifier_rel=None,
                        quals=quals,triple_index=freq) 
            node_embedding.append(x)
            x, r = self.conv_7(x=x, edge_index=edge_index,
                        edge_type=edge_type, rel_embed=r,
                        qualifier_ent=None,
                        qualifier_rel=None,
                        quals=quals,triple_index=freq) 
            node_embedding.append(x)
        if self.config['LAYERS'] >= 3:
            outputs = []
            pps = torch.stack(node_embedding, dim=1)
            # print(pps.shape)
            # for i in range(pps.shape[0]):
            retain_score = self.proj(pps)
            retain_score = retain_score.squeeze()
            retain_score = torch.sigmoid(retain_score)
            retain_score = retain_score.unsqueeze(1)
            # print(retain_score.shape)
            # print(pps.shape)
            x = torch.matmul(retain_score, pps).squeeze(1)
            # print(x.shape)
            # print()
        if self.return_type == 'feature':
            return x,x2
        x = global_add_pool(F.relu(x), batch)
        if self.config['ESTIMATE'] == 'combine':
            x1 = global_add_pool(F.relu(x1), batch)
        # x1 = x
            x = torch.cat((x,x1),dim=0)
            # print(x.shape)
            x = torch.sum(x,dim=0).view(1,200)
        
        x = self.lin(x)
        if self.lin3 is not None:
            x = F.relu(x)
            x = self.lin3(x)
        x = F.relu(x)
        x = self.lin2(x)
        if self.config['ESTIMATE_GATE'] == 'none':
            if self.config['RETURN_EMBED']:
                return torch.abs(x),x1
            else:
                # if tpents is not None:
                if self.config['PRINT_VECTOR']:
                    return torch.abs(x), return_emb
                else:
                    return torch.abs(x)
        else:
            return torch.abs(x),self.sigma,self.mu

    def coalesce_quals(self, qual_embeddings, qual_index, num_edges, fill=0):
        """

        before:
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type          :   [0,0,0,0,0,0,0, .....]               (empty array of size num_edges)

        After:
            edge_type          :   [a+b+d,c+e+g,f ......]        (here each element in the list is of 200 dim)

        :param qual_embeddings: shape of [1, N_QUALS]
        :param qual_index: shape of [1, N_QUALS] which states which quals belong to which main relation from the index,
            that is, all qual_embeddings that have the same index have to be summed up
        :param num_edges: num_edges to return the appropriate tensor
        :param fill: fill value for the output matrix - should be 0 for sum/concat and 1 for mul qual aggregation strat
        :return: [1, N_EDGES]
        """
        # print(qual_index)
        # print(qual_embeddings.shape)
        # print(num_edges)
        if self.config['STAREARGS']['QUAL_N'] == 'sum':
            output = scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges)
        elif self.config['STAREARGS']['QUAL_N'] == 'mean':
            output = scatter_mean(qual_embeddings, qual_index, dim=0, dim_size=num_edges)

        if fill != 0:
            # by default scatter_ functions assign zeros to the output, so we assign them 1's for correct mult
            mask = output.sum(dim=-1) == 0
            output[mask] = fill

        return output

class FC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FC, self).__init__()
        self.fc = torch.nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super(MLP, self).__init__()
        self.fc1 = FC(in_ch, hid_ch)
        self.fc2 = FC(hid_ch, out_ch)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GAT(nn.Module):
    def __init__(self, input_feat_dim, out_dim, train_eps=True):
        super(GAT, self).__init__()
        # we can change the sequential nn
        self.GAT_layer = GATConv(input_feat_dim, out_dim, add_self_loops=False)

    def forward(self, in_feat, edge_list):
        # print(edge_list)
        # print(edge_list.shape)
        x = self.GAT_layer(in_feat.float(), edge_list)
        # x = self.GIN_layer_2(x, edge_list)

        return x
    
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=True)

    def forward(self, z1, z2):
        # 计算余弦相似度
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        dot_products = torch.mm(z1, z2.t())
        diag = torch.diag(dot_products)
        exp_diag = torch.exp(diag / self.temperature)
        
        # 计算负样本损失
        dot_products = dot_products - torch.diag(torch.diag(dot_products))
        exp_dot_products = torch.exp(dot_products / self.temperature)
        
        # 计算InfoNCE损失
        positive_loss = -torch.log(exp_diag / (exp_diag + exp_dot_products.sum(dim=1)))
        loss = positive_loss.mean()

        return loss

class Gbrfe(nn.Module):
    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.mu = nn.Embedding(1,dim)
        self.sigma = nn.Embedding(1,dim)

class PredictYModel(nn.Module):
    def __init__(self,device,num_components=7, num_features=800):
        super(PredictYModel, self).__init__()
        self.num_components = num_components
        self.num_features = num_features
        self.device= device
        # Parameters for the Gaussian Mixture Model
        self.means = nn.Parameter(torch.zeros(num_components, num_features))
        self.covariances = nn.Parameter(torch.eye(num_features).expand(num_components, num_features, num_features))
        self.weights = nn.Parameter(torch.ones(num_components) / num_components)
        self.distri = torch.distributions.MultivariateNormal(self.means, self.covariances)

    def forward(self, x,sample):
        # Calculate responsibilities using Gaussian Mixture Model
        with torch.no_grad():
            x = torch.cat((x,sample),dim=1).to(self.device)
            probabilities = self.distri.log_prob(x).to(self.device)
            log_weights = torch.log(self.weights).to(self.device)
            responsibilities = torch.softmax(probabilities + log_weights, dim=0).to(self.device)
            y = torch.sum(torch.mm(responsibilities.view(1,-1), self.means), dim=0).to(self.device)
        return y

    def compute_dist(self):
        self.distri = torch.distributions.MultivariateNormal(self.means, self.covariances)