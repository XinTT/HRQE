import os.path

import torch
from torch_geometric.data import Data, HeteroData, Dataset
import torch_geometric.transforms as T

import numpy as np
import json
import random

from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader
from typing import Union,List,Dict
from torch import Tensor
import copy
import warnings

class ToUndirectedCustom(BaseTransform):
    r"""
    Custom ToUndirected transform that does not merge the reverse edges,
    but changes the last dimension of the edge attributes to -1


    Converts a homogeneous or heterogeneous graph to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge
    :math:`(i,j) \in \mathcal{E}` (functional name: :obj:`to_undirected`).
    In heterogeneous graphs, will add "reverse" connections for *all* existing
    edge types.

    Args:
        reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)
        merge (bool, optional): If set to :obj:`False`, will create reverse
            edge types for connections pointing to the same source and target
            node type.
            If set to :obj:`True`, reverse edges will be merged into the
            original relation.
            This option only has effects in
            :class:`~torch_geometric.data.HeteroData` graph data.
            (default: :obj:`True`)
    """
    def __init__(self, reduce: str = "add", merge: bool = True):
        self.reduce = reduce
        self.merge = merge

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue
            # print(store)
            nnz = store.edge_index.size(1)

            if isinstance(data, HeteroData) and (store.is_bipartite()
                                                 or not self.merge):
                src, rel, dst = store._key
                # print('here')
                # Just reverse the connectivity and add edge attributes:
                row, col = store.edge_index
                rev_edge_index = torch.stack([col,row], dim=0)

                inv_store = data[dst, f'rev_{rel}', src]
                inv_store.edge_index = rev_edge_index
                for key, value in store.items():
                    if key == 'edge_index':
                        continue
                    if isinstance(value, Tensor) and value.size(0) == nnz:
                        inv_value = copy.deepcopy(value)
                        inv_value[0, -1] = -1
                        inv_store[key] = inv_value

                        #value[0, -1] = -1
                        #print(value.shape)
                        # print(value)
                        # print(inv_value)

            else:
                keys, values = [], []
                for key, value in store.items():
                    if key == 'edge_index':
                        continue

                    if store.is_edge_attr(key):
                        keys.append(key)
                        values.append(value)
                
                store.edge_index, values = to_undirected(
                    store.edge_index, values, reduce=self.reduce)
                
                for key, value in zip(keys, values):
                    store[key] = value

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


# class ToUndirectedCustom(BaseTransform):
#     r"""
#     Custom ToUndirected transform that does not merge the reverse edges,
#     but changes the last dimension of the edge attributes to -1


#     Converts a homogeneous or heterogeneous graph to an undirected graph
#     such that :math:`(j,i) \in \mathcal{E}` for every edge
#     :math:`(i,j) \in \mathcal{E}` (functional name: :obj:`to_undirected`).
#     In heterogeneous graphs, will add "reverse" connections for *all* existing
#     edge types.

#     Args:
#         reduce (string, optional): The reduce operation to use for merging edge
#             features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
#             :obj:`"mul"`). (default: :obj:`"add"`)
#         merge (bool, optional): If set to :obj:`False`, will create reverse
#             edge types for connections pointing to the same source and target
#             node type.
#             If set to :obj:`True`, reverse edges will be merged into the
#             original relation.
#             This option only has effects in
#             :class:`~torch_geometric.data.HeteroData` graph data.
#             (default: :obj:`True`)
#     """
#     def __init__(self, reduce: str = "add", merge: bool = True):
#         self.reduce = reduce
#         self.merge = merge

#     def __call__(self, data: Union[Data, HeteroData]):
#         for store in data.edge_stores:
#             if 'edge_index' not in store:
#                 continue
#             # print(store)
#             nnz = store.edge_index.size(-1)

#             if isinstance(data, HeteroData) and (store.is_bipartite()
#                                                  or not self.merge):
#                 src, rel, dst = store._key
#                 # print('here')
#                 # Just reverse the connectivity and add edge attributes:
#                 temp = []
#                 for item in store.edge_index:
#                     temp.append(item)
#                 rev_edge_index = torch.stack(list(reversed(temp)), dim=0)

#                 inv_store = data[dst, f'rev_{rel}', src]
#                 inv_store.edge_index = rev_edge_index
#                 for key, value in store.items():
#                     if key == 'edge_index':
#                         continue
#                     if isinstance(value, Tensor) and value.size(0) == nnz:
#                         inv_value = copy.deepcopy(value)
#                         inv_value[0, -1] = -1
#                         inv_store[key] = inv_value
#                         #value[0, -1] = -1
#                         #print(value.shape)
#                         #rint(value)

#             else:
#                 keys, values = [], []
#                 for key, value in store.items():
#                     if key == 'edge_index':
#                         continue

#                     if store.is_edge_attr(key):
#                         keys.append(key)
#                         values.append(value)

#                 store.edge_index, values = to_undirected(
#                     store.edge_index, values, reduce=self.reduce)

#                 for key, value in zip(keys, values):
#                     store[key] = value

#         return data

#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}()'


class StatisticsLoader():
    """
    This class is used to load the statistics of the dataset
    It wraps the folder containing the statistics in a dict like object
    """
    def __init__(self, statistics_path,length=200):
        self.statistics_path = statistics_path
        self.length = length
    def __getitem__(self, item):
        try:
            # print(f'{item} {self.statistics_path}')
            with open(os.path.join(self.statistics_path, item.replace("/", "|") + ".json")) as f:
                return json.load(f)

        except FileNotFoundError:
            print("Cant find embedding for", item)
            # Return a random embedding
            statistic_dict = {"embedding": [random.uniform(0, 1) for i in range(100)], "occurence": 0}
            return statistic_dict


def get_query_graph_data_nx(query_graph, statistics, device):
    """
    This function is used to get the graph data object from a query graph
    :param query_graph: Dict representing the query graph of the form {"triples": [triple1, triple2, ...], "y": cardinality,
    "query": String of sparql query, "x": List of occurring entities}
    :param statistics: Dict or dict like loader for the embeddings of entities
    :param device: cpu or cuda
    :return: Graph data object
    """
    data = HeteroData()
    data = data.to(device)
    node_mapping = {}
    n_entities = 0
    node_embeddings = []
    # Embeddings for variables in edges or nodes
    # edge has an additional dimension to indicate the direction of the edge
    variable_embedding_edge = np.ones(102)
    variable_embedding = np.ones(101)
    feature_vector = None
    
    for (n1,n2) in query_graph.edges():
        edge_data = query_graph.get_edge_data(n1,n2)
        s = n1
        o = n2
        triple = [0,0,0]
        
        for k in edge_data:
            triple[1] = edge_data[k]['label']

            # If predicate is a variable, add the variable embedding to the edge attributes
            if "?" in triple[1]:
                p = 237
                try:
                    ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([variable_embedding_edge])), dim=0)
                    data["entity", p, "entity"].edge_attr = ea
                except:
                    # If the edge does not exist yet, create it
                    data["entity", p, "entity"].edge_attr = torch.tensor([variable_embedding_edge])

            else:
                try:
                    p = triple[1].replace("<http://example.com/", "").replace(">", "")
                    # p = triple[1].replace("<http://ex.org/", "").replace(">", "")

                except:
                    print(triple)
                    raise
                try:
                    # Get the embedding of the predicate
                    feature_vector = statistics[triple[1].replace("<", "").replace(">", "")]["embedding"].copy()
                    # Add the occurence of the predicate to the embedding
                    feature_vector.append(statistics[triple[1].replace("<", "").replace(">", "")]["occurence"] / 16018)
                    # Add a dimension for the direction of the edge
                    feature_vector.append(1)

                    ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([feature_vector])), dim=0)
                    data["entity", p, "entity"].edge_attr = ea
                except:
                    # Case if edge does not exist yet
                    feature_vector = statistics[triple[1].replace("<", "").replace(">", "")]["embedding"].copy()
                    feature_vector.append(statistics[triple[1].replace("<", "").replace(">", "")]["occurence"] / 16018)
                    feature_vector.append(1)

                    data["entity", p, "entity"].edge_attr = torch.tensor([feature_vector])

        # Adding the embeddings of s and o
        # print(s)
        # print(n1)
            if not s in node_mapping.keys():
                node_mapping[s] = n_entities
                n_entities += 1
                if "?" in s:
                    emb = variable_embedding
                    # indicate index of variable in query
                    emb[0] = node_mapping[s]
                    node_embeddings.append(emb)
                else:
                    # Get the embedding of the entity
                    feature_vector = statistics[s]["embedding"].copy()
                    # Add the occurence of the entity to the embedding
                    feature_vector.append(statistics[s]["occurence"] / 16018)
                    node_embeddings.append(feature_vector)
            if not o in node_mapping.keys():
                node_mapping[o] = n_entities
                n_entities += 1
                if "?" in o:
                    emb = variable_embedding
                    emb[0] = node_mapping[o]
                    node_embeddings.append(variable_embedding)
                else:
                    feature_vector = statistics[o]["embedding"].copy()
                    feature_vector.append(statistics[o]["occurence"] / 16018)
                    node_embeddings.append(feature_vector)

            # Finally, add the edge to the graph
            try:
                tp = torch.cat(
                    (data['entity', p, 'entity'].edge_index, torch.tensor([[node_mapping[s]], [node_mapping[o]]])), dim=1)
                data["entity", p, "entity"].edge_index = tp

            except:
                tp = torch.tensor([[node_mapping[s]], [node_mapping[o]]])
                data["entity", p, "entity"].edge_index = tp

    data["entity"].x = torch.tensor(node_embeddings)
    return data

def get_query_graph_data_new(query_graph, statistics, device):
    """
    This function is used to get the graph data object from a query graph
    :param query_graph: Dict representing the query graph of the form {"triples": [triple1, triple2, ...], "y": cardinality,
    "query": String of sparql query, "x": List of occurring entities}
    :param statistics: Dict or dict like loader for the embeddings of entities
    :param device: cpu or cuda
    :return: Graph data object
    """
    data = HeteroData()
    data = data.to(device)
    node_mapping = {}
    n_entities = 0
    node_embeddings = []
    # Embeddings for variables in edges or nodes
    # edge has an additional dimension to indicate the direction of the edge
    variable_embedding_edge = np.ones(102)
    variable_embedding = np.ones(101)
    feature_vector = None
    for triple in query_graph["triples"]:

        s = triple[0].replace("<", "").replace(">", "")
        o = triple[2].replace("<", "").replace(">", "")

        # If predicate is a variable, add the variable embedding to the edge attributes
        if "?" in triple[1]:
            p = 237
            try:
                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([variable_embedding_edge])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                # If the edge does not exist yet, create it
                data["entity", p, "entity"].edge_attr = torch.tensor([variable_embedding_edge])

        else:
            try:
                #p = int(triple[1].replace("<http://example.com/", "").replace(">", ""))
                p = triple[1].replace("<http://example.com/", "").replace(">", "")

            except:
                print(triple)
                raise
            try:
                # Get the embedding of the predicate
                feature_vector = statistics[triple[1].replace("<", "").replace(">", "")]["embedding"].copy()
                # Add the occurence of the predicate to the embedding
                feature_vector.append(statistics[triple[1].replace("<", "").replace(">", "")]["occurence"] / 16018)
                # Add a dimension for the direction of the edge
                feature_vector.append(1)

                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([feature_vector])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                # Case if edge does not exist yet
                feature_vector = statistics[triple[1].replace("<", "").replace(">", "")]["embedding"].copy()
                feature_vector.append(statistics[triple[1].replace("<", "").replace(">", "")]["occurence"] / 16018)
                feature_vector.append(1)

                data["entity", p, "entity"].edge_attr = torch.tensor([feature_vector])

        # Adding the embeddings of s and o
        if not s in node_mapping.keys():
            node_mapping[s] = n_entities
            n_entities += 1
            if "?" in s:
                emb = variable_embedding
                # indicate index of variable in query
                emb[0] = node_mapping[s]
                node_embeddings.append(emb)
            else:
                # Get the embedding of the entity
                feature_vector = statistics[s]["embedding"].copy()
                # Add the occurence of the entity to the embedding
                feature_vector.append(statistics[s]["occurence"] / 16018)
                node_embeddings.append(feature_vector)
        if not o in node_mapping.keys():
            node_mapping[o] = n_entities
            n_entities += 1
            if "?" in o:
                emb = variable_embedding
                emb[0] = node_mapping[o]
                node_embeddings.append(variable_embedding)
            else:
                feature_vector = statistics[o]["embedding"].copy()
                feature_vector.append(statistics[o]["occurence"] / 16018)
                node_embeddings.append(feature_vector)

        # Finally, add the edge to the graph
        try:
            tp = torch.cat(
                (data['entity', p, 'entity'].edge_index, torch.tensor([[node_mapping[s]], [node_mapping[o]]])), dim=1)
            data["entity", p, "entity"].edge_index = tp

        except:
            tp = torch.tensor([[node_mapping[s]], [node_mapping[o]]])
            data["entity", p, "entity"].edge_index = tp

    data["entity"].x = torch.tensor(node_embeddings)
    return data


def get_query_graph_data(query_graph, statistics, device):
    data = HeteroData()
    data = data.to(device)
    
    node_mapping = {}
    n_entities = 0
    node_embeddings = []
    variable_embedding = np.ones(101)
    feature_vector = None
    for triple in query_graph["triples"]:

        s = triple[0].replace("<", "").replace(">", "")
        o = triple[2].replace("<", "").replace(">", "")

        # If predicate is a variable
        if "?" in triple[1]:
            p = 237
            try:
                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([variable_embedding])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                data["entity", p, "entity"].edge_attr = torch.tensor([variable_embedding])

        else:
            try:
                p = int(triple[1].replace("<http://ex.org/", "").replace(">", ""))
            except:
                print(triple)
                raise
            try:
                feature_vector = statistics[triple[1].replace("<", "").replace(">", "")]["embedding"].copy()
                feature_vector.append(statistics[triple[1].replace("<", "").replace(">", "")]["occurence"] / 16018)

                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([feature_vector])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                feature_vector = statistics[triple[1].replace("<", "").replace(">", "")]["embedding"].copy()
                feature_vector.append(statistics[triple[1].replace("<", "").replace(">", "")]["occurence"] / 16018)
                data["entity", p, "entity"].edge_attr = torch.tensor([feature_vector])

        # Adding the embeddings of s and o to
        if not s in node_mapping.keys():
            node_mapping[s] = n_entities
            n_entities += 1
            if "?" in s:
                emb = variable_embedding
                emb[0] = node_mapping[s]
                node_embeddings.append(emb)
            else:
                feature_vector = statistics[s]["embedding"].copy()
                feature_vector.append(statistics[s]["occurence"] / 16018)
                node_embeddings.append(feature_vector)
        if not o in node_mapping.keys():
            node_mapping[o] = n_entities
            n_entities += 1
            if "?" in o:
                emb = variable_embedding
                emb[0] = node_mapping[s]
                node_embeddings.append(variable_embedding)
            else:
                feature_vector = statistics[o]["embedding"].copy()
                feature_vector.append(statistics[o]["occurence"] / 16018)
                node_embeddings.append(feature_vector)
        try:
            tp = torch.cat(
                (data['entity', p, 'entity'].edge_index, torch.tensor([[node_mapping[s]], [node_mapping[o]]])), dim=1)
            data["entity", p, "entity"].edge_index = tp

        except:
            tp = torch.tensor([[node_mapping[s]], [node_mapping[o]]])
            data["entity", p, "entity"].edge_index = tp

    data["entity"].x = torch.tensor(node_embeddings)
    return data

def get_query_graph_data_new_nary(query_graph,ent,rel, mapping, edge_mapping,device):
    """
    This function is used to get the graph data object from a query graph
    :param query_graph: Dict representing the query graph of the form {"triples": [triple1, triple2, ...], "y": cardinality,
    "query": String of sparql query, "x": List of occurring entities}
    :param statistics: Dict or dict like loader for the embeddings of entities
    :param device: cpu or cuda
    :return: Graph data object
    """
    data = HeteroData()
    data = data.to(device)
    node_mapping = {}
    n_entities = 0
    node_embeddings = []
    # Embeddings for variables in edges or nodes
    # edge has an additional dimension to indicate the direction of the edge
    variable_embedding_edge = np.ones(202)
    variable_embedding = np.ones(201)
    feature_vector = None
    for triple in query_graph["triples"]:

        s = triple[0].replace("<", "").replace(">", "")
        o = triple[2].replace("<", "").replace(">", "")

        # If predicate is a variable, add the variable embedding to the edge attributes
        if "?" in triple[1]:
            p = 0
            try:
                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([variable_embedding_edge])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                # If the edge does not exist yet, create it
                data["entity", p, "entity"].edge_attr = torch.tensor([variable_embedding_edge])

        else:
            try:
                #p = int(triple[1].replace("<http://example.com/", "").replace(">", ""))
                p = edge_mapping[triple[1].replace("<http://example.com/", "").replace(">", "")]

            except:
                print(triple)
                raise
            try:
               
                feature_vector = torch.cat((rel[p],torch.tensor([1],device=device))).view(1,202)

                ea = torch.cat((data["entity", p, "entity"].edge_attr, feature_vector), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                # Case if edge does not exist yet
                feature_vector = torch.cat((rel[p],torch.tensor([1],device=device))).view(1,202)

                data["entity", p, "entity"].edge_attr = feature_vector

        # Adding the embeddings of s and o
        if not s in node_mapping.keys():
            node_mapping[s] = n_entities
            n_entities += 1
            if "?" in s:
                emb = ent[0]
                # indicate index of variable in query
                emb[0] = node_mapping[s]
                node_embeddings.append(emb)
            else:
                # Get the embedding of the entity
                feature_vector = ent[mapping[s]]
                # Add the occurence of the entity to the embedding
                # feature_vector.append(statistics[s]["occurence"] / 16018)
                node_embeddings.append(feature_vector)
        if not o in node_mapping.keys():
            node_mapping[o] = n_entities
            n_entities += 1
            if "?" in o:
                emb = ent[0]
                # emb[0] = node_mapping[o]
                node_embeddings.append(emb)
            else:
                feature_vector = ent[mapping[o]]
                node_embeddings.append(feature_vector)

        # Finally, add the edge to the graph
        try:
            tp = torch.cat(
                (data['entity', p, 'entity'].edge_index, torch.tensor([[node_mapping[s]], [node_mapping[o]]])), dim=1)
            data["entity", p, "entity"].edge_index = tp

        except:
            tp = torch.tensor([[node_mapping[s]], [node_mapping[o]]])
            data["entity", p, "entity"].edge_index = tp

    data["entity"].x = torch.stack([t for t in node_embeddings])
    return data

def get_query_graph_data_new_nary_rdf(dataset, query_graph, statistics, dim,device):
    """
    This function is used to get the graph data object from a query graph
    :param query_graph: Dict representing the query graph of the form {"triples": [triple1, triple2, ...], "y": cardinality,
    "query": String of sparql query, "x": List of occurring entities}
    :param statistics: Dict or dict like loader for the embeddings of entities
    :param device: cpu or cuda
    :return: Graph data object
    """
    data = HeteroData()
    data = data.to(device)
    node_mapping = {}
    n_entities = 0
    node_embeddings = []
    # Embeddings for variables in edges or nodes
    # edge has an additional dimension to indicate the direction of the edge
    variable_embedding_edge = np.ones(dim+2)
    variable_embedding = np.ones(dim+1)
    if dataset == 'wd50k':
        with open('/export/data/kb_group_shares/GNCE/wd50k/data_graph/id_to_id_mapping_predicate.json', 'r') as f:
            pred = json.load(f)
        with open('/export/data/kb_group_shares/GNCE/wd50k/data_graph/id_to_id_mapping.json', 'r') as f:
            ent = json.load(f)
    else:
        pred = {}
        ent = {}
    feature_vector = None
    for triple in query_graph["triples"]:
        if triple[0] in ent:
            s = str(ent[triple[0]])  
        else:
            s = triple[0]
        if triple[2] in ent:
            o = str(ent[triple[2]])
        else:
            o = triple[2]

        # If predicate is a variable, add the variable embedding to the edge attributes
        if "?" in triple[1]:
            p = '0487'
            try:
                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([variable_embedding_edge])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                # If the edge does not exist yet, create it
                data["entity", p, "entity"].edge_attr = torch.tensor([variable_embedding_edge])

        else:
            try:
                #p = int(triple[1].replace("<http://example.com/", "").replace(">", ""))
                if dataset == 'wd50k':
                    p = '0'+str(pred[triple[1]])
                else:
                    p = triple[1]

            except:
                print(triple)
                raise
            try:
                # Get the embedding of the predicate
                if type(statistics[p]["embedding"].copy()[0]) == list:
                    feature_vector = statistics[p]["embedding"].copy()[0]
                else:
                    feature_vector = statistics[p]["embedding"].copy()
                # Add the occurence of the predicate to the embedding
                
                feature_vector.append(statistics[p]["occurence"] / 16018)
                # Add a dimension for the direction of the edge
                feature_vector.append(1)

                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([feature_vector])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                # Case if edge does not exist yet
                if type(statistics[p]["embedding"].copy()[0]) == list:
                    feature_vector = statistics[p]["embedding"].copy()[0]
                else:
                    feature_vector = statistics[p]["embedding"].copy()
                feature_vector.append(statistics[p]["occurence"] / 16018)
                feature_vector.append(1)
                # print(feature_vector)
                data["entity", p, "entity"].edge_attr = torch.tensor([feature_vector])

        # Adding the embeddings of s and o
        if not s in node_mapping.keys():
            node_mapping[s] = n_entities
            n_entities += 1
            if "?" in s:
                emb = variable_embedding
                # indicate index of variable in query
                emb[0] = node_mapping[s]
                node_embeddings.append(emb)
            else:
                # Get the embedding of the entity
                if type(statistics[s]["embedding"].copy()[0]) == list:
                    feature_vector = statistics[s]["embedding"].copy()[0]
                else:
                    feature_vector = statistics[s]["embedding"].copy()
                # feature_vector = statistics[s]["embedding"].copy()
                # Add the occurence of the entity to the embedding
                feature_vector.append(statistics[s]["occurence"] / 16018)
                node_embeddings.append(feature_vector)
        if not o in node_mapping.keys():
            node_mapping[o] = n_entities
            n_entities += 1
            if "?" in o:
                emb = variable_embedding
                emb[0] = node_mapping[o]
                node_embeddings.append(variable_embedding)
            else:
                if type(statistics[o]["embedding"].copy()[0]) == list:
                    feature_vector = statistics[o]["embedding"].copy()[0]
                else:
                    feature_vector = statistics[o]["embedding"].copy()
                # feature_vector = statistics[o]["embedding"].copy()
                feature_vector.append(statistics[o]["occurence"] / 16018)
                node_embeddings.append(feature_vector)

        # Finally, add the edge to the graph
        try:
            tp = torch.cat(
                (data['entity', p, 'entity'].edge_index, torch.tensor([[node_mapping[s]], [node_mapping[o]]])), dim=1)
            data["entity", p, "entity"].edge_index = tp

        except:
            tp = torch.tensor([[node_mapping[s]], [node_mapping[o]]])
            data["entity", p, "entity"].edge_index = tp

    data["entity"].x = torch.tensor(node_embeddings)
    return data

def get_query_graph_data_nary(query_graph, mapping,edge_mapping,config):
    # data = HeteroData()
    # data = data.to(device)
    # data1 = {}
    # ent_mapping = {}
    # ent_embeddings = {}
    # node_mapping = {}
    # n_entities = 0
    # node_embeddings = []
    # variable_embedding = np.ones(201)

    # feature_vector = None
    # print(query_graph)
    ent = {}
    rel = {}
    temp = []
    # with open('/export/data/kb_group_shares/wd50k/wd50k_qe/vocab.txt','r') as f:
    #     lines = f.readlines()
    # mapping = {}
    # edge_mapping = {}
    # for i,line in enumerate(lines):
    #     if line.startswith('Q'):
    #         mapping[line.strip()] = i-533
    #     if line.startswith('P'):
    #         edge_mapping[line.strip()] = i-2
    ent[0] = []
    rel[0] = []
    var_mapping = {}
    # print(mapping)
    triple_f = True
    for triple in query_graph:
        cnt = 0
        for i,item in enumerate(triple):
            
            if item != 'direct' and item != 'and':
                cnt += 1
        if cnt > 3:
            triple_f = False


    for triple in query_graph:
        # print(triple[:-1])
        # print(triple)
        triple_temp = []
        for i,item in enumerate(triple):
            
            if item != 'direct' and item != 'and':
                triple_temp.append(item)
                if config['HAS_QUAL']:
                    if i % 2 == 0:
                        # print(f'{item} {item in mapping} {ent[0]}')
                        if item in mapping and item.startswith('?') is False:
                            ent[mapping[item]] = [item]
                        else:
                            if item not in ent[0]:
                                ent[0].append(item)
                    else:
                        if item.startswith('?'):
                            if item not in rel[0]:
                                rel[0].append(item)
                        else:
                            
                            rel[edge_mapping[item]] = [item]
                elif i < 3:
                    if i % 2 == 0:
                        # print(f'{item} {item in mapping} {ent[0]}')
                        if item in mapping and item.startswith('?') is False:
                            ent[mapping[item]] = [item]
                        else:
                            if item not in ent[0]:
                                ent[0].append(item)
                    else:
                        if item.startswith('?'):
                            if item not in rel[0]:
                                rel[0].append(item)
                        else:
                            
                            rel[edge_mapping[item]] = [item]
            else:
                break
        temp.append(triple_temp)
    dic,node,edge = get_alternative_graph_repr(temp,mapping,edge_mapping,config['HAS_QUAL'])
    var_idx = 0
    for item in ent[0]:
        var_mapping[item] = var_idx
        var_idx += 1
    for item in rel[0]:
        var_mapping[item] = var_idx
        var_idx += 1
    return ent,rel,dic,node,edge,var_mapping,triple_f

def get_query_graph_data_tp(query_graph, mapping,edge_mapping,config):
    # data = HeteroData()
    # data = data.to(device)
    # data1 = {}
    # ent_mapping = {}
    # ent_embeddings = {}
    # node_mapping = {}
    # n_entities = 0
    # node_embeddings = []
    # variable_embedding = np.ones(201)

    # feature_vector = None
    # print(query_graph)
    ent = {}
    rel = {}
    temp = []
    # with open('/export/data/kb_group_shares/wd50k/wd50k_qe/vocab.txt','r') as f:
    #     lines = f.readlines()
    # mapping = {}
    # edge_mapping = {}
    # for i,line in enumerate(lines):
    #     if line.startswith('Q'):
    #         mapping[line.strip()] = i-533
    #     if line.startswith('P'):
    #         edge_mapping[line.strip()] = i-2
    ent[0] = []
    rel[0] = []
    var_mapping = {}
    # print(mapping)
    
    for triple in query_graph:
        # print(triple[:-1])
        triple_temp = []
        for i,item in enumerate(triple[:3]):
            
            if item != 'direct' and item != 'and':
                triple_temp.append(item)
                if config['HAS_QUAL']:
                    if i % 2 == 0:
                        # print(f'{item} {item in mapping} {ent[0]}')
                        if item in mapping and item.startswith('?') is False:
                            ent[mapping[item]] = [item]
                        else:
                            if item not in ent[0]:
                                ent[0].append(item)
                    else:
                        if item.startswith('?'):
                            if item not in rel[0]:
                                rel[0].append(item)
                        else:
                            
                            rel[edge_mapping[item]] = [item]
                elif i < 3:
                    if i % 2 == 0:
                        # print(f'{item} {item in mapping} {ent[0]}')
                        if item in mapping and item.startswith('?') is False:
                            ent[mapping[item]] = [item]
                        else:
                            if item not in ent[0]:
                                ent[0].append(item)
                    else:
                        if item.startswith('?'):
                            if item not in rel[0]:
                                rel[0].append(item)
                        else:
                            
                            rel[edge_mapping[item]] = [item]
            else:
                break
        temp.append(triple_temp)
    dic,node,edge = get_alternative_graph_repr(temp,mapping,edge_mapping,False)
    var_idx = 0
    for item in ent[0]:
        var_mapping[item] = var_idx
        var_idx += 1
    for item in rel[0]:
        var_mapping[item] = var_idx
        var_idx += 1
    return ent,rel,dic,node,edge,var_mapping

def get_alternative_graph_repr(raw,node_mapping,edge_mapping,has_qualifiers):
    """
    Decisions:

        Quals are represented differently here, i.e., more as a coo matrix
        s1 p1 o1 qr1 qe1 qr2 qe2    [edge index column 0]
        s2 p2 o2 qr3 qe3            [edge index column 1]

        edge index:
        [ [s1, s2],
            [o1, o2] ]

        edge type:
        [ p1, p2 ]

        quals will looks like
        [ [qr1, qr2, qr3],
            [qe1, qr2, qe3],
            [0  , 0  , 1  ]       <- obtained from the edge index columns

    :param raw: [[s, p, o, qr1, qe1, qr2, qe3...], ..., [...]]
        (already have a max qualifier length padded data)
    :param config: the config dict
    :return: output dict
    """
    
    try:
        nr = 532
    except KeyError:
        raise AssertionError("Function called too soon. Num relations not found.")
    # print(raw)
    edge_index, edge_type = np.zeros((2, len(raw) * 2), dtype=int), np.zeros((len(raw) * 2), dtype=int)
    # qual_rel = np.zeros(((len(raw[0]) - 3) // 2, len(raw) * 2), dtype=int)
    # qual_ent = np.zeros(((len(raw[0]) - 3) // 2, len(raw) * 2), dtype=int)
    qualifier_rel = []
    qualifier_ent = []
    qualifier_edge = []
    # node_mapping = {}
    # edge_mapping = {}
    node_emd_mapping = {}
    edge_emd_mapping = {}
    node_cnt = 0
    edge_cnt = 0
    # qualifer_mapping
    # Add actual data
    for i, data in enumerate(raw):
        # print(data)
        if data[0] not in node_mapping:
            node_mapping[data[0]] = 0
            # node_cnt += 1
        if data[2] not in node_mapping:
            node_mapping[data[2]] = 0
        #     node_cnt += 1
        if data[1] not in edge_mapping:
            if data[1].startswith('?'):
                edge_mapping[data[1]] = 0
        if data[0] not in node_emd_mapping:
            node_emd_mapping[data[0]] = node_cnt
            node_cnt += 1  
        if data[2] not in node_emd_mapping:
            node_emd_mapping[data[2]] = node_cnt
            node_cnt += 1   
        if data[1] not in edge_emd_mapping:
            edge_emd_mapping[data[1]] = edge_cnt
            edge_cnt += 1    
            # edge_cnt += 1
        edge_index[:, i] = [node_emd_mapping[data[0]], node_emd_mapping[data[2]]]
        edge_type[i] = edge_emd_mapping[data[1]]

        # @TODO: add qualifiers
        if has_qualifiers:
            # print(data[3::2])
            # print(f'{data[3::2]} {data[4::2]}')
            qual_rel = []
            for item in data[3::2]:
                if item not in edge_emd_mapping:
                    edge_emd_mapping[item] = edge_cnt
                    edge_cnt += 1
                qual_rel.append(edge_emd_mapping[item])
            qual_rel = np.array(qual_rel)
            # print(qual_rel)
            qual_ent = []
            for item in data[4::2]:
                if item not in node_emd_mapping:
                    node_emd_mapping[item] = node_cnt
                    node_cnt += 1
                qual_ent.append(node_emd_mapping[item])
            qual_ent = np.array(qual_ent)
            non_zero_rels = qual_rel
            non_zero_ents = qual_ent
            # print(qual_ent)
            # print(qual_rel)
            
            # print(non_zero_ents)
            # print(non_zero_rels)
            for j in range(non_zero_ents.shape[0]):
                qualifier_rel.append(non_zero_rels[j])
                qualifier_ent.append(non_zero_ents[j])
                qualifier_edge.append(i)
            
    if has_qualifiers:
        quals = np.stack((qualifier_rel, qualifier_ent, qualifier_edge), axis=0)
    else:
        quals = None
    num_triples = len(raw)

    # Add inverses
    edge_index[1, len(raw):] = edge_index[0, :len(raw)]
    edge_index[0, len(raw):] = edge_index[1, :len(raw)]
    edge_type[len(raw):] = edge_type[:len(raw)] + edge_cnt 

    if has_qualifiers:
        full_quals = np.hstack((quals, quals))
        full_quals[2, quals.shape[1]:] = quals[2, :quals.shape[1]] #+ len(raw) # TODO: might need to + num_triples

        return {'edge_index': edge_index,
                'edge_type': edge_type,
                'quals': full_quals},node_emd_mapping,edge_emd_mapping
    else:
        return {'edge_index': edge_index,
                'edge_type': edge_type,
                'quals': None},node_emd_mapping,edge_emd_mapping

    #     edge_index[:, i] = [node_mapping[data[0]], node_mapping[data[2]]]
    #     edge_type[i] = edge_mapping[data[1]]

    #     # @TODO: add qualifiers
    #     if has_qualifiers:
    #         # print(data[3::2])
    #         qual_rel = []
    #         for item in data[3::2]:
    #             if item not in edge_mapping:
    #                 edge_mapping[item] = 0
    #             qual_rel.append(edge_mapping[item])
    #         qual_rel = np.array(qual_rel)
    #         # print(qual_rel)
    #         qual_ent = []
    #         for item in data[4::2]:
    #             if item not in node_mapping:
    #                 node_mapping[item] = 0
    #             qual_ent.append(node_mapping[item])
    #         qual_ent = np.array(qual_ent)
    #         non_zero_rels = qual_rel[np.nonzero(qual_rel)]
    #         non_zero_ents = qual_ent[np.nonzero(qual_ent)]
    #         for j in range(non_zero_ents.shape[0]):
    #             qualifier_rel.append(non_zero_rels[j])
    #             qualifier_ent.append(non_zero_ents[j])
    #             qualifier_edge.append(i)

    # quals = np.stack((qualifier_rel, qualifier_ent, qualifier_edge), axis=0)
    # num_triples = len(raw)

    # # Add inverses
    # edge_index[1, len(raw):] = edge_index[0, :len(raw)]
    # edge_index[0, len(raw):] = edge_index[1, :len(raw)]
    # edge_type[len(raw):] = edge_type[:len(raw)] + nr

    # if has_qualifiers:
    #     full_quals = np.hstack((quals, quals))
    #     full_quals[2, quals.shape[1]:] = quals[2, :quals.shape[1]]  # TODO: might need to + num_triples

    #     return {'edge_index': edge_index,
    #             'edge_type': edge_type,
    #             'quals': full_quals}
    # else:
    #     return {'edge_index': edge_index,
    #             'edge_type': edge_type}

def create_edge_list(query2data,nodem,dnodem,config):
    edge_index = [[],[]]
    num_vertices = len(nodem)
    raw = 1
    for k in query2data:
        raw += len(query2data[k])
    #     for data in query2data[k]:
    #         raw += 1
    edge_index =  np.zeros((2, raw * 2), dtype=int)
    # print(query2data)
    cnt = 0
    for k in query2data:
        for data in query2data[k]:
            edge_index[:,cnt] = [nodem[k],num_vertices+dnodem[data]]
            # edge_index[0].append(nodem[k])
            # edge_index[1].append(num_vertices+dnodem[data])
            cnt += 1
    # print(edge_index)
    # raw = len(edge_index[0]) 
    edge_index[1, raw:] = edge_index[0, :raw]
    edge_index[0, raw:] = edge_index[1, :raw]
    return torch.tensor([edge_index]).to(config['DEVICE']).view(2,-1)

def _get_uniques_(train_data, valid_data, test_data): #-> (
# list, list):
    """ Throw in parsed_data/wd50k/ files and we'll count the entities and predicates"""

    statement_entities, statement_predicates = [], []

    for statement in train_data + valid_data + test_data:
        statement_entities += statement[::2]
        statement_predicates += statement[1::2]

    statement_entities = sorted(list(set(statement_entities)))
    statement_predicates = sorted(list(set(statement_predicates)))

    return statement_entities, statement_predicates

def _conv_to_our_format_(data, filter_literals=True):
    conv_data = []
    dropped_statements = 0
    dropped_quals = 0
    for datum in data:
        try:
            conv_datum = []

            # Get head and tail rels
            head, tail, rel_h, rel_t = None, None, None, None
            for rel, val in datum.items():
                if rel[-2:] == '_h' and type(val) is str:
                    head = val
                    rel_h = rel[:-2]
                if rel[-2:] == '_t' and type(val) is str:
                    tail = val
                    rel_t = rel[:-2]
                    if filter_literals and "http://" in tail:
                        dropped_statements += 1
                        raise Exception

            assert head and tail and rel_h and rel_t, f"Weird data point. Some essentials not found. Quitting\nD:{datum}"
            assert rel_h == rel_t, f"Weird data point. Head and Tail rels are different. Quitting\nD: {datum}"

            # Drop this bs
            datum.pop(rel_h + '_h')
            datum.pop(rel_t + '_t')
            datum.pop('N')
            conv_datum += [head, rel_h, tail]

            # Get all qualifiers
            for k, v in datum.items():
                for _v in v:
                    if filter_literals and "http://" in _v:
                        dropped_quals += 1
                        continue
                    conv_datum += [k, _v]

            conv_data.append(list(conv_datum))
        except Exception:
            continue
    print(f"\n Dropped {dropped_statements} statements and {dropped_quals} quals with literals \n ")
    return conv_data

def clean_literals(data):
    """

    :param data: triples [s, p, o] with possible literals
    :return: triples [s,p,o] without literals

    """
    result = []
    for triple in data:
        if "http://" not in triple[2]:
            result.append(list(triple))

    return result

def convert_nicely(arg, possible_types=(bool, float, int, str)):
    """ Try and see what sticks. Possible types can be changed. """
    for data_type in possible_types:
        try:

            if data_type is bool:
                # Hard code this shit
                if arg in ['T', 'True', 'true']: return True
                if arg in ['F', 'False', 'false']: return False
                raise ValueError
            else:
                proper_arg = data_type(arg)
                return proper_arg
        except ValueError:
            continue
    # Here, i.e. no data type really stuck
    warnings.warn(f"None of the possible datatypes matched for {arg}. Returning as-is")
    return arg

def parse_args(raw_args: List[str], compulsory: List[str] = (), compulsory_msg: str = "",
               types: Dict[str, type] = None, discard_unspecified: bool = False):
    """
        I don't like argparse.
        Don't like specifying a complex two liner for each every config flag/macro.

        If you maintain a dict of default arguments, and want to just overwrite it based on command args,
        call this function, specify some stuff like

    :param raw_args: unparsed sys.argv[1:]
    :param compulsory: if some flags must be there
    :param compulsory_msg: what if some compulsory flags weren't there
    :param types: a dict of confignm: type(configvl)
    :param discard_unspecified: flag so that if something doesn't appear in config it is not returned.
    :return:
    """

    # parsed_args = _parse_args_(raw_args, compulsory=compulsory, compulsory_msg=compulsory_msg)
    #
    # # Change the "type" of arg, anyway

    parsed = {}

    while True:

        try:                                        # Get next value
            nm = raw_args.pop(0)
        except IndexError:                          # We emptied the list
            break

        # Get value
        try:
            vl = raw_args.pop(0)
        except IndexError:
            raise ImproperCMDArguments(f"A value was expected for {nm} parameter. Not found.")

        # Get type of value
        if types:
            try:
                parsed[nm] = types[nm](vl)
            except ValueError:
                raise ImproperCMDArguments(f"The value for {nm}: {vl} can not take the type {types[nm]}! ")
            except KeyError:                    # This name was not included in the types dict
                if not discard_unspecified:     # Add it nonetheless
                    parsed[nm] = convert_nicely(vl)
                else:                           # Discard it.
                    continue
        else:
            parsed[nm] = convert_nicely(vl)

    # Check if all the compulsory things are in here.
    for key in compulsory:
        try:
            assert key in parsed
        except AssertionError:
            raise ImproperCMDArguments(compulsory_msg + f"Found keys include {[k for k in parsed.keys()]}")

    # Finally check if something unwanted persists here
    return parsed

class ImproperCMDArguments(Exception): pass

def get_query_graph_data_new_batch(query_graph, statistics, device, unknown_entity='false', n_atoms: int = None):
    """
    This function is used to get the graph data object from a query graph
    :param query_graph: Dict representing the query graph of the form {"triples": [triple1, triple2, ...], "y": cardinality,
    "query": String of sparql query, "x": List of occurring entities}
    :param statistics: Dict or dict like loader for the embeddings of entities
    :param device: cpu or cuda
    :param unknown entity: Determines if an entity receives its embedding('false'), randomly, embedding
    or random vector('random') or always a random embedding('true')
    :return: Graph data object
    """

    data = HeteroData()
    data = data.to(device)
    node_mapping = {}
    n_edge_variables = 0 # todo remove, for testing
    n_entities = 0 # todo remove, only for testing
    n_edge_variables_random = random.randint(0, 5) # todo remove, for testing
    n_entities_random = random.randint(0, 5) # todo remove, only for testing
    node_embeddings = []
    # Embeddings for variables in edges or nodes
    # edge has an additional dimension to indicate the direction of the edge
    variable_embedding_edge = np.ones(102)
    variable_embedding = np.ones(101)
    np.random.seed(0)
    unknown_entity_embedding = list(np.random.rand(100))
    # Reset the seed

    # Random number indicating if to use embeddings or not
    rand_num = random.random()


    np.random.seed(None)

    # Set to count the total unique atoms in the query
    atom_set = set()

    feature_vector = None

    USE_EMBEDDING = True
    USE_OCCURRENCE = True

    # Whether to shuffle triples and start with a random integer for variable denoting
    shuffled =True

    triple_list = query_graph["triples"]

    # Todo remove, for testing effect of variable enumeration
    if shuffled:
        random.shuffle(triple_list)
    for triple in query_graph["triples"]:

        atom_set.update(triple)
        try:
            s = triple[0].replace("<", "").replace(">", "")
            o = triple[2].replace("<", "").replace(">", "")
        except:
            print(query_graph)
            raise
        # If predicate is a variable, add the variable embedding to the edge attributes
        if "?" in triple[1]:
            p = 237
            try:
                v = variable_embedding_edge
                if shuffled:
                    v[-1] = n_edge_variables_random #todo
                else:
                    v[-1] = n_edge_variables
                n_edge_variables_random +=1 #todo
                n_edge_variables += 1
                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([v])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                # If the edge does not exist yet, create it
                v = variable_embedding_edge
                if shuffled:
                    v[-1] = n_edge_variables_random #todo
                else:
                    v[-1] = n_edge_variables #todo

                n_edge_variables_random += 1 #todo
                n_edge_variables += 1
                data["entity", p, "entity"].edge_attr = torch.tensor([v])

        else:
            try:
                #p = int(triple[1].replace("<http://example.com/", "").replace(">", ""))
                p = triple[1].replace("<http://example.com/", "").replace(">", "")

            except:
                print(triple)
                raise
            try:
                # Get the embedding of the predicate
                if USE_EMBEDDING:
                    feature_vector = statistics[triple[1].replace("<", "").replace(">", "")]["embedding"].copy()
                else:
                    idx = int(triple[1].replace(">", "").split("/")[-1])
                    idx_bin = bin(idx)[2:].zfill(100)
                    feature_vector = [float(i) for i in idx_bin]
                # Add the occurence of the predicate to the embedding
                if USE_OCCURRENCE:
                    feature_vector.append(statistics[triple[1].replace("<", "").replace(">", "")]["occurence"] / 16018)
                else:
                    feature_vector.append(1)
                # Add a dimension for the direction of the edge
                feature_vector.append(1)

                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([feature_vector])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                # Case if edge set does not exist yet
                if USE_EMBEDDING:
                    feature_vector = statistics[triple[1].replace("<", "").replace(">", "")]["embedding"].copy()
                else:
                    idx = int(triple[1].replace(">", "").split("/")[-1])
                    idx_bin = bin(idx)[2:].zfill(100)
                    feature_vector = [float(i) for i in idx_bin]
                if USE_OCCURRENCE:
                    feature_vector.append(statistics[triple[1].replace("<", "").replace(">", "")]["occurence"] / 16018)
                else:
                    feature_vector.append(1)
                feature_vector.append(1)

                data["entity", p, "entity"].edge_attr = torch.tensor([feature_vector])

        # Adding the embeddings of s and o
        if not s in node_mapping.keys():
            node_mapping[s] = n_entities
            n_entities += 1
            if "?" in s:
                emb = variable_embedding
                # indicate index of variable in query

                if shuffled:
                    emb[0] = n_entities_random #todo
                else:
                    emb[0] = node_mapping[s]  # todo
                n_entities_random +=1 #todo
                node_embeddings.append(emb)
            else:
                if unknown_entity == 'false':
                    feature_vector = statistics[s]["embedding"].copy()
                    feature_vector.append(statistics[s]["occurence"] / 16018)
                elif unknown_entity == 'true':
                    feature_vector = unknown_entity_embedding.copy()
                    feature_vector.append(1)
                elif unknown_entity == 'random':
                    # Generate a random float between 0 and 1

                    # 70% chance of being True
                    if rand_num < 0.7:
                        feature_vector = statistics[s]["embedding"].copy()
                        feature_vector.append(statistics[s]["occurence"] / 16018)
                    else:
                        feature_vector = unknown_entity_embedding.copy()
                        feature_vector.append(1)
                else:
                    idx = int(s.replace(">", "").split("/")[-1])
                    idx_bin = bin(idx)[2:].zfill(100)
                    feature_vector = [float(i) for i in idx_bin]

                node_embeddings.append(feature_vector)
        if not o in node_mapping.keys():
            node_mapping[o] = n_entities
            n_entities += 1
            if "?" in o:
                emb = variable_embedding
                if shuffled:
                    emb[0] = n_entities_random #todo
                else:
                    emb[0] = node_mapping[o]  # todo
                n_entities_random +=1 #todo
                node_embeddings.append(emb)
            else:
                if unknown_entity == 'false':
                    feature_vector = statistics[o]["embedding"].copy()
                    feature_vector.append(statistics[o]["occurence"] / 16018)
                elif unknown_entity == 'true':
                    feature_vector = unknown_entity_embedding.copy()
                    feature_vector.append(1)
                elif unknown_entity == 'random':
                    # Generate a random float between 0 and 1
                    rand_num = random.random()
                    # 70% chance of being True
                    if rand_num < 0.7:
                        feature_vector = statistics[o]["embedding"].copy()
                        feature_vector.append(statistics[o]["occurence"] / 16018)
                    else:
                        feature_vector = unknown_entity_embedding.copy()
                        feature_vector.append(1)
                else:
                    idx = int(o.replace(">", "").split("/")[-1])
                    idx_bin = bin(idx)[2:].zfill(100)
                    feature_vector = [float(i) for i in idx_bin]

                node_embeddings.append(feature_vector)


        # Finally, add the edge to the graph
        try:
            tp = torch.cat(
                (data['entity', p, 'entity'].edge_index, torch.tensor([[node_mapping[s]], [node_mapping[o]]])), dim=1)
            data["entity", p, "entity"].edge_index = tp

        except:
            tp = torch.tensor([[node_mapping[s]], [node_mapping[o]]])
            data["entity", p, "entity"].edge_index = tp

    data["entity"].x = torch.tensor(node_embeddings)

    if n_atoms is not None:
        n_atoms += len(atom_set)
        return data, n_atoms
    else:
        return data

class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # 单进程情况下，使用默认的数据加载器
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers).__iter__()
        else:
            # 多进程情况下，使用数据集的迭代器来加载和处理数据
            return self._worker_iter(worker_info)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def _worker_iter(self, worker_info):
        # 获取当前进程的数据集分片
        dataset = self.dataset
        indices = dataset.indices
        start = worker_info.id * len(indices) // worker_info.num_workers
        end = (worker_info.id + 1) * len(indices) // worker_info.num_workers
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[start:end])

        # 使用数据集的迭代器来加载和处理数据
        batch_sampler = torch.utils.data.BatchSampler(sampler, self.batch_size, drop_last=False)
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=self.num_workers)
        for batch in loader:
            yield batch