import os
os.environ['MKL_NUM_THREADS'] = '1'
from collections import OrderedDict
from functools import partial
import random
import math
import pickle
# import wandb
import sys
import collections

import torch
import numpy as np

# Local imports
import json
from utils import get_query_graph_data_nary,_get_uniques_,get_query_graph_data_tp,_conv_to_our_format_
from torchutils import parse_args
from torch.nn import MSELoss, NLLLoss
import torch.nn.functional as F
from models_batch import NaryModel,InfoNCELoss,Gbrfe
from tqdm import tqdm
from scipy.stats import entropy
import time
"""
    CONFIG Things
"""

# Clamp the randomness
# np.random.seed(42)
# random.seed(42)
# torch.manual_seed(132)
# torch.cuda.manual_seed_all(132) 
# torch.backends.cudnn.enabled = True   # 默认值
# torch.backends.cudnn.benchmark = False  # 默认为False
# torch.backends.cudnn.deterministic = True #
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
import warnings
warnings.filterwarnings("ignore")
# sum: learning_rate = 0.0001
#cat: learning_rate = 0.00005

DEFAULT_CONFIG = {
    'BATCH_SIZE': 32,
    'DATASET': 'wd50k',
    'DEVICE': 'cuda',
    'EMBEDDING_DIM': 200,
    'INIT_EMBED':'stare',
    'MULTI_DIM':False,
    'MULTI_DIM_SIZE':1,
    'ENT_POS_FILTERED': True,
    'EPOCHS': 100,
    'EVAL_EVERY': 5,
    'LEARNING_RATE': 0.0005,
    'OCCURENCES':'no',
    'MAX_QPAIRS': 15,
    'MODEL_NAME': 'stare_transformer',
    'CORRUPTION_POSITIONS': [0, 2],
    'REVERSE':True,
    'USE_VAR':False,
    'ACTIVE':False,
    'CLASSES':10,
    'UNCERTAIN':'entropy',
    'coeff':0.5,
    'var':'zero',
    'DISTILL':False,
    'DISTILL_AGGREGATION':'cat',
    'INPUT_DIM':400,
    'DISTILL_DIM':100,
    'HAS_QUAL':True,
    'SUBTYPE':'statements',
    'USE_ATTENTION':False,
    'TRIPLE':False,
    'REMOVE_QUAL':False,
    'QT':'both',
    'SEPARATE':False,
    'SPLIT':False,
    'ESTIMATE':'none',
    'LAYERS':2,
    'RELATION_TRANS':True,
    'RUN':0,
    # # not used for now
    # 'MARGIN_LOSS': 5,
    # 'NARY_EVAL': False,
    # 'NEGATIVE_SAMPLING_PROBS': [0.3, 0.0, 0.2, 0.5],
    # 'NEGATIVE_SAMPLING_TIMES': 10,
    # 'NORM_FOR_NORMALIZATION_OF_ENTITIES': 2,s
    # 'NORM_FOR_NORMALIZATION_OF_RELATIONS': 2,
    # 'NUM_FILTER': 5,
    # 'PROJECT_QUALIFIERS': False,
    # 'PRETRAINED_DIRNUM': '',
    # 'RUN_TESTBENCH_ON_TRAIN': False,
    # 'SAVE': False,
    # 'SELF_ATTENTION': 0,
    # 'SCORING_FUNCTION_NORM': 1,

    # important args
    'SAVE': True,
    'STATEMENT_LEN': -1,
    'USE_TEST': False,
    'WANDB': False,
    'LABEL_SMOOTHING': 0.1,
    'SAMPLER_W_QUALIFIERS': True,
    'OPTIMIZER': 'adam',
    'CLEANED_DATASET': True,  # should be false for WikiPeople and JF17K for their original data
    'HID_DIM':200,
    'RANDOM':False,
    'GRAD_CLIPPING': True,
    'LR_SCHEDULER': True,
    'budget':400,
    'biased_sample':True,
    'active_iters':3,
    'REMOVE_SELF':False,
    'ESTIMATE_GATE':'none',
    'PRINT_VECTOR':False,
    'RETURN_EMBED':False
}

STAREARGS = {
    'LAYERS': 2,
    'N_BASES': 0,
    'GCN_DIM': 200,
    'GCN_DROP': 0.1,
    'HID_DROP': 0.3,
    'BIAS': False,
    'OPN': 'rotate',
    'TRIPLE_QUAL_WEIGHT': 0.8,
    'QUAL_AGGREGATE': 'sum',  # or concat or mul
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

DEFAULT_CONFIG['STAREARGS'] = STAREARGS
def getModel(device,model):
    # MODEL = 
    model_old = torch.load('model_ocr_nary.pth', map_location=device)
    new_state_dict = OrderedDict()
    for k, v in model_old.items():
        # print(f'{k} {v.shape}')
        # name = k[7:] # remove `module.` origin
        # if 'classifier' not in k: #fine-tuned
        #     name = k
        # else:
        #     name = k.replace('classifier','out')
        new_state_dict[k] = v
        # load params
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def make_print_to_file(config, model,path='./'):
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
    if config['USE_TEST'] == True:
        sys.stdout = Logger(fileName + '_'+model.replace('.pth','')+'eval.log', path=path)
    else:
        sys.stdout = Logger(fileName + '_'+model.replace('.pth','')+'.log', path=path)
  
    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60,'*'))

def batch_queries(data,config):
    batch_data = {'x':[],'edge_index':[[],[]],'edge_type':[],'r':[],'qual':[[],[],[]],'y':[],'batch':[],'triples':[]}
    x_offset = 0
    r_offset = 0
    edge_index_offset = 0
    inv_type = []
    inv_qr = []
    inv_qe = []
    inv_qindex = []
    for idx,d in enumerate(data):
    #     print(f'node:{len(d[0])}')
    #     print(f'edge:{len(d[1])}')
    #     print(d[0][0].shape)
        batch_data['x'].extend(d[0])
        batch_data['r'].extend(d[1])
        # print(d[3])
        # print(f'nodenum:{d[2]}')
        d[2] += x_offset
        d[2] = d[2].tolist()
        d[3] += r_offset
        d[3] = d[3].tolist()
        # print(d[3])
        batch_data['edge_index'][0].extend(d[2][0][:len(d[2][0])//2])
        batch_data['edge_index'][1].extend(d[2][1][:len(d[2][1])//2])
        batch_data['edge_type'].extend(d[3][:len(d[3])//2])
        inv_type.extend(d[3][len(d[3])//2:])
        d[4][0] += r_offset
        d[4][1] += x_offset
        d[4][2] += edge_index_offset
        d[4] = d[4].tolist()
        # print(d[4])
        batch_data['qual'][0].extend(d[4][0][:len(d[4][0])//2])
        inv_qr.extend(d[4][0][len(d[4][0])//2:])
        batch_data['qual'][1].extend(d[4][1][:len(d[4][1])//2])
        inv_qe.extend(d[4][1][len(d[4][1])//2:])
        batch_data['qual'][2].extend(d[4][2][:len(d[4][2])//2])
        inv_qindex.extend(d[4][2][len(d[4][2])//2:])
        batch_data['y'].append(d[5])
        batch_data['batch'].extend([idx for i in range(len(d[0]))])
        x_offset += len(d[0])
        r_offset += len(d[1])
        edge_index_offset += len(d[2][0]) // 2
        # if d[6] != '':
        #     batch_data['triples']
        # print(d[-1])
        batch_data['triples'].append(d[-1])
    batch_data['edge_index'][0].extend(batch_data['edge_index'][1])
    batch_data['edge_index'][1].extend(batch_data['edge_index'][0][:len(batch_data['edge_index'][0])//2])
    batch_data['edge_type'].extend(inv_type)
    batch_data['qual'][0].extend(inv_qr)
    batch_data['qual'][1].extend(inv_qe)
    batch_data['qual'][2].extend(inv_qindex)
    # print(batch_data['x'])
    batch_data['x'] = torch.cat(batch_data['x'],dim=0).to(config['DEVICE'])
    
    batch_data['r'] = torch.cat(batch_data['r'],dim=0).to(config['DEVICE'])
    
    batch_data['edge_index'] = torch.tensor(batch_data['edge_index']).to(config['DEVICE'])
    batch_data['edge_type'] = torch.tensor(batch_data['edge_type']).to(config['DEVICE'])
    # print(batch_data['qual'])

    batch_data['qual'] = torch.tensor(batch_data['qual']).to(config['DEVICE'])
    batch_data['y'] = torch.tensor(batch_data['y']).to(config['DEVICE'])
    batch_data['batch'] = torch.tensor(batch_data['batch']).to(config['DEVICE'])
    # print(batch_data['batch'])
    return batch_data

def q_error(pred, gt):
    gt_exp = gt
    pred_exp = torch.exp(pred.squeeze())
    return torch.max(gt_exp / pred_exp, pred_exp / gt_exp)

def build_training_data(train_data, config,model):
     with torch.no_grad():
        for epoch in tqdm(range(config['EPOCHS'])):
            points_processed = 0
            input_data = train_data
            random.Random(random.randint(0,config['EPOCHS'])).shuffle(input_data)
            # print(input_data)
            # input_data = [{'triples':[['Q38111', 'P166', '?v0', 'P1686', 'Q892735', 'direct'], ['Q117315', 'P166', '?v0', 'P1346', '?v1', 'direct'], ['?v1', 'P1411', 'Q106291', 'P1686', '?v2', 'P805', 'Q857047', 'direct'], ['?v2', 'P2747', 'Q23830578', 'direct', '?v3']],'gt':1}]
        
            time1 = time.time()
            query_per_batch = []
            count = 0
            training_data_per_epoch = []
            for d in tqdm(input_data):
                f = 0
                for triple in d['triples']:
                    if len(triple) > 3:
                        f = 1
                        break
                if config['QT'] == 'triple' and f == 1 or config['QT'] == 'qual' and f == 0:
                    # if f == 1:
                    continue
                if d['gt'] == 0:
                    continue
                convert_triple = None
                ents,rels,dat,nodem,edgem,var,triple_f = get_query_graph_data_nary(d['triples'],mapping,edge_mapping,config)
                

                x, r, edge_index, edge_type, quals = model.load_queries(ents,rels,nodem,edgem,var,dat)
                # print(edge_index)
                query_per_batch.append([x, r, edge_index, edge_type, quals,d['gt'],d['triples']])
                count += 1
                if count == config['BATCH_SIZE']:
                    data = batch_queries(query_per_batch,config)
                    training_data_per_epoch.append(data)
                    query_per_batch = []
                    count = 0
                    # print()
            # training_data.append(training_data_per_epoch)
            if str(config['STAREARGS']['QUAL_AGGREGATE']) == 'cat':
                with open(config['DATASET']+'/train_'+str(config['BATCH_SIZE'])+'/'+str(epoch)+'.pkl','wb') as f:
                    pickle.dump(training_data_per_epoch,f)
            else:
                with open(config['DATASET']+'/train_'+str(config['BATCH_SIZE'])+'_'+str(config['STAREARGS']['QUAL_AGGREGATE'])+'/'+str(epoch)+'.pkl','wb') as f:
                    pickle.dump(training_data_per_epoch,f)
if __name__ == "__main__":

    # Get parsed arguments
    config = DEFAULT_CONFIG.copy()
    gcnconfig = STAREARGS.copy()
    parsed_args = parse_args(sys.argv[1:])
    # print(parsed_args)

    # Superimpose this on default config
    for k, v in parsed_args.items():
        # If its a generic arg
        if k in config.keys():
            default_val = config[k.upper()]
            if default_val is not None:
                needed_type = type(default_val)
                config[k.upper()] = needed_type(v)
            else:
                config[k.upper()] = v
        # If its a starearg
        elif k.lower().startswith('gcn_') and k[4:] in gcnconfig:
            default_val = gcnconfig[k[4:].upper()]
            if default_val is not None:
                needed_type = type(default_val)
                gcnconfig[k[4:].upper()] = needed_type(v)
            else:
                gcnconfig[k[4:].upper()] = v

        else:
            config[k.upper()] = v

    config['STAREARGS'] = gcnconfig
    

    """
        Custom Sanity Checks
    """
    # If we're corrupting something apart from S and O
    if max(config['CORRUPTION_POSITIONS']) > 2:
        assert config['ENT_POS_FILTERED'] is False, \
            f"Since we're corrupting objects at pos. {config['CORRUPTION_POSITIONS']}, " \
            f"You must allow including entities which appear exclusively in qualifiers, too!"

    """
        Loading and preparing data
        
        Typically, sending the config dict, and executing the returned function gives us data,
        in the form of
            -> train_data (list of list of 43 / 5 or 3 elements)
            -> valid_data
            -> test_data
            -> n_entities (an integer)
            -> n_relations (an integer)
            -> ent2id (dictionary to interpret the data above, if needed)
            -> rel2id

       
    """
    data_ext = []
    data1_ext = []
    data2_ext = []
    if config['DATASET'] == 'wd50k':
        documents = os.listdir('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_nary_test/')
        # if documents == []:
        #     documents = os.listdir('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_total/')
        #     for doc in documents:
        #         if 'test' in doc:
        #             with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_total/'+doc,'r') as f:
        #                 temp = json.load(f)
        #                 data = temp #random.sample(temp,int(0.6*len(temp)))
                    
        #             with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_nary/'+doc,'w') as f:
        #                 json.dump(data,f)
        #         elif 'valid' in doc:
        #             with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_total/'+doc,'r') as f:
        #                 temp = json.load(f)
        #                 data = temp
                    
        #             with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_nary/'+doc,'w') as f:
        #                 json.dump(data,f)
        #         else:
        #             with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_total/'+doc,'r') as f:
        #                 temp = json.load(f)
        #                 data = temp #random.sample(temp,int(0.1*len(temp)))
                    
        #             with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_nary/'+doc,'w') as f:
        #                 json.dump(data,f)
        documents = os.listdir('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_nary_test/')
        for doc in documents:
            if 'test' in doc:
                with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_nary_test/'+doc,'r') as f:
                    
                    data2_ext.extend(json.load(f))
            elif 'valid' in doc:
                with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_nary_test/'+doc,'r') as f:
                    data1_ext.extend(json.load(f))
            else:
                with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_nary_test/'+doc,'r') as f:
                    data_ext.extend(json.load(f))
    elif config['DATASET'] == 'wd50k_nary':
        documents = os.listdir('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_origin/')
        for doc in documents:
            if 'test' in doc:
                with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_origin/'+doc,'r') as f:
                    
                    data2_ext.extend(json.load(f))
            elif 'valid' in doc:
                with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_origin/'+doc,'r') as f:
                    data1_ext.extend(json.load(f))
            else:
                with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wd50k_origin/'+doc,'r') as f:
                    data_ext.extend(json.load(f))
    elif config['DATASET'] == 'jf17k':
        documents = os.listdir('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/jf17k_nary_test/')
        for doc in documents:
            if 'test' in doc:
                with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/jf17k_nary_test/'+doc,'r') as f:
                    data2_ext.extend(json.load(f))
            elif 'valid' in doc:
                with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/jf17k_nary_test/'+doc,'r') as f:
                    data1_ext.extend(json.load(f))
            else:
                with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/jf17k_nary_test/'+doc,'r') as f:
                    data_ext.extend(json.load(f))
    elif config['DATASET'] == 'wikipeople':
        documents = os.listdir('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wikipeople_nary_test/')
        for doc in documents:
            if 'test' in doc:
                with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wikipeople_nary_test/'+doc,'r') as f:
                    data2_ext.extend(json.load(f))
            elif 'valid' in doc:
                with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wikipeople_nary_test/'+doc,'r') as f:
                    data1_ext.extend(json.load(f))
            else:
                with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/wikipeople_nary_test/'+doc,'r') as f:
                    data_ext.extend(json.load(f))
        # Break down the data
    try:
        # train_data, valid_data, test_data = data1[:int(len(data)*0.6)]+data,data1[int(len(data)*0.6):],data2[int(len(data)*0.8):]
        # train_data, valid_data, test_data = data[int(len(data)*0.2):int(len(data)*0.95)]+data_ext,data1[int(len(data1)*0.85):]+data1_ext,data2+data2_ext
        print(f'train:{len(data_ext)} test:{len(data1_ext)+len(data2_ext)}')
        train_data, valid_data, test_data = data_ext,data1_ext,data2_ext
        # print(f'{len(data_ext)} {len(data1_ext)} {len(data2_ext)}')
        # train_data, valid_data, test_data = random.sample(data_ext,min(4000,len(data_ext))),random.sample(data1_ext,min(1479,len(data1_ext))),random.sample(data2_ext,min(1479,len(data2_ext)))
        # train_data, valid_data, test_data = random.sample(data_ext+data1_ext[:int(0.4*len(data1_ext))]+data2_ext[:int(0.4*len(data2_ext))],min(1413,len(data_ext+data1_ext[:int(0.4*len(data1_ext))]+data2_ext[:int(0.4*len(data2_ext))]))),random.sample(data1_ext[int(0.4*len(data1_ext)):],min(475,len(data1_ext[int(0.4*len(data1_ext)):]))),random.sample(data2_ext[int(0.4*len(data2_ext)):],min(488,len(data2_ext[int(0.4*len(data2_ext)):])))
    except ValueError:
        raise ValueError(f"Honey I broke the loader for {config['DATASET']}")
    if config['SUBTYPE'] == 'cleaned_statements_removeloop':
        config['NUM_ENTITIES'] = 46885
        config['NUM_RELATIONS'] = 1038
        with open('/export/data/kb_group_shares/wd50k/wd50k_qe/wd50k_clean_occurences.json','r') as f:
            occurences = json.load(f)
        import copy
        occurences_single = copy.deepcopy(occurences)
        with open('/export/data/kb_group_shares/wd50k/wd50k_qe/wd50k_clean_occurences_pos.json','r') as f:
            occurences_pos = json.load(f)
        max_ocr = {'single':0,'qual':0,'s':0,'o':0,'s+o':0, 'r':0,'main':0,'qual_ent':0,'qual_rel':0}
        for k in occurences:
            if occurences[k] > max_ocr['single']:
                max_ocr['single'] = occurences[k]
        for k in occurences_pos:
            if k.startswith('Q'):
                if occurences_pos[k]['qual'] > max_ocr['qual']:
                    max_ocr['qual'] = occurences_pos[k]['qual']
                if occurences_pos[k]['s'] > max_ocr['s']:
                    max_ocr['s'] = occurences_pos[k]['s']
                if occurences_pos[k]['s'] + occurences_pos[k]['o']> max_ocr['s+o']:
                    max_ocr['s+o'] = occurences_pos[k]['s'] + occurences_pos[k]['o']
                if occurences_pos[k]['o'] > max_ocr['o']:
                    max_ocr['o'] = occurences_pos[k]['o']
                if occurences_pos[k]['qual'] > max_ocr['qual_ent']:
                    max_ocr['qual_ent'] = occurences_pos[k]['qual']
                if occurences_pos[k]['main'] > max_ocr['main']:
                    max_ocr['main'] = occurences_pos[k]['main']
            else:
                if occurences_pos[k]['qual'] > max_ocr['qual']:
                    max_ocr['qual'] = occurences_pos[k]['qual']
                if occurences_pos[k]['main'] > max_ocr['r']:
                    max_ocr['r'] = occurences_pos[k]['main']
                if occurences_pos[k]['qual'] > max_ocr['qual_rel']:
                    max_ocr['qual_rel'] = occurences_pos[k]['qual']
                if occurences_pos[k]['main'] > max_ocr['main']:
                    max_ocr['main'] = occurences_pos[k]['main']
        # Exclude entities which don't appear in the dataset. E.g. entity nr. 455 may never appear.
        # always off for wikipeople and jf17k
        
        with open('/export/data/kb_group_shares/wd50k/WD50K/wd50k/cleaned_statements_removeloop/train.txt', 'r') as f:
            raw_trn = []
            for line in f.readlines():
                if config['HAS_QUAL']:
                    raw_trn.append(line.strip("\n").split(","))
                else:
                    raw_trn.append(line.strip("\n").split(",")[0:3])

        with open('/export/data/kb_group_shares/wd50k/WD50K/wd50k/cleaned_statements_removeloop/test.txt', 'r') as f:
            raw_tst = []
            for line in f.readlines():
                if config['HAS_QUAL']:
                    raw_tst.append(line.strip("\n").split(","))
                else:
                    raw_tst.append(line.strip("\n").split(",")[0:3])

        with open('/export/data/kb_group_shares/wd50k/WD50K/wd50k/cleaned_statements_removeloop/valid.txt', 'r') as f:
            raw_val = []
            for line in f.readlines():
                if config['HAS_QUAL']:
                    raw_val.append(line.strip("\n").split(","))
                else:
                    raw_val.append(line.strip("\n").split(",")[0:3])
    elif config['SUBTYPE'] == 'cleaned_statements_removeloopqual':
        config['NUM_ENTITIES'] = 46885
        config['NUM_RELATIONS'] = 1038
        with open('/export/data/kb_group_shares/wd50k/wd50k_qe/wd50k_'+config['SUBTYPE']+'_occurences.json','r') as f:
            occurences = json.load(f)
        import copy
        occurences_single = copy.deepcopy(occurences)
        with open('/export/data/kb_group_shares/wd50k/wd50k_qe/wd50k_'+config['SUBTYPE']+'_occurences_pos.json','r') as f:
            occurences_pos = json.load(f)
        max_ocr = {'single':0,'qual':0,'s':0,'o':0,'s+o':0, 'r':0,'main':0,'qual_ent':0,'qual_rel':0}
        for k in occurences:
            if occurences[k] > max_ocr['single']:
                max_ocr['single'] = occurences[k]
        for k in occurences_pos:
            if k.startswith('Q'):
                if occurences_pos[k]['qual'] > max_ocr['qual']:
                    max_ocr['qual'] = occurences_pos[k]['qual']
                if occurences_pos[k]['s'] > max_ocr['s']:
                    max_ocr['s'] = occurences_pos[k]['s']
                if occurences_pos[k]['s'] + occurences_pos[k]['o']> max_ocr['s+o']:
                    max_ocr['s+o'] = occurences_pos[k]['s'] + occurences_pos[k]['o']
                if occurences_pos[k]['o'] > max_ocr['o']:
                    max_ocr['o'] = occurences_pos[k]['o']
                if occurences_pos[k]['qual'] > max_ocr['qual_ent']:
                    max_ocr['qual_ent'] = occurences_pos[k]['qual']
                if occurences_pos[k]['main'] > max_ocr['main']:
                    max_ocr['main'] = occurences_pos[k]['main']
            else:
                if occurences_pos[k]['qual'] > max_ocr['qual']:
                    max_ocr['qual'] = occurences_pos[k]['qual']
                if occurences_pos[k]['main'] > max_ocr['r']:
                    max_ocr['r'] = occurences_pos[k]['main']
                if occurences_pos[k]['qual'] > max_ocr['qual_rel']:
                    max_ocr['qual_rel'] = occurences_pos[k]['qual']
                if occurences_pos[k]['main'] > max_ocr['main']:
                    max_ocr['main'] = occurences_pos[k]['main']
        # Exclude entities which don't appear in the dataset. E.g. entity nr. 455 may never appear.
        # always off for wikipeople and jf17k
        
        with open('/export/data/kb_group_shares/wd50k/WD50K/wd50k/cleaned_statements_removeloopqual/train.txt', 'r') as f:
            raw_trn = []
            for line in f.readlines():
                if config['HAS_QUAL']:
                    raw_trn.append(line.strip("\n").split(","))
                else:
                    raw_trn.append(line.strip("\n").split(",")[0:3])

        with open('/export/data/kb_group_shares/wd50k/WD50K/wd50k/cleaned_statements_removeloopqual/test.txt', 'r') as f:
            raw_tst = []
            for line in f.readlines():
                if config['HAS_QUAL']:
                    raw_tst.append(line.strip("\n").split(","))
                else:
                    raw_tst.append(line.strip("\n").split(",")[0:3])

        with open('/export/data/kb_group_shares/wd50k/WD50K/wd50k/cleaned_statements_removeloopqual/valid.txt', 'r') as f:
            raw_val = []
            for line in f.readlines():
                if config['HAS_QUAL']:
                    raw_val.append(line.strip("\n").split(","))
                else:
                    raw_val.append(line.strip("\n").split(",")[0:3])
    elif config['SUBTYPE'] == 'statements':
        if config['DATASET'] == 'wd50k' or config['DATASET'] == 'wd50k_nary':
            config['NUM_ENTITIES'] = 47156
            config['NUM_RELATIONS'] = 1064
            with open('/export/data/kb_group_shares/wd50k/wd50k_qe/wd50k_occurences.json','r') as f:
                occurences = json.load(f)
            import copy
            occurences_single = copy.deepcopy(occurences)
            with open('/export/data/kb_group_shares/wd50k/wd50k_qe/wd50k_occurences_pos.json','r') as f:
                occurences_pos = json.load(f)
            max_ocr = {'single':0,'qual':0,'s':0,'o':0,'s+o':0, 'r':0,'main':0,'qual_ent':0,'qual_rel':0}
            for k in occurences:
                if occurences[k] > max_ocr['single']:
                    max_ocr['single'] = occurences[k]
            for k in occurences_pos:
                if k.startswith('Q'):
                    if occurences_pos[k]['qual'] > max_ocr['qual']:
                        max_ocr['qual'] = occurences_pos[k]['qual']
                    if occurences_pos[k]['s'] > max_ocr['s']:
                        max_ocr['s'] = occurences_pos[k]['s']
                    if occurences_pos[k]['s'] + occurences_pos[k]['o']> max_ocr['s+o']:
                        max_ocr['s+o'] = occurences_pos[k]['s'] + occurences_pos[k]['o']
                    if occurences_pos[k]['o'] > max_ocr['o']:
                        max_ocr['o'] = occurences_pos[k]['o']
                    if occurences_pos[k]['qual'] > max_ocr['qual_ent']:
                        max_ocr['qual_ent'] = occurences_pos[k]['qual']
                    if occurences_pos[k]['main'] > max_ocr['main']:
                        max_ocr['main'] = occurences_pos[k]['main']
                else:
                    if occurences_pos[k]['qual'] > max_ocr['qual']:
                        max_ocr['qual'] = occurences_pos[k]['qual']
                    if occurences_pos[k]['main'] > max_ocr['r']:
                        max_ocr['r'] = occurences_pos[k]['main']
                    if occurences_pos[k]['qual'] > max_ocr['qual_rel']:
                        max_ocr['qual_rel'] = occurences_pos[k]['qual']
                    if occurences_pos[k]['main'] > max_ocr['main']:
                        max_ocr['main'] = occurences_pos[k]['main']
            # Exclude entities which don't appear in the dataset. E.g. entity nr. 455 may never appear.
            # always off for wikipeople and jf17k
            
            with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/train.txt', 'r') as f:
                raw_trn = []
                triple_trn = []
                for line in f.readlines():
                    if config['HAS_QUAL']:
                        raw_trn.append(line.strip("\n").split(","))
                        if config['TRIPLE']:
                            triple_trn.append(line.strip("\n").split(",")[0:3])
                    else:
                        raw_trn.append(line.strip("\n").split(",")[0:3])

            with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/test.txt', 'r') as f:
                raw_tst = []
                triple_tst = []
                for line in f.readlines():
                    if config['HAS_QUAL']:
                        raw_tst.append(line.strip("\n").split(","))
                        if config['TRIPLE']:
                            triple_tst.append(line.strip("\n").split(",")[0:3])
                    else:
                        raw_tst.append(line.strip("\n").split(",")[0:3])

            with open('/export/data/kb_group_shares/wd50k/StarE-master/data/clean/wd50k/statements/valid.txt', 'r') as f:
                raw_val = []
                triple_val = []
                for line in f.readlines():
                    if config['HAS_QUAL']:
                        raw_val.append(line.strip("\n").split(","))
                        if config['TRIPLE']:
                            triple_val.append(line.strip("\n").split(",")[0:3])
                    else:
                        raw_val.append(line.strip("\n").split(",")[0:3])
        
        # Get uniques
            statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
                                                                test_data=raw_tst,
                                                                valid_data=raw_val)
        elif config['DATASET'] == 'jf17k':
            config['NUM_ENTITIES'] = 28646
            config['NUM_RELATIONS'] = 1004
            
            # Exclude entities which don't appear in the dataset. E.g. entity nr. 455 may never appear.
            # always off for wikipeople and jf17k
            
            with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/train.txt', 'r') as f:
                raw_trn = []
                triple_trn = []
                for line in f.readlines():
                    if config['HAS_QUAL']:
                        raw_trn.append(line.strip("\n").split(","))
                        if config['TRIPLE']:
                            triple_trn.append(line.strip("\n").split(",")[0:3])
                    else:
                        raw_trn.append(line.strip("\n").split(",")[0:3])

            with open('/export/data/kb_group_shares/wd50k/StarE-master/data/parsed_data/jf17k/test.txt', 'r') as f:
                raw_tst = []
                triple_tst = []
                for line in f.readlines():
                    if config['HAS_QUAL']:
                        raw_tst.append(line.strip("\n").split(","))
                        if config['TRIPLE']:
                            triple_tst.append(line.strip("\n").split(",")[0:3])
                    else:
                        raw_tst.append(line.strip("\n").split(",")[0:3])

            
            
            
            
            # Get uniques
            statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
                                                                    test_data=raw_tst,
                                                                    valid_data=[])
            # print(statement_predicates)
        elif config['DATASET'] == 'wikipeople':
            config['NUM_ENTITIES'] = 34826
            config['NUM_RELATIONS'] = 358
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
        else:
            print('not support!')
    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates
    entoid = {i: pred for i, pred in enumerate(st_entities)}
    prtoid = {i: pred for i, pred in enumerate(st_predicates)}
    mapping = {pred:i for i, pred in enumerate(st_entities)}
    edge_mapping = {pred:i for i, pred in enumerate(st_predicates)}
    """
     However, when we want to run a GCN based model, we also work with
            COO representations of triples and qualifiers.
    
            In this case, for each split: [train, valid, test], we return
            -> edge_index (2 x n) matrix with [subject_ent, object_ent] as each row.
            -> edge_type (n) array with [relation] corresponding to sub, obj above
            -> quals (3 x nQ) matrix where columns represent quals [qr, qv, k] for each k-th edge that has quals
    
        So here, train_data_gcn will be a dict containing these ndarrays.
    """
    # qual_ratio = {'wd50k':1,'wikipeople':0,'jf17k':1}
    qual_ratio = {'wd50k':0.136,'wikipeople':0.029,'jf17k':0.452}
    # qual_ratio = {'wd50k':0.301,'wikipeople':0.064,'jf17k':1}
    # print(f"Training on {n_entities} entities")

    """
        Make the model.
    """
    config['DEVICE'] = torch.device(config['DEVICE'])
    if config['OCCURENCES'] == 'position':
        occurences = occurences_pos
        
    else:
        occurences = {}
        max_ocr = {}
    tp_prtoid = None
    tp_entoid = None
    # occurences = {}
    if config['SPLIT']:
        v = 'splitv2'
    else:
        v = 'nosplit'
    if config['ESTIMATE'] != 'none':
        est1 = 'estimator'
        if config['ESTIMATE_GATE'] != 'none':
            est2 = config['ESTIMATE_GATE']
        else:
            est2 = ''
    else:
        est1 = 'base'
        est2 = ''
    if config['QT'] == 'both':
        query_group = ''
    elif config['QT'] == 'triple':
        query_group = 'triplequery'
    elif config['QT'] == 'qual':
        query_group = 'qualquery'
    print(est2)
    if config['REMOVE_QUAL']:
        noqual = 'noqual'
    else:
        noqual = ''
    if config['RELATION_TRANS']:
        relation = 'relationtrans'
    else:
        relation = 'norelationtrans'
    model_name = "model_qe_"+config['DATASET']+"_p_nary_"+config['STAREARGS']['QUAL_AGGREGATE']+est1+"200"+str(config['LEARNING_RATE'])+"_"+relation+"_"+est2+"_"+config['ESTIMATE']+"_"+query_group+"_"+v+"_"+str(config['RUN'])+"_"+config['var']+"_"+noqual+"_"+str(config['LAYERS'])+"layers_nary.pth"
    print(config)
    make_print_to_file(config, model_name, path='/export/data/kb_group_shares/GNCE/GNCE/training_logs/')
    
    # for train_j in tqdm(range(config['RUN'])):
    if not config['USE_TEST']:
        if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
            # config['LEARNING_RATE'] = 0.0001
            # config['LEARNING_RATE'] = 0.0001
            config['var'] = 'zero'
        elif config['STAREARGS']['QUAL_AGGREGATE'] == 'sum':
            # config['LEARNING_RATE'] = 0.001
            # config['var'] = 'zero'
            # config['LEARNING_RATE'] = 0.0001
            config['var'] = 'one'
        model = NaryModel(config,entoid,prtoid,occurences,max_ocr,'',tp_entoid,tp_prtoid,nodem=mapping,
        edgem=edge_mapping,qual_ratio=qual_ratio)
        # model = StarEEncoder(config,entoid,prtoid,occurences,max_ocr,train_data)
        model.to(config['DEVICE'])

        
        # print(model.parameters())
        print("Model params: ",sum([param.nelement() for param in model.parameters()]))

        if config['OPTIMIZER'] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=config['LEARNING_RATE'])
        elif config['OPTIMIZER'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
        else:
            print("Unexpected optimizer, we support `sgd` or `adam` at the moment")
            raise NotImplementedError

    test_mae = []
    test_q_error = []
    min_q_error = 9999999
    min_mae = 99999999
    model_name_to_save = ''
    model_name_to_save1 = ''
    if config['OCCURENCES'] == 'single':
        model_name = "model_ocr_qe_"+config['STAREARGS']['QUAL_OPN']+"_"+config['STAREARGS']['QUAL_AGGREGATE']+"_"+config['STAREARGS']['QUAL_N']+"_nary.pth"
        model_name_mae = "model_ocr_"+config['STAREARGS']['QUAL_OPN']+"_"+config['STAREARGS']['QUAL_AGGREGATE']+"_"+config['STAREARGS']['QUAL_N']+"_nary.pth"
        if config['USE_VAR']:
            model_name_mae = 'var_' + model_name_mae
            model_name = 'var_' + model_name
    elif config['OCCURENCES'] == 'position':
        model_name = "model_pos_qe_"+config['STAREARGS']['QUAL_OPN']+"_"+config['STAREARGS']['QUAL_AGGREGATE']+"_"+config['STAREARGS']['QUAL_N']+"_rdfmarkov_nary.pth"
        model_name_mae = "model_pos_"+config['STAREARGS']['QUAL_OPN']+"_"+config['STAREARGS']['QUAL_AGGREGATE']+"_"+config['STAREARGS']['QUAL_N']+"_rdfmarkov_nary.pth"
        if config['USE_VAR']:
            model_name = 'var_' + model_name
            model_name_mae = 'var_' + model_name_mae
    else:
        
        if config['SPLIT']:
            v = 'splitv2'
        else:
            v = 'nosplit'
        if config['ESTIMATE'] != 'none':
            est1 = 'estimator'
            if config['ESTIMATE_GATE'] != 'none':
                est2 = config['ESTIMATE_GATE']
            else:
                est2 = ''
        else:
            est1 = 'base'
            est2 = ''
        if config['QT'] == 'both':
            query_group = ''
        elif config['QT'] == 'triple':
            query_group = 'triplequery'
        elif config['QT'] == 'qual':
            query_group = 'qualquery'
        print(est2)
        if config['REMOVE_QUAL']:
            noqual = 'noqual'
        else:
            noqual = ''
        if config['RELATION_TRANS']:
            relation = 'relationtrans'
        else:
            relation = 'norelationtrans'
        # model_name = "model_qe_"+config['STAREARGS']['QUAL_AGGREGATE']+"_"+v+"_stare_"+config['var']+"_distill7_"+config['DISTILL_AGGREGATION']+"_nary.pth"
        # model_name_mae = "model_"+config['STAREARGS']['QUAL_AGGREGATE']+"_"+v+"_stare_"+config['var']+"_distill7_"+config['DISTILL_AGGREGATION']+"_nary.pth"
        model_name = "model_qe_"+config['DATASET']+"_p_nary_"+config['STAREARGS']['QUAL_AGGREGATE']+est1+"200"+str(config['LEARNING_RATE'])+"_"+relation+"_"+est2+"_"+config['ESTIMATE']+"_"+query_group+"_"+v+"_"+config['var']+"_"+noqual+"_"+str(config['LAYERS'])+"layers_nary.pth"
        model_name_mae = "model_"+config['DATASET']+"_p_nary_"+config['STAREARGS']['QUAL_AGGREGATE']+est1+"200"+str(config['LEARNING_RATE'])+"_"+relation+"_"+est2+"_"+config['ESTIMATE']+"_"+query_group+"_"+v+"_"+config['var']+"_"+noqual+"_"+str(config['LAYERS'])+"layers_nary.pth"
        # model_name = "model_qe_"+config['STAREARGS']['QUAL_AGGREGATE']+"_"+config['STAREARGS']['QUAL_N']+"_"+config['UNCERTAIN']+"_denseedge_400dim_changekb_nary.pth"
        # model_name_mae = "model_"+config['STAREARGS']['QUAL_AGGREGATE']+"_"+config['STAREARGS']['QUAL_N']+"_"+config['UNCERTAIN']+"_denseedge_400dim_changekb_nary.pth"
        
        if config['USE_VAR']:
            model_name = 'var_' + model_name
            model_name_mae = 'var_' + model_name_mae
    
    # make_print_to_file(config, model_name, path='/export/data/kb_group_shares/GNCE/GNCE/training_logs/')
    
    
    # 
    if not config['USE_TEST']:
        loss = MSELoss()
        cla_loss = NLLLoss()
        info_loss = InfoNCELoss()
        config['RETURN_EMBED'] = False
        model_active = ''
        valid_q_error = 0
        valid_mae = 0
        if config['ESTIMATE_GATE'] != 'none' and config['ESTIMATE_GATE'] != 'linear' and config['ESTIMATE_GATE'] != 'freq':
        
            gbrfe_optim = torch.optim.Adam(model.G_BRFE.parameters(),lr=0.01)
        training_data = []
        # if config['BATCH_SIZE'] > 1:
        # with torch.no_grad():
        #     for epoch in tqdm(range(config['EPOCHS'])):
        #         points_processed = 0
        #         input_data = train_data
        #         random.Random(random.randint(0,config['EPOCHS'])).shuffle(input_data)
        #         # print(input_data)
        #         # input_data = [{'triples':[['Q38111', 'P166', '?v0', 'P1686', 'Q892735', 'direct'], ['Q117315', 'P166', '?v0', 'P1346', '?v1', 'direct'], ['?v1', 'P1411', 'Q106291', 'P1686', '?v2', 'P805', 'Q857047', 'direct'], ['?v2', 'P2747', 'Q23830578', 'direct', '?v3']],'gt':1}]
            
        #         time1 = time.time()
        #         query_per_batch = []
        #         count = 0
        #         training_data_per_epoch = []
        #         for d in tqdm(input_data):
        #             f = 0
        #             for triple in d['triples']:
        #                 if len(triple) > 3:
        #                     f = 1
        #                     break
        #             if config['QT'] == 'triple' and f == 1 or config['QT'] == 'qual' and f == 0:
        #                 # if f == 1:
        #                 continue
                    
        #             convert_triple = None
        #             ents,rels,dat,nodem,edgem,var,triple_f = get_query_graph_data_nary(d['triples'],mapping,edge_mapping,config)
                    

        #             x, r, edge_index, edge_type, quals = model.load_queries(ents,rels,nodem,edgem,var,dat)
        #             # print(edge_index)
        #             query_per_batch.append([x, r, edge_index, edge_type, quals,d['gt'],''])
        #             count += 1
        #             if count == config['BATCH_SIZE']:
        #                 data = batch_queries(query_per_batch,config)
        #                 training_data_per_epoch.append(data)
        #                 query_per_batch = []
        #                 count = 0
        #                 # print()
        #         # training_data.append(training_data_per_epoch)
        #         if str(config['STAREARGS']['QUAL_AGGREGATE']) == 'cat':
        #             with open(config['DATASET']+'/train_'+str(config['BATCH_SIZE'])+'/'+str(epoch)+'.pkl','wb') as f:
        #                 pickle.dump(training_data_per_epoch,f)
        #         else:
        #             with open(config['DATASET']+'/train_'+str(config['BATCH_SIZE'])+'_'+str(config['STAREARGS']['QUAL_AGGREGATE'])+'/'+str(epoch)+'.pkl','wb') as f:
        #                 pickle.dump(training_data_per_epoch,f)
        test_query_per_batch = []
        count = 0
        test_queries = []
        all_test_data = valid_data+test_data

        for datapoint in valid_data+test_data:
            f = 0
            for triple in datapoint['triples']:
                if len(triple) > 3:
                    f = 1
                    break
            if config['QT'] == 'triple' and f == 1 or config['QT'] == 'qual' and f == 0:
                # if f == 1:
                continue
            convert_triple = None
            ents,rels,dat,nodem,edgem,var,triple_f = get_query_graph_data_nary(datapoint['triples'],mapping,edge_mapping,config)

            x, r, edge_index, edge_type, quals = model.load_queries(ents,rels,nodem,edgem,var,dat)
            count += 1
            test_query_per_batch.append([x, r, edge_index, edge_type, quals,datapoint['gt'],datapoint['triples']])
            if count == config['BATCH_SIZE']:
                data = batch_queries(test_query_per_batch,config)
                test_queries.append(data)
                test_query_per_batch = []
                count = 0
            
        #     with open(config['DATASET']+'/test.pkl','wb') as f:
        #         pickle.dump(test_queries,f)
        # with open(config['DATASET']+'/test.pkl','rb') as f:
        #     test_queries = pickle.load(f)
        #     test_data = valid_data + test_data
        
        for epoch in tqdm(range(config['EPOCHS'])):
            model.train()
            try:
                if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
                    with open(config['DATASET']+'/train_'+str(config['BATCH_SIZE'])+'/'+str(epoch)+'.pkl','rb') as f:
                        train_data = pickle.load(f)
                else:
                    with open(config['DATASET']+'/train_'+str(config['BATCH_SIZE'])+'_'+str(config['STAREARGS']['QUAL_AGGREGATE'])+'/'+str(epoch)+'.pkl','rb') as f:
                        train_data = pickle.load(f)
            except:
                build_training_data(train_data,config,model)

                if config['STAREARGS']['QUAL_AGGREGATE'] == 'cat':
                    with open(config['DATASET']+'/train_'+str(config['BATCH_SIZE'])+'/'+str(epoch)+'.pkl','rb') as f:
                        train_data = pickle.load(f)
                else:
                    with open(config['DATASET']+'/train_'+str(config['BATCH_SIZE'])+'_'+str(config['STAREARGS']['QUAL_AGGREGATE'])+'/'+str(epoch)+'.pkl','rb') as f:
                        train_data = pickle.load(f)
            
            abs_errors = []
            train_q_error = 0
            train_loss = 0
            num_batches = 0
            preds = []
            gts = []
            points_processed = 0
            for data in tqdm(train_data):
                # print(f'qual: {data["qual"]}')
                if config['ESTIMATE_GATE'] == 'none':
                    out = model(data['x'].float(), data['r'].float(), data['edge_index'], data['edge_type'], data['qual'].long(),data['batch'])
                else:
                    out,sigma,mu = model(data['x'].float(), data['r'].float(), data['edge_index'], data['edge_type'], data['qual'].long(),data['batch'])
                    if config['ESTIMATE_GATE'] != 'none' and config['ESTIMATE_GATE'] != 'linear' and config['ESTIMATE_GATE'] != 'freq':
                        loss_bayesian = torch.tensor(0.).to(config['DEVICE'])
                        loss_bayesian += torch.sum(torch.log((sigma+1e-10)/(model.G_BRFE.sigma.weight+1e-10)+1e-10))-600
                        # print(loss_bayesian.item())
                        loss_bayesian += torch.sum((model.G_BRFE.mu.weight-mu)**2/(sigma+1e-10))
                        # print((torch.sum((G_BRFE.mu.weight-mu)**2/(sigma+1e-10))).item())
                        loss_bayesian += torch.sum((model.G_BRFE.sigma.weight)/(sigma+1e-10))
                        # print((torch.sum((G_BRFE.sigma.weight)/(sigma+1e-10))).item())
                        loss_bayesian = loss_bayesian/2
            
                if config['ESTIMATE_GATE'] != 'none' and config['ESTIMATE_GATE'] != 'linear' and config['ESTIMATE_GATE'] != 'freq':
                    l = loss(out, torch.log(data['y']).to(config['DEVICE']).float()) + loss_bayesian
                else:
                    # l = loss(out, torch.log(data['y']).to(config['DEVICE']).float()) 
                    loss_i = None
                    for batch_i in range(len(data['y'])):
                        label_i = torch.log(data['y'])[batch_i]
                        output_i = out[batch_i]
                        if loss_i is None:
                            loss_i = loss(output_i, label_i)
                        else:
                            loss_i += loss(output_i, label_i)
                        # print(loss_i.item())
                        # loss_i.backward()
                        
                # pred = out.detach().cpu().numpy()
                batch_q_error = q_error(out, data['y'])
                train_loss += loss_i.item()
                loss_i.backward()
                # print(batch_q_error)
                # print(f'{pred} {y}')
                # y = d["gt"]
                # pred = np.exp(pred)
                # print(f'{pred} {y}')
                # pred = np.exp(pred)
                # if config['BATCH_SIZE'] ==1:
                #         preds.append(out.squeeze().tolist())
                #     else:
                #         preds += out.squeeze().tolist()

                #     gts += data['y'].tolist()
                # abs_errors.append(np.abs(pred - y))
                
                # train_q_errors.append(np.max([np.abs(pred) / y, y / np.abs(pred)]))
                # print(f'{batch_q_error} {out} {torch.log(data["y"])} {data["y"]}')
                points_processed += 1
                # print(l.item())
                # train_loss += l.item()
                # l.backward()
                train_q_error += torch.sum(batch_q_error).item()
                # print(len(data['y']))
                num_batches += 1
                # Gradient Accumulation
                # print(train_loss)
                # if points_processed*config['BATCH_SIZE'] == 32:
                #     # print(abs_errors[0])
                #     if config['ESTIMATE_GATE'] != 'none' and config['ESTIMATE_GATE'] != 'linear' and config['ESTIMATE_GATE'] != 'freq':
                #         gbrfe_optim.step()
                #         model.G_BRFE.sigma.weight.data = F.relu(model.G_BRFE.sigma.weight.data)
                    
                optimizer.step()
                optimizer.zero_grad()
                points_processed = 0

            # print(gts)
            # print(preds)
            
            avg_train_loss = train_loss / num_batches
            avg_train_q_error = train_q_error / num_batches
            print('Train Loss: ', avg_train_loss)
            print('Train Qerror: ', avg_train_q_error)
            # print('Time: ',time2-time1)
            model.eval()
            time2 = time.time()
            abs_errors = []
            q_errors = []
            preds = []
            gts = []
            error = []
            total = []
            error_cnt = 0
            num_batches = 0
            test_loss = 0
            test_q_error = 0
            with torch.no_grad():
                for data in test_queries:
                    f = 0
                    # for triple in data['triples']:
                    #     if len(triple) > 3:
                    #         f = 1
                    #         break
                    # if config['QT'] == 'triple' and f == 1 or config['QT'] == 'qual' and f == 0:
                    #     # if f == 1:
                    #     continue
                    convert_triple = None
                    

                    
                    # ents,rels,dat,nodem,edgem,var,triple_f = get_query_graph_data_nary(datapoint['triples'],mapping,edge_mapping,config)

                    # x, r, edge_index, edge_type, quals = model.load_queries(ents,rels,nodem,edgem,var,dat)
                    
                    if config['ESTIMATE_GATE'] == 'none':
                        out = model(data['x'].float(), data['r'].float(), data['edge_index'], data['edge_type'], data['qual'].long(), data['batch'])
                    else:
                        out,sigma,mu = model(data['x'].float(), data['r'].float(), data['edge_index'], data['edge_type'], data['qual'].long(), data['batch'])
                    # y = datapoint["gt"]
                    # pred = out.detach().cpu().numpy()
                    # # print(f'{pred} {y}')
                    # pred = np.exp(pred)
                    # print(f'{pred} {y}')
                    # pred = np.exp(pred)
                    try:
                        batch_q_error = q_error(out, data['y'])
                    except:
                        print('prapa')
                    test_loss += loss(out, torch.log(data['y'])).item()
                    test_q_error += torch.sum(batch_q_error).item()  # Sum q-errors in the batch
                    

                    if config['BATCH_SIZE'] ==1:
                        preds.append(out.squeeze().tolist())
                    else:
                        preds += out.squeeze().tolist()

                    gts += data['y'].tolist()
                    # abs_errors.append(np.abs(pred - y))
                    # q_errors.append(np.max([np.abs(pred) / y, y / np.abs(pred)]))
                    # if np.max([np.abs(pred) / y, y / np.abs(pred)]) > 5:
                        
                    #     error.append([datapoint['triples'],datapoint["gt"],float(pred[0])])
                    #     # print()
                    #     error_cnt += 1
                    for i in range(len(data['y'].tolist())):
                        
                        total.append([data['triples'][i],data['y'].tolist()[i],float(out.squeeze().tolist()[i])])
                    num_batches += len(data['y'])
            # print(gts)
            # print(preds)
            avg_test_loss = config['BATCH_SIZE']*test_loss / num_batches
            
            avg_test_q_error = test_q_error / num_batches
            print('Loss: ', avg_test_loss)
            print('Qerror: ', avg_test_q_error)
            print('Min Qerror: ', min_q_error)
            time1 = time.time()
            print('Time: ',time2-time1)
            if (avg_test_q_error < min_q_error) and (avg_train_q_error <  avg_test_q_error or epoch > 17):
                # torch.save(model.state_dict(), "models_extend/"+model_name)
                # with open(config['DATASET']+'/extend/analyse/extra/'+model_name.replace('.pth','')+'_error_query.json','w') as f:
                #     json.dump(error,f,indent=2)
                # with open(config['DATASET']+'/extend/analyse/extra/'+model_name.replace('.pth','')+'_total_query.json','w') as f:
                #     json.dump(total,f,indent=2)
                min_q_error = avg_test_q_error
            # if (np.mean(abs_errors) < min_mae) and (np.mean(avg_train_q_error) <  np.mean(avg_test_q_errors) or epoch > 17):
            #     torch.save(model.state_dict(), "models_extend/"+model_name_mae)
                
            #     min_mae = np.mean(abs_errors)
        print()
        print('Valid Qerror: ', valid_q_error / config['EPOCHS'])
        print('Valid MAE: ', valid_mae / config['EPOCHS'])
        print('Min Qerror: ', min_q_error)
        
        print()
        print()
        print()
                
    else:
        models = ['model_qe_wd50k_catestimator200_pattern1__vae__splitv2_stare_zero__nary.pth']
        models = ['model_qe_wd50k_catestimator200_degree2__vae__splitv2_stare_zero__nary.pth']
        info_loss = InfoNCELoss()
        for m in tqdm(models):
            # print(m)
            # if 'dense' not in m or 'nondense' in m or 'inverseremoval' in m:
            #     continue
            if m.endswith('.pth') is False:
                continue
            config['EMBEDDING_DIM'] = 200
            config['HID_DIM'] = 200
            config['var'] = 'zero'
            config['SPLIT'] = True
            config['ESTIMATE'] = 'vae'
            config['INIT_EMBED'] = 'stare'
            config['STAREARGS']['QUAL_AGGREGATE'] = 'cat'
            config['PRINT_VECTOR'] = True
        
            print(f'{m} {config}')
            model = NaryModel(config,entoid,prtoid,occurences,max_ocr,'',tp_entoid,tp_prtoid)
            model.to(config['DEVICE'])
            dic = torch.load('models_extend/'+m)
            # state_dic  = OrderedDict()
            # for k,v in dic.items():
                # if k.startswith('cla') or k.startswith('ent_trans') or k.startswith('rel_trans'):
                #     continue
                # else:
                #     state_dic[k] = v
                # pass
            model.load_state_dict(dic)
            model.eval()
            
            # for idx in range(0,6):
            #     with open('/export/data/kb_group_shares/wd50k/wd50k_qe/extend/dense_grouped/'+str(idx)+'.json','r') as f:
            #         test_data = json.load(f)
            #     # if idx == 2:
            #     #     continue
            #     if len(test_data) == 0:
            #         continue
            #     test_data = random.sample(test_data,min(778,len(test_data)))
                
            abs_errors = []
            q_errors = []
            consistency_error = []
            preds = []
            gts = []
            error = []
            total = []
            error_cnt = 0
            test_uncertainties = []
            testset_dict = {}
            word_list = {}
            qual_rotate_list = {}
            for i,datapoint in tqdm(enumerate(valid_data+test_data)):
                # print(datapoint['triples'])
                ents,rels,dat,nodem,edgem,var,triple_f = get_query_graph_data_nary(datapoint['triples'],mapping,edge_mapping,config)
                # tpents,tprels,tpdat,tpnodem,tpedgem,tpvar = get_query_graph_data_tp(datapoint['triples'],tp_mapping,tp_edge_mapping,config)
            
                with torch.no_grad():
                    if config['PRINT_VECTOR']:
                        out, qual_emb = model(ents,rels,nodem,edgem,var,dat)
                    else:
                        out = model(ents,rels,nodem,edgem,var,dat)
                    # out = model(ents,rels,nodem,edgem,var,dat,tpents,tprels,tpnodem,tpedgem,tpvar,tpdat,triple_f)
                y = datapoint["gt"]
                pred = out.detach().cpu().numpy()
                # print(f'{pred} {y}')
                if config['PRINT_VECTOR']:
                    # print(f"{qual_emb[2]['qual_emb'][qual_emb[2]['qual_idx']].shape} {qual_emb[2]['qual_emb'][1].shape} {qual_emb[2]['qual_idx']}")
                    # word_list[i] = {'freq':len(d[3:]) // 2 if len(d) > 3 else 0,'norm':float(torch.norm(recon_x)),'fact':d}
                    # qual_rotate_list[i] = {'freq':len(d[3:]) // 2 if len(d) > 3 else 0,'norm':float(torch.norm(qual)),'fact':d}
                    word_list[i] = {}
                    qual_rotate_list[i] = {}
                    word_list[i][1] = []
                    word_list[i][2] = []
                    qual_rotate_list[i][1] = []
                    qual_rotate_list[i][2] = []
                    for qualidx in range(0,qual_emb[1]['qual_idx'].shape[0]):
                        if qual_emb[2]['qual_idx'][qualidx]:
                            word_list[i][1].append({'norm':float(torch.norm(qual_emb[1]['qual_emb'][qualidx])),'fact':datapoint['triples'][qualidx]})
                        else:
                            qual_rotate_list[i][1].append({'norm':float(torch.norm(qual_emb[1]['qual_emb'][qualidx])),'fact':datapoint['triples'][qualidx]})
                    for qualidx in range(0,qual_emb[2]['qual_idx'].shape[0]):
                        if qual_emb[2]['qual_idx'][qualidx]:
                            word_list[i][2].append({'norm':float(torch.norm(qual_emb[2]['qual_emb'][qualidx])),'fact':datapoint['triples'][qualidx]})
                        else:
                            qual_rotate_list[i][2].append({'norm':float(torch.norm(qual_emb[2]['qual_emb'][qualidx])),'fact':datapoint['triples'][qualidx]})
                pred = np.exp(pred)
                # print(f'{pred} {y}')
                # pred = np.exp(pred)
                preds.append(pred)
                gts.append(y)
                abs_errors.append(np.abs(pred - y))
                q_errors.append(np.max([np.abs(pred) / y, y / np.abs(pred)]))
                if np.max([np.abs(pred) / y, y / np.abs(pred)]) > 5:
                    error.append([datapoint['triples'],datapoint["gt"],float(pred[0])])
                    # print()
                    error_cnt += 1
                total.append([datapoint['triples'],datapoint["gt"],float(pred[0])])
            
            print('MAE: ', np.mean(abs_errors))
            test_mae.append(np.mean(abs_errors))
            print('consistency:',np.mean(consistency_error))
            print('Qerror: ', np.mean(q_errors))
            test_q_error.append(np.mean(q_errors))
            print(f'Error Triple:{error_cnt} {error_cnt/len(test_data+valid_data)}')
            # print(f'uncertain error: {overlap}')
            with open('wd50k/'+m.replace('.pth','')+'_qual_rotate.json','w') as f:
                json.dump(qual_rotate_list,f,indent=2)
            with open('wd50k/'+m.replace('.pth','')+'_recon_x.json','w') as f:
                json.dump(word_list,f,indent=2)
            with open('wd50k/extend/analyse/extra/'+m.replace('.pth','')+'_error_query_test.json','w') as f:
                json.dump(error,f,indent=2)
            with open('wd50k/extend/analyse/extra/'+m.replace('.pth','')+'_total_query_test.json','w') as f:
                json.dump(total,f,indent=2)
                # with open('wd50k/extend/analyse/extra/'+m.replace('.pth','')+'_selected_uncertain_query.json','w') as f:
                #     json.dump(selected,f,indent=2)
            print()

