import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

# path
Dataset_Path = './Data/'
Model_Path = './Model/'
Result_Path = './Result/'
#Fasta_Path = './Data/Tmfasta/'
#Pssm_Path = './Data/Tmpssm/'
#Hhm_Path = './Data/PSSM_HHM_A3M/Tmhhm/'
#Spd33_Path = './Data/Tmspd33/'

Fasta_Path = './Data/last/fasta/'
Pssm_Path = './Data/last/pssm/'
Hhm_Path = './Data/last/PSSM_HHM_A3M/hhm/'
Spd33_Path = './Data/last/spd33/'

Node_Feature_num = 135

aalist = list('ACDEFGHIKLMNPQRSTVWY')
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
polar_aa = 'AVLIFWMP'
nonpolar_aa = 'GSTCYNQHKRDE'

with open('./Data/aa_phy7','r') as f:
    pccp = f.read().splitlines()
    pccp = [i.split() for i in pccp]
    pccp_dic = {i[0]: np.array(i[1:]).astype(float) for i in pccp}
fastalist = []
with open('./fastalist.txt','r') as f:
    for line in f:
        fastalist.append(line[:-1])

def read_pccp(seq):
    return np.array([pccp_dic[i] for i in seq])

def read_fasta(fname):
    with open(fname,'r') as f:
        sequence = f.read().splitlines()[1]
    return sequence

def read_pssm(fname,seq):
    num_pssm_cols = 44
    pssm_col_names = [str(j) for j in range(num_pssm_cols)]
    with open(fname,'r') as f:
        tmp_pssm = pd.read_csv(f,delim_whitespace=True,names=pssm_col_names).dropna().values[:,2:22].astype(float)
    if tmp_pssm.shape[0] != len(seq):
        raise ValueError('PSSM file is in wrong format or incorrect!')
    return tmp_pssm

def read_hhm(fname,seq):
    num_hhm_cols = 22
    hhm_col_names = [str(j) for j in range(num_hhm_cols)]
    with open(fname,'r') as f:
        hhm = pd.read_csv(f,delim_whitespace=True,names=hhm_col_names)
    pos1 = (hhm['0']=='HMM').idxmax()+3
    num_cols = len(hhm.columns)
    hhm = hhm[pos1:-1].values[:,:num_hhm_cols].reshape([-1,44])
    hhm[hhm=='*']='9999'
    if hhm.shape[0] != len(seq):
        raise ValueError('HHM file is in wrong format or incorrect!')
    return hhm[:,2:-12].astype(float)

def spd3_feature_sincos(x,seq):
    ASA = x[:,0]
    rnam1_std = "ACDEFGHIKLMNPQRSTVWYX"
    ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                        185, 160, 145, 180, 225, 115, 140, 155, 255, 230,1)
    dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))
    ASA_div =  np.array([dict_rnam1_ASA[i] for i in seq])
    ASA = (ASA/ASA_div)[:,None]
    angles = x[:,1:5]
    HSEa = x[:,5:7]
    HCEprob = x[:,-3:]
    angles = np.deg2rad(angles)
    angles = np.concatenate([np.sin(angles),np.cos(angles)],1)
    return np.concatenate([ASA,angles,HSEa,HCEprob],1)

def read_spd33(fname,seq):
    with open(fname,'r') as f:
        spd3_features = pd.read_csv(f,delim_whitespace=True).values[:,3:].astype(float)
    tmp_spd3 = spd3_feature_sincos(spd3_features,seq)
    if tmp_spd3.shape[0] != len(seq):
        raise ValueError('SPD3 file is in wrong format or incorrect!')
    return tmp_spd3

def get_AAfq(seq):
    AAfq_dic = dict()
    AAfq = np.array([seq.count(x) for x in aalist])/len(seq)
    for (key,value) in zip(aalist,AAfq):
         AAfq_dic[key] = value
    seq_AAfq = np.array([AAfq_dic[x] for x in seq])
    return seq_AAfq

def do_count(seq):
    dimers = Counter()
    for i in range(len(seq)-1):
        dimers[seq[i:i+2]] += 1.0
    return dimers

def get_dipfq(seq):
    result = do_count(seq)
    dimers = sum(result.values())
    dimers_fq = dict()
    for a1 in amino_acids:
        for a2 in amino_acids:
            dimers_fq[a1+a2] = (result[a1+a2]*1.0)/dimers
    dipfq ={}
    for x in aalist:
        a=[]
        b=[]
        for y in aalist:
            a.append(dimers_fq[x+y])
            b.append(dimers_fq[y+x])
        dipfq[x] = np.hstack((a,b))
    seq_dipfq = []
    for x in seq:
        seq_dipfq.append(dipfq[x].tolist())
    return seq_dipfq

def load_blosum():
    with open(Dataset_Path + 'BLOSUM62_dim23.txt', 'r') as f:
        result = {}
        next(f)
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            result[line[0]] = [int(i) for i in line[1:]]
    return result

def get_blosum(sequence):
    blosum_dic = load_blosum()
    blosum = np.array([blosum_dic[i] for i in sequence])
    return blosum

def get_matrix():
    for uniprot_id in tqdm(fastalist):
        sequence = read_fasta(Fasta_Path+uniprot_id)
        # L * 23
        blosum = get_blosum(sequence)
        # L * 1
        AAfq = get_AAfq(sequence)
        # L * 40
        dipfq = get_dipfq(sequence)
        # L * 20
        pssm = read_pssm(Pssm_Path + uniprot_id + '.pssm',sequence)
        # L * 30
        hhm = read_hhm(Hhm_Path + uniprot_id + '.hhm',sequence)
        # L * 14
        spd33 = read_spd33(Spd33_Path + uniprot_id + '.spd33',sequence)
        # L * 7
        PP7 = read_pccp(sequence)
        #matrix = np.concatenate([blosum,pssm,hhm,spd33,PP7,np.array(AAfq).reshape(-1,1),np.array(dipfq)],axis=1)
        matrix = np.concatenate([blosum,pssm,hhm,spd33,PP7,np.array(AAfq).reshape(-1,1),np.array(dipfq)],axis=1)
        np.save('./Data/node_features/' + uniprot_id + '.npy',matrix)

def cal_mean_std():
    total_length = 0
    oneD_mean = np.zeros(Node_Feature_num)
    oneD_mean_square = np.zeros(Node_Feature_num)
    for name in tqdm(fastalist):
        matrix = np.load('./Data/node_features/' + name+'.npy')
        total_length += matrix.shape[0]
        oneD_mean += np.sum(matrix, axis=0)
        oneD_mean_square += np.sum(np.square(matrix),axis=0)
    oneD_mean /= total_length  # EX
    oneD_mean_square /= total_length  # E(X^2)
    oneD_std = np.sqrt(np.subtract(oneD_mean_square, np.square(oneD_mean)))  # DX = E(X^2)-(EX)^2, std = sqrt(DX)
    np.save('./Data/oneD_mean.npy', oneD_mean)
    np.save('./Data/oneD_std.npy', oneD_std)

if __name__ == "__main__":
    get_matrix()
    #cal_mean_std()
