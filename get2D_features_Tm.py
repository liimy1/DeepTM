import os
import numpy as np
from tqdm import tqdm
fastalist = []
with open('./fastalist.txt','r') as f:
    for line in f:
        fastalist.append(line[:-1])


def read_spotcon(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    for index, line in enumerate(lines):
        if index < 5:
            continue
        elif index == 5:
            sequence = line.strip()
            matrix = np.ones((len(sequence), len(sequence)), dtype=float)
            mask1 = np.tril(np.ones((len(sequence), len(sequence)), dtype=float), -3)
            mask2 = np.triu(np.ones((len(sequence), len(sequence)), dtype=float), 3)
            mask = mask1 + mask2
            matrix -= mask
        else:
            data = line.strip().split()
            matrix[int(data[0])][int(data[1])] = float(data[2])
            matrix[int(data[1])][int(data[0])] = float(data[2])
    return matrix


def read_fasta(fname):
    with open(fname,'r') as f:
        sequence = f.read().splitlines()[1]
    return sequence


# spotcon = [name.split('.')[0] for name in os.listdir('./Data/linshispotcon/')]
for name in tqdm(fastalist):
    # name = name.split('.')[0]
    # if name in spotcon:
    matrix = read_spotcon('./Data/spotcon/'+name+'.spotcon')
    # else:
    #     sequence = read_fasta('./Data/linshifasta/'+name)
    #     matrix = np.ones((len(sequence), len(sequence)), dtype=float)
    #     mask1 = np.tril(np.ones((len(sequence), len(sequence)), dtype=float), -3)
    #     mask2 = np.triu(np.ones((len(sequence), len(sequence)), dtype=float), 3)
    #     mask = mask1 + mask2
    #     matrix -= mask
    np.save('./Data/edge_features_Tm/'+name+'.npy', matrix)



