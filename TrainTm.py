import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import sys
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
np.set_printoptions(threshold=sys.maxsize)
# path
Dataset_Path = './Data/'
Model_Path = './Model/'
Result_Path = './Result/'
Fasta_Path = './Data/fasta/'
Pssm_Path = './Data/pssm/'
TrainData_Path = './Data/Train.csv'


amino_acid = list("ACDEFGHIKLMNPQRSTVWY")
amino_dict = {aa: i for i, aa in enumerate(amino_acid)}

# Seed
SEED = 2333
np.random.seed(SEED)
torch.manual_seed(SEED)
print(torch.cuda.is_available())
device_ids = [0]
if torch.cuda.is_available():
    # torch.cuda.set_device(1)
    torch.cuda.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Model parameters
NUMBER_EPOCHS = 50
LEARNING_RATE = 1E-4
WEIGHT_DECAY = 1E-3
BATCH_SIZE = 32
NUM_CLASSES = 1
LENG_SIZE = 1028

# GCN parameters
GCN_FEATURE_DIM = 133
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 64

# Attention parameters
DENSE_DIM = 16
ATTENTION_HEADS = 4

ogt_dic = {}
Ogt = pd.read_csv(TrainData_Path,sep=',')
ogt_dic_uniprotid = np.array([str(n) for n in Ogt['uniprot_id'].values])
#ogt_dic_uniprotid = Ogt['uniprot_id'].values
ogt_dic_topt = Ogt['ogt'].values
for i in range(len(ogt_dic_uniprotid)):
    ogt_dic[ogt_dic_uniprotid[i]] = ogt_dic_topt[i]

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def load_sequences(sequence_path):
    names, sequences, labels = ([] for i in range(3))
    for file_name in tqdm(os.listdir(sequence_path)):
        with open(sequence_path + file_name, 'r') as file_reader:
            lines = file_reader.read().split('\n')
            names.append(file_name)
            sequences.append(lines[1])
            labels.append(int(lines[2]))
    return pd.DataFrame({'names': names, 'sequences': sequences, 'labels': labels})


def load_features(uniprot_id, mean, std):
    # len(sequence) * 94
    feature_matrix1 = np.load(Dataset_Path + 'node_features/' + uniprot_id + '.npy')
    feature_matrix2 = np.ones(feature_matrix1.shape[0])
    feature_matrix2 = feature_matrix2 * ogt_dic[uniprot_id]
    feature_matrix = np.concatenate((feature_matrix1, feature_matrix2.reshape(-1,1)),axis=1)
    feature_matrix = (feature_matrix - mean) / std
    part1 = feature_matrix[:,0:20]
    part2 = feature_matrix[:,23:]

    # len(sequence) * 91
    feature_matrix = np.concatenate([part1,part2],axis=1)
    return feature_matrix


aalist = list('ACDEFGHIKLMNPQRSTVWY')
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
polar_aa = 'AVLIFWMP'
nonpolar_aa = 'GSTCYNQHKRDE'

def load_graph(sequence_name):
    matrix = np.load(Dataset_Path + 'edge_features/' + sequence_name + '.npy').astype(np.float32)
    matrix = normalize(matrix)
    return matrix


def load_values():
    # (94,)
    mean1 = np.load(Dataset_Path + 'oneD_mean.npy')
    std1 = np.load(Dataset_Path + 'oneD_std.npy')
    # (1,)
    df = pd.read_csv(TrainData_Path,sep=',')
    ogt = df['ogt'].values
    mean2 = []
    mean2.append(np.mean(ogt))
    std2 = []
    std2.append(np.std(ogt))
    mean2 = np.array(mean2)
    std2 = np.array(std2)
    mean = np.concatenate([mean1, mean2])
    std = np.concatenate([std1, std2])
    return mean, std


def load_blosum():
    with open(Dataset_Path + 'BLOSUM62_dim23.txt', 'r') as f:
        result = {}
        next(f)
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            result[line[0]] = [int(i) for i in line[1:]]
    return result


class ProDataset(Dataset):

    def __init__(self, dataframe):
        self.names = np.array([str(n) for n in dataframe['uniprot_id'].values])
        #self.names = dataframe['uniprot_id'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['tm'].values
        self.mean, self.std = load_values()
        self.blosum = load_blosum()

    def __getitem__(self, index):
        uniprot_id = self.names[index]
        sequence = self.sequences[index]
        label = self.labels[index]
        # L * 94
        sequence_feature = load_features(uniprot_id, self.mean, self.std)
        sequence_feature = np.pad(sequence_feature,((0,LENG_SIZE-len(sequence)),(0,0)),'constant')
        
        # L * L
        sequence_graph = load_graph(uniprot_id)
        sequence_graph=np.pad(sequence_graph,(0,LENG_SIZE-len(sequence)),'constant')
        
        return uniprot_id, sequence, label, sequence_feature, sequence_graph

    def __len__(self):
        return len(self.labels)


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = input @ self.weight    # X * W
        output = adj @ support           # A * X * W
        if self.bias is not None:        # A * X * W + b
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(GCN_FEATURE_DIM, GCN_HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(GCN_HIDDEN_DIM)
        self.gc2 = GraphConvolution(GCN_HIDDEN_DIM, GCN_OUTPUT_DIM)
        self.ln2 = nn.LayerNorm(GCN_OUTPUT_DIM)
        self.relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x, adj):  			# x.shape = (seq_len, GCN_FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = self.gc1(x, adj)  				# x.shape = (seq_len, GCN_HIDDEN_DIM)
        x = self.relu1(self.ln1(x))
        x = self.gc2(x, adj)
        output = self.relu2(self.ln2(x))	# output.shape = (seq_len, GCN_OUTPUT_DIM)
        return output


class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  				# input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  	# x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  					# x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		# attention.shape = (1, attention_hops, seq_len)
        return attention


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.gcn = GCN()
        self.attention = Attention(GCN_OUTPUT_DIM, DENSE_DIM, ATTENTION_HEADS)
        self.fc_final = nn.Linear(GCN_OUTPUT_DIM, NUM_CLASSES)
        self.criterion = nn.MSELoss()
        weight_p, bias_p = [],[]
        for name, p in self.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': WEIGHT_DECAY},
                                            {'params': bias_p, 'weight_decay': 0}
                                            ], lr=LEARNING_RATE)

    def forward(self, x, adj):  											# x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = x.float()
        x = self.gcn(x, adj)  												# x.shape = (seq_len, GAT_OUTPUT_DIM)
        l = len(x.shape)
        if(l<3):
            x = x.unsqueeze(0).float()  										# x.shape = (1, seq_len, GAT_OUTPUT_DIM)
        att = self.attention(x)  											# att.shape = (1, ATTENTION_HEADS, seq_len)
        node_feature_embedding = att @ x 									# output.shape = (1, ATTENTION_HEADS, GAT_OUTPUT_DIM)
        node_feature_embedding_avg = torch.sum(node_feature_embedding,
                                               1) / self.attention.n_heads  # node_feature_embedding_avg.shape = (1, GAT_OUTPUT_DIM)
        output = torch.sigmoid(self.fc_final(node_feature_embedding_avg))  	# output.shape = (1, NUM_CLASSES)
        return output.squeeze(0)


def train_one_epoch(model, data_loader, epoch):

    epoch_loss_train = 0.0
    n_batches = 0
    for data in tqdm(data_loader):
        model.module.optimizer.zero_grad()
        _, _, labels, sequence_features, sequence_graphs = data

        sequence_features = torch.squeeze(sequence_features)
        sequence_graphs = torch.squeeze(sequence_graphs)

        if torch.cuda.is_available():
            features = Variable(sequence_features.cuda())
            graphs = Variable(sequence_graphs.cuda())
            y_true = Variable(labels.cuda())
        else:
            features = Variable(sequence_features)
            graphs = Variable(sequence_graphs)
            y_true = Variable(labels)
        y_pred = model(features, graphs)
        y_pred = torch.squeeze(y_pred)
        y_true = y_true.float()/120.0
        # calculate loss
        if(len(y_pred.size())==0):
            y_pred = y_pred.unsqueeze(0)
        loss = model.module.criterion(y_pred, y_true)
        # backward gradient
        loss.backward()

        # update all parameters
        model.module.optimizer.step()

        epoch_loss_train += loss.item()
        n_batches += 1

    epoch_loss_train_avg = epoch_loss_train / n_batches
    return epoch_loss_train_avg


def evaluate(model, data_loader):
    model.module.eval()

    epoch_loss = 0.0
    n_batches = 0
    valid_pred = []
    valid_true = []
    valid_name = []

    for data in tqdm(data_loader):
        with torch.no_grad():
            sequence_names, _, labels, sequence_features, sequence_graphs = data

            sequence_features = torch.squeeze(sequence_features)
            sequence_graphs = torch.squeeze(sequence_graphs)

            if torch.cuda.is_available():
                features = Variable(sequence_features.cuda())
                graphs = Variable(sequence_graphs.cuda())
                y_true = Variable(labels.cuda())
            else:
                features = Variable(sequence_features)
                graphs = Variable(sequence_graphs)
                y_true = Variable(labels)
            y_pred = model(features, graphs)
            y_pred = torch.squeeze(y_pred)
            y_true = y_true.float()/120.0
            if(len(y_pred.size())==0):
                y_pred = y_pred.unsqueeze(0)
            loss = model.module.criterion(y_pred, y_true)
            y_pred = y_pred.cpu().detach().numpy().tolist()
            y_true = y_true.cpu().detach().numpy().tolist()
            flag = isinstance(y_pred,float)
            if(flag):
                a = []
                a.append(y_pred)
                y_pred = a
            valid_pred.extend(y_pred)
            valid_true.extend(y_true)
            valid_name.extend(sequence_names)

            epoch_loss += loss.item()
            n_batches += 1
    epoch_loss_avg = epoch_loss / n_batches

    return epoch_loss_avg, valid_true, valid_pred, valid_name


def train(model, train_dataframe, valid_dataframe, fold=0):
    train_loader = DataLoader(dataset=ProDataset(train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    train_losses = []
    train_pearson = []
    train_r2 = []
    
    valid_losses = []
    valid_pearson = []
    valid_r2 = []

    best_val_loss = 1000
    best_train_loss = 1000
    best_epoch = 0
    best_epoch_train = 0

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.module.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader, epoch + 1)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, train_name = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        print("Train loss: ", np.sqrt(epoch_loss_train_avg))
        print("Train pearson:", result_train['pearson'])
        print("Train r2:", result_train['r2'])

        train_losses.append(np.sqrt(epoch_loss_train_avg))
        train_pearson.append(result_train['pearson'])
        train_r2.append(result_train['r2'])
        
        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, valid_name = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred)
        print("Valid loss: ", np.sqrt(epoch_loss_valid_avg))
        print("Valid pearson:", result_valid['pearson'])
        print("Valid r2:", result_valid['r2'])
        
        valid_losses.append(np.sqrt(epoch_loss_valid_avg))
        valid_pearson.append(result_valid['pearson'])
        valid_r2.append(result_valid['r2'])

        if best_val_loss > epoch_loss_valid_avg:
            best_val_loss = epoch_loss_valid_avg
            best_epoch = epoch + 1
            torch.save(model.module.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))
            valid_true_T=list(np.array(valid_true) * 120)
            valid_pred_T=list(np.array(valid_pred) * 120)

            valid_detail_dataframe = pd.DataFrame({'uniprot_id': valid_name, 'y_true': valid_true, 'y_pred': valid_pred, 'Tm':valid_true_T, 'prediction':valid_pred_T})
            valid_detail_dataframe.sort_values(by=['y_true'], inplace=True)
            valid_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_valid_detail.csv", header=True, sep=',')
            train_true_T=list(np.array(train_true) * 120)
            train_pred_T=list(np.array(train_pred) * 120)

            train_detail_dataframe = pd.DataFrame({'uniprot_id': train_name, 'y_true': train_true, 'y_pred': train_pred, 'Tm':train_true_T, 'prediction':train_pred_T})
            train_detail_dataframe.sort_values(by=['y_true'], inplace=True)
            train_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_train_detail.csv", header=True, sep=',')

    # save calculation information
    result_all = {
        'Train_loss': train_losses,
        'Train_pearson': train_pearson,
        'Train_r2': train_r2,
        
        'Valid_loss': valid_losses,
        'Valid_pearson': valid_pearson,
        'Valid_r2': valid_r2,
        
        'Best_epoch': [best_epoch for _ in range(len(train_losses))]
    }
    result = pd.DataFrame(result_all)
    print("Fold", str(fold), "Best epoch at", str(best_epoch))
    result.to_csv(Result_Path + "Fold" + str(fold) + "_result.csv", sep=',')

def analysis(y_true, y_pred):
    pearson = pearsonr(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    
    result = {
        'pearson': pearson,
        'r2': r2,
    }
    return result
def cross_validation(all_dataframe,fold_number=10):
    print("split_seed: ", SEED)
    sequence_names = all_dataframe['uniprot_id'].values
    sequence_labels = all_dataframe['tm'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Training on", str(train_dataframe.shape[0]), "examples, Validation on", str(valid_dataframe.shape[0]),
              "examples")
        model = Model()
        # if torch.cuda.is_available():
        #     model.cuda()
        if torch.cuda.device_count()>1:
            print(f"use {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model,device_ids=device_ids)
        model.to(device)

        train(model, train_dataframe, valid_dataframe, fold + 1)
        fold += 1


if __name__ == "__main__":
    train_dataframe = pd.read_csv(Dataset_Path + 'Tm50Train.csv',sep=',')
    cross_validation(train_dataframe,fold_number=5)
