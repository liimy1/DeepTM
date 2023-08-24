import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr

# path
Dataset_Path = './Data/'
Model_Path = './Model_OGT/'
Result_Path = './Result_OGT/'
Fasta_Path = './Data/fasta/'
Pssm_Path = './Data/pssm/'
TestData_Path = './test.csv'
amino_acid = list("ACDEFGHIKLMNPQRSTVWY")
amino_dict = {aa: i for i, aa in enumerate(amino_acid)}

# Seed
SEED = 2333
np.random.seed(SEED)
torch.manual_seed(SEED)
print(torch.cuda.is_available())
device_ids = [0]
if torch.cuda.is_available():
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
GCN_FEATURE_DIM = 118
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 64

# Attention parameters
DENSE_DIM = 16
ATTENTION_HEADS = 4

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
    feature_matrix = np.load(Dataset_Path + 'node_features_OGT/' + uniprot_id + '.npy')
    feature_matrix = (feature_matrix - mean) / std
    part1 = feature_matrix[:,0:20]
    part2 = feature_matrix[:,23:]
    feature_matrix = np.concatenate([part1,part2],axis=1)
    return feature_matrix


def load_graph(sequence_name):
    matrix = np.load(Dataset_Path + 'edge_features_OGT/' + sequence_name + '.npy').astype(np.float32)
    matrix = normalize(matrix)
    return matrix

def load_values():
    mean = np.load(Dataset_Path + 'oneD_mean_OGT.npy')
    std = np.load(Dataset_Path + 'oneD_std_OGT.npy')
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
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['ogt'].values
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
        support = input @ self.weight
        output = adj @ support
        if self.bias is not None:
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

    def forward(self, x, adj):              # x.shape = (seq_len, GCN_FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = self.gc1(x, adj)                # x.shape = (seq_len, GCN_HIDDEN_DIM)
        x = self.relu1(self.ln1(x))
        x = self.gc2(x, adj)
        output = self.relu2(self.ln2(x))    # output.shape = (seq_len, GCN_OUTPUT_DIM)
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

    def forward(self, input):               # input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))     # x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)                     # x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)       # attention.shape = (1, attention_hops, seq_len)
        return attention


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.gcn = GCN()
        self.attention = Attention(GCN_OUTPUT_DIM, DENSE_DIM, ATTENTION_HEADS)
        self.fc_final = nn.Linear(GCN_OUTPUT_DIM, NUM_CLASSES)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def forward(self, x, adj):                                              # x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = x.float()
        x = self.gcn(x, adj)                                                # x.shape = (seq_len, GAT_OUTPUT_DIM)
        l = len(x.shape)
        if(l<3):
            x = x.unsqueeze(0).float()  										# x.shape = (1, seq_len, GAT_OUTPUT_DIM)
        att = self.attention(x)                                             # att.shape = (1, ATTENTION_HEADS, seq_len)
        node_feature_embedding = att @ x                                    # output.shape = (1, ATTENTION_HEADS, GAT_OUTPUT_DIM)
        node_feature_embedding_avg = torch.sum(node_feature_embedding,
                                               1) / self.attention.n_heads  # node_feature_embedding_avg.shape = (1, GAT_OUTPUT_DIM)
        output = torch.sigmoid(self.fc_final(node_feature_embedding_avg))   # output.shape = (1, NUM_CLASSES)
        return output.squeeze(0)


def evaluate(model, data_loader):
    model.eval()

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
            y_true = y_true.float()/110.0
            if(len(y_pred.size())==0):
                y_pred = y_pred.unsqueeze(0)
            loss = model.criterion(y_pred, y_true)
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


def test(test_dataframe):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_result = {}

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = Model()
        if torch.cuda.is_available():
            model.cuda()
        state_dict = model.state_dict()
        model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0'))

        epoch_loss_test_avg, test_true, test_pred, test_name = evaluate(model, test_loader)
        result_test = analysis(test_true, test_pred)
        print("\n========== Evaluate Test set ==========")
        print("Test loss: ", np.sqrt(epoch_loss_test_avg))
        print("Test pearson:", result_test['pearson'])
        print("Test r2:", result_test['r2'])

        test_result[model_name] = [
            np.sqrt(epoch_loss_test_avg),
            result_test['pearson'],
            result_test['r2'],
        ]
        test_true_T=list(np.array(test_true) * 110)
        test_pred_T=list(np.array(test_pred) * 110)
        test_detail_dataframe = pd.DataFrame({'uniprot_id': test_name, 'y_true': test_true, 'y_pred': test_pred,'ogt':test_true_T,'prediction':test_pred_T})
        test_detail_dataframe.sort_values(by=['uniprot_id'], inplace=True)
        test_detail_dataframe.to_csv(Result_Path + model_name + "_test_detail.csv", header=True, sep=',')

    test_result_dataframe = pd.DataFrame.from_dict(test_result, orient='index',
                                                   columns=['loss', 'pearson', 'r2'])
    test_result_dataframe.to_csv(Result_Path + "test_result.csv", index=True, header=True, sep=',')

def analysis(y_true, y_pred):
    # continous evaluate
    #pearson = pearsonr(y_true, y_pred)
    #r2 = metrics.r2_score(y_true, y_pred)
    pearson = 0
    r2 = 0

    result = {
        'pearson': pearson,
        'r2': r2,
    }
    return result


if __name__ == "__main__":
    test_dataframe = pd.read_csv(TestData_Path, sep=',')
    test(test_dataframe) 
   
