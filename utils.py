import numpy as np
import scipy.sparse as sp
import torch

def load_file_as_Adj_matrix(Alledge,features):
    import scipy.sparse as sp
    relation_matrix = np.zeros((len(features),len(features)))
    for i, j in np.array(Alledge):
        lnc, mi = int(i), int(j)
        relation_matrix[lnc, mi] = 1
    Adj = sp.csr_matrix(relation_matrix, dtype=np.float32)
    return Adj
    
def load_data(edgelist,node_features,node_labels):
    features = sp.csr_matrix(node_features, dtype=np.float32)
    idx_train = range(500)
    idx_val = range(500, 660)
    idx_test = range(660, int(node_features.shape[0]))  
    features = torch.FloatTensor(np.array(features.todense()))  
    labels = torch.LongTensor(np.array(node_labels))
    adj = load_file_as_Adj_matrix(edgelist,node_features)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test
