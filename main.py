import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, precision_recall_curve
from lightgbm import LGBMClassifier
# from dataset import dataset
import warnings
from utils import *
from model import UNIFY

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


warnings.filterwarnings('ignore')


AllNode = pd.read_csv('data/Allnode_DrPr.csv', names=[0, 1], header=None)
Alledge = pd.read_csv('data/DrPrNum_DrPr.csv', header=None)
features = pd.read_csv('data/features.csv', header=None)
AllNegative = pd.read_csv('data/AllNegative.csv', header=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'use {device} ! ')

features = features.iloc[:, 1:]

labels = pd.DataFrame(np.random.rand(len(AllNode), 1))
labels.iloc[0:549, 0] = 0
labels.iloc[549:, 0] = 1
labels = labels[0]

adj, features, labels, idx_train, idx_val, idx_test = load_data(Alledge, features, labels)
edge_index = torch.tensor(Alledge.values.T, dtype=torch.long).to(device)
features = features.to(device)

print("start training...")

nhid = 512
k_size = 7
nclass = 64
heads = 32
layers = 3
model = UNIFY(nfeat=features.shape[1], nhid=64, k_size=k_size, nclass=nclass, dropout=0.5, heads=heads,
                   layers=layers).to(device)
model.eval()
with torch.no_grad():
    Embedding = model(features, edge_index)

Embedding_GCN = pd.DataFrame(Embedding.cpu().numpy())

Positive = Alledge.copy()
Negative = AllNegative.sample(n=1923, random_state=8)
Positive[2] = 1
Negative[2] = 0
result = pd.concat([Positive, Negative]).reset_index(drop=True)

X = pd.concat([
    Embedding_GCN.loc[result[0].values.tolist()].reset_index(drop=True),
    Embedding_GCN.loc[result[1].values.tolist()].reset_index(drop=True)
], axis=1)
Y = result[2]

k_fold = 10

print(f"{k_fold} fold CV")
i = 0
tprs = []
aucs = []
auprs = []
f1_scores = []
mean_fpr = np.linspace(0, 1, 1000)
AllResult = []
skf = StratifiedKFold(n_splits=k_fold, random_state=8, shuffle=True)

for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    model = LGBMClassifier(
        n_estimators=800,
        max_depth=10,
        subsample=0.85,
        learning_rate=0.1,
        random_state=8,
        verbosity=-1,
        class_weight='balanced'
    )

    model.fit(np.array(X_train), np.array(Y_train))

    y_score0 = model.predict(np.array(X_test))
    y_score_RandomF = model.predict_proba(np.array(X_test))

    fpr, tpr, thresholds = roc_curve(Y_test, y_score_RandomF[:, 1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    precision, recall, _ = precision_recall_curve(Y_test, y_score_RandomF[:, 1])
    aupr = average_precision_score(Y_test, y_score_RandomF[:, 1])
    f1 = f1_score(Y_test, y_score0)
    auprs.append(aupr)
    f1_scores.append(f1)
    print('%d fold (AUC=%0.4f, AUPR=%0.4f, F1=%0.4f)' % (i, roc_auc, aupr, f1))
    i += 1

print("Mean ROC-AUC: {:.4f} ± {:.4f}".format(np.mean(aucs), np.std(aucs)))
print("Mean AUPR: {:.4f} ± {:.4f}".format(np.mean(auprs), np.std(auprs)))
print("Mean F1 Score: {:.4f} ± {:.4f}".format(np.mean(f1_scores), np.std(f1_scores)))
