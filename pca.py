import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
import os
import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    PowerTransformer,
)


def replace_inf(obj_df):
    lsa_max = obj_df.loc[obj_df['LSA'] != np.inf, 'LSA'].max()
    obj_df['LSA'].replace(np.inf, lsa_max, inplace=True)

save_path = "./PCA/"

use_metrics = [
    "VBCount",
    "NNCount",
    "JJCount",
    "RBCount",
    "ConjCount",
    "SentCount",
    "Length",
    "Polysemy",
    "DependDist",
    "SentiScore",
    "SentiFlip",
    "ConstHeight",
    "TerminalRatio",
    "NC0.1",
    "NC0.3",
    "NC0.5",
    "NC0.7",
    "NC0.9",
    # "KMNC",
    # "NBC",
    # "SNAC",
    "X_al",
    "DeepGini",
    "LSA",
    "DSA"
]

net_list = [
    "CNNLSTM",
    "BiLSTM",
    "LSTM",
    "CNN"
]
data_list = ["SST", "IMDB"]
data_name = "SST"
scaler = QuantileTransformer(output_distribution='normal', random_state=0)

res_df = pd.DataFrame()
cov_df = pd.DataFrame()

# 读input指标
for n in net_list:
    for d in data_list:
        path = "./Metrics/" + n + "_" + data_name + "/"
        dirs = os.listdir(path)
        for i, m in enumerate(dirs):
            if os.path.splitext(m)[1] == ".csv":
                # print("model " + str(i) + " " + m)
                temp_df = pd.read_csv(path + m, index_col=0)
                res_df = pd.concat([res_df, temp_df])
                # print(res_df)


# res_df = res_df.replace([np.inf, -np.inf], np.nan)
# res_df = res_df.dropna()
# res_df = res_df.reset_index(drop=True)

# 读cov指标
for n in net_list:
    for d in data_list:
        path = "./Coverage/" + n + "_" + data_name + "/"
        dirs = os.listdir(path)
        for i, m in enumerate(dirs):
            if os.path.splitext(m)[1] == ".csv":
                # print("model " + str(i) + " " + m)
                temp_df = pd.read_csv(path + m, index_col=0)
                cov_df = pd.concat([cov_df, temp_df])
                # print(res_df)

#
# cov_df = cov_df.replace([np.inf, -np.inf], np.nan)
# cov_df = cov_df.dropna()
# cov_df = cov_df.reset_index(drop=True)

cov_df = cov_df.drop(columns=["isRight", "PredVal", "PredRes", "TrueRes", "Sentence"])
comb_df = pd.concat([res_df, cov_df], axis=1)
comb_df = comb_df.sample(frac=1).reset_index(drop=True)

X_al = res_df["PredVal"]

# print(X_al.shape)
print(comb_df.shape)
X_al = X_al / (1 - X_al)
X_al = np.log(X_al)
X_al = abs(X_al)
X_al = X_al.values.reshape(-1, 1)
X_al = pd.DataFrame(X_al, columns=["X_al"])
# print(X_al)
print(X_al.shape)

comb_df = pd.concat([comb_df, X_al], axis=1)
comb_df = comb_df.replace([np.inf, -np.inf], np.nan)
comb_df = comb_df.dropna()
comb_df = comb_df.reset_index(drop=True)
print(comb_df.isnull().sum().any())


X = scaler.fit_transform(comb_df.reindex(columns=use_metrics))
print(np.isnan(X).sum().any())
X = X[~np.isnan(X).any(axis=1)]
print(np.isnan(X).sum().any())
print(X.shape)
# print(X)

X_train, X_test = train_test_split(X, test_size=0.3)

pca = PCA(n_components=0.85)
pca.fit(X_train)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
result_df = pd.DataFrame(pca.components_, columns=use_metrics)
result_df = abs(result_df)


# print(result_df)
result_df.to_csv(save_path + "0.85.csv")
