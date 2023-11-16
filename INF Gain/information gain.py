import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import os
import numpy as np

net_list = [
    "CNNLSTM",
    "BiLSTM",
    "LSTM",
    "CNN"
]
data_list = ["SST", "IMDB"]
input_metrics = [
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
]

cov_metrics = ["NC0.1", "NC0.3", "NC0.5", "NC0.7", "NC0.9"]
# cov_metrics = ["NC0.1", "NC0.3", "NC0.5", "NC0.7", "NC0.9", "KMNC","NBC","SNAC"]
out_metrics = ["X_al", "DeepGini", "LSA", "DSA"]

mul_info = []
mul_ave = []
mul_max = []
mul_min = []


# print(out_metrics[1])

file = open("information gain.txt", 'a')

# 读input指标
for n in net_list:
    for d in data_list:
        mul_info = []
        path = "../Coverage/" + n + "_" + d + "/"
        dirs = os.listdir(path)
        for i, m in enumerate(dirs):
            if os.path.splitext(m)[1] == ".csv":
                # print("model " + str(i) + " " + m)
                temp_df = pd.read_csv(path + m, index_col=0)
                X_al = temp_df["PredVal"]
                X_al = X_al / (1 - X_al)
                X_al = np.log(X_al)
                X_al = abs(X_al)
                X_al = X_al.values.reshape(-1, 1)
                X = X_al
                # temp_df = temp_df.replace([np.inf, -np.inf], np.nan)
                # temp_df = temp_df.dropna()
                # temp_df.reset_index(drop=True)
                # X = temp_df["NC0.9"].values.reshape(-1, 1)
                # print(X.shape)
                Y = temp_df["isRight"]
                # print(Y.shape)
                mutual_info = mutual_info_classif(X, Y)
                print(mutual_info)
                mul_info.append(mutual_info)
        print(len(mul_info))
        mutual_max = np.max(mul_info)
        mutual_min = np.min(mul_info)
        mul_max.append(mutual_max)
        mul_min.append(mutual_min)
        ave = np.mean(mul_info)
        ave = (ave - mutual_min) / (mutual_max - mutual_min)
        mul_ave.append(ave)
        print(ave)



mutual_ave = np.mean(mul_ave)
file.write("AL: " + str(mutual_ave) + "\n" + "\n")
# info_max = np.max(mul_info)
# info_min = np.min(mul_info)
# ave = (ave - info_min) / (info_max - info_min)
#
# file.write("TerminalRatio: " + str(ave) + "\n")
# file.write("max: " + str(info_max) + "\n")
# file.write("min: " + str(info_min) + "\n" + "\n")
print(mutual_ave)
