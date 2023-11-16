import pandas as pd
import os
import numpy as np

LR_path = "./Coverage/10%Compare/LR/上/"
GaussianNB_path = "./Coverage/10%Compare/GaussianNB/上/"
KMeans_path = "./Coverage/10%Compare/KMeans/上/"
MiniBatchKMeans_path = "./Coverage/10%Compare/MiniBatchKmeans/上/"
Agglomerative_path = "./Coverage/Agglomerative/"
DecisionTree_path = "./Coverage/10%Compare/DecisionTree/上/"
BernoulliNB_path = "./Coverage/BernoulliNB/"

dirs = os.listdir(KMeans_path)

col_mean = []

for i in dirs:
    if os.path.splitext(i)[1] == ".csv":
        print(i)
        df = pd.read_csv(KMeans_path + i)
        col_m = df.mean(axis=0)
        col_mean.append(col_m)

# col_mean = np.array(col_mean)

print(col_mean)
out_df = pd.DataFrame(col_mean)

out_df.to_csv(KMeans_path + "all.csv")


