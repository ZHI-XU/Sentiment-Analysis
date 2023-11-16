import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os


def val_to_res(val):
    if val > 0.5:
        return 1
    else:
        return 0


pre_net = "BiLSTM"
new_net = "CNN"
data = "SST"

pre_path = "./Metrics/" + pre_net + "_" + data + "/"
cov_path = "./Coverage/" + pre_net + "_" + data + "/"
new_path = "./Model/" + new_net + "_" + data + "/"
new_path_data = "./Metrics/" + new_net + "_" + data + "/"
new_path_cov = "./Coverage/" + new_net + "_" + data + "/"
outpath = "./CrossProject/"

# M2
model_dirs = os.listdir(new_path)
model = load_model(new_path + model_dirs[0])
new_dirs = os.listdir(new_path_data)
new_cov_dirs = os.listdir(new_path_cov)
new_pd = pd.read_csv(new_path_data + new_dirs[0])
new_cov = pd.read_csv(new_path_cov + new_cov_dirs[0])
new_pd = new_pd.drop(columns=["isRight", "PredVal", "PredRes", "Unnamed: 0", "TrueRes", "Sentence"])
new_pd = pd.concat([new_pd, new_cov], axis=1)
x_test = new_pd[["VBCount", "NNCount", "JJCount", "RBCount", "ConjCount", "SentCount", "Length",
              "Polysemy", "DependDist", "SentiScore", "SentiFlip", "ConstHeight", "TerminalRatio",
              "DeepGini", "LSA", "DSA", "NC0.1", "NC0.3", "NC0.5", "NC0.7", "NC0.9"]]

# PM1
data_dirs = os.listdir(pre_path)
pre_pd = pd.read_csv(pre_path + data_dirs[0])
pre_pd = pre_pd.drop(columns=["isRight", "PredVal", "PredRes", "Unnamed: 0", "TrueRes", "Sentence"])
dirs = os.listdir(cov_path)
cov_pd = pd.read_csv(cov_path + dirs[0])
m1_pd = pd.concat([pre_pd, cov_pd], axis=1)
print(pd)

x_train = m1_pd[["VBCount", "NNCount", "JJCount", "RBCount", "ConjCount", "SentCount", "Length",
              "Polysemy", "DependDist", "SentiScore", "SentiFlip", "ConstHeight", "TerminalRatio",
              "DeepGini", "LSA", "DSA", "NC0.1", "NC0.3", "NC0.5", "NC0.7", "NC0.9"]]
y_train = m1_pd["PredRes"]
print(x_train.shape)
x_train = x_train.values.reshape(-1,49)
print(x_train)
model.summary()



# model.fit(x_train, y_train,batch_size=50,)
# cross_pred = model.predict(x_test)
# cross_pred = pd.DataFrame(cross_pred, columns=["CrossPred"])
# out_pd = pd.concat([x_test,cross_pred],axis=1)
# out_pd.to_csv(outpath + pre_net + "_" + new_net + "_" + data + ".csv")

# predval = pre_pd["PredVal"]
# predval_res = [val_to_res(n) for n in predval]
# predval = pd.DataFrame(predval_res, columns=["PredVal"])
#
# sentence = pre_pd["Sentence"]
# data = pd.concat([sentence,predval],axis=1)


# right_number = pre_pd["isRight"].sum()
# print(right_number)
# print(right_number/len(predval))


# model.summary()
# pre_val = model.predict(predval)
# print(pre_val)
# pre_val = [val_to_res(n) for n in pre_val]
# print(pre_val)
