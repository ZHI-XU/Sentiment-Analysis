import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE


def val_to_res(val):
    if val > 0.5:
        return 1
    else:
        return 0


# 计算混淆矩阵
def compute_confusion_matrix(pred, expected):
    part = pred ^ expected  # 对结果进行分类，亦或使得判断正确的为0,判断错误的为1
    pcount = np.bincount(part)  # 分类结果统计，pcount[0]为0的个数，pcount[1]为1的个数
    tp_list = list(pred & expected)  # 将TP的计算结果转换为list
    fp_list = list(pred & ~expected)  # 将FP的计算结果转换为list
    tp = tp_list.count(1)  # 统计TP的个数
    fp = fp_list.count(1)  # 统计FP的个数
    tn = pcount[0] - tp  # 统计TN的个数
    fn = pcount[1] - fp  # 统计FN的个数
    return tp, fp, tn, fn


# 计算常用指标
def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)  # 准确率
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)  # 精确率
    recall = tp / (tp + fn)  # 召回率
    F1 = (2 * precision * recall) / (precision + recall)  # F1
    return accuracy, precision, recall, F1


netname = "BiLSTM"
dataname = "IMDB"
pre_path = "./Metrics/" + netname + "_" + dataname + "/"
dirs = os.listdir(pre_path)

# 初始化结果
acc_deepgini = []
acc_lsa = []
acc_dsa = []
acc_al =[]

f1_deepgini = []
f1_lsa = []
f1_dsa = []
f1_al = []

auc_deepgini = []
auc_lsa = []
auc_dsa = []
auc_al = []

mcc_deepgini = []
mcc_lsa = []
mcc_dsa = []
mcc_al = []

for i in range(10):
    df = pd.read_csv(pre_path + dirs[i])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(dirs[i])

    # 归一化
    deepgini = df["DeepGini"]
    lsa = df["LSA"]
    dsa = df["DSA"]
    y = df["isRight"]

    # al
    al = df["PredVal"]
    al = al / (1 - al)
    al = np.log(al)
    al = abs(al)
    al = al.values.reshape(-1, 1)
    al = [val_to_res(n) for n in al]
    al = np.array(al)

    deepgini_max = deepgini.max()
    deepgini_min = deepgini.min()
    lsa_max = lsa.max()
    lsa_min = lsa.min()
    dsa_max = dsa.max()
    dsa_min = dsa.min()

    deepgini = (deepgini - deepgini_min) / (deepgini_max - deepgini_min)
    lsa = (lsa - lsa_min) / (lsa_max - lsa_min)
    dsa = (dsa - dsa_min) / (dsa_max - dsa_min)

    deepgini_pred_res = [val_to_res(n) for n in deepgini]
    deepgini_pred_res = np.array(deepgini_pred_res)
    lsa_pred_res = [val_to_res(n) for n in lsa]
    lsa_pred_res = np.array(lsa_pred_res)
    dsa_pred_res = [val_to_res(n) for n in dsa]
    dsa_pred_res = np.array(dsa_pred_res)


    # 计算DeepGini相关指标
    tp_d, fp_d, tn_d, fn_d = compute_confusion_matrix(deepgini_pred_res, y)
    acc_d, precision_d, recall_d, f1_d = compute_indexes(tp_d, fp_d, tn_d, fn_d)
    accuracy_d = accuracy_score(y, deepgini_pred_res)
    auc_d = roc_auc_score(y, deepgini_pred_res)
    mcc_d = matthews_corrcoef(y, deepgini_pred_res)
    print("DeepGini has been calculated")

    acc_deepgini.append(accuracy_d)
    f1_deepgini.append(f1_d)
    auc_deepgini.append(auc_d)
    mcc_deepgini.append(mcc_d)

    # 计算LSA相关指标
    tp_l, fp_l, tn_l, fn_l = compute_confusion_matrix(lsa_pred_res, y)
    acc_l, precision_l, recall_l, f1_l = compute_indexes(tp_l, fp_l, tn_l, fn_l)
    accuracy_l = accuracy_score(y, lsa_pred_res)
    auc_l = roc_auc_score(y, lsa_pred_res)
    mcc_l = matthews_corrcoef(y, lsa_pred_res)
    print("LSA has been calculated")

    acc_lsa.append(accuracy_l)
    f1_lsa.append(f1_l)
    auc_lsa.append(auc_l)
    mcc_lsa.append(mcc_l)

    # 计算DSA相关指标
    tp_dsa, fp_dsa, tn_dsa, fn_dsa = compute_confusion_matrix(dsa_pred_res, y)
    acc_ds, precision_ds, recall_ds, f1_ds = compute_indexes(tp_dsa, fp_dsa, tn_dsa, fn_dsa)
    accuracy_ds = accuracy_score(y, dsa_pred_res)
    auc_ds = roc_auc_score(y, dsa_pred_res)
    mcc_ds = matthews_corrcoef(y, dsa_pred_res)
    print("DSA has been calculated")

    acc_dsa.append(accuracy_ds)
    f1_dsa.append(f1_ds)
    auc_dsa.append(auc_ds)
    mcc_dsa.append(mcc_ds)

    # al
    tp_al, fp_al, tn_al, fn_al = compute_confusion_matrix(al, y)
    acc_a, precision_a, recall_a, f1_a = compute_indexes(tp_al, fp_al, tn_al, fn_al)
    accuracy_a = accuracy_score(y, al)
    auc_a = roc_auc_score(y, al)
    mcc_a = matthews_corrcoef(y, al)
    print("AL has been calculated")

    acc_al.append(accuracy_a)
    f1_al.append(f1_a)
    auc_al.append(auc_a)
    mcc_al.append(mcc_a)



    out_df = pd.DataFrame(
        {
            "acc_deepgini": acc_deepgini,
            "acc_lsa": acc_lsa,
            "acc_dsa": acc_dsa,
            "acc_al": acc_al,

            "f1_deepgini": f1_deepgini,
            "f1_lsa": f1_lsa,
            "f1_dsa": f1_dsa,
            "f1_al": f1_al,

            "auc_deepgini": auc_deepgini,
            "auc_lsa": auc_lsa,
            "auc_dsa": auc_dsa,
            "auc_al": auc_al,

            "mcc_deepgini": mcc_deepgini,
            "mcc_lsa": mcc_lsa,
            "mcc_dsa": mcc_dsa,
            "mcc_al": mcc_al
        }
    )

    out_df.to_csv("./Coverage/DeepGini/" + netname + "_" + dataname + ".csv")
