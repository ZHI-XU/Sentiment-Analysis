import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
import os
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model, Model
from imblearn.over_sampling import SMOTE


def val_to_res(val):
    if val > 0.5:
        return 1
    else:
        return 0


def under_sampling(X, y):
    t1_df = pd.concat([X, y], axis=1)
    # 划分
    features_train, feature_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2)

    number_neg = len(labels_train[labels_train == 1])  # 出错样本个数
    # print("训练集中出错样本数为：")
    # print(number_neg)
    # print("总样本数为:")
    # print(len(t1_df))
    # print("t1_df.shape:")
    # print(t1_df.shape)

    pos_indices = labels_train[labels_train == 0].index
    neg_indices = labels_train[labels_train == 1].index

    random_pos_indices = np.random.choice(pos_indices, number_neg * 3, replace=False)
    random_pos_indices = np.array(random_pos_indices)

    # 合并index，下采样
    under_sample_indices = np.concatenate([neg_indices, random_pos_indices])
    final_train_df = t1_df.loc[under_sample_indices, :]
    final_train_df = final_train_df.sample(frac=1).reset_index(drop=True)
    return final_train_df, feature_test, labels_test


def get_last_layer_model(model):
    layer_names = [layer.name for layer in model.layers]
    layer_output = model.get_layer(layer_names[-2]).output
    ret = Model(model.input, layer_output)

    return ret


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
    precision = tp / (tp + fp)  # 精确率
    recall = tp / (tp + fn)  # 召回率
    F1 = (2 * precision * recall) / (precision + recall)  # F1
    return accuracy, precision, recall, F1


def upper_sampling_KMeans(X, y):
    y = np.bitwise_not(y) * 1  # 反转标签

    # 上采样(100%)
    features_train, feature_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2)
    oversampler = SMOTE()
    os_features, os_labels = oversampler.fit_resample(features_train, labels_train)

    features_train = pd.DataFrame(features_train)
    labels_train = pd.DataFrame(labels_train)
    train_df = pd.concat([features_train,labels_train],axis=1)
    print("--------训练集---------")
    print(train_df.shape)

    model = KMeans(n_clusters=2)
    model.fit(os_features)

    y_pred = model.predict(feature_test)
    y_pred_res = np.array(y_pred)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)

    # 10%
    train_df = train_df.sample(frac=0.1).reset_index(drop=True)
    print("---------10%-----------")
    print(train_df.shape)
    features_train_10 = train_df.drop(columns=["isRight"])
    labels_train_10 = train_df["isRight"]

    oversampler = SMOTE()
    os_features_10, os_labels_10 = oversampler.fit_resample(features_train_10, labels_train_10)

    model_10 = KMeans(n_clusters=2)
    model_10.fit(os_features_10)

    y_pred = model_10.predict(feature_test)
    y_pred_res = np.array(y_pred)

    tp_10, fp_10, tn_10, fn_10 = compute_confusion_matrix(y_pred_res, labels_test)
    acc_10, precision_10, recall_10, f1_10 = compute_indexes(tp_10, fp_10, tn_10, fn_10)
    accuracy_10 = accuracy_score(labels_test, y_pred_res)
    auc_10 = roc_auc_score(labels_test, y_pred_res)
    mcc_10 = matthews_corrcoef(labels_test, y_pred_res)

    return accuracy, f1, auc, mcc, accuracy_10, f1_10, auc_10, mcc_10


def under_sampling_KMeans(X,y):
    y = np.bitwise_not(y) * 1  # 反转标签

    # 下采样
    under_sampling_train_df, feature_test, labels_test = under_sampling(X, y)
    feature_test = np.array(feature_test).reshape(-1, X.shape[1])
    X_train = under_sampling_train_df.drop(columns=["isRight"])
    y_train = under_sampling_train_df["isRight"]

    model = KMeans(n_clusters=2)
    model.fit(X_train)

    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)

    # 10%
    under_sampling_train_df_10 = under_sampling_train_df.sample(frac=0.1).reset_index(drop=True)
    X_train_10 = under_sampling_train_df_10.drop(columns=["isRight"])
    y_train_10 = under_sampling_train_df_10["isRight"]

    model_10 = KMeans(n_clusters=2)
    model_10.fit(X_train_10)

    y_pred = model_10.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp_10, fp_10, tn_10, fn_10 = compute_confusion_matrix(y_pred_res, labels_test)
    acc_10, precision_10, recall_10, f1_10 = compute_indexes(tp_10, fp_10, tn_10, fn_10)
    accuracy_10 = accuracy_score(labels_test, y_pred_res)
    auc_10 = roc_auc_score(labels_test, y_pred_res)
    mcc_10 = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc, accuracy_10, f1_10, auc_10, mcc_10


netname = "LSTM"
dataname = "SST"
test_path = "。./Coverage/" + netname + "_" + dataname + "/"
pre_path = "。./Metrics/" + netname + "_" + dataname + "/"
dirs = os.listdir(test_path)

i = 9

# 合并coverage指标与输入输出端指标
pre_df_cov = pd.read_csv(test_path + dirs[i])
pre_df_input = pd.read_csv(pre_path + dirs[i])
pre_df_input = pre_df_input.drop(columns=["isRight", "PredVal", "PredRes", "Unnamed: 0", "TrueRes", "Sentence"])
comb_df = pd.concat([pre_df_cov, pre_df_input], axis=1)

# 初始化结果
acc_cov = []
acc_input = []
acc_output = []
acc_all = []
acc_cov_10 = []
acc_input_10 = []
acc_output_10 = []
acc_all_10 = []

f1_cov = []
f1_input = []
f1_output = []
f1_all = []
f1_cov_10 = []
f1_input_10 = []
f1_output_10 = []
f1_all_10 = []

auc_cov = []
auc_input = []
auc_output = []
auc_all = []
auc_cov_10 = []
auc_input_10 = []
auc_output_10 = []
auc_all_10 = []

mcc_cov = []
mcc_input = []
mcc_output = []
mcc_all = []
mcc_cov_10 = []
mcc_input_10 = []
mcc_output_10 = []
mcc_all_10 = []

# 10次减少随机性
for epoch in range(10):
    # 打乱, 预处理
    df = comb_df.sample(frac=1).reset_index(drop=True)
    # cov_df = cov_df.dropna(axis = 0, subset = [''] )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.reset_index(drop=True)
    cov_df = df

    # 读取coverage指标
    X_cov = cov_df[["NC0.1", "NC0.3", "NC0.5", "NC0.7", "NC0.9"]]
    # X_cov = cov_df[["NC0.1", "NC0.3", "NC0.5", "NC0.7", "NC0.9", "KMNC", "NBC", "SNAC"]]
    y_cov = cov_df["isRight"]

    acc_c, f1_c, auc_c, mcc_c, acc_c_10, f1_c_10, auc_c_10, mcc_c_10 = upper_sampling_KMeans(X_cov, y_cov)
    acc_cov.append(acc_c)
    f1_cov.append(f1_c)
    auc_cov.append(auc_c)
    mcc_cov.append(mcc_c)
    acc_cov_10.append(acc_c_10)
    f1_cov_10.append(f1_c_10)
    auc_cov_10.append(auc_c_10)
    mcc_cov_10.append(mcc_c_10)


    # input指标
    X_input = cov_df[["VBCount", "NNCount", "JJCount", "RBCount", "ConjCount", "SentCount", "Length",
                      "Polysemy", "DependDist", "SentiScore", "SentiFlip", "ConstHeight", "TerminalRatio"]]
    y_input = cov_df["isRight"]

    accuracy_i, f1_i, auc_i, mcc_i, acc_i_10, f1_i_10, auc_i_10, mcc_i_10 = upper_sampling_KMeans(X_input, y_input)
    acc_input.append(accuracy_i)
    f1_input.append(f1_i)
    auc_input.append(auc_i)
    mcc_input.append(mcc_i)
    acc_input_10.append(acc_i_10)
    f1_input_10.append(f1_c_10)
    auc_input_10.append(auc_i_10)
    mcc_input_10.append(mcc_i_10)

    # DeepGini + LSA + DSA
    X_baseline = cov_df[["DeepGini", "LSA", "DSA"]]
    y_baseline = cov_df["isRight"]

    # al指标
    X_al = cov_df["PredVal"]
    X_al = X_al / (1 - X_al)
    X_al = np.log(X_al)
    X_al = abs(X_al)
    y_al = cov_df["isRight"]
    X_al = X_al.values.reshape(-1, 1)
    X_al = pd.DataFrame(X_al, columns=["X_al"])

    # output
    X_al_output = pd.concat([X_al, X_baseline], axis=1)
    y_al_output = cov_df["isRight"]

    accuracy_a_o, f1_a_o, auc_a_o, mcc_a_o, acc_o_10, f1_o_10, auc_o_10, mcc_o_10 = upper_sampling_KMeans(X_al_output, y_al_output)
    acc_output.append(accuracy_a_o)
    f1_output.append(f1_a_o)
    auc_output.append(auc_a_o)
    mcc_output.append(mcc_a_o)
    acc_output_10.append(acc_o_10)
    f1_output_10.append(f1_o_10)
    auc_output_10.append(auc_o_10)
    mcc_output_10.append(mcc_o_10)

    # all
    X_cov_al_input = pd.concat([X_cov, X_al_output, X_input], axis=1)
    y_cov_al_input = cov_df["isRight"]

    accuracy_cov_al_input, f1_c_a_i, auc_c_a_i, mcc_c_a_i, acc_a_10, f1_a_10, auc_a_10, mcc_a_10 = upper_sampling_KMeans(X_cov_al_input, y_cov_al_input)
    acc_all.append(accuracy_cov_al_input)
    f1_all.append(f1_c_a_i)
    auc_all.append(auc_c_a_i)
    mcc_all.append(mcc_c_a_i)
    acc_all_10.append(acc_a_10)
    f1_all_10.append(f1_a_10)
    auc_all_10.append(auc_a_10)
    mcc_all_10.append(mcc_a_10)

    # 保存
    out_df = pd.DataFrame(
        {
            "acc_input": acc_input,
            "acc_output": acc_output,
            "acc_all": acc_all,
            "acc_cov": acc_cov,
            "acc_input_10": acc_input_10,
            "acc_output_10": acc_output_10,
            "acc_all_10": acc_all_10,
            "acc_cov_10": acc_cov_10,

            "f1_input": f1_input,
            "f1_output": f1_output,
            "f1_all": f1_all,
            "f1_cov": f1_cov,
            "f1_input_10": f1_input_10,
            "f1_output_10": f1_output_10,
            "f1_all_10": f1_all_10,
            "f1_cov_10": f1_cov_10,

            "auc_input": auc_input,
            "auc_output": auc_output,
            "auc_all": auc_all,
            "auc_cov": auc_cov,
            "auc_input_10": auc_input_10,
            "auc_output_10": auc_output_10,
            "auc_all_10": auc_all_10,
            "auc_cov_10": auc_cov_10,

            "mcc_input": mcc_input,
            "mcc_output": mcc_output,
            "mcc_all": mcc_all,
            "mcc_cov": mcc_cov,
            "mcc_input_10": mcc_input_10,
            "mcc_output_10": mcc_output_10,
            "mcc_all_10": mcc_all_10,
            "mcc_cov_10": mcc_cov_10,
        }
    )
    out_df.to_csv("./Coverage/10%Compare/KMeans/上/" + netname + "_" + dataname + "_" + dirs[i])
