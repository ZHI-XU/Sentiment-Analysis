import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
import os
import numpy as np
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn import tree


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
    precision = tp / (tp + fp)  # 精确率
    recall = tp / (tp + fn)  # 召回率
    F1 = (2 * precision * recall) / (precision + recall)  # F1
    return accuracy, precision, recall, F1


def under_sampling(X, y):
    t1_df = pd.concat([X, y], axis=1)
    # 划分
    features_train, feature_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2)

    number_neg = len(labels_train[labels_train == 0])  # 出错样本个数
    # print("训练集中出错样本数为：")
    # print(number_neg)
    # print("总样本数为:")
    # print(len(t1_df))
    # print("t1_df.shape:")
    # print(t1_df.shape)

    pos_indices = labels_train[labels_train == 1].index
    neg_indices = labels_train[labels_train == 0].index

    random_pos_indices = np.random.choice(pos_indices, number_neg * 3, replace=False)
    random_pos_indices = np.array(random_pos_indices)

    # 合并index，下采样
    under_sample_indices = np.concatenate([neg_indices, random_pos_indices])
    final_train_df = t1_df.loc[under_sample_indices, :]
    final_train_df = final_train_df.sample(frac=1).reset_index(drop=True)
    return final_train_df, feature_test, labels_test


def upper_sampling_lr(X, y):
    y = np.bitwise_not(y) * 1  # 反转标签

    # 上采样
    features_train, feature_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2)
    oversampler = SMOTE()
    os_features, os_labels = oversampler.fit_resample(features_train, labels_train)

    # 保存
    features_train = pd.DataFrame(features_train)
    labels_train = pd.DataFrame(labels_train)
    train_df = pd.concat([features_train,labels_train],axis=1)
    print("--------训练集---------")
    print(train_df.shape)
    # print(train_df)

    model = LogisticRegression()
    model.fit(os_features, os_labels)

    # 预测
    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc, train_df, feature_test, labels_test


def upper_sampling_lr_10(train_df, feature_test, labels_test):
    # y = np.bitwise_not(y) * 1  # 反转标签

    print("--------原训练集--------------")
    print(train_df.shape)
    train_df = train_df.sample(frac=0.1).reset_index(drop=True)
    print("----------10%---------------")
    print(train_df.shape)

    # print(train_df)

    features_train = train_df.drop(columns=["isRight"])
    labels_train = train_df["isRight"]


    # 上采样
    oversampler = SMOTE()
    os_features, os_labels = oversampler.fit_resample(features_train, labels_train)

    model = LogisticRegression()
    model.fit(os_features, os_labels)

    # 预测
    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc


def under_sampling_lr(X, y):
    # y = np.bitwise_not(y) * 1  # 反转标签

    # 下采样
    under_sampling_train_df, feature_test, labels_test = under_sampling(X, y)
    feature_test = np.array(feature_test).reshape(-1, X.shape[1])
    X_train = under_sampling_train_df.drop(columns=["isRight"])
    y_train = under_sampling_train_df["isRight"]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc, under_sampling_train_df, feature_test, labels_test


def under_sampling_lr_10(X, feature_test, labels_test):
    # y = np.bitwise_not(y) * 1  # 反转标签

    print("--------原训练集--------------")
    print(X.shape)
    X = X.sample(frac=0.1).reset_index(drop=True)
    print("----------10%---------------")
    print(X.shape)

    X_train = X.drop(columns=["isRight"])
    y_train = X["isRight"]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc


def upper_sampling_GaussianNB(X, y):
    y = np.bitwise_not(y) * 1  # 反转标签

    # 上采样
    features_train, feature_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2)
    oversampler = SMOTE()
    os_features, os_labels = oversampler.fit_resample(features_train, labels_train)

    # 保存
    features_train = pd.DataFrame(features_train)
    labels_train = pd.DataFrame(labels_train)
    train_df = pd.concat([features_train, labels_train], axis=1)
    print("--------训练集---------")
    print(train_df.shape)
    # print(train_df)

    model = LogisticRegression()
    model.fit(os_features, os_labels)

    # 预测
    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc, train_df, feature_test, labels_test


def upper_sampling_GaussianNB_10(train_df, feature_test, labels_test):
    # y = np.bitwise_not(y) * 1  # 反转标签

    print("--------原训练集--------------")
    print(train_df.shape)
    train_df = train_df.sample(frac=0.1).reset_index(drop=True)
    print("----------10%---------------")
    print(train_df.shape)

    # print(train_df)

    features_train = train_df.drop(columns=["isRight"])
    labels_train = train_df["isRight"]


    # 上采样
    oversampler = SMOTE()
    os_features, os_labels = oversampler.fit_resample(features_train, labels_train)

    model = GaussianNB()
    model.fit(os_features, os_labels)

    # 预测
    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc


def under_sampling_GaussianNB(X, y):
    # y = np.bitwise_not(y) * 1  # 反转标签

    # 下采样
    under_sampling_train_df, feature_test, labels_test = under_sampling(X, y)
    feature_test = np.array(feature_test).reshape(-1, X.shape[1])
    X_train = under_sampling_train_df.drop(columns=["isRight"])
    y_train = under_sampling_train_df["isRight"]

    model = GaussianNB()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc, under_sampling_train_df, feature_test, labels_test


def under_sampling_GaussianNB_10(X, feature_test, labels_test):
    # y = np.bitwise_not(y) * 1  # 反转标签

    print("--------原训练集--------------")
    print(X.shape)
    X = X.sample(frac=0.1).reset_index(drop=True)
    print("----------10%---------------")
    print(X.shape)

    X_train = X.drop(columns=["isRight"])
    y_train = X["isRight"]

    model = GaussianNB()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc


def upper_sampling_DecisionTree(X, y):
    y = np.bitwise_not(y) * 1  # 反转标签

    # 上采样
    features_train, feature_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2)
    oversampler = SMOTE()
    os_features, os_labels = oversampler.fit_resample(features_train, labels_train)

    # 保存
    features_train = pd.DataFrame(features_train)
    labels_train = pd.DataFrame(labels_train)
    train_df = pd.concat([features_train, labels_train], axis=1)
    print("--------训练集---------")
    print(train_df.shape)
    # print(train_df)

    model = tree.DecisionTreeClassifier()
    model.fit(os_features, os_labels)

    # 预测
    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc, train_df, feature_test, labels_test


def upper_sampling_DecisionTree_10(train_df, feature_test, labels_test):
    # y = np.bitwise_not(y) * 1  # 反转标签

    print("--------原训练集--------------")
    print(train_df.shape)
    train_df = train_df.sample(frac=0.1).reset_index(drop=True)
    print("----------10%---------------")
    print(train_df.shape)

    # print(train_df)

    features_train = train_df.drop(columns=["isRight"])
    labels_train = train_df["isRight"]


    # 上采样
    oversampler = SMOTE()
    os_features, os_labels = oversampler.fit_resample(features_train, labels_train)

    model = tree.DecisionTreeClassifier()
    model.fit(os_features, os_labels)

    # 预测
    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc


def under_sampling_DecisionTree(X, y):
    # y = np.bitwise_not(y) * 1  # 反转标签

    # 下采样
    under_sampling_train_df, feature_test, labels_test = under_sampling(X, y)
    feature_test = np.array(feature_test).reshape(-1, X.shape[1])
    X_train = under_sampling_train_df.drop(columns=["isRight"])
    y_train = under_sampling_train_df["isRight"]

    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc, under_sampling_train_df, feature_test, labels_test


def under_sampling_DecisionTree_10(X, feature_test, labels_test):
    # y = np.bitwise_not(y) * 1  # 反转标签

    print("--------原训练集--------------")
    print(X.shape)
    X = X.sample(frac=0.1).reset_index(drop=True)
    print("----------10%---------------")
    print(X.shape)

    X_train = X.drop(columns=["isRight"])
    y_train = X["isRight"]

    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(feature_test)
    y_pred_res = [val_to_res(n) for n in y_pred]
    y_pred_res = np.array(y_pred_res)

    tp, fp, tn, fn = compute_confusion_matrix(y_pred_res, labels_test)
    acc, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)
    accuracy = accuracy_score(labels_test, y_pred_res)
    auc = roc_auc_score(labels_test, y_pred_res)
    mcc = matthews_corrcoef(labels_test, y_pred_res)
    return accuracy, f1, auc, mcc


netname = "BiLSTM"
dataname = "IMDB"
test_path = ".。/Coverage/" + netname + "_" + dataname + "/"
pre_path = ".。/Metrics/" + netname + "_" + dataname + "/"
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
# acc_baseline = []
acc_all = []
# acc_input_al = []
# acc_al_baseline = []
# acc_deepgini = []
# acc_lsa = []
# acc_dsa = []
# acc_cov_input = []
# acc_cov_output = []
acc_input_10 = []
acc_output_10 = []
acc_all_10 = []
acc_cov_10 = []

f1_cov = []
f1_input = []
f1_output = []
# f1_baseline = []
f1_all = []
# f1_input_al = []
# f1_al_baseline = []
# f1_deepgini = []
# f1_lsa = []
# f1_dsa = []
# f1_cov_input = []
# f1_cov_output = []
f1_input_10 = []
f1_output_10 = []
f1_all_10 = []
f1_cov_10 = []

auc_cov = []
auc_input = []
auc_output = []
# auc_baseline = []
auc_all = []
# auc_input_al = []
# auc_al_baseline = []
# auc_deepgini = []
# auc_lsa = []
# auc_dsa = []
# auc_cov_input = []
# auc_cov_output = []
auc_input_10 = []
auc_output_10 = []
auc_all_10 = []
auc_cov_10 = []

mcc_cov = []
mcc_input = []
mcc_output = []
# mcc_baseline = []
mcc_all = []
# mcc_input_al = []
# mcc_al_baseline = []
# mcc_deepgini = []
# mcc_lsa = []
# mcc_dsa = []
# mcc_cov_input = []
# mcc_cov_output = []
mcc_input_10 = []
mcc_output_10 = []
mcc_all_10 = []
mcc_cov_10 = []

# 10次减少随机性
for epoch in range(10):
    # 打乱, 预处理
    df = comb_df.sample(frac=1).reset_index(drop=True)
    # cov_df = cov_df.dropna(axis = 0, subset = [''] )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.reset_index(drop=True)
    cov_df = df
    print("原样本总数据为：")
    print(cov_df.shape)

    # 下采样预处理
    # y = df["isRight"]
    # cov_df = under_sampling(df, y)

    # cov
    X_cov = cov_df[["NC0.1", "NC0.3", "NC0.5", "NC0.7", "NC0.9"]]
    # X_cov = cov_df[["NC0.1", "NC0.3", "NC0.5", "NC0.7", "NC0.9", "KMNC", "NBC", "SNAC"]]
    y_cov = cov_df["isRight"]

    accuracy_cov, f1_c, auc_c, mcc_c, lr_train_cov, feature_test_cov, labels_test_cov = upper_sampling_DecisionTree(X_cov, y_cov)
    acc_cov.append(accuracy_cov)
    f1_cov.append(f1_c)
    auc_cov.append(auc_c)
    mcc_cov.append(mcc_c)
    print("cov:  ")
    print(accuracy_cov)
    print(f1_c)

    accuracy_cov_10, f1_c_10, auc_c_10, mcc_c_10 = upper_sampling_DecisionTree_10(lr_train_cov, feature_test_cov, labels_test_cov)
    acc_cov_10.append(accuracy_cov_10)
    f1_cov_10.append(f1_c_10)
    auc_cov_10.append(auc_c_10)
    mcc_cov_10.append(mcc_c_10)

    # input
    X_input = cov_df[["VBCount", "NNCount", "JJCount", "RBCount", "ConjCount", "SentCount", "Length",
                      "Polysemy", "DependDist", "SentiScore", "SentiFlip", "ConstHeight", "TerminalRatio"]]
    y_input = cov_df["isRight"]

    accuracy_i, f1_i, auc_i, mcc_i, lr_train_i, feature_test_i, labels_test_i = upper_sampling_DecisionTree(X_input, y_input)
    acc_input.append(accuracy_i)
    f1_input.append(f1_i)
    auc_input.append(auc_i)
    mcc_input.append(mcc_i)

    accuracy_i_10, f1_i_10, auc_i_10, mcc_i_10 = upper_sampling_DecisionTree_10(lr_train_i, feature_test_i, labels_test_i)
    acc_input_10.append(accuracy_i_10)
    f1_input_10.append(f1_i_10)
    auc_input_10.append(auc_i_10)
    mcc_input_10.append(mcc_i_10)

    # DeepGini + LSA + DSA
    X_baseline = cov_df[["DeepGini", "LSA", "DSA"]]
    y_baseline = cov_df["isRight"]

    # accuracy_b, f1_b, auc_b, mcc_b = upper_sampling_BernoulliNB(X_baseline, y_baseline)
    # acc_baseline.append(accuracy_b)
    # f1_baseline.append(f1_b)
    # auc_baseline.append(auc_b)
    # mcc_baseline.append(mcc_b)

    # al
    X_al = cov_df["PredVal"]
    X_al = X_al / (1 - X_al)
    X_al = np.log(X_al)
    X_al = abs(X_al)
    y_al = cov_df["isRight"]
    X_al = X_al.values.reshape(-1, 1)
    X_al = pd.DataFrame(X_al, columns=["X_al"])
    print("传参前数据shape为：")
    print(X_al.shape)
    print(y_al.shape)

    # accuracy_output, f1_o, auc_o, mcc_o = under_sampling_lr(X_al, y_al)
    # acc_output.append(accuracy_output)
    # f1_output.append(f1_o)
    # auc_output.append(auc_o)
    # mcc_output.append(mcc_o)

    # output
    X_al_output = pd.concat([X_al, X_baseline], axis=1)
    y_al_output = cov_df["isRight"]

    accuracy_o, f1_o, auc_o, mcc_o, lr_train_o, feature_test_o, labels_test_o = upper_sampling_DecisionTree(X_al_output, y_al_output)
    acc_output.append(accuracy_o)
    f1_output.append(f1_o)
    auc_output.append(auc_o)
    mcc_output.append(mcc_o)

    accuracy_o_10, f1_o_10, auc_o_10, mcc_o_10 = upper_sampling_DecisionTree_10(lr_train_o, feature_test_o, labels_test_o)
    acc_output_10.append(accuracy_o_10)
    f1_output_10.append(f1_o_10)
    auc_output_10.append(auc_o_10)
    mcc_output_10.append(mcc_o_10)

    # all
    X_cov_al_input = pd.concat([X_cov, X_al_output, X_input], axis=1)
    y_cov_al_input = cov_df["isRight"]

    accuracy_a, f1_a, auc_a, mcc_a, lr_train_a, feature_test_a, labels_test_a = upper_sampling_DecisionTree(X_cov_al_input, y_cov_al_input)
    acc_all.append(accuracy_a)
    f1_all.append(f1_a)
    auc_all.append(auc_a)
    mcc_all.append(mcc_a)

    accuracy_a_10, f1_a_10, auc_a_10, mcc_a_10 = upper_sampling_DecisionTree_10(lr_train_a, feature_test_a, labels_test_a)
    acc_all_10.append(accuracy_a_10)
    f1_all_10.append(f1_a_10)
    auc_all_10.append(auc_a_10)
    mcc_all_10.append(mcc_a_10)

    # input + output
    # X_input_al = pd.concat([X_al_output, X_input], axis=1)
    # y_input_al = cov_df["isRight"]
    # accuracy_input_al, f1_i_a, auc_i_a, mcc_i_a = upper_sampling_BernoulliNB(X_input_al, y_input_al)
    # acc_input_al.append(accuracy_input_al)
    # f1_input_al.append(f1_i_a)
    # auc_input_al.append(auc_i_a)
    # mcc_input_al.append(mcc_i_a)

    # DeepGini
    # X_deepgini = cov_df["DeepGini"]
    # X_deepgini = X_deepgini.values.reshape(-1, 1)
    # X_deepgini = pd.DataFrame(X_deepgini, columns=["X_deepgini"])
    # y_deepgini = cov_df["isRight"]
    # accuracy_deep, f1_deep, auc_deep, mcc_deep = upper_sampling_BernoulliNB(X_deepgini, y_deepgini)
    # acc_deepgini.append(accuracy_deep)
    # f1_deepgini.append(f1_deep)
    # auc_deepgini.append(auc_deep)
    # mcc_deepgini.append(mcc_deep)

    # LSA
    # X_lsa = cov_df["LSA"]
    # X_lsa = X_lsa.values.reshape(-1, 1)
    # X_lsa = pd.DataFrame(X_lsa, columns=["X_lsa"])
    # y_lsa = cov_df["isRight"]
    # accuracy_l, f1_l, auc_l, mcc_l = upper_sampling_BernoulliNB(X_lsa, y_lsa)
    # acc_lsa.append(accuracy_l)
    # f1_lsa.append(f1_l)
    # auc_lsa.append(auc_l)
    # mcc_lsa.append(mcc_l)
    #
    # # DSA
    # X_dsa = cov_df["DSA"]
    # X_dsa = X_dsa.values.reshape(-1, 1)
    # X_dsa = pd.DataFrame(X_dsa, columns=["X_dsa"])
    # y_dsa = cov_df["isRight"]
    # accuracy_d, f1_d, auc_d, mcc_d = upper_sampling_BernoulliNB(X_dsa, y_dsa)
    # acc_dsa.append(accuracy_d)
    # f1_dsa.append(f1_d)
    # auc_dsa.append(auc_d)
    # mcc_dsa.append(mcc_d)

    # cov + input
    # X_c_i = pd.concat([X_cov, X_input], axis=1)
    # y_c_i = cov_df["isRight"]
    # accuracy_c_i, f1_c_i, auc_c_i, mcc_c_i = upper_sampling_BernoulliNB(X_c_i, y_c_i)
    # acc_cov_input.append(accuracy_c_i)
    # f1_cov_input.append(f1_c_i)
    # auc_cov_input.append(auc_c_i)
    # mcc_cov_input.append(mcc_c_i)
    #
    # # cov + output
    # X_c_o = pd.concat([X_cov, X_al_output], axis=1)
    # y_c_o = cov_df["isRight"]
    # accuracy_c_o, f1_c_o, auc_c_o, mcc_c_o = upper_sampling_BernoulliNB(X_c_o, y_c_o)
    # acc_cov_output.append(accuracy_c_o)
    # f1_cov_output.append(f1_c_o)
    # auc_cov_output.append(auc_c_o)
    # mcc_cov_output.append(mcc_c_o)

    # 保存
    out_df = pd.DataFrame(
        {
            # "acc_baseline": acc_baseline,
            # "acc_DeepGini": acc_deepgini,
            # "acc_LSA": acc_lsa,
            # "acc_DSA": acc_dsa,
            "acc_input": acc_input,
            # "acc_al": acc_output,
            "acc_output": acc_output,
            # "acc_input_output": acc_input_al,
            "acc_cov_al_input": acc_all,
            "acc_cov": acc_cov,
            # "acc_cov_input": acc_cov_input,
            # "acc_cov_output": acc_cov_output,
            "acc_input_10": acc_input_10,
            "acc_output_10": acc_output_10,
            "acc_all_10": acc_all_10,
            "acc_cov_10": acc_cov_10,

            # "f1_baseline": f1_baseline,
            # "f1_DeepGini": f1_deepgini,
            # "f1_LSA": f1_lsa,
            # "f1_DSA": f1_dsa,
            "f1_input": f1_input,
            # "f1_al": f1_output,
            "f1_output": f1_output,
            # "f1_input_output": f1_input_al,
            "f1_cov_al_input": f1_all,
            "f1_cov": f1_cov,
            # "f1_cov_input": f1_cov_input,
            # "f1_cov_output": f1_cov_output,
            "f1_input_10": f1_input_10,
            "f1_output_10": f1_output_10,
            "f1_all_10": f1_all_10,
            "f1_cov_10": f1_cov_10,

            # "auc_baseline": auc_baseline,
            # "auc_DeepGini": auc_deepgini,
            # "auc_LSA": auc_lsa,
            # "auc_DSA": auc_dsa,
            "auc_input": auc_input,
            # "auc_al": auc_output,
            "auc_output": auc_output,
            # "auc_input_output": auc_input_al,
            "auc_cov_al_input": auc_all,
            "auc_cov": auc_cov,
            # "auc_cov_input": auc_cov_input,
            # "auc_cov_outout": auc_cov_output,
            "auc_input_10": auc_input_10,
            "auc_output_10": auc_output_10,
            "auc_all_10": auc_all_10,
            "auc_cov_10": auc_cov_10,

            # "mcc_baseline": mcc_baseline,
            # "mcc_DeepGini": mcc_deepgini,
            # "mcc_LSA": mcc_lsa,
            # "mcc_DSA": mcc_dsa,
            "mcc_input": mcc_input,
            # "mcc_al": mcc_output,
            "mcc_output": mcc_output,
            # "mcc_input_output": mcc_input_al,
            "mcc_cov_al_input": mcc_all,
            "mcc_cov": mcc_cov,
            # "mcc_cov_input": mcc_cov_input,
            # "mcc_cov_output": mcc_cov_output
            "mcc_input_10": mcc_input_10,
            "mcc_output_10": mcc_output_10,
            "mcc_all_10": mcc_all_10,
            "mcc_cov_10": mcc_cov_10
        }
    )
    out_df.to_csv("./Coverage/10%Compare/DecisionTree/上/" + netname + "_" + dataname + "_" + dirs[i])
