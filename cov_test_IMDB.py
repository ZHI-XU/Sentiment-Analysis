import warnings
import sys
import time
from openpyxl import Workbook, load_workbook
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import numpy as np
from coverage import Coverage
import os
import pymannkendall as mk
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

# warnings.filterwarnings("ignore")
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)

# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#    tf.config.experimental.set_memory_growth(device, True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))



max_len = 200


def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding="post")
    return padded


def pre_proc(sentence):
    words = sentence.split(" ")
    if len(words) > max_len:
        sentence = " ".join(words[:max_len])
    return sentence


def load_data():
    train_df = pd.read_csv(
        "./IMDB/train.csv",
    )

    test_df = pd.read_csv(
        "./IMDB/test.csv",
    )

    train_x = train_df.text.tolist()
    train_y = train_df.label

    test_x = test_df.text.tolist()
    test_y = test_df.label

    # tokenizer = create_tokenizer(train_x)
    with open("./tokenizer/200in_IMDB.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

    train_x = encode_text(tokenizer, train_x, max_len)
    test_x = encode_text(tokenizer, test_x, max_len)

    le = LabelEncoder()
    train_y = le.fit_transform(train_y).reshape(-1, 1)

    test_y = le.transform(test_y).reshape(-1, 1)

    test_sentence = test_df.text.tolist()

    test_x = tokenizer.texts_to_sequences(test_sentence)
    test_x = pad_sequences(test_x, maxlen=max_len, padding="post")
    test_y = test_df.label.to_numpy().reshape(-1, 1)
    test_sentence = list(map(pre_proc, test_sentence))

    return train_x, train_y, test_x, test_y, test_sentence


if __name__ == '__main__':

    netname = "BiLSTM"
    dataname = "IMDB"

    num = 25000
    T = 0
    flag = 0

    x_train, y_train, x_test, y_test, test_sentence = load_data()

    result_df = pd.DataFrame(
        {"Sentence": test_sentence}
    )

    data_shape = [x_train.shape]
    print(data_shape)
    data_shape = [x_test.shape]
    print(data_shape)

    # x_train, y_train = shuffle(x_train, y_train, random_state=seed)

    model_path = './Model/' + netname + '_' + dataname + '/'
    dirs = os.listdir(model_path)
    out_path = "./Coverage/" + netname + "_" + dataname + "/" + os.path.splitext(dirs[T])[0]

    pre_df = pd.read_csv("./Metrics/" + netname + "_" + dataname + "/" + os.path.splitext(dirs[T])[0] + ".csv")

    model = load_model(model_path + dirs[T])

    model_layer = len(model.layers) - 1
    data_shape = x_train.shape
    # print(data_shape)
    # print(model_layer)

    l = []
    model.summary()
    for i in range(1, model_layer):
        if model.layers[i].output.shape[1] != None:
            l.append(i)

    # 记录每次迭代的coverage list
    # trend_test_nc1 = []
    # trend_test_nc2 = []
    # trend_test_nc3 = []
    # trend_test_nc4 = []
    # trend_test_nc5 = []
    trend_test_tknc = []
    trend_test_kmnc = []
    # trend_test_tknp = []
    trend_test_nbc = []
    trend_test_snac = []

    for i in x_test:
        print(time.asctime())
        print(flag)
        flag = flag + 1
        # 计算test_data上的coverage
        coverage = Coverage(model, x_train[0: num], i[None, :])
        # nc1, activate_num1, total_num1 = coverage.NC(l, threshold=0.1)
        # nc2, activate_num2, total_num2 = coverage.NC(l, threshold=0.3)
        # nc3, activate_num3, total_num3 = coverage.NC(l, threshold=0.5)
        # nc4, activate_num4, total_num4 = coverage.NC(l, threshold=0.7)
        # nc5, activate_num5, total_num5 = coverage.NC(l, threshold=0.9)
        tknc, pattern_num, total_num6 = coverage.TKNC(l)
        # tknp = coverage.TKNP(l)
        kmnc, nbc, snac, covered_num, l_covered_num, u_covered_num, neuron_num = coverage.KMNC(l)

        # trend_test_nc1.append(nc1)
        # trend_test_nc2.append(nc2)
        # trend_test_nc3.append(nc3)
        # trend_test_nc4.append(nc4)
        # trend_test_nc5.append(nc5)
        trend_test_tknc.append(tknc)
        trend_test_kmnc.append(kmnc)
        # trend_test_tknp.append(tknp)
        trend_test_nbc.append(nbc)
        trend_test_snac.append(snac)

        K.clear_session()

    # 数据保存
    # result_df["NC0.1"] = trend_test_nc1
    # result_df["NC0.3"] = trend_test_nc2
    # result_df["NC0.5"] = trend_test_nc3
    # result_df["NC0.7"] = trend_test_nc4
    # result_df["NC0.9"] = trend_test_nc5
    result_df["TKNC"] = trend_test_tknc
    result_df["KMNC"] = trend_test_kmnc
    result_df["NBC"] = trend_test_nbc
    result_df["SNAC"] = trend_test_snac
    #
    # result_df["TrueRes"] = pre_df["TrueRes"]
    # result_df["PredVal"] = pre_df["PredVal"]
    # result_df["PredRes"] = pre_df["PredRes"]
    # result_df["isRight"] = pre_df["isRight"]

    result_df.to_csv(out_path + "_add.csv")
    print("model " + dirs[T] + " is saved")
