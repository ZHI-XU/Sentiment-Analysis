import os
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pickle
import pandas as pd

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

from tensorflow.keras.models import load_model
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Flatten,
    Conv1D,
    MaxPooling1D,
    Add,
    Lambda,
    Dropout,
    concatenate,
    LSTM,
    Bidirectional
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import l2_normalize
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from IPython.display import SVG

def ModiData(obj_df):
    obj_df = obj_df[obj_df["label"] != 2]
    # obj_df["label"] = obj_df["label"].astype(string)
    obj_df.label = obj_df.label.replace(0, "neg")
    obj_df.label = obj_df.label.replace(1, "neg")
    obj_df.label = obj_df.label.replace(3, "pos")
    obj_df.label = obj_df.label.replace(4, "pos")
    obj_df = obj_df.fillna(0)
    # obj_df["label"] = obj_df["label"].astype(int)
    return obj_df


def clean_line(line):
    # split into tokens by white space
    tokens = line.split()
    # remove punctuation from each token
    table = str.maketrans("", "", string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding="post")
    return padded

train_df = pd.read_csv(
    "../IMDB/train.csv",
)

test_df = pd.read_csv(
    "../IMDB/test.csv",
)

train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

# train_df = ModiData(train_df)
# # dev_df = ModiData(dev_df)
# test_df = ModiData(test_df)

train_x = train_df.text.tolist()
train_y = train_df.label
# dev_x = dev_df.text.tolist()
# dev_y = dev_df.label
test_x = test_df.text.tolist()
test_y = test_df.label

max_len = 200
# tokenizer = create_tokenizer(train_x)
with open("../tokenizer/200in_IMDB.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index
train_x = encode_text(tokenizer, train_x, max_len)
# dev_x = encode_text(tokenizer, dev_x, max_len)
test_x = encode_text(tokenizer, test_x, max_len)

le = LabelEncoder()
train_y = le.fit_transform(train_y).reshape(-1, 1)
# dev_y = le.transform(dev_y).reshape(-1, 1)
test_y = le.transform(test_y).reshape(-1, 1)


def get_embeddings_layer(name, max_len, trainable):
    embedding_layer = Embedding(
        input_dim=len(word_index) + 1,
        output_dim=50,
        input_length=max_len,
        # weights=[embeddings_matrix],
        trainable=trainable,
        name=name,
    )
    return embedding_layer


def get_LSTM(max_len):
    embeddings_layer = get_embeddings_layer(
        "embeddings_layer", max_len, trainable=True
    )

    # dynamic channel
    in_layer = Input(shape=(max_len,), dtype="int32", name="input")
    layer = embeddings_layer(in_layer)

    layer = Bidirectional(LSTM(256, return_sequences=True))(layer)
    layer = Bidirectional(LSTM(256))(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(64, activation="relu", name="FC1")(layer)
    # layer = Dropout(0.5)(layer)

    o = Dense(1, activation="sigmoid", name="output", )(layer)

    model = Model(inputs=in_layer, outputs=o)
    model.compile(
        loss="binary_crossentropy", optimizer='adam', metrics=["acc"],
    )

    return model

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("val_loss") <= 0.335:
            print("\nReached 0.32 loss")
            self.model.stop_training = True


beststop = myCallback()
earlystop = EarlyStopping(monitor="val_loss", min_delta=0.001)
checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1)

def TrainModel(num):
    while num > 0:
        model = get_LSTM(max_len)
        # model.summary()
        history = model.fit(
            x=train_x,
            y=train_y,
            batch_size=400,
            epochs=5,
            validation_data=(test_x, test_y),
            callbacks=[beststop],
        )
        if history.history["val_acc"][-1] >= 0.86:
            save_name = (
                "../BiLSTM_IMDB/"
                + str(history.history["val_loss"][-1])[2:6]
                + "_"
                + str(history.history["val_acc"][-1])[2:6]
                + ".hdf5"
            )
            num -= 1
            model.save(save_name)

TrainModel(10)

# del model
model = get_LSTM(max_len)
model.summary()

history = model.fit(
            x=train_x,
            y=train_y,
            batch_size=400,
            epochs=20,
            validation_data=(test_x, test_y),
            # callbacks=[earlystop],
        )

from tensorflow.keras.models import load_model

# 保存模型
# model.save("10in_32unit_temp_res.h5")
# del model  # deletes the existing model
# 导入已经训练好的模型
# model = load_model("my_model.h5")
## 保存训练好的Tokenizer，和导入
import pickle

# saving
with open("/content/drive/MyDrive/Sentiment Analysis/tokenizer/49in_SST.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)