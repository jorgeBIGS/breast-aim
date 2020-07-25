from datetime import *
import numpy as np
import pandas as pd
import tensorflow as tf
import random

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from sklearn.feature_selection import SelectKBest, SelectFdr
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from utils import *
import os

GPU_DEVICE = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE)  # specify which GPU(s) to be used


def extract_measures(classifiers, x, y, repeats, splits, shuffle, encoding_dim, epochs_ae, batch_ae, use_auto_encoder,
                     use_fs):
    results = {c.__class__.__name__: 0.0 for c in classifiers}
    for i in range(1, repeats + 1):
        os.environ['PYTHONHASHSEED'] = str(i)
        random.seed(i)
        np.random.seed(i)
        tf.set_random_seed(i)

        sk_fold = StratifiedKFold(n_splits=splits, shuffle=shuffle, random_state=i)

        for idx, (train_index, test_index) in enumerate(sk_fold.split(x, y)):

            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]

            step = MinMaxScaler()
            x_train = step.fit_transform(x_train)
            x_test = step.transform(x_test)

            step = SelectFdr()
            x_train = step.fit_transform(x_train, y_train)
            x_test = step.transform(x_test)

            number_instances, num_attr = x_train.shape[0], x_train.shape[1]

            if use_auto_encoder:
                with tf.device('/GPU:' + str(GPU_DEVICE)):
                    # this is our input placeholder
                    input_ = Input(shape=(num_attr,))
                    # "encoded" is the encoded representation of the input
                    encoded = Dense(encoding_dim, activation='relu')(input_)
                    # encoded = Dense(int(encoding_dim/2), activation='relu')(encoded)
                    # encoded = Dense(encoding_dim, activation='relu')(encoded)
                    decoded = Dense(num_attr, activation='sigmoid')(encoded)

                    auto_encoder = Model(input_, decoded)

                    auto_encoder.compile(optimizer='adam', loss='mse')

                    auto_encoder.fit(x_train, x_train, epochs=epochs_ae,
                                     batch_size=batch_ae)

                    encoder = Model(input_, encoded)

                    x_train = pd.np.concatenate([x_train, encoder.predict(x_train)], axis=1)
                    x_test = pd.np.concatenate([x_test, encoder.predict(x_test)], axis=1)
                    reset_encoder(GPU_DEVICE)

            if use_fs:
                model = SelectKBest(k=min(number_instances, num_attr))
                x_train = model.fit_transform(x_train, y_train)
                x_test = model.transform(x_test)

            for c in classifiers:
                c.fit(x_train, y_train)
                results[c.__class__.__name__] += c.score(x_test, y_test) / (splits * repeats)
    return results


time_rep = datetime.now()
SPLITS = 5
REPEATS = 10
SHUFFLE = True
ENCODING_DIM = 100
EPOCHS_AE = 100
BATCH_AE = 5

BREAST_FILE = 'data/brca - copia.csv'#'data/wdbc.csv'  #

FIELD_SEPARATOR = ','

df_main = pd.read_csv(BREAST_FILE, sep=FIELD_SEPARATOR, index_col=0)
df_main['class'] = df_main['class'].map({'YES': 1, 'NO': 0})
Y = df_main.pop('class').values
X = df_main.values

classify = [
    SVC(), KNeighborsClassifier(),
    NuSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB()]

results1 = extract_measures(classify, X, Y, REPEATS, SPLITS, SHUFFLE, ENCODING_DIM, EPOCHS_AE, BATCH_AE, False, False)
results2 = extract_measures(classify, X, Y, REPEATS, SPLITS, SHUFFLE, ENCODING_DIM, EPOCHS_AE, BATCH_AE, False, True)
results3 = extract_measures(classify, X, Y, REPEATS, SPLITS, SHUFFLE, ENCODING_DIM, EPOCHS_AE, BATCH_AE, True, False)
results4 = extract_measures(classify, X, Y, REPEATS, SPLITS, SHUFFLE, ENCODING_DIM, EPOCHS_AE, BATCH_AE, True, True)

print("NONE", str(results1))
print("FS", str(results2))
print("AUTO_NO_FS", str(results3))
print("AUTO_FS", str(results4))

f = open(str(datetime.now()) + "_resultsByClassifier.csv", "w")
f.write("NONE," + str(results1) + "\n")
f.write("FS," + str(results2) + "\n")
f.write("AUTO_NO_FS," + str(results3) + "\n")
f.write("AUTO_FS," + str(results4) + "\n")
f.close()

print(str(datetime.now() - time_rep))
