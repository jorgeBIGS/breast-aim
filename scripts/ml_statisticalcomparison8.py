from datetime import *
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from sklearn.feature_selection import SelectFpr, SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import regularizers
from utils import *
import os

GPU = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)  # specify which GPU(s) to be used


def extract_measures(classifiers, file, splits, repeats, use_auto, use_fs, encoding_dim, epochs, batch):
    BREAST_FILE = file
    FIELD_SEPARATOR = ','

    df_main = pd.read_csv(BREAST_FILE, sep=FIELD_SEPARATOR, index_col=0)
    df_main['class'] = df_main['class'].map({'YES': 1, 'NO': 0})
    y = df_main.pop('class').values
    x = df_main.values

    result = []

    for i in range(1, repeats + 1):
        os.environ['PYTHONHASHSEED'] = str(i)
        random.seed(i)
        np.random.seed(i)
        tf.set_random_seed(i)

        sk_fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=i)

        for idx, (train_index, test_index) in enumerate(sk_fold.split(x, y)):

            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]

            step = MinMaxScaler()
            x_train = step.fit_transform(x_train)
            x_test = step.transform(x_test)

            step = SelectFpr()
            x_train = step.fit_transform(x_train, y_train)
            x_test = step.transform(x_test)

            number_instances, num_attr = x_train.shape[0], x_train.shape[1]

            if use_auto:
                try:
                    with tf.device('/GPU:' + str(GPU)):
                        x_train_aux = x_train
                        x_test_aux = x_test
                        es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
                        list_callbacks = [es]

                        # this is our input placeholder
                        input_ = Input(shape=(num_attr,))
                        # "encoded" is the encoded representation of the input
                        encoded = Dense(encoding_dim, activation='relu', kernel_initializer=glorot_uniform(i),
                                        activity_regularizer=regularizers.l1(10e-5))(input_)
                        decoded = Dense(num_attr, activation='sigmoid', kernel_initializer=glorot_uniform(i))(
                            encoded)

                        auto_encoder = Model(input_, decoded)

                        auto_encoder.compile(optimizer='adam', loss='mse')

                        auto_encoder.fit(x_train_aux, x_train_aux, epochs=epochs,
                                         batch_size=batch, shuffle=True,
                                         validation_split=0.3,
                                         callbacks=list_callbacks)

                        encoder = Model(input_, encoded)

                        x_train = pd.np.concatenate([x_train_aux, encoder.predict(x_train_aux)], axis=1)
                        x_test = pd.np.concatenate([x_test_aux, encoder.predict(x_test_aux)], axis=1)
                        # X_train = encoder.predict(X_train_aux)
                        # X_test = encoder.predict(X_test_aux)
                        reset_encoder(GPU)
                except RuntimeError as e1:
                    print(e1)

                if use_fs:
                    model = SelectKBest(k=min(number_instances, num_attr))
                    x_train = model.fit_transform(x_train, y_train)
                    x_test = model.transform(x_test)

            for c in classifiers:
                c.fit(x_train, y_train)
                result += [c.score(x_test, y_test)]
    return result


CLASSIFIERS = [
    SVC(), KNeighborsClassifier(),
    NuSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB()]

FILE = 'data/brca - copia.csv'#'data/wdbc.csv'  #
s = 5
r = 10
n = 100
b = 5
e = 100

duration_1 = datetime.now()

results = dict()
results["NONE"] = extract_measures(CLASSIFIERS, FILE, 5, 10, False, False, n, b, e)
results["FS"] = extract_measures(CLASSIFIERS, FILE, 5, 10, False, True, n, b, e)
results["AUTO_NO_FS"] = extract_measures(CLASSIFIERS, FILE, 5, 10, True, False, n, b, e)
results["AUTO_FS"] = extract_measures(CLASSIFIERS, FILE, 5, 10, True, True, n, b, e)

ac1 = sum(results["NONE"]) / len(results["NONE"])
ac2 = sum(results["AUTO_NO_FS"]) / len(results["AUTO_NO_FS"])
ac3 = sum(results["FS"]) / len(results["FS"])
ac4 = sum(results["AUTO_FS"]) / len(results["AUTO_FS"])

print("NONE", ac1)
print("FS", ac3)
print("AUTO_NO_FS", ac2)
print("AUTO_FS", ac4)

f = open(str(datetime.now()) + "_results.csv", "w")
f.write("NONE," + str(results["NONE"]) + "\n")
f.write("FS," + str(results["FS"]) + "\n")
f.write("AUTO_NO_FS," + str(results["AUTO_NO_FS"]) + "\n")
f.write("AUTO_FS," + str(results["AUTO_FS"]) + "\n")
f.close()

print(datetime.now() - duration_1)
