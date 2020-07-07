from datetime import *

from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform
from sklearn.metrics import confusion_matrix, f1_score

from utils import *
import pandas as pd
from keras import Input, Model
from keras.layers import Dense
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import numpy as np

time_init = datetime.now()

config = Config(0, True, 0, 100)
BREAST_FILE = 'data/brca - copia.csv'  # 'data/wdbc.csv'  #
FIELD_SEPARATOR = ','

message = "Original + FS(300) + Auto (100) + FS(#training)"
SPLITS = 5
REPEATS = 10
SHUFFLE = True
USE_AUTO_ENCODER = True
USE_FEATURE_SELECTION = True
USE_PCA = False
SEED = 100
EPOCHS_AE = 100
BATCH_AE = 5
GPU_DEVICE = 1

df_main = pd.read_csv(BREAST_FILE, sep=FIELD_SEPARATOR, index_col=0)
df_main['class'] = df_main['class'].map({'YES': 1, 'NO': 0})
y = df_main.pop('class').values
x = df_main.values

classifiers = [
    SVC(random_state=SEED), KNeighborsClassifier(),
    NuSVC(random_state=SEED),
    DecisionTreeClassifier(random_state=SEED),
    RandomForestClassifier(random_state=SEED),
    AdaBoostClassifier(random_state=SEED),
    GradientBoostingClassifier(random_state=SEED),
    GaussianNB()]

results = {c.__class__.__name__: 0.0 for c in classifiers}

for i in range(0, REPEATS):

    sk_fold = StratifiedKFold(n_splits=SPLITS, shuffle=SHUFFLE, random_state=i)

    for idx, (train_index, test_index) in enumerate(sk_fold.split(x, y)):

        X_train = x[train_index]
        Y_train = y[train_index]
        X_test = x[test_index]
        Y_test = y[test_index]

        step = MinMaxScaler()
        X_train = step.fit_transform(X_train)
        X_test = step.transform(X_test)

        if USE_AUTO_ENCODER:
            es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
            list_callbacks = [es]

            ENCODING_DIM = config.dim
            step = SelectKBest(k=3 * ENCODING_DIM)
            X_train_aux = step.fit_transform(X_train, Y_train)
            X_test_aux = step.transform(X_test)
            number_instances, num_attr = X_train_aux.shape[0], X_train_aux.shape[1]

            with tf.device('/gpu:' + str(GPU_DEVICE)):
                # this is our input placeholder
                input_ = Input(shape=(num_attr,))
                # "encoded" is the encoded representation of the input
                encoded = Dense(ENCODING_DIM, activation='relu', kernel_initializer=glorot_uniform(i))(input_)
                # encoded = Dense(int(ENCODING_DIM//2), activation='relu')(encoded)
                # encoded = Dense(encoding_dim, activation='relu')(encoded)
                decoded = Dense(num_attr, activation='sigmoid', kernel_initializer=glorot_uniform(i))(encoded)

                auto_encoder = Model(input_, decoded)

                auto_encoder.compile(optimizer='adadelta', loss='mse')

                auto_encoder.fit(X_train_aux, X_train_aux, epochs=EPOCHS_AE,
                                 batch_size=BATCH_AE, shuffle=SHUFFLE,
                                 validation_split=0.2,
                                 callbacks=list_callbacks)

                encoder = Model(input_, encoded)

                X_train = np.concatenate([X_train_aux, encoder.predict(X_train_aux)], axis=1)
                X_test = np.concatenate([X_test_aux, encoder.predict(X_test_aux)], axis=1)
                # X_train = encoder.predict(X_train)
                # X_test = encoder.predict(X_test)
                reset_encoder(GPU_DEVICE)
        elif USE_PCA:
            pca = PCA()
            X_train = pca.fit_transform(X_train, Y_train)
            X_test = pca.transform(X_test)

        if USE_FEATURE_SELECTION:
            number_instances, num_attr = X_train.shape[0], X_train.shape[1]
            step = SelectKBest(k=min(number_instances, num_attr))
            X_train = step.fit_transform(X_train, Y_train)
            X_test = step.transform(X_test)

        for c in classifiers:
            try:
                c.fit(X_train, Y_train)
                results[c.__class__.__name__] += c.score(X_test, Y_test) / (SPLITS * REPEATS)
            except:
                results[c.__class__.__name__] += 0.0
                print("ERROR on FIT")


def print_beauty(dic):
    result = ''
    for key in dic:
        result += str(key) + "," + str(dic[key]) + '\n'
    return result


print(
    "Jorge" + message + "\n" + BREAST_FILE + '\n' + str(print_beauty(results)) + "\n" + str(
        datetime.now() - time_init))
