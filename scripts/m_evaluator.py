import gc

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import History
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
import tensorflow as tf
from datetime import *
from utils import *

time_init = datetime.now()
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BREAST_FILE = 'data/brca - copia.csv'  # 'data/wdbc.csv'  #
FIELD_SEPARATOR = ','
SPLITS = 5
REPEAT = 10
SHUFFLE = True
configs = [
    Config(25, True, 0, 325),
    Config(20, True, 0, 300),
    Config(15, True, 0, 275),
    Config(10, True, 0, 250),
    Config(1, True, 0, 225),
    Config(2, True, 0, 200),
    Config(3, True, 0, 175),
    Config(3, True, 0, 150),
    Config(3, True, 0, 125),
    Config(4, True, 0, 100),
    Config(5, True, 0, 75),
    Config(6, True, 0, 50),
    Config(7, True, 0, 25),
    Config(8, True, 0, 20),
    Config(9, True, 0, 15),
    Config(10, True, 0, 10),
    Config(11, True, 0, 5)
]

m_evaluation = {c.dim: 0.0 for c in configs}

df_main = pd.read_csv(BREAST_FILE, sep=FIELD_SEPARATOR, index_col=0)
df_main['class'] = df_main['class'].map({'YES': 1, 'NO': 0})
y = df_main.pop('class').values
x = df_main.values

for config in configs:
    print(config.fold)
    NUMBER = config.fold
    USE_AUTO_ENCODER = config.autoencoder
    GPU_DEVICE = config.gpu
    encoding_dim = config.dim  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    batch_AE = 10
    epochs_AE = 20
    DEVICE_NAME = '/GPU:' + str(GPU_DEVICE)  # '/device:CPU'  #

    es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
    list_callbacks = [es]

    for i in range(0, REPEAT):

        sk_fold = StratifiedKFold(n_splits=SPLITS, shuffle=SHUFFLE, random_state=i)

        for idx, (train_index, test_index) in enumerate(sk_fold.split(x, y)):
            X_train = x[train_index]
            Y_train = y[train_index]

            step = MinMaxScaler()  # StandardScaler()  #
            X_train = step.fit_transform(X_train)

            ##### AUTOENCODER #####

            if USE_AUTO_ENCODER:
                with tf.device(DEVICE_NAME):
                    number_instances, num_attr = X_train.shape[0], X_train.shape[1]

                    # this is our input placeholder
                    input = Input(shape=(num_attr,))
                    # "encoded" is the encoded representation of the input
                    encoded = Dense(encoding_dim, activation='relu', kernel_initializer=glorot_uniform(i))(input)
                    decoded = Dense(num_attr, activation='sigmoid', kernel_initializer=glorot_uniform(i))(encoded)

                    auto_encoder = Model(input, decoded)

                    auto_encoder.compile(optimizer='adadelta', loss='mse')

                    history: History = auto_encoder.fit(X_train, X_train,
                                                        epochs=epochs_AE,
                                                        batch_size=batch_AE,
                                                        shuffle=SHUFFLE,
                                                        validation_split=0.2,
                                                        callbacks=list_callbacks
                                                        )

                    score = auto_encoder.evaluate(X_train, X_train)
                    m_evaluation[encoding_dim] = m_evaluation[encoding_dim] + (score / (SPLITS * REPEAT))

                    reset_encoder(GPU_DEVICE)

ordered = sorted(m_evaluation, key=lambda val: m_evaluation[val])
print(
    "Jorge\n" + BREAST_FILE + '\n' + str(m_evaluation) + "\n" + str(ordered) + "\n" + str(datetime.now() - time_init))
