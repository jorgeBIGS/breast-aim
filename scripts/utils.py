import os
import requests
import json
import gc
import collections

from tensorflow.python.keras import backend as k

Config = collections.namedtuple('Config', 'fold autoencoder gpu dim')

auto_encoder = None
encoder = None

def reset_encoder(gpu):
    sess = k.get_session()
    k.clear_session()
    sess.close()
    sess = k.get_session()

    try:
        del auto_encoder  # this is from global space - change this as you need
        del encoder
    except:
        pass

    print(gc.collect())  # if it's done something you should see a number being outputted
