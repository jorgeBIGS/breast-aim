import os
import requests
import json
import gc
import collections

from tensorflow.python.keras import backend as k

Config = collections.namedtuple('Config', 'fold autoencoder gpu dim')

auto_encoder = None
encoder = None


def notify_slack(msg, web_hook='https://hooks.slack.com/services/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'):
    if web_hook is None:
        web_hook = os.environ.get('webhook_slack')
    if web_hook is not None:
        try:
            requests.post(web_hook, json.dumps({'text': msg}))
        except:
            print('Error while notifying slack')
            print(msg)
    else:
        print("NO WEBHOOK FOUND")


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
