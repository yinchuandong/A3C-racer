from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from PIL import Image
from io import BytesIO
import base64
import time
import numpy as np
import os

from action_helper import decode_action
from a3c import A3C

# python xxx.py
# 127.0,0.1:5000
app = Flask(__name__, static_url_path='', static_folder='static-ddpg')
app.config['SECRET_KEY'] = 'secret!'
app.debug = False  # you need to cancel debug mode when you run it on gpu
socketio = SocketIO(app)

net = A3C()
# four threads
state_list = [[], [], [], []]


def getTime():
    return int(round(time.time() * 1000))


@app.route('/train', methods=['post'])
def train():
    data = request.form
    thread_id = int(data['thread_id'])
    image = Image.open(BytesIO(base64.b64decode(data['img']))).convert('L')
    global state_list
    if len(state_list[thread_id]) == 0:
        state_list[thread_id] = np.stack((image, image, image, image), axis=2)
    else:
        image = np.reshape(image, (84, 84, 1))
        state_list[thread_id] = np.append(state_list[thread_id][:, :, 1:], image, axis=2)
    reward = float(data['reward'])
    terminal = data['terminal'] == 'true'

    # print reward, terminal, np.shape(state_list[thread_id])
    # action_id = 2
    action_id = net.train_function(thread_id, state_list[thread_id], reward, terminal)
    return jsonify(decode_action(action_id))


@app.route('/')
def index_final():
    return app.send_static_file('v4.final.html')


if __name__ == '__main__':
    socketio.run(app)
