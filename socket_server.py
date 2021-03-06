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
app.debug = True  # you need to cancel debug mode when you run it on gpu
socketio = SocketIO(app)

net = A3C()
# four threads
state = None
next_state = None


def getTime():
    return int(round(time.time() * 1000))


@app.route('/train', methods=['post'])
def train():
    data = request.form
    thread_id = int(data['thread_id'])
    image = Image.open(BytesIO(base64.b64decode(data['img']))).convert('L')
    reward = float(data['reward'])
    terminal = data['terminal'] == 'true'
    start_frame = data['start_frame'] == 'true'

    global state, next_state
    if start_frame or state is None:
        state = np.stack((image, image, image, image), axis=2)

    image = np.reshape(image, (84, 84, 1))
    next_state = np.append(state[:, :, 1:], image, axis=2)

    # print reward, terminal, np.shape(state), np.shape(next_state)
    action_id = net.train_function(thread_id, state, reward, next_state, terminal, start_frame)
    state = next_state
    return jsonify(decode_action(action_id))
    # image = Image.fromarray(state[:, :, 0])
    # image.save('img/' + str(getTime()) + '.png')
    # return jsonify(decode_action(0))


@app.route('/')
def index_final():
    return app.send_static_file('v4.final.html')


if __name__ == '__main__':
    socketio.run(app)
