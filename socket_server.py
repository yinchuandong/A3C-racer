from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
import base64
import time
import numpy as np
import os


# python xxx.py
# 127.0,0.1:5000
app = Flask(__name__, static_url_path='', static_folder='static-ddpg')
app.config['SECRET_KEY'] = 'secret!'
# app.debug = False  # you need to cancel debug mode when you run it on gpu
app.debug = True  # you need to cancel debug mode when you run it on gpu
socketio = SocketIO(app)


def getTime():
    return int(round(time.time() * 1000))


@socketio.on('init')
def handle_init(msg):
    print msg
    action = {'keyLeft': False, 'keyRight': False, 'keyFaster': True, 'keySlower': False}
    emit('init', action)
    return


@app.route('/train', methods=['post'])
def req_train():
    data = request.form
    image = Image.open(BytesIO(base64.b64decode(data['img']))).convert('L')

    # return jsonify(decode_action)
    return data['reward']


@app.route('/')
def index_final():
    return app.send_static_file('v4.final.html')


@app.route('/test')
def test():
    return 'test1'

if __name__ == '__main__':
    socketio.run(app)
