import base64
import io

import numpy as np
from PIL import Image
from flask import Flask, render_template, send_file
from flask_socketio import SocketIO, emit

from core.network import *

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
net = Network()
net.deserialize("../trained/handwritten_digits.json")


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('image_data')
def handle_image(data_url):
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image_stream = io.BytesIO(image_bytes)

    img = Image.open(image_stream)
    img = img.convert('RGBA')
    img = img.getchannel('A')

    pixels = np.array(img)
    pixels = np.append([], pixels)
    pixels /= 255

    result = net.predict(np.array([pixels]).T)
    emit("image_data", list(result))


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)

