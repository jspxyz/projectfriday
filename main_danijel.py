from flask import Flask, render_template, logging, request, jsonify
from subprocess import run, PIPE

import tensorflow as tf
import numpy as np
import re
import os
import base64
import uuid

## https://github.com/danijel3/audio_gui/blob/master/main.py
## updated on 2020.12.05
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/audio', methods=['POST'])
def audio():
    with open('/tmp/audio.wav', 'wb') as f:
        f.write(request.data)
    proc = run(['ffprobe', '-of', 'default=noprint_wrappers=1', '/tmp/audio.wav'], text=True, stderr=PIPE)
    return proc.stderr


if __name__ == "__main__":
    app.logger = logging.getLogger('audio-gui')
    app.run(debug=True)