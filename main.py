# https://stackoverflow.com/questions/60032983/record-voice-with-recorder-js-and-upload-it-to-python-flask-server-but-wav-file
# added on 2020.12.06
# linked to index_addpipe_simple.html

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from subprocess import run, PIPE
from flask import logging, Flask, render_template, request
import datetime
import os


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/audio', methods=['POST'])
def audio():
    basename = "audio.wav"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = "_".join([timestamp, basename]) # e.g. '20201207_1714_audio'
    savepath = './entries/audio'
    filepath = "/".join([savepath, filename])
    with open(filepath, 'wb') as f:
        f.write(request.data)
    # with open('./entries/audio.wav', 'wb') as f:
    #     f.write(request.data)
    proc = run(['ffprobe', '-of', 'default=noprint_wrappers=1', filepath], text=True, stderr=PIPE)
    return proc.stderr


if __name__ == "__main__":
    app.logger = logging.getLogger('audio-gui')
    app.run(debug=True)

# import datetime
# basename = "audio"
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
# filename = "_".join([timestamp, basename]) # e.g. '20201207_1714_audio'
# savepath = './entries/audio'
# filepath = "/".join([savepath, filename]) # outputs: './entries/audio/20201207_1142_audio'