# my personal working main.py
# saved on December 7 at 1034
# https://stackoverflow.com/questions/60032983/record-voice-with-recorder-js-and-upload-it-to-python-flask-server-but-wav-file
# added on 2020.12.06
# linked to index_addpipe_simple.html

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask
from flask import request
from flask import render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/audio", methods=['POST', 'GET'])
def audio():
    if request.method == "POST":
        f = open('./file.wav', 'wb')
        f.write(request.get_data("audio_data"))
        f.close()
        if os.path.isfile('./file.wav'):
            print("./file.wav exists")

        return render_template('index.html', request="POST")   
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run()