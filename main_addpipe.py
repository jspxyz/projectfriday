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

@app.route("/", methods=['POST', 'GET'])
def index():
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

# possible solution
# @app.route("/", methods=['POST', 'GET'])
# def index():
#     if request.method == "POST":
#         f = request.files['audio_data']
#         with open('audio.wav', 'wb') as audio:
#             f.save(audio)
#         print('file uploaded successfully')

#         return render_template('index.html', request="POST")
#     else:
#         return render_template("index.html")


# if __name__ == "__main__":
#     app.run(debug=True)