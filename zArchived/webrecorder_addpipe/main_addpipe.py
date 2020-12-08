# https://stackoverflow.com/questions/60032983/record-voice-with-recorder-js-and-upload-it-to-python-flask-server-but-wav-file
# added on 2020.12.06
# linked to index_addpipe_simple.html, app_addpipe.js, upload.php, style_addpipe.css
# https://addpipe.com/simple-recorderjs-demo/
# https://blog.addpipe.com/using-recorder-js-to-capture-wav-audio-in-your-html5-web-site/
# https://github.com/mattdiamond/Recorderjs

'''
current issue on December 6, 2020 at 1541
does not upload file to anywhere. can download
potential fix:
I have this problem and it takes me 2 days for finding the solution :)) .
In flask server you can use request.files['audio_data'] to get wav audio file. 
You can pass and use it as an audio variable too. Hope this can help you
'''

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