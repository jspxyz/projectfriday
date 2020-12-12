# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask_migrate import Migrate
from os import environ
from sys import exit
from decouple import config

from config_flask import config_dict
from app import create_app, db

from subprocess import run, PIPE
from flask import logging, Flask, render_template, request
import datetime
import os

# WARNING: Don't run with debug turned on in production!
DEBUG = config('DEBUG', default=True)

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try:
    
    # Load the configuration using the default values 
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app( app_config ) 
Migrate(app, db)

@app.route('/recorder', methods=['POST'])
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
    app.logger = logging.getLogger() # 'audio-gui'
    app.run()
