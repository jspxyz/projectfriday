# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from app.home import blueprint
from flask import render_template, redirect, url_for
from flask_login import login_required, current_user
from app import login_manager
from jinja2 import TemplateNotFound

@blueprint.route('/index')
@login_required
def index():
    # create lists
        # x-axis labels, date
        # series a data, audio polarity
        # series b data, text polarity
        # pie chart lists
    # create queries here


    return render_template('index.html')

@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith( '.html' ):
            template += '.html'

        return render_template( template )

    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
    except:
        return render_template('page-500.html'), 500

import requests
import json
from flask import jsonify
from flask import request
 
# DARKSKY_API_KEY = "1a317f487c751fc2d46d0c17b671300e"
 
# @blueprint.route('/get_data_api.html', methods=['GET'])
# def get_data_api():
#     ## add get model tf stuff here
#     ## look at VND classifier for example
#     lat = request.args.get('lat')
#     lon = request.args.get('lon')
#     our_request = 'https://api.darksky.net/forecast/{}/{},{}'.format(DARKSKY_API_KEY,lat,lon)
#     # print("Request: ",our_request)
#     res = requests.get(our_request)  
#     # print("Result: ",res.json())
 
#     return jsonify(res.json())