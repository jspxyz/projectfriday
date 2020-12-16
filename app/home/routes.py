# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from app.home import blueprint
from flask import render_template, redirect, url_for
from flask_login import login_required, current_user
from app import login_manager
from jinja2 import TemplateNotFound
import numpy as np
import pandas as pd
import sqlite3

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chart_studio
import chart_studio.plotly as py

@blueprint.route('/index')
@login_required
def index():
    # create lists
        # x-axis labels, date
        # series a data, audio polarity
        # series b data, text polarity
        # pie chart lists
    # create queries here

    ##### to start queries
    conn = sqlite3.connect('journal.db')
    # cur = conn.cursor()
    #####

    ##### START date list
    # date = pd.read_sql_query('''SELECT date 
    #                             FROM journal_entries''', conn)

    # date['date'] = pd.to_datetime(date['date'], yearfirst=True, format='%Y%m%d')
    # date_list = date['date'].to_list()
    ##### END date list

    ##### start text polarity score list start
    text_pol_query = pd.read_sql_query('''SELECT text_polarity_prob 
                                            FROM journal_entries''', conn)

    text_pol_table = text_pol_query['text_polarity_prob'].map(eval)
    text_pol_forchart = text_pol_table.apply(pd.Series)
    text_pol_score = text_pol_forchart['score'].to_list()

    # print(text_pol_score)
    #### end text polarity score list done

    ##### START audio polarity score list
    audio_pol_query = pd.read_sql_query('''SELECT audio_polarity_prob 
                                            FROM journal_entries''', conn)
    audio_df = audio_pol_query['audio_polarity_prob'].map(eval)
    audio_df = audio_df.apply(pd.Series)
    print(audio_df)

    def get_pol_score(x):
        pol_index = np.argmax(x.values)
        if pol_index == 0:
            return x[pol_index] * -1
        elif pol_index == 1: 
            return 0
        else:
            return x[pol_index]

    audio_df['audio_score'] = audio_df.apply(get_pol_score, axis=1)

    # Create figure with secondary y-axis
    # fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces: Heatmap and Linechart
    fig = go.Figure(data=[go.Heatmap(z=text_pol_forchart[['score']].T, colorscale='Blues'),
                           go.Scatter(x=audio_df['audio_score'].index, y=audio_df['audio_score'].values)])

    fig.update_layout(yaxis_range=[-1,1])

    username = 'jswpark'
    api_key = 'rFaFTWTzVKoGmnTfmzhl'

    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    path = py.plot(fig, filename='demo', auto_open=False)

    return render_template('index.html', path=path)

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