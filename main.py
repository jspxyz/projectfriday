from flask import Flask, render_template, render_template, request, jsonify
 
import tensorflow as tf
import numpy as np
import re
import os
import base64
import uuid

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 