# https://stackoverflow.com/questions/51401989/is-there-a-way-to-send-audio-files-to-a-flask-server

@app.route('/uploadfile',methods=['GET','POST'])
def uploadfile():
    if request.method == 'PUT':
        f = request.files['file']
        filePath = "./somedir/"+secure_filename(f.filename)
        f.save(filePath)
        return "success"
