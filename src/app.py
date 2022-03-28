import os
from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from predict_nieuw import predict

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    image = ''
    error = ''
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)

        file = request.files['file']
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        if file and (file_extension == 'jpg' or file_extension == 'jpeg' or file_extension == 'png'):
            filename = secure_filename(file.filename)
            file.save(os.path.join("upload/", filename))
            image = predict("upload/" + filename, False)
        else:
            print('Only jpg or png is supported')
            error = 'Het bestandstype moet jpg of png zijn.'

    return render_template("index.html", image=image, error=error)


@app.route('/processed-image/<path:path>')
def send_report(path):
    return send_from_directory('predicted_images', path)
