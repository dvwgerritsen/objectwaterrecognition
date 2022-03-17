import os
from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    result = ''
    image = ''
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file and file.filename.rsplit('.', 1)[1].lower() == 'jpg':
            filename = secure_filename(file.filename)
            file.save(os.path.join(filename))
            image = filename
            # todo do something with the uploaded image and set the result
        else:
            print('Only jpg is supported')

    return render_template("index.html", result=result, image=image)


# todo files need to be uploaded to a separate directory
@app.route('/<path:path>')
def send_report(path):
    return send_from_directory('', path)
