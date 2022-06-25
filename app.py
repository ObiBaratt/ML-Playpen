from flask import Flask, url_for, render_template, request, redirect, flash
from textblob import TextBlob
from joblib import load

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename

from deepface_recognition import deepface_analyze

import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'super secret key'


sk_model = load('sk_sentiment.joblib')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class FaceImgForm(FlaskForm):
    face_img = FileField('image', validators=[
                        FileRequired(),
                        FileAllowed(['jpg', 'jpeg', 'png'], 'jpg/jpeg/png only!')])


def basic_text_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/text', methods=['GET', 'POST'])
def text_sentiment():
    if request.method == 'POST':
        input_text = request.form['text_input']
        analyzed = basic_text_sentiment(input_text)
        return render_template('text.html', analyzed=analyzed, display=input_text)
    return render_template('text.html')


@app.route('/sk_sentiment', methods=['GET', 'POST'])
def sk_sentiment():
    if request.method == 'POST':
        input_text = request.form['text_input']
        iterable_text = [input_text]
        prediction_val = sk_model.predict(iterable_text)
        if prediction_val == 1:
            prediction = 'Positive'
        else:
            prediction = 'Negative'
        return render_template('sk_text.html', analyzed=prediction, display=input_text)
    return render_template('sk_text.html')


@app.route('/deepface', methods=['GET', 'POST'])
def deepface_analysis():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            filepath = f'static/uploads/{filename}'
            analyzed = deepface_analyze(filepath)
            return render_template('face_recognition.html', filename=filename, analyzed=analyzed)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return render_template('face_recognition.html')
    else:
        return render_template('face_recognition.html')


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    app.run(debug=True)
