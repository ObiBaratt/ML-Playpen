from flask import Flask, url_for, render_template, request
from textblob import TextBlob
from joblib import load


def basic_text_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment


app = Flask(__name__)

sk_model = load('sk_sentiment.joblib')


@app.route('/')
def index():
    return f"<a href='{url_for('text_sentiment')}'>Text Sentiment Analysis using TextBlob</a>" \
           f"<a href='{url_for('sk_sentiment')}'>Text Sentiment Analysis with sklearn</a>"


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
        return render_template('text.html', analyzed=prediction, display=input_text)
    return render_template('text.html')


if __name__ == '__main__':
    app.run(debug=True)

