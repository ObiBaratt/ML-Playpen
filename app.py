from flask import Flask, url_for, render_template, request

from textblob import TextBlob


def basic_text_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment


app = Flask(__name__)


@app.route('/')
def index():
    return f"<a href='{url_for('text_sentiment')}'>Text Sentiment Analysis</a>"


@app.route('/text', methods=['GET', 'POST'])
def text_sentiment():
    if request.method == 'POST':
        input_text = request.form['text_input']
        analyzed = basic_text_sentiment(input_text)
        return render_template('text.html', analyzed=analyzed, display=input_text)
    return render_template('text.html')


if __name__ == '__main__':
    app.run(debug=True)

