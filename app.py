import numpy as np
from flask import Flask, request, render_template
from model import get_recommendation

app = Flask(__name__)


@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/nlp-cosine')
def nlp_route():
    return render_template('movieInput.html')


@app.route('/nlp-cosine', methods=['POST'])
def cosine_model():
    movie = request.form.get('movie', "The Dark Knight Rises")
    number = request.form.get('number', 10)
    movies = get_recommendation(str(movie), int(number))
    return render_template('movieInput.html', size=len(movies), movies=movies)


if __name__ == '__main__':
    app.run(port='5000', debug=True)
