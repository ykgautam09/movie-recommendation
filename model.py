import numpy as np

MOVIE_NAME = "The Dark Knight Rises"
PREDICTION_COUNT = 10

cosine_model = np.load('./cosine_model.npy', allow_pickle=True)
data = np.load('./raw_data.npz', allow_pickle=True)


def get_recommendation(movie, n):
    index = np.where(data['movie'] == movie)[0]
    result = np.argsort(cosine_model[index])[0][1:int(n) + 1]
    movies = data['movie'][result]  # recommended movies
    return movies


if __name__ == '__main__':
    print(get_recommendation(MOVIE_NAME, PREDICTION_COUNT))
