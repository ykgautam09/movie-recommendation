import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


def generate_model():
    raw_data = pd.read_csv('./dataset/tmdb_5000_movies.csv')  # read data
    # column selection
    data_to_consider = raw_data[['overview', 'original_title']]
    np.savez('raw_data.npz', movie=data_to_consider['original_title'])
    vectorizer = TfidfVectorizer(
        stop_words='english', analyzer='word', encoding='utf-8', ngram_range=(1, 3))
    feature_sm = vectorizer.fit_transform(
        data_to_consider['overview'].values.astype('U'))
    cosines = pairwise_distances(feature_sm, metric='cosine', n_jobs=-1)
    np.save('cosine_model.npy', cosines)


if __name__ == '__main__':
    generate_model()
