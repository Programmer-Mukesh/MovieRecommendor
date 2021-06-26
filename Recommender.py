import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

movies= pd.read_csv('/movie.csv')
tags= pd.read_csv('/tag.csv')
ratings= pd.read_csv('/rating.csv')

movies.head()

tags.head()

ratings.head()

movies['genres']= movies['genres'].str.replace('|',' ')

len(movies.movieId.unique())

len(ratings.movieId.unique())

ratings_f = ratings.groupby('userId').filter(lambda x: len(x)>= 55)

movie_list_rating = ratings_f.movieId.unique().tolist()

len(ratings_f.movieId.unique())/len(movies.movieId.unique()) * 100

len(ratings_f.movieId.unique())/len(ratings.userId.unique()) * 100

movies = movies[movies.movieId.isin(movie_list_rating)]

movies.head()

Mapping_file= dict(zip(movies.title.tolist(),movies.movieId.tolist()))

tags.drop(['timestamp'],1, inplace=True)
ratings_f.drop(['timestamp'],1,inplace=True)

"""Merging the movies and the tag dataframe

"""

mixed= pd.merge(movies, tags, on='movieId', how='left')
mixed.head()

mixed.fillna("", inplace= True)
mixed= pd.DataFrame(mixed.groupby('movieId')['tag'].apply(lambda x:"%s" % ' '.join(x)))

Final= pd.merge(movies, mixed, on='movieId', how='left')
Final['metadata']= Final[['tag', 'genres']].apply(lambda x: ' '.join(x), axis=1)

Final[['movieId','title','metadata']].head()

Final.shape

Final.loc[1,"metadata"]

"""Creating a content latent matrix from movie metadeta

**tf-idf vectors and truncated SVD:**
"""

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf= TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(Final['metadata'])
tfidf_df= pd.DataFrame(tfidf_matrix.toarray(),index= Final.index.tolist())
print(tfidf_df.shape)

tfidf_df.shape

tfidf_df.loc[0]


