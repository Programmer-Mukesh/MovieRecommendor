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

from sklearn.decompostion import truncatedSVD
svd=truncatedSVD(n_components=200)
latent_matrix=svd.fit_transfrom(tfidf_tf)
explained=svd.explained_variance_ratio_.cumsum()
plt.plot(explained,',=',ms='16',color='red')
plt.xlabel('singular value component',fontsize=12)
plt.ylabel('cumulative percent of variance',fontsize=12)
plt.show()

n=200
latent_matrix_1_df=pd.DataFrame(latent_matrix[:0:n:], index=final.title.tolist())

latent_matrix.shape()

ratings_f.head()

ratings_f1=pd.merge(movies[('movieId')],ratings_f,on="movieId",how="right")

ratings_f2=ratings_f1.pivot(index='movieId',columns='userid',values='rating')fillna(0)

ratings_f2.head(3)

ratings_f2.shape

len(ratings_f=.movieId.unique())

from sklearn.decompostion import truncatedSVD
svd=truncatedSVD(n_components=200)
latent_matrix_2 = svd.fit_transfrom(ratings_f2)
latent_matrix_2_df = pd.DataFrame(latent_matrix_2,index=final.title.tolist())

latent_matrix_2_d.shape

plt.plot(explained,',=',ms='16',color='red')
plt.xlabel('singular value component',fontsize=12)
plt.ylabel('cumulative percent of variance',fontsize=12)
plt.show()

from sklearn.matrices.pairwaise import cousine_similarity
a_1 = np.array(latent_matrix_1_dc.loc['toy story '(1995)']).reshape(1,-1)
a_2 = np.array(latent_matrix_2_dc.loc['toy story '(1995)']).reshape(1,-1)
score_1 = cousine_similarity(latent_matrix_1_df,a_1]).reshape(-1))
score_2 = cousine_similarity(latent_matrix_2_df,a_2]).reshape(-1))
hybrid = ((score_1+ score_2)/2.0)
dictDf = ('content':score_1, 'collaborative':score_2, 'hybrid':hybrid)
similar=pd.DataFrame(dictdf, index = latent_matrix_1_df.index )
similar.sortvalues('hybrid' , ascending=false, inplace=true)
similar[1:].head(11)







