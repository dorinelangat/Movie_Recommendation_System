import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import ast
import json

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Reader, Dataset 
from surprise.model_selection import cross_validate

# This code defines a class called DatasetInfo that provides several methods for analyzing and processing datasets. 
# Let's go through the different methods:
class DatasetInfo:
    # This is the constructor method that initializes the class instance with a dataset.
    def __init__(self, dataset):
        self.dataset = dataset

    # This method checks if the dataset is a Pandas DataFrame and
    # returns the information about the dataset using the info() method of the DataFrame.
    def check_dataset_info(self):
        if isinstance(self.dataset, pd.DataFrame):
            return self.dataset.info()
        else:
            return "Invalid dataset type. Please provide a Pandas DataFrame."

    # This method checks if the dataset is either a NumPy array or a Pandas DataFrame and 
    # prints the shape of the dataset using the shape attribute.
    def check_dataset_shape(self):
        if isinstance(self.dataset, (np.ndarray, pd.DataFrame)):
            print("Dataset shape:", self.dataset.shape)
        else:
            print("Invalid dataset type. Please provide a NumPy array or a Pandas DataFrame.")

    #This method checks if the dataset is a Pandas DataFrame and 
    # returns the descriptive statistics of the dataset using the describe() method of the DataFrame.
    def get_dataset_statistics_describe(self):
        if isinstance(self.dataset, pd.DataFrame):
            return self.dataset.describe()
        else:
            return "Invalid dataset type. Please provide a Pandas DataFrame."

    # This is a static method that takes a string as input and converts it into a list of genres.
    # It assumes that the input string is in JSON format and extracts the genre names from the JSON objects.       
    @staticmethod
    def convert(self):
        result = []
        if isinstance(self, str):
            genres_list = json.loads(self)
            for genre in genres_list:
                result.append(genre['name'])
        return result
    
#This is another static method that takes an object as input and converts it into a list of names. It assumes that the input object is a string representation of a list of dictionaries.
#It iterates over the dictionaries and extracts the names, limiting the result to the first three names encountered.
    @staticmethod
    def convert3(obj):
        result = []
        count = 0
        for i in ast.literal_eval(obj):
            if count != 5:
                result.append(i['name'])
                count += 1
            else:
                break
        return result

# This is also a static method that takes a string as input and returns a list of directors' names. 
# It assumes that the input string is a string representation of a list of dictionaries. 
# It iterates over the dictionaries and extracts the names of directors based on the 'job' key being set to 'Director'.
    @staticmethod
    def get_directors(text):
        result = []
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                result.append(i['name'])
        return result
    @staticmethod
    def get_keywords(text):
        result = []
        for item in ast.literal_eval(text):
            result.append(item['name'])
        return result
	


	
# Calculate score for each qualified movie
def movie_score(x):
    v=x['vote_count']
    m=movies_credits['vote_count'].quantile(q=0.9)
    R=x['vote_average']
    C=movies_credits['vote_average'].mean()
    return ((R*v)/(v+m))+((C*m)/(v+m))

#Define a function to get movie recommendations for a user
def get_user_recommendations(user_Id, user_item_matrix, similarity_matrix, movies_credits, top_n=10):
    # Get the index of the user in the user-item matrix
    user_index = user_item_matrix.index.get_loc(user_Id)

    # Compute the similarity scores between the user and other users
    sim_scores = list(enumerate(similarity_matrix[user_index]))

    # Sort the users based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top N similar users
    top_user_indices = [i[0] for i in sim_scores[1:top_n+1]]

    # Get the movies that the top N similar users have rated highly
    top_movies = user_item_matrix.iloc[top_user_indices].sum(axis=0)
    top_movies = top_movies[top_movies == 0]  # Exclude movies already rated by the user

    # Get the movie details from the movie dataset
    recommendations = movies_credits.loc[top_movies.index]

    return recommendations
