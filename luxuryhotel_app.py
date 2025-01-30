# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 07:48:58 2021

@author: manis
"""
# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Step 1: Load Dataset
file_path = '/content/Top_50_hotels_dataset.csv'  # Replace with your actual file path
df = pd.read_csv(file_path, encoding='latin-1') # or 'iso-8859-1' or 'cp1252'

# Step 2: Combine relevant columns into a single feature
df['Features'] = df['Overview'] + " " + df['Dining_Area'] + " " + df['Drinking_Area'] + " " + df['Hotel_Ammenties']

# Step 3: Transform text data into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Features'])

# Step 4: Compute cosine similarity between hotels
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 5: Define recommendation function
def recommend_hotel(hotel_name, cosine_sim=cosine_sim, df=df):
    if hotel_name not in df['Name'].values:
        return f"Hotel '{hotel_name}' not found in the dataset."

    # Find the index of the hotel
    idx = df[df['Name'] == hotel_name].index[0]

    # Get similarity scores and sort them
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 5 similar hotels (excluding the queried hotel itself)
    sim_scores = sim_scores[1:6]

    # Get hotel names
    recommended_hotels = [df['Name'].iloc[i[0]] for i in sim_scores]

    return recommended_hotels

# Step 6: Streamlit app for serving recommendations
st.title("Luxury Hotel Recommendation System")

hotel_name = st.text_input("Enter a hotel name:")

if hotel_name:
    recommendations = recommend_hotel(hotel_name)
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.subheader("Recommended Hotels:")
        for rec in recommendations:
            st.write(f"- {rec}")