#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: @DanKellen MABA6490 Final Project
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
import warnings
import pickle

header = st.container()
user_input = st.container()
playlist = st.container()
data = st.container()
cluster = st.container()

#DataFrame
with open("tswift_df.pkl", "rb") as file:
    data = pickle.load(file)

#recommender
with open("recs.pkl", "rb") as file1:
    recs = pickle.load(file1)
#Plot
with open("cluster_plot.pkl", "rb") as file2:
    cluster_image = pickle.load(file2)

#what cluster does the input belong to
with open("tswift.pkl", "rb") as file3:
    cluster_model = pickle.load(file3)


with header:
    st.header("TSwift playlist Builder")

with user_input:

    col1, col2 = st.columns(2)

    popularity = col1.slider("How hipster are you?", min_value = 0, max_value = 100, value = 50, step = 5)
    danceability = col1.slider("How bad do you want to dance?", min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
    energy = col1.slider("Going out or staying in?", min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
    valence = col1.slider("Good vibes or bad", min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
    tempo = col1.slider("Moving slow, or going quick?", min_value = 0, max_value = 100, value = 50, step = 50)

    song_input = [popularity, danceability, energy, valence, tempo]

    recommendations = recs.kneighbors([song_input])

    one = recommendations[1][0,0]
    two = recommendations[1][0,1]
    thr = recommendations[1][0,2]
    playlist = pd.DataFrame(data[["name", "album"]].iloc[[one, two, thr]])
    col2.write(playlist)

    from googlesearch import search
    for url in search(playlist['name'][0], stop=1):
        url_link = url
    st.video(url_link)

with cluster:
    st.header("The TSwift EcoSystem")
    user_cluster = cluster_model.predict(np.array(song_input).reshape(1,-1))
    st.write("Check out cluster {}".format(user_cluster))
    st.plotly_chart(cluster_image, user_container_width = True)
