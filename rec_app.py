#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: @DanKellen MABA6490 Final Project
"""

import streamlit as st
st.set_page_config(layout="wide")
import os
import numpy as np
import pandas as pd
import warnings
import pickle
from googlesearch import search
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

header = st.container()
user_input = st.container()
video = st.container()
data = st.container()
cluster = st.container()

#DataFrame
with open("full_dataset_df.pkl", "rb") as file:
    data = pickle.load(file)

#recommender
with open("full_recs.pkl", "rb") as file1:
    recs = pickle.load(file1)
#Plot
with open("cluster_plot.pkl", "rb") as file2:
    cluster_image = pickle.load(file2)

#what cluster does the input belong to
with open("full_data_cluster.pkl", "rb") as file3:
    cluster_model = pickle.load(file3)


with header:
    st.header("Playlist Builder")

with user_input:
    dance1 = ['I hate dancing', 'Not Now', 'Maybe a little', 'Trying to catch a vibe', "Where's the Tequila"]
    pop1 = ["You probably haven't heard of them", "I saw them at 1st Ave before the were big", "I like experimenting", "Where's JBiebs", "FM Radio" ]
    temp1 = ['slowest', 'slower', 'even keel', 'fast', 'fastest']
    val1 = ["In the feels", "angsty teen", "keeping it even", "good vibes only", "Where's the Tequila"]
    nrg1 = ["no energy", "low energy", "net neutral", "harness the good energy, block the bad", "Hand me the Aux"]
    score = [-5,-2,0,2,5]
    dance_dict = dict(zip(dance1,score))
    pop_dict = dict(zip(pop1,score))
    tempo_dict = dict(zip(temp1,score))
    val_dict = dict(zip(val1,score))
    energy_dict = dict(zip(nrg1,score))


    col1, col2 = st.columns(2)

    list_length = col1.slider("How Many Songs do you want in your Playlist?", min_value = 2, max_value = 25, value = 15, step = 1)

    pop = col1.select_slider("Hipster or Mainstream?",options = pop1,value = pop1[2])
    dance = col1.select_slider('How bad do you want to dance?', options = dance1, value = dance1[2])
    nrg = col1.select_slider("Energy", options = nrg1, value = nrg1[2])
    val= col1.select_slider("Mood",options = val1, value = val1[2])
    temp = col1.select_slider("Tempo, Tempo, Tempo", options = temp1, value = temp1[2])

    danceability = dance_dict.get(dance)
    popularity = pop_dict.get(pop)
    energy = energy_dict.get(nrg)
    valence = val_dict.get(val)
    tempo = tempo_dict.get(temp)

    song_input = [popularity, danceability, energy, valence, tempo]

    playlist_length = list_length

    neigh = NearestNeighbors(n_neighbors=playlist_length)
    neigh.fit(data[["popularity", "danceability", "energy", "valence", "tempo"]])
    NearestNeighbors(algorithm='auto', leaf_size=30)

    recs = neigh.kneighbors([song_input])
    song = []

    for i in range(playlist_length):
        song.append(recs[1][0,i])

    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    playlist = pd.DataFrame(data[["song name", "artist"]].iloc[song])
    col2.table(playlist)

col3, col4 = st.columns(2)
with cluster:
    st.header("The Song EcoSystem")
    user_cluster = cluster_model.predict(np.array(song_input).reshape(1,-1))
    col3.plotly_chart(cluster_image, user_container_width = True)

    col4.write("You may also like songs from cluster {}".format(user_cluster[0]))
    user_cluster_df = data[data['cluster'] == user_cluster[0]]
    col4.table(user_cluster_df[['song name', 'artist']][:50])

with video:
    search_term = playlist['song name'].values[0] +" "+ playlist["artist"].values[0] + " " + "youtube"
    for url in search(search_term, stop=1):
        url_link = url
    st.video(url_link)
