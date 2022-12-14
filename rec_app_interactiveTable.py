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
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

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
    st.markdown("This app has a 1500 song library that will create a playlist based on your current mood.\
                The data was extracted from Spotify (via kaggle) and have audio features that characterize each song based on several parameters. \
                My idea was to create an app that would create a playlist of songs that are closely aligned with the users current mood, to help them \
                discover new music, or current music that hits the spot. Further developments would center around linking the app with the Spotify API to dynamically keep \
                up with the top music (S/O TSwift) and connect to the users account where the app would write the playist into the users account and would be rewritten each time.\
                There are more features available for each track, the ones with the highest positive correlation to popularity were chosen")
    st.header("Features")
    st.subheader("Description of features from Spotify documentation")
    st.markdown("**Popularity** - The popularity of the artist. The artist's popularity is calculated from the popularity of all the artist's tracks.")
    st.markdown("**Danceability** - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.")
    st.markdown("**Energy** - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.")
    st.markdown("**Tempo** - The overall estimated tempo of a track. In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.")
    st.markdown("**Valence** - A measure describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).)")


with user_input:
    col1, col2 = st.columns(2)
    with col1:
        dance1 = ['I hate dancing', 'Not Now', 'Maybe a little', 'Trying to catch a vibe', "Where's the Tequila"]
        pop1 = ["You probably haven't heard of them", "I saw them at 1st Ave before the were big", "I like experimenting", "Where's JBiebs", "FM Radio" ]
        temp1 = ['slowest', 'slower', 'even keel', 'fast', 'fastest']
        val1 = ["In the feels", "angsty teen", "keeping it even", "good vibes only", "Where's the Tequila"]
        nrg1 = ["no energy", "low energy", "net neutral", "harness the good energy, block the bad", "Hand me the Aux"]
        score = [-5,-2,1,2,5]
        song_input = []

        dance_dict = dict(zip(dance1,score))
        pop_dict = dict(zip(pop1,score))
        tempo_dict = dict(zip(temp1,score))
        val_dict = dict(zip(val1,score))
        energy_dict = dict(zip(nrg1,score))

        st.header("Use the following sliders to build your playlist!")
        list_length = st.slider("How Many Songs do you want in your Playlist?", min_value = 2, max_value = 25, value = 0, step = 1)
        pop = st.select_slider("Hipster or Mainstream?",options = pop1,value = pop1[2])
        dance = st.select_slider('How bad do you want to dance?', options = dance1, value = dance1[2])
        nrg = st.select_slider("Energy", options = nrg1, value = nrg1[2])
        val= st.select_slider("Mood",options = val1, value = val1[2])
        temp = st.select_slider("Tempo, Tempo, Tempo", options = temp1, value = temp1[2])

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

        # playlist = pd.DataFrame(data[["song name", "artist"]].iloc[song])
        # st.table(playlist)
    with col2:
        st.header("Your Playlist")
        playlist2 = pd.DataFrame(data[["song name", "artist"]].iloc[song])

        def aggrid_interactive_table(df: pd.DataFrame):
            options = GridOptionsBuilder.from_dataframe(
                df,
                enableRowGroup=True,
                enableValue=True,
                enablePivot=False,

            )
            options.configure_default_column()
            options.configure_side_bar()

            options.configure_selection("single")
            selection = AgGrid(
                df,
                enable_enterprise_modules=True,
                gridOptions=options.build(),
                update_mode=GridUpdateMode.MODEL_CHANGED,
                allow_unsafe_jscode=True,
            )

            return selection


        playlist2 = pd.DataFrame(data[["song name", "artist"]].iloc[song])
        st.write("Select a Song from the playlist to listen via YouTube")
        selection = aggrid_interactive_table(df=playlist2)

        if selection["selected_rows"] == []:
            st.write("")
            # st.json(selection["selected_rows"])
        else:
            sn = selection["selected_rows"][0]["song name"]
            art = selection["selected_rows"][0]["artist"]
            st.write(selection["selected_rows"][0]["song name"])
            term = sn + " " + art + " " + "youtube"
            for url2 in search(term, stop=1):
                url_link2 = url2
            st.video(url_link2)

with cluster:
    st.header("The Song EcoSystem")
    col1, col2 = st.columns(2)
    user_cluster = cluster_model.predict(np.array(song_input).reshape(1,-1))
    col1.write("You may also like songs from cluster {}".format(user_cluster[0]))
    col1.plotly_chart(cluster_image)
    user_cluster_df = data[data['cluster'] == user_cluster[0]]
    col2.table(user_cluster_df[['song name', 'artist']][:15])




# with video:
#     search_term = playlist['song name'].values[0] +" "+ playlist["artist"].values[0] + " " + "youtube"
#     for url in search(search_term, stop=1):
#         url_link = url
#     st.video(url_link)
