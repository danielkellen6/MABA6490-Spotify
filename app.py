#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DanKellen@ MABA CLASS
"""

import streamlit as st
import pandas as pd
import string
import spacy
import pickle as pkl
from sentence_transformers import SentenceTransformer, util
import torch


st.image('miami.jpg')

header = st.container()


with open("miami_df.pkl", "rb") as file1: # "rb" because we want to read in binary mode
    df = pkl.load(file1)

with open("corpus_embeddings.pkl", "rb") as file2: # "rb" because we want to read in binary mode
    corpus_embeddings = pkl.load(file2)

with open("corpus.pkl", "rb") as file3: # "rb" because we want to read in binary mode
    corpus = pkl.load(file3)

@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedder = load_model()

with header:
    st.title("Miami Hotel Search")
    st.markdown("As Will Smith so eloquently stated: *Welcome to Miami - Bienvenidos a Miami*")


query = st.text_input("What are you looking for in a hotel?", placeholder = 'Enter your search here')
top_k = min(1, len(corpus))
query_embedding = embedder.encode(query, convert_to_tensor=True)

# We use cosine-similarity and torch.topk to find the highest 5 scores
cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
top_results = torch.topk(cos_scores, k=top_k)

for score, idx in zip(top_results[0], top_results[1]):
    row_dict = df.loc[df['review_body']== corpus[idx]]
    Hotel = " ".join(row_dict['hotelName'])
    Summary = " ".join(row_dict['Review_Summary'])

    st.header("The best hotel for your stay: ")
    st.write(Hotel)
    st.header("What other guests had to say:")
    st.write(Summary)



from googlesearch import search
for url in search(Hotel, stop=1):
    url_link = url
st.header("Book your stay today!")
st.write("[Click here to be redirected to the Hotel Webpage](%s)" % url_link)
