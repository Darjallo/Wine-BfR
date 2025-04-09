# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 17:15:54 2025

@author: Daria Savvateeva
"""

import streamlit as st

# 1  data upload form
def data_upload_form():
    """
    Displays the file upload form and returns user inputs.
    """
    with st.form("data_upload_form"):
        st.write("### Upload and Configure Your Data")

        # Data type selection
        data_type = st.radio("Please select data type of your CSV:", 
                             ["American", "German"], 
                             index=0)  

        # File uploader
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "xlsx"])

        # Data classification selection
        selected_data_type = st.radio("Please select one of the following options:",
                                      ["This is the data with unknown labels", 
                                       "This is the test data with known labels"],
                                      index=None)

        # Submit button
        submit_button = st.form_submit_button("Submit")

    # Return the collected values
    return data_type, uploaded_file, selected_data_type, submit_button

# 2 data normalisation form
def data_norm_form(max_val):
    with st.form("data_normalise_form"):
        st.write("### Normalise Your Data (Pretreatment and Scaling)")
    
        # Select data normalisation algorithm
        dataprep1 = st.selectbox("Select an action", 
                                 ["None", "splmp", "Other?"], 
                                 index=None, # no option is selected initially
                                 )
       # max_val = len(df)
        sample_idx = st.slider("Select a sample to display", 
                           min_value=0, max_value=max_val-1)
            
        # Submit button
        submit_button = st.form_submit_button("Submit")
    return dataprep1, sample_idx, submit_button
        

# 3 or 4? UMAP parameters
def umap_params_form():
    with st.form("umap_params"):
        st.write("#### UMAP parameters")
        neigh = st.slider("Select the number of neighbors", 
                          min_value=1, max_value=10, value=5)
        # should be dependent on the data size (up to 1/4)
        
        min_dist = st.slider("Select the distance", 
                             min_value=0.0, max_value=1.0, value=0.5)
        # controls how tightly UMAP is allowed to pack points together
        
        # ncomp = st.slider("Select the distance", 
        #                      min_value=1, max_value=3, value=2)
        # dimensionality of the reduced dimension space 
        
        metric = st.selectbox('Metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski', \
                                         'canberra', 'braycurtis', 'haversine',\
                                         'mahalanobis', 'wminkowski', 'seuclidean',\
                                         'cosine', 'correlation'\
                                         'hamming', 'jaccard', 'dice', 'russellrao', 'kulsinski',\
                                         'rogerstanimoto', 'sokalmichener', 'sokalsneath', 'yule'])
        # how distance is computed
        
        st.session_state["umap_neigh"] = int(neigh)
        st.session_state["umap_min_dist"] = float(min_dist)
        #st.session_state["umap_ncomp"] = int(ncomp)
        st.session_state["umap_metric"] = str(metric)
        submit_button = st.form_submit_button("Submit")
    return neigh, min_dist, submit_button


# 5? HDBSCAN parameters
def hdbscan_params_form():
    with st.form("hdbscan_params"):
        st.write("#### HDBSCAN parameters")
        min_cluster_size = st.slider("Select the minimum cluster size", \
                                     min_value=2, max_value=20)
        cluster_selection_epsilon = st.slider("Select cluster selection epsilon", \
                                     min_value=0.0, max_value=1.0, value=0.0)

        st.session_state["min_cluster_size"] = int(min_cluster_size)
        st.session_state["cluster_selection_epsilon"] = float(cluster_selection_epsilon)
        submit_button = st.form_submit_button("Submit")
    return min_cluster_size, cluster_selection_epsilon, submit_button
