# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 17:15:54 2025

@author: Daria Savvateeva
"""

import streamlit as st
import re

# 1  data upload form
def data_upload_form():
    """
    Displays the file upload form and returns user inputs.
    """
    with st.form("data_upload_form"):
        st.write("### Upload and configure your data")
        # Data type selection
        data_type = st.radio("Please select data type of your file:", 
                             ["American", "German"], 
                             index=0)  
        # File uploader
        uploaded_file = st.file_uploader("Choose a file", 
                                         type=["csv", "txt", "xlsx"])
        # Submit button
        submit_button = st.form_submit_button("Submit")
    # Return the collected values
    return data_type, uploaded_file, submit_button #selected_data_type, 


# 1.2 metadata and labels
def data_metadata_labels(df):

    # Get columns that contain at least 2 letters
    cols = list(df.columns)
    st.warning('The table has to contain at least one column with labels or metadata.')
    n = st.number_input('The measurements start from column number ', min_value=2, 
                    max_value=len(cols), value="min", step=1,) - 1
    metadata = cols[:n]
    st.session_state['full_metadata'] = metadata
    st.write('Columns with labels and/or metadata:')
    st.write(', '.join(metadata))
    # Let the user select label columns
    label_main = st.selectbox(
        "Select the column that shall be used as the main label",
        metadata,
        index=None,
        placeholder="",
    )
    
    # Let the user select label columns
    labels = st.multiselect(
        "Select column(s) that shall be used as additional labels, groups or unique identifiers",
        metadata,
        []
    )
    
    # Dynamically show metadata as "the rest" of the columns
    metadata_unused = [c for c in metadata if c not in labels and c not in [label_main]]
    
    st.write("Metadata columns that are not relevant for visual analysis:")
    st.write(', '.join(metadata_unused))
    
    # Optional: use a form just for final confirmation
    with st.form("confirm_form"):
        submit_button = st.form_submit_button("Submit")

    return [label_main], list(set(labels+[label_main])), metadata_unused, submit_button

# 1.3 treshold for features
def treshold_form():
    with st.form("set feature treshold"):
        percent = st.slider("% of featre value considered as noize", 
                           min_value=0, max_value=100, value=0)
        submit_button = st.form_submit_button("Submit")
    return percent, submit_button

# 2 data normalisation form
def data_norm_form(max_val):
    with st.form("data_normalise_form"):
        st.write("### Normalise Your Data (Pretreatment and Scaling)")
    
        # Select data normalisation algorithm
        dataprep1 = st.selectbox("Select an action", 
                                 ["None", "SPIMP", "Other?"], 
                                 index=None, # no option is selected initially
                                 )
       # max_val = len(df)
        sample_idx = st.slider("Select a sample to display", 
                           min_value=0, max_value=max_val-1)
            
        # Submit button
        submit_button = st.form_submit_button("Submit")
    return dataprep1, sample_idx, submit_button

def scaling_form(df):        
    with st.form("scaling"):
        st.write("### Scaling")
        scaler = st.selectbox("Select scaling", 
                              ["None", "StandardScaler", "Other?"], 
                              index=None)            
        submit_button = st.form_submit_button("Submit")
    return scaler, submit_button
   

def algorithm_form():
    """
    select UMAP or HDBSCAN first
    """
    with st.form("algorithm selection"):
        st.write("### UMAP or HDBSCAN?")
        method = st.selectbox("Select algorithm", 
                              ["UMAP", "HDBSCAN"], 
                              index=None)        
        submit_button = st.form_submit_button("Submit")
    return method, submit_button


# 3 or 4? UMAP parameters
def umap_params_form():
    data_len = len(st.session_state['df_vals'])
    max_val = int(data_len/4)
    v = int(max_val/2)
    with st.form("umap_params"):
        st.write("#### UMAP parameters")
        neigh = st.slider("Select the number of neighbors", 
                          min_value=1, max_value=max_val, value=v, help="Balances of the local structure versus global structure of the data. \
                        Low values: Focus on the local structure present in the data.\
                        High values: Focus on the larger neighborhoods losing the fine structure.")
        # should be dependent on the data size (up to 1/4)
        
        min_dist = st.slider("Select the distance", 
                             min_value=0.0, max_value=1.0, value=0.5, help="Controls how closely the points will be represented on the dimensional reduced plot.\
                                 Low values: show more clustered points.\
                                High values: tend to prevent clusters from forming, preserving the broad topological structure.")
        # controls how tightly UMAP is allowed to pack points together
        
        # ncomp = st.slider("Select the distance", 
        #                      min_value=1, max_value=3, value=2)
        # dimensionality of the reduced dimension space 
        
        metric = st.selectbox('Metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski', \
                                         'canberra', 'braycurtis', 'haversine',\
                                         'mahalanobis', 'wminkowski', 'seuclidean',\
                                         'cosine', 'correlation'\
                                         'hamming', 'jaccard', 'dice', 'russellrao', 'kulsinski',\
                                         'rogerstanimoto', 'sokalmichener', 'sokalsneath', 'yule'],
                              help="Determines how the distances between the points in the multidimensional space are calculated.")
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
                                     min_value=2, max_value=20,
                                     help="Sets the smallest group of samples for it to be considered a cluster.")
        min_samples = st.slider("Select minimum amount of samples in a cluster", \
                                     min_value=1, max_value=20,
                                     help="Controls how conservative the clustering will be.\
                                         Low values: Tends to include all samples in a cluster.\
                                        High values: will mark more and more samples as being “noise” not including them in a cluster.")
        cluster_selection_epsilon = st.slider("Select cluster selection epsilon", \
                                     min_value=0.0, max_value=1.0, value=0.5,
                                     help="This parameter affects the cluster merging, being the minimum distance for two clusters to be \
                                         separate from each other. This parameter is very useful when using a low Minimum Cluster Size, \
                                             but too many small clusters are formed, high values here would merge these clusters into \
                                                 bigger ones. ")
        cluster_sel_method = st.selectbox('Method to select clusters (Excess of Mass or leaf)', ['eom', 'leaf'],
                                          help="Excess of Mass tend to produce 1 or 2 big clusters and many smaller ones, \
                                              Leaf will tend to produce more smaller clusters like the leaves in a tree.")
        allow_single_cluster = st.selectbox('Allow single cluster', [False, True],
                                            help="Allows for the identification of one large cluster present in the data, \
                                                useful in cases where a good structure is not present in the data or when there \
                                                are too many small clusters.")

        st.session_state["hdbscan_min_cluster_size"] = int(min_cluster_size)
        st.session_state["hdbscan_min_samples"] = int(min_samples)
        st.session_state["hdbscan_cluster_selection_epsilon"] = float(cluster_selection_epsilon)
        st.session_state["hdbscan_cluster_selection_method"] = str(cluster_sel_method)
        st.session_state["hdbscan_allow_single_cluster"] = allow_single_cluster
        submit_button = st.form_submit_button("Submit")
    return submit_button

def select_label():
    with st.form("clustering label"):
        l = st.selectbox('Choose a label for your data:', ['']+st.session_state["labels"])
        submit_button = st.form_submit_button("Submit")
    return l, submit_button

def select_hover():
    with st.form("label on hovering"):
        h = st.selectbox('Choose a label on hovering:', st.session_state['full_metadata'])
        submit_button = st.form_submit_button("Submit")
    return h, submit_button



def svg_save_form():
    """
    select figure parameters
    """
    with st.form("svg parameters"):
        st.write("### Define parameters of the svg image")
        new_file_name = st.text_input("Filename (without extension)", "figure")
        width = st.number_input("Export width (px)", min_value=100, value=1500)
        height = st.number_input("Export height (px)", min_value=100, value=1500)
        scale = st.slider("Export scale (multiplier)", min_value=1, max_value=5, value=1)
              
        submit_button = st.form_submit_button("Submit")
    return new_file_name, width, height, scale, submit_button