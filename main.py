#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:01:27 2024

@author: Daria Savvateeva

"""

import streamlit as st
st.set_page_config(layout="wide")

#______________________________________________________________________________
"""
Pipeline:
1) data entry
2) pretreatment (different types of centering & standardization - 
                 including normalization, pareto scaling, medoid scaling, etc)
3) feature selection (SpImp, Missing Values Ratio (MVR), Threshold Setting, 
                      Low Variance Filter, High Correlation Filter etc)
4) dimension reduction (all the matrix factoring methods - PCA, LDA, MDS, NMDS, 
                        proximity matrix factoring, etc; then the embedding 
                        methods - UMAP, TriMAP, tSNE etc)
5) clustering (K-means, K-medoids, various forms of hierachical clustering, 
               BIRCH, DBScan, HDBScan etc)
"""
#______________________________________________________________________________


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import src.customtools as ct
import src.forms as f

    
# Initialize session state variables
for key, default_value in {
                           "df_raw": pd.DataFrame(), # original data from file
                           "main_label": [], 
                           "labels": [], # all colums names that can be used as labels
                           "labeled_data": pd.DataFrame(), # data of the labels
                           "df_vals": pd.DataFrame(), # measurements only
                           "data_type": '',
                           
                           "processed_treshold": None, 
                           "processed_splmp": None,
                           "step_1_ok": False,
                           "step_2_ok": False,
                           "step_3_1_ok": False, # algorithm selection (dim red)
                           "step_3_2_ok": False, # parameter selection
                           "df_norm1": pd.DataFrame(),
                           "df_dimred": pd.DataFrame(),
                           
                           "dimred": None, # dimention reduction algorithm selected
                           "umap_neigh": 15,
                           "umap_min_dist": 0.5,
                           "umap_ncomp": 2,
                           "umap_metric": 'euclidian',
                           
                           "step_4_1_ok": False, # clustering alg selection
                           "step_4_2_ok": False, # params for clustering
                           
                           "hdbscan_min_cluster_size": 5,
                           "hdbscan_min_samples": 5,
                           "hdbscan_cluster_selection_epsilon":0.5,
                           "hdbscan_cluster_selection_method": "eom",
                           "hdbscan_allow_single_cluster": False,
                           
                           "class_categories": [], # labels predicted by model
                           "class_proba": [], # prediction probabilities
                           "cluster_label": '',
                           }.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# Title of the app
st.title("App for spectra classification")
#______________________________________________________________________________
# Step 1: File Upload
st.info('Some details on how the file with the data should look like')

#______________________________________________________________________________
# Create two columns for the layout
st.divider()
col_upl_1, col_upl_2 = st.columns(2)
with col_upl_1:
    # 📌 Step 1: Data upload
    
    # data type German or American
    # file formats csv, txt, xlsx
    
    data_type, uploaded_file, submit_button_1 = f.data_upload_form()
    st.session_state['data_type'] = data_type

#______________________________________________________________________________

@st.cache_data
def file_upload(uploaded_file):
    """
    Uploads file, verifies format, processes it, and handles preprocessing.

    Args:
        uploaded_file (file): The uploaded file object.
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if not uploaded_file:
        st.warning("Please upload a file.")
        return pd.DataFrame()

    
    # Verify file format and read data
    if not ct.file_format_verification(uploaded_file):
        st.error("Invalid file format. Please upload a valid CSV or Excel file.")
        return pd.DataFrame()
    
    # Convert file to DataFrame
    try:
        df = ct.file2df(uploaded_file)
        st.session_state['df_raw'] = df
        return df
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return pd.DataFrame()

@st.cache_data
def file_vals_labels(data_type, labels, metadata):
    df = st.session_state['df_raw'] # original data
    cols_vals = [c for c in list(df.columns) if c not in labels and c not in metadata]

    labels_vals = df[labels]
    if st.session_state['data_type']=='German':
        df_vals = ct.german_df_processing(df[cols_vals])
    else:
        df_vals = df[cols_vals]
    st.session_state['df_vals'] = df_vals
    return df_vals


with col_upl_2:
    if submit_button_1:
        df_raw = file_upload(uploaded_file)
        st.session_state['df_raw'] = df_raw
    if len(st.session_state['df_raw'])>0:
        st.success("✅ File successfully uploaded.")
        st.write(st.session_state['df_raw'].head())


with col_upl_1:
    if len(st.session_state['df_raw'])>0:
        main_label, labels, metadata, submit_button_1_2 = f.data_metadata_labels(st.session_state['df_raw'])

        df_raw = st.session_state['df_raw']
        st.session_state['main_label'] = main_label
        st.session_state['labels'] = labels

if 'submit_button_1_2' in locals(): #??
    if submit_button_1_2: # labels selected
        st.session_state['labeled_data'] =df_raw[labels]     
        # dataframe with the measurements only
        df_vals = file_vals_labels(data_type, labels, metadata)
        # convert negative data to 0
        st.session_state['df_vals'] = ct.treshold(df_vals)
        st.warning('Attention! Negative values are converted to 0! Do we need to discuss it?', icon="⚠️")
        st.session_state["step_1_ok"] = True

# st.write('df' in locals())
# df disappears after the second button is pressed
# df = st.session_state['df_raw'] 
#groups = st.session_state['groups'] 
#col = st.session_state['col'] 

#______________________________________________________________________________
# 📌 Step 2: First dataprocessing step (splmp, for example)
@st.cache_data
def first_processing(df_data, dataprep1, sample_idx):
    """
    Feature Selection (splmp)

    """   
    if dataprep1 == "Other?":
        #ct.display_other_plot()
        df_norm=df_data
    elif dataprep1 == "splmp":
        df_norm = ct.splmp(df_data)
        #ct.splmp_plot(df_data, df_norm, sample_idx)
    
    else:
        df_norm=df_data
        #st.warning("⚠ Are you sure you want no processing at this step?")

    return df_norm

@st.cache_data
def first_processing_vis(df_norm, dataprep1, sample_idx):
    if dataprep1 == "Other?":
        ct.display_other_plot()
    elif dataprep1 == "splmp":
        ct.splmp_plot(df_norm, sample_idx)   
    else:

        st.warning("⚠ Are you sure you want no processing at this step?")
    pass

st.divider()
col_norm_1, col_norm_2 = st.columns(2)

with col_norm_1:
    if st.session_state["step_1_ok"] == True:
        dataprep1, sample_idx, submit_button_2 = f.data_norm_form(len(st.session_state['df_vals']))
    
       
    
with col_norm_2:    
    if 'submit_button_2' in locals(): #??
        if submit_button_2:
            df_norm1 = first_processing(st.session_state['df_vals'], dataprep1, sample_idx)   # after slpm
            #first_processing_vis(df_norm1, dataprep1, sample_idx)
            st.session_state["step_2_ok"] = True
            st.session_state["df_norm1"] = df_norm1
    if st.session_state["step_2_ok"] == True:
        first_processing_vis(st.session_state["df_norm1"], dataprep1, sample_idx)
    
    df_norm1 = st.session_state["df_norm1"]
    if len(df_norm1)>0:
        st.write(df_norm1.head())

#______________________________________________________________________________
# 📌 Step 3: Standard Scaler and dimention reduction
st.divider()
col_dimred_1, col_dimred_2 = st.columns(2)
st.write(st.session_state["step_2_ok"])

with col_dimred_1:
    if st.session_state["step_2_ok"] == True:
        
        with st.form("dimention_reduction"):
            st.write("### Standard Scaler ?")
            scaler = st.selectbox("Select normalization", 
                                  ["None", "StandardScaler", "Other?"], 
                                  index=None)
            
            st.write("### Dimention reduction")
        
            # Select dim red algorithm
            dimred = st.selectbox("Select dimention reduction method", 
                                     ["None", "UMAP", "Other?"], 
                                     index=None, # no option is selected initially
                                     )
            st.session_state["dimred"] = dimred
                
            # Submit button
            submit_button_3_1 = st.form_submit_button("Submit")
            
if 'submit_button_3_1' in locals():
    if submit_button_3_1:
        st.session_state["step_3_1_ok"] = True
  
with col_dimred_1:
    st.write(st.session_state["step_3_1_ok"])
    if st.session_state["step_3_1_ok"] == True:
        # depending on selected dimention reduction method define parameters
        if st.session_state["dimred"]=="UMAP":
            umap_neigh, umap_min_dist, submit_button_3_2 = f.umap_params_form()
            st.session_state["umap_neigh"] = umap_neigh
            st.session_state["umap_min_dist"] = umap_min_dist
        else:
            # add here for another dim red algorithms
            pass

@st.cache_data
def scale_dimred(df, scaler): # make kwargs
    """
    do we need scaling here or before feature selection??

    """
    vals = ct.normal(df, scaler)
    #output_dimred = ct.dim_reduction(vals, dimred, neigh, min_dist)
    #st.session_state["step_3_2_ok"] = True
    return vals # scaled (or not scaled) data before dimred

@st.cache_data
def plot_dimred(vals, paras):
# 'paras' is introduced here only make the function run once new values are available
# alternatively it does not use the new values stored in session state
# therefore all possible paras are passed to this function
    output_dimred = ct.dim_reduction(vals)
    return output_dimred
    
with col_dimred_2:
    if 'submit_button_3_2' in locals():
        if submit_button_3_2:
            st.write(submit_button_3_2)
            st.write(st.session_state["dimred"])
            scaled_dimred = scale_dimred(df_norm1, scaler,)
            dimred_paras = [st.session_state["umap_neigh"], st.session_state["umap_min_dist"],
                     st.session_state["umap_ncomp"], st.session_state["umap_metric"]]
            df_dimred = plot_dimred(scaled_dimred, dimred_paras)
            st.session_state["df_dimred"] = df_dimred
            st.session_state["step_3_2_ok"] = True
        
df_dimred = st.session_state["df_dimred"]

st.divider()
col_clust_1, col_clust_2 = st.columns(2)
with col_clust_1:
    if st.session_state["step_3_2_ok"] == True:
        # Select clustering algorithm
        with st.form("clustering"):
            st.write("### Clustering method")
            cluster = st.selectbox("Select clustering algorithm", 
                                     ["None", "HDBSCAN", "Other?"], index=None)
            st.session_state["cluster_alg"] = cluster
            submit_button_4_1 = st.form_submit_button("Submit")
            
    if 'submit_button_4_1' in locals():
        if submit_button_4_1:
            # clustering algorithm was selected
            st.session_state["step_4_1_ok"] = True 
    
    if st.session_state["step_4_1_ok"] == True:
        # depending on selected clustering method define parameters
        if st.session_state["cluster_alg"]=="HDBSCAN":
            min_cluster_size, cluster_selection_epsilon, submit_button_4_2 = f.hdbscan_params_form()
            #st.session_state["umap_neigh"] = umap_neigh
            #st.session_state["umap_min_dist"] = umap_min_dist
            # dublication, check also for umap
        else:
            # add here for another clustering algorithms
            pass
with col_clust_2:
    if st.session_state["step_4_1_ok"] == True:
        l, l_button = f.select_label()
        st.session_state["cluster_label"] = l

if 'submit_button_4_2' in locals():
    if submit_button_4_2:
        st.session_state["step_4_2_ok"] = True 
        
with col_clust_2:
    if st.session_state["step_4_2_ok"] == True:
        ct.clustering(df_dimred, st.session_state["cluster_alg"], 
                      st.session_state['df_raw'][st.session_state['main_label']],
                      st.session_state["cluster_label"])
        #st.session_state["step_4_2_ok"] = True
        
#@st.cache_data
#def do_clustering():  #(df_dimred, st.session_state["cluster_alg"], groups):
    
    
#         st.write('4. Select clustering algorithm')
#         dataprep4 = st.selectbox("Select step", ["None", "HDBSCAN", "Other?"], index=None)
        
# #______________________________________________________________________________
#         # Step 7: Apply HDBSCAN
#         if 'dataprep4' in locals() and dataprep4 is not None:
#             ct.clustering(output, dataprep4, groups)
                
#             st.success('Analysis finished')

