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
for key, default_value in {"sep": ",", 
                           "df": None, # original data from file
                           "groups": [], # column with data on entry labels
                           "col": None,  # name of the column that contains info about labels
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
    # with or without labels 
    # file formats csv, txt, xlsx
    
    data_type, uploaded_file, selected_data_type, submit_button_1 = f.data_upload_form()

#______________________________________________________________________________

@st.cache_data
def file_upload(data_type, uploaded_file, selected_data_type):
    """
    Uploads file, verifies format, processes it, and handles preprocessing.

    Args:
        data_type (str): Type of data ('American' or 'German').
        uploaded_file (file): The uploaded file object.
        selected_data_type (str): The type of data (labeled or unlabeled).
        
    Returns:
        pd.DataFrame, list, list: Processed DataFrame, groups (if labeled), column names for labels
    """
    if not uploaded_file:
        st.warning("Please upload a file.")
        return pd.DataFrame(), None, None
    
    # Set CSV separator based on data type
    st.session_state['sep'] = ',' if data_type == 'American' else ';' #German

    
    # Verify file format and read data
    if not ct.file_format_verification(uploaded_file):
        st.error("Invalid file format. Please upload a valid CSV or Excel file.")
        return pd.DataFrame(), None, None
    
    # Convert file to DataFrame
    try:
        df = ct.file2df(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return pd.DataFrame(), None, None

    st.success("✅ File successfully uploaded.")
    st.write(df.head())
    
    
    # Handling labeled vs unlabeled data
    col, groups = [], None
    if selected_data_type == "This is the data with unknown labels":
        if data_type == 'German':
            df = ct.german_df_processing(df, col)
    elif selected_data_type == "This is the test data with known labels":
        col = [df.columns[0]] # name of the column that contains info about labels
        groups = df[df.columns[0]]
        if data_type == 'German':
            df = ct.german_df_processing(df, col)
    else:
        st.warning("⚠ Please select a valid data type option.")
        return pd.DataFrame(), None, None

    # groups is a column with labels
    return df, groups, col

with col_upl_2:
    if submit_button_1:
        df, groups, col = file_upload(data_type, uploaded_file, selected_data_type)
        st.session_state['df'] = df
        st.session_state['groups'] = groups
        st.session_state['col'] = col
              
        if df is not None:
            st.session_state["step_1_ok"] = True

# st.write('df' in locals())
# df disappears after the second button is pressed
df = st.session_state['df'] 
groups = st.session_state['groups'] 
col = st.session_state['col'] 

#______________________________________________________________________________
# 📌 Step 2: First dataprocessing step (splmp, for example)
@st.cache_data
def first_processing(df, dataprep1, col, sample_idx):
    """
    Feature Selection (splmp)

    """
    #remove column "group" if present
    df_data = ct.treshold(df, col) 
    
    if dataprep1 == "Other?":
        #ct.display_other_plot()
        df_norm=df_data
    elif dataprep1 == "splmp":
        df_norm = ct.splmp(df_data)
        #ct.splmp_plot(df_data, df_norm, sample_idx)
    
    else:
        df_norm=df_data
        #st.warning("⚠ Are you sure you want no processing at this step?")

    return df_data, df_norm

@st.cache_data
def first_processing_vis(df_data, df_norm, dataprep1, sample_idx):
    if dataprep1 == "Other?":
        ct.display_other_plot()
    elif dataprep1 == "splmp":
        ct.splmp_plot(df_data, df_norm, sample_idx)   
    else:

        st.warning("⚠ Are you sure you want no processing at this step?")
    pass

st.divider()
col_norm_1, col_norm_2 = st.columns(2)

with col_norm_1:
    if st.session_state["step_1_ok"] == True:
        dataprep1, sample_idx, submit_button_2 = f.data_norm_form(len(df))
    
       
    
with col_norm_2:    
    if 'submit_button_2' in locals(): #??
        if submit_button_2:
            df_tresh, df_norm1 = first_processing(df, dataprep1, col, sample_idx)   # after slpm
            first_processing_vis(df_tresh, df_norm1, dataprep1, sample_idx)
            st.session_state["step_2_ok"] = True
            st.session_state["df_norm1"] = df_norm1
    
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
            paras = [st.session_state["umap_neigh"], st.session_state["umap_min_dist"],
                     st.session_state["umap_ncomp"], st.session_state["umap_metric"]]
            df_dimred = plot_dimred(scaled_dimred, paras)
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

if 'submit_button_4_2' in locals():
    if submit_button_4_2:
        st.session_state["step_4_2_ok"] = True 
        
with col_clust_2:
    if st.session_state["step_4_2_ok"] == True:
        ct.clustering(df_dimred, st.session_state["cluster_alg"], 
                      st.session_state['groups'])
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

