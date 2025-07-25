#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:01:27 2024

@author: Daria Savvateeva

"""

import streamlit as st
st.set_page_config(layout="wide")

#______________________________________________________________________________
# """
# Pipeline:
# 1) data entry
# 2) pretreatment (different types of centering & standardization - 
#                  including normalization, pareto scaling, medoid scaling, etc)
# 3) feature selection (SpImp, Missing Values Ratio (MVR), Threshold Setting, 
#                       Low Variance Filter, High Correlation Filter etc)
# 4) dimension reduction (all the matrix factoring methods - PCA, LDA, MDS, NMDS, 
#                         proximity matrix factoring, etc; then the embedding 
#                         methods - UMAP, TriMAP, tSNE etc)
# 5) clustering (K-means, K-medoids, various forms of hierachical clustering, 
#                BIRCH, DBScan, HDBScan etc)
# """
#______________________________________________________________________________


import pandas as pd
import src.customtools as ct
import src.forms as f
import src.visual as v

    
# Initialize session state variables
for key, default_value in {
                           "df_raw": pd.DataFrame(), # original data from file
                           "main_label": [], 
                           "labels": [], # all colums names that can be used as labels
                           "labeled_data": pd.DataFrame(), # data of the labels
                           "df_vals_no_filter": pd.DataFrame(), # measurements only
                           "df_vals": pd.DataFrame(), # after filtering
                           "data_type": '',
                           
                           "processed_treshold": None, 
                           "processed_splmp": None,
                           "step_1_ok": False,
                           "step_2_ok": False,
                           #"step_3_1_ok": False, # algorithm selection (dim red)
                           "step_3_2_ok": False, # umap 1 completed
                           "step_3_fig_ok": False, # figure is created
                           "step_3_3_ok": False, # hdbscan 1 completed
                           "df_norm": pd.DataFrame(),#normalised data (1 step)
                           "df_scaled": pd.DataFrame(),#scaled data (1 step)
                           
                           "df_dimred": [], # ndarray values
                           "df_clust": pd.DataFrame(), #??
                           
                           "method": None, # 1st selected algorithm
                           
                           "umap_neigh": 15,
                           "umap_min_dist": 0.5,
                           "umap_ncomp": 2,
                           "umap_metric": 'euclidian',
                           
                           "step_4_1_ok": False, # clustering alg selection
                           "step_4_2_ok": False, # params for clustering
                           "step_4_fig_ok": False, # ready for the last step
                           
                           "hdbscan_min_cluster_size": 5,
                           "hdbscan_min_samples": 5,
                           "hdbscan_cluster_selection_epsilon":0.5,
                           "hdbscan_cluster_selection_method": "eom",
                           "hdbscan_allow_single_cluster": False,
                           
                           "class_categories": [], # labels predicted by model
                           "class_proba": [], # prediction probabilities
                           "cluster_label": '',
                           "hover_label": '',
                           }.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

col_ttl_1, col_ttl_2 = st.columns(2)
with col_ttl_1:
    # Title of the app
    st.title("Graph theory based pipeline for data exploration")

    st.info("""
    ✅ Accepted file formats: xlsx, csv, txt  
    📋 The data must be arranged as a table where the first few columns are metadata.  
    💡 If the decimal part is separated with a comma, select German data type.
    """)
with col_ttl_2:
    #st.title("")
    st.image("BfR_Logo.png", width=400, )

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

    if st.session_state['data_type']=='German':
        df_vals = ct.german_df_processing(df[cols_vals])
    else:
        df_vals = df[cols_vals]

    st.session_state['df_vals_no_filter'] = ct.treshold(df_vals) # convert negatives to 0
    return


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
        # dataframe with the measurements only in session_state df_vals
        file_vals_labels(data_type, labels, metadata)
        df_vals = st.session_state['df_vals_no_filter']
        with col_upl_2:
            st.subheader(':blue[Selected data for processing:]')
            st.dataframe(df_vals, key='row_values')
            st.warning('Attention! Negative values are converted to 0!', icon="⚠️")
        st.session_state["step_1_ok"] = True
st.divider()
# st.write('df' in locals())
# df disappears after the second button is pressed
# df = st.session_state['df_raw'] 
#groups = st.session_state['groups'] 
#col = st.session_state['col'] 

@st.cache_data
def feature_treshold(t):
    st.session_state['df_vals'] = ct.feature_tr(t) 
    return
    
    
if st.session_state["step_1_ok"] == True:
    st.subheader('optional step')
    st.subheader('use treshold for features')
    col_tresh1, col_tresh2=st.columns(2)
    with col_tresh1:
        percent, treshold_submit = f.treshold_form()
        #form treshold input 
    with col_tresh2:
        feature_treshold(percent)
        st.subheader(':blue[Filtered data:]')
        st.dataframe(st.session_state['df_vals'], key='row_values')
        #display table
    st.divider()
#______________________________________________________________________________
# 📌 Step 2: First dataprocessing step (splmp)
@st.cache_data
def first_processing(df_data, dataprep1):
    """
    Feature Selection (spimp)
    """   
    if dataprep1 == "Other?":
        df_norm=df_data # nothing happens
        
    elif dataprep1 == "SPIMP":
        df_norm = ct.splmp(df_data)
    
    else:
        df_norm=df_data
    
    st.session_state["df_norm"] = df_norm
    return

@st.cache_data
def first_processing_vis(df_norm, dataprep1, sample_idx):
    if dataprep1 == "Other?":
        ct.display_other_plot()
    elif dataprep1 == "SPIMP":
        v.splmp_plot(df_norm, sample_idx)   
    else:
        st.warning("⚠ Are you sure you want no processing at this step?")
    pass


col_norm_1, col_norm_2 = st.columns(2)

if st.session_state["step_1_ok"] == True:
    st.divider()

    with col_norm_1:
        dataprep1, sample_idx, submit_button_2 = f.data_norm_form(len(st.session_state['df_vals']))
        
       
    
with col_norm_2:    
    if 'submit_button_2' in locals(): #??
        if submit_button_2:
            first_processing(st.session_state['df_vals'], dataprep1)   # after slpm
            st.session_state["step_2_ok"] = True
            df_norm = st.session_state["df_norm"]
            if len(df_norm)>0:
                st.subheader(':blue[Data after SPIMP processing:]')
                st.dataframe(df_norm)

    if st.session_state["step_2_ok"] == True:
        first_processing_vis(st.session_state["df_norm"], dataprep1, sample_idx)



# @st.cache_data
# def scale(df, scaler): # make kwargs
#     """
#     do we need scaling here or later??
#     """
#     ct.normal(df, scaler)

#     return ct.normal(df, scaler) # scaled (or not scaled) data before dimred

# with col_norm_1:
#     if st.session_state["step_2_ok"] == True:
#         scaling, but_scaling = f.scaling_form(st.session_state["df_norm"])

# if 'but_scaling' in locals():
#     if but_scaling:
#         st.session_state['df_scaled'] = scale(st.session_state["df_norm"], scaling)
#         st.success('Scaling completed')



#______________________________________________________________________________
# 📌 Step 3: Standard Scaler and dimention reduction
@st.cache_data
def scale_umap(df, scaler="StandardScaler"): # make kwargs
    """
    do we need scaling here or before feature selection??
    return dataframe
    """
    return ct.normal(df, scaler)

@st.cache_data
def plot_umap(vals, paras):
# 'paras' is introduced here only make the function run once new values are available
# alternatively it does not use the new values stored in session state
# therefore all possible paras are passed to this function
    output_dimred = ct.dim_reduction(vals)
    return output_dimred # ndarray

col_step1_1, col_step1_2 = st.columns(2)
if st.session_state["step_2_ok"] == True:
    st.divider()

    with col_step1_1:
        if st.session_state["step_2_ok"] == True:
            method, submit_button_3_1 = f.algorithm_form()
            if method != st.session_state["method"]:
                st.session_state["method"] = method
                st.session_state["step_3_2_ok"] = False
                st.session_state["step_3_fig_ok"] = False
                st.session_state["step_4_fig_ok"] = False
            # umap or hdbscan
            
            
# if 'submit_button_3_1' in locals():
#     if submit_button_3_1:
#         st.session_state["step_3_1_ok"] = True
  
if st.session_state["method"] is not None:
    # branching depending on 1st selected algorithm
    if st.session_state["method"]=="UMAP":
        # input umap params
        with col_step1_1:
            umap_neigh, umap_min_dist, submit_button_3_2 = f.umap_params_form()
        st.session_state["umap_neigh"] = umap_neigh
        st.session_state["umap_min_dist"] = umap_min_dist
        if 'submit_button_3_2' in locals():
            if submit_button_3_2:
                st.session_state["step_3_2_ok"] = True
        
        st.warning("Data is normalised with StandardScaler before applying UMAP")
        scaled_dimred = scale_umap(st.session_state["df_norm"])
        umap_paras = [st.session_state["umap_neigh"], st.session_state["umap_min_dist"],
                 st.session_state["umap_ncomp"], st.session_state["umap_metric"]]


        with col_step1_2:
            if st.session_state["step_3_2_ok"]:
                df_dimred = plot_umap(scaled_dimred, umap_paras)
                st.session_state["df_dimred"] = df_dimred # ndarray values
                st.session_state["step_3_fig_ok"] = True
        
    elif st.session_state["method"]=="HDBSCAN":
        with col_step1_1:
            submit_button_3_3 = f.hdbscan_params_form()
            
        
        
        vals = st.session_state["df_norm"].values
        with col_step1_1:
            ct.clustering(vals, 
                          st.session_state['df_raw'][st.session_state['main_label']],
                          st.session_state["cluster_label"])
            
        if 'submit_button_3_3' in locals():
            if submit_button_3_3:
                    st.session_state["step_3_3_ok"] = True
                    with col_step1_1:
                        st.success("clustering completed")
            
        


# ! umap plotting with plotly and various labels
#! umap 2D and 3D options


col_step2_1, col_step2_2 = st.columns(2)

if (st.session_state["step_3_fig_ok"] == True) or (st.session_state["step_3_3_ok"] == True):
    # 2nd step
    if st.session_state["method"]=="UMAP":
        # do hdbscan
        with col_step2_1:
            submit_button_4 = f.hdbscan_params_form()
        vals = st.session_state["df_dimred"]
        if 'submit_button_4' in locals():
            if submit_button_4:
                ct.clustering(vals, 
                              st.session_state['df_raw'][st.session_state['main_label']],
                              st.session_state["cluster_label"])
                st.session_state["step_4_fig_ok"] = True
                
        
    else:
        with col_step2_1:
            umap_neigh, umap_min_dist, submit_button_4 = f.umap_params_form()
        st.session_state["umap_neigh"] = umap_neigh
        st.session_state["umap_min_dist"] = umap_min_dist
        scaled_dimred = scale_umap(st.session_state["df_norm"])
        umap_paras = [st.session_state["umap_neigh"], st.session_state["umap_min_dist"],
                 st.session_state["umap_ncomp"], st.session_state["umap_metric"]]
        if 'submit_button_4' in locals():
            if submit_button_4:
                with col_step2_2:
                    df_dimred = plot_umap(scaled_dimred, umap_paras)
                st.session_state["df_dimred"] = df_dimred
                st.session_state["step_4_fig_ok"] = True
            
        pass
        # do umap

if st.session_state["step_4_fig_ok"]:
    #if submit_button_4_1:
    with col_step2_2:
        l, l_button = f.select_label()
        st.session_state["cluster_label"] = l
        h, h_button = f.select_hover()
        st.session_state["hover_label"] = h

    if 'l_button' in locals():
        #if l_button:
        vals = st.session_state["df_dimred"]
       # st.write(st.session_state["cluster_label"])
        fig = v.clustering_visual(vals[:,0], vals[:,1], "HDBSCAN", st.session_state["cluster_label"])
# in the very end clustering of hdbscan + umap
    
# download svg image

if 'fig' in locals():
   
    new_file_name, width, height, scale, submit_button_fig = f.svg_save_form()
    if submit_button_fig:
        with st.spinner("Generating SVG..."):
            image = fig.to_image(format='svg', width=width, height=height, scale=scale)
    
        # Download the image with the new file name
        st.download_button(label='Download svg', data=image, file_name=f"{new_file_name}.svg", mime='image/svg+xml')       

    


        