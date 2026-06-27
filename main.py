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
DEFAULT_STATES = {
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
                           "custom_group_colors": {}, # dictionary
                           }
 
    
for key, default_value in DEFAULT_STATES.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Helper functions
@st.cache_data
def file_upload(uploaded_file):
    """
    Uploads file, verifies format, processes it, and handles preprocessing.

    Args:
        uploaded_file (file): The uploaded file object.
        
    Returns:
        pd.DataFrame: Processed DataFrame or None
    """
  
    if not uploaded_file:
        return None
    if not ct.file_format_verification(uploaded_file):
        return "invalid_format"
    try:
        df = ct.file2df(uploaded_file)
        return df
    except Exception as e:
        return f"error: {e}"
    
    

@st.cache_data
def file_vals_labels(df_raw, data_type, labels, metadata):

    cols_vals = [c for c in list(df_raw.columns) if c not in labels and c not in metadata]

    if data_type=='German':
        df_vals = ct.german_df_processing(df_raw[cols_vals])
    else:
        df_vals = df_raw[cols_vals]

    return ct.treshold(df_vals) # convert negatives to 0


@st.cache_data
def first_processing(df_data, dataprep1):
    """
    Feature Selection (spimp)
    """   
        
    if dataprep1 == "SPIMP":
        return ct.splmp(df_data)


@st.cache_data
def scale_umap(df, scaler="StandardScaler"): # make kwargs
    """
    return dataframe
    """
    return ct.normal(df, scaler)


@st.cache_data
def plot_umap(vals, paras, color_dict, label):
# 'paras' is introduced here only make the function run once new values are available
# alternatively it does not use the new values stored in session state
# therefore all possible paras are passed to this function
    fig, embedding = ct.dim_reduction(vals) # ndarray
    return fig, embedding

# _____________________________________________________________________________

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

if submit_button_1 and uploaded_file:
    result = file_upload(uploaded_file)
    if isinstance(result, pd.DataFrame):
        st.session_state['df_raw'] = result
        if len(result.columns) == 1:
            st.error('Check data format. The data must contain at least one column with labels or metadata.')
    elif result == "invalid_format":
        st.error("Invalid file format. Please upload a valid CSV or Excel file.")
    else:
        st.error(f"Error reading the file: {result}")
 
       
if len(st.session_state['df_raw'])>0:
    with col_upl_2:
        st.success("✅ File successfully uploaded. First 5 rows:")
        st.write(st.session_state['df_raw'].head())


    with col_upl_1:
        main_label, labels, metadata, submit_button_1_2 = f.data_metadata_labels(st.session_state['df_raw'])
        
    if submit_button_1_2:
        st.session_state['main_label'] = main_label
        st.session_state['labels'] = labels
        st.session_state['labeled_data'] =st.session_state['df_raw'][labels]

        df_vals = file_vals_labels(st.session_state['df_raw'], data_type, labels, metadata)
        st.session_state['df_vals_no_filter'] = df_vals
        st.session_state["step_1_ok"] = True
        
        with col_upl_2:
            st.subheader(':blue[Selected data for processing:]')
            st.dataframe(df_vals, key='row_values')
            st.warning('Attention! Negative values are converted to 0!', icon="⚠️")

    
    
if st.session_state["step_1_ok"]:
    st.divider()
    st.subheader('Optional step: use treshold for features')
    col_tresh1, col_tresh2=st.columns(2)
    
    with col_tresh1:
        percent, treshold_submit = f.treshold_form()
        #form treshold input 
        
    st.session_state['df_vals'] = ct.feature_tr(st.session_state['df_vals_no_filter'], percent)
    
    with col_tresh2:
        st.subheader(':blue[Filtered data:]')
        st.dataframe(st.session_state['df_vals'], key='row_values')
        #display table
#______________________________________________________________________________
# 📌 Step 2: First dataprocessing step (splmp)

if st.session_state["step_1_ok"]:
    st.divider()
    col_norm_1, col_norm_2 = st.columns(2)

    with col_norm_1:
        dataprep1, sample_idx, submit_button_2 = f.data_norm_form(len(st.session_state['df_vals']))
        
    if submit_button_2:
        st.session_state["df_norm"] = first_processing(st.session_state['df_vals'], dataprep1) 
        st.session_state["step_2_ok"] = True
    
    with col_norm_2: 
        if st.session_state["step_2_ok"]:
            if len(st.session_state["df_norm"])>0:
                st.subheader(':blue[Data after SPIMP processing:]')
                st.dataframe(st.session_state["df_norm"])
                
                # Visualization
                if dataprep1 == "Other?":
                    ct.display_other_plot()
                elif dataprep1 == "SPIMP":
                    fig_splmp = v.splmp_plot(st.session_state['df_vals'], st.session_state["df_norm"], sample_idx)   
                    st.plotly_chart(fig_splmp)
                else:
                    st.warning("⚠ Are you sure you want no processing at this step?")


if st.session_state["step_2_ok"]:
    st.divider()
    col_step1_1, col_step1_2 = st.columns(2)

    with col_step1_1:      
        method, submit_button_3_1 = f.algorithm_form()
        if method != st.session_state["method"]:
            st.session_state["method"] = method
            st.session_state["step_3_2_ok"] = False
            st.session_state["step_3_fig_ok"] = False
            st.session_state["step_4_fig_ok"] = False
            
    if st.session_state["method"]=="UMAP":
        with col_step1_1:
            umap_neigh, umap_min_dist, submit_button_3_2 = f.umap_params_form()
        
        if submit_button_3_2:
            st.session_state["umap_neigh"] = umap_neigh
            st.session_state["umap_min_dist"] = umap_min_dist
            st.session_state["step_3_2_ok"] = True
            
        st.warning("Data is normalised with StandardScaler before applying UMAP")
        scaled_dimred = scale_umap(st.session_state["df_norm"])
        umap_paras = [st.session_state["umap_neigh"], st.session_state["umap_min_dist"],
                 st.session_state["umap_ncomp"], st.session_state["umap_metric"]]
        
        
        if st.session_state["step_3_2_ok"]:
            fig_umap, df_dimred = plot_umap(scaled_dimred, umap_paras, 
                                  st.session_state["custom_group_colors"],
                                  st.session_state["cluster_label"])
            st.session_state["df_dimred"] = df_dimred # ndarray values
            st.plotly_chart(fig_umap)
            st.session_state["step_3_fig_ok"] = True
            
    elif st.session_state["method"] == "HDBSCAN":
        with col_step1_1:
            submit_button_3_3 = f.hdbscan_params_form()
            
        if submit_button_3_3:
            st.session_state["step_3_3_ok"] = True
            vals = st.session_state["df_norm"].values
            ct.clustering(vals, 
                          st.session_state['df_raw'][st.session_state['main_label']],
                          st.session_state["cluster_label"])
            
            
            with col_step1_1:
                st.success("clustering completed")
            

if st.session_state["step_3_fig_ok"] or st.session_state["step_3_3_ok"]:
    st.divider()
    col_step2_1, col_step2_2 = st.columns(2)
    # 2nd step
    if st.session_state["method"]=="UMAP":
        # do hdbscan
        with col_step2_1:
            submit_button_4 = f.hdbscan_params_form()
        if submit_button_4 and st.session_state["df_dimred"] is not None:
            ct.clustering(st.session_state["df_dimred"], 
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
        fig_umap, df_dimred = plot_umap(scaled_dimred, umap_paras, 
                                  st.session_state["custom_group_colors"],
                                  st.session_state["cluster_label"])
        st.session_state["df_dimred"] = df_dimred
        st.plotly_chart(fig_umap)
        st.session_state["step_4_fig_ok"] = True
        

if st.session_state["step_4_fig_ok"]:
    col_step2_1, col_step2_2 = st.columns(2)
    with col_step2_2:
        l, l_button = f.select_label()
        st.session_state["cluster_label"] = l
        h, h_button = f.select_hover()
        st.session_state["hover_label"] = h
        # add color and countur selection
        # if user wants change colors:
        with st.expander(label="Customise colours of the data", expanded=False):
            st.write("define colours here")
            _, unique_groups = v.get_groups(st.session_state["cluster_label"])
            col_drop1, col_drop2 = st.columns([1,2])
            with col_drop1:
                selected_group_name = st.selectbox("Search for a label:", options=unique_groups)
            with col_drop2:
                custom_color = st.text_input(f"Custom Color for :rainbow[{selected_group_name}]", value="", #label_visibility='collapsed',
                                             help="Format: #000000")
            # ckeck the color format
            if custom_color:
                st.session_state["custom_group_colors"][selected_group_name]=custom_color
                
                
            # create dictionary with new colours, save to session state
            # add option to reset the dictionary with colors
            
            
        vals = st.session_state["df_dimred"]
       # st.write(st.session_state["cluster_label"])
        fig_hdbscan = v.clustering_visual(vals[:,0], vals[:,1], "HDBSCAN", st.session_state["cluster_label"])
    st.plotly_chart(fig_hdbscan, use_container_width=True)
# in the very end clustering of hdbscan + umap
    

# download final table
    with st.expander(label="Do you wish to display the data after processing and download it?", expanded=False):
        st.subheader("Projected data and its classification:")
        st.write("Navigate to the right upper corner of the table to download it")
        final_table = pd.DataFrame(st.session_state["df_dimred"])
        final_table['category'] = st.session_state["class_categories"]
        final_table['probability'] = st.session_state["class_proba"]
        st.dataframe(final_table)
                
# download svg image

    if 'fig_hdbscan' in locals():
       
        new_file_name, width, height, scale, submit_button_fig = f.svg_save_form()
        if submit_button_fig:
            with st.spinner("Generating SVG..."):
                image = fig_hdbscan.to_image(format='svg', width=width, height=height, scale=scale)
        
            # Download the image with the new file name
            st.download_button(label='Download svg', data=image, file_name=f"{new_file_name}.svg", mime='image/svg+xml')       
    
        


        