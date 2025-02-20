#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:01:27 2024

@author: daria

"""
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

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import src.customtools as ct

    
# Initialize session state variables
for key, default_value in {"sep": ",", 
                           "df": None,
                           "groups": None,
                           "col": None,
                           "processed_treshold": None, 
                           "processed_splmp": None,
                           "step_1_ok": False,
                           "step_2_ok": False,
                           "step_3_ok": False,
                           "df_norm1": None,
                           "df_dimred": None,
                           }.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# Title of the app
st.title("App for KIDA conference")
#______________________________________________________________________________
# Step 1: File Upload
st.info('Some details on how the file with the data should look like')

#______________________________________________________________________________
# 📌 Step 1: Data upload
# Uploading form
with st.form("data_upload_form"):
    st.write("### Upload and Configure Your Data")

    # Data type selection
    data_type = st.radio("Please select data type of your CSV:", 
                         ["American", "German"], 
                         index=0, # no option is selected initially
                         )

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "xlsx"])

    # Data classification selection
    selected_data_type = st.radio("Please select one of the following options:",
                                  ["This is the data with unknown labels", 
                                   "This is the test data with known labels"],
                                  index=None)

    # Submit button
    submit_button_1 = st.form_submit_button("Submit")
    
#______________________________________________________________________________

@st.cache_data
def file_upload(data_type, uploaded_file, selected_data_type):
    """
    Uploads file, verifies format, processes it, and handles preprocessing.

    """
    if not uploaded_file:
        st.warning("Please upload a file.")
        return None, None
    
    # Set CSV separator based on data type
    st.session_state['sep'] = ',' if data_type == 'American' else ';' #German

    
    # Verify file format and read data
    if not ct.file_format_verification(uploaded_file):
        st.error("Invalid file format. Please upload a valid CSV or Excel file.")
        return None, None
    
    df = ct.file2df(uploaded_file)
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
        return None, None

    # groups is a column with labels
    return df, groups, col


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
st.write(st.session_state["step_1_ok"])
if st.session_state["step_1_ok"] == True:
    
    with st.form("data_normalise_form"):
        st.write("### Normalise Your Data (Pretreatment and Scaling)")
    
        # Select data normalisation algorithm
        dataprep1 = st.selectbox("Select an action", 
                                 ["None", "splmp", "Other?"], 
                                 index=None, # no option is selected initially
                                 )
        max_val = len(df)
        number = st.slider("Select a sample to display", 
                           min_value=1, max_value=max_val)
            
        # Submit button
        submit_button_2 = st.form_submit_button("Submit")
        
        
    @st.cache_data
    def first_processing(df, dataprep1, col, number):
        """
        Feature Selection (splmp)
    
        """
        #remove column "group" if present
        df_data = ct.treshold(df, col) 
        
        if dataprep1 == "Other?":
            ct.display_other_plot()
            df_norm=df_data
        elif dataprep1 == "splmp":
            df_norm = ct.splmp(df_data)
            ct.splmp_plot(df_data, df_norm, number)
        
        else:
            df_norm=None

        return df_norm
    
if 'submit_button_2' in locals(): #??
    df_norm1 = first_processing(df, dataprep1, col, number)   #  after slpm
    st.session_state["step_2_ok"] = True
    st.session_state["df_norm1"] = df_norm1

df_norm1 = st.session_state["df_norm1"]
st.write(df_norm1.head())

#______________________________________________________________________________
# 📌 Step 3: Standard Scaler and dimention reduction
st.write(st.session_state["step_2_ok"])
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
            
        # Submit button
        submit_button_3 = st.form_submit_button("Submit")
        
        @st.cache_data
        def scale_dimred(df, scaler, dimred):
            """
            do we need scaling here or before feature selection??

            """
            vals = ct.normal(df, scaler)
            output_dimred = ct.dim_reduction(vals, dimred)
            st.session_state["step_3_ok"] = True
            return output_dimred


if 'submit_button_3' in locals():
    if submit_button_3:
        st.write(submit_button_3)
        df_dimred = scale_dimred(df_norm1, scaler, dimred)
        st.session_state["df_dimred"] = df_dimred
        st.session_state["step_3_ok"] = True
        
df_dimred = st.session_state["df_dimred"]



#         st.write('4. Select clustering algorithm')
#         dataprep4 = st.selectbox("Select step", ["None", "HDBSCAN", "Other?"], index=None)
        
# #______________________________________________________________________________
#         # Step 7: Apply HDBSCAN
#         if 'dataprep4' in locals() and dataprep4 is not None:
#             ct.clustering(output, dataprep4, groups)
                
#             st.success('Analysis finished')

