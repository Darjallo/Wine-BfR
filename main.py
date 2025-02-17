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
from src.customtools import file_format_verification, file2df, treshold, splmp,\
    normal, dim_reduction, clustering, german_df_processing

if 'sep' not in st.session_state:
    st.session_state['sep']=','
if 'processed_treshold' not in st.session_state:
    st.session_state['processed_treshold']=None
if 'processed_splmp' not in st.session_state:
    st.session_state['processed_splmp']=None

# Title of the app
st.title("App for KIDA conference")
#______________________________________________________________________________
# Step 1: File Upload
st.info('Some details on how the file with the data should look like')
data_type = st.radio("Please select data type of your csv:",\
                           ["American",\
                            "German"],
                            #index=None  # No option is selected initially
                            index=0
                            )

if data_type=='American':
    st.session_state['sep']=','
elif data_type=='German':
    st.session_state['sep']=';'

uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "xlsx"])

#______________________________________________________________________________
# Step 2: Check if the file format is correct
if file_format_verification(uploaded_file):
    df = file2df(uploaded_file)
    st.success("File is uploaded:")
    # Show the DataFrame if the file is valid
    st.write(df.head())
else:
    st.info("Please upload a file.")
  
# Radio button for the user to select between two options
if 'df' in locals():
    selected_data_type = st.radio("Please select one of the following options:",\
                               ["This is the data with unknown labels",\
                                "This is the test data with known labels"],
                                index=None  # No option is selected initially
                                )
    
    # Ensuring the user selects an option and confirms the selection
    if selected_data_type:
        st.success(f"You selected: {selected_data_type}")
        if selected_data_type=="This is the data with unknown labels":
            col=[]
            groups=None
            if data_type=='German':
                df = german_df_processing(df, col)
        else:
            col=[df.columns[0]]
            groups=df[df.columns[0]] # groups with labels
            if data_type=='German':
                df = german_df_processing(df, col)

    else:
        st.warning("Please select an option.")
        
        



#______________________________________________________________________________
# Step 3: User selects first step (splmp)
    st.write('1. Select feature selection algorithm')
    if selected_data_type:
        dataprep1 = st.selectbox("Select an action", ["None", "splmp", "Other?"], index=None)

#______________________________________________________________________________
# Step 4: User selects an integer number from a range to plot the data
if 'dataprep1' in locals() and dataprep1 is not None:
    #if any preprocessing procedure was selected
    if dataprep1 == "splmp":
        # if splmp step was selected
        max_val=len(df)
        number = st.slider("Select the sample to display", min_value=1, max_value=max_val)
        
        
        
        if st.session_state['processed_splmp'] is None:
            df_data = treshold(df, col)
            df = splmp(df_data)
            st.session_state['processed_treshold']=df_data
            st.session_state['processed_splmp']=df
        else: 
            df_data = st.session_state['processed_treshold']
            df = st.session_state['processed_splmp']

        # Display corresponding plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # Data for the plots
        x = list(df_data.columns)
        y = df_data.iloc[number, :].to_list()
        y_new = df.iloc[number, :].to_list()       
        # Plot for the original data on the left
        ax[0].plot(x, y)
        ax[0].set_title('Original')
        ax[0].set_xlabel('X-Axis Label')
        ax[0].set_ylabel('Y-Axis Label')       
        # Plot for the modified data on the right
        ax[1].plot(x, y_new, 'r')
        ax[1].set_title('Modified')
        ax[1].set_xlabel('X-Axis Label')
        ax[1].set_ylabel('Y-Axis Label') 
        # Adjust layout to prevent overlap
        plt.tight_layout()     
        st.pyplot(fig)
    elif dataprep1 == "Other?":
        # Display corresponding plot
        fig, ax = plt.subplots()
        x = np.arange(0, 10, 0.1)
        y = np.cos(1 * x)  # Use the selected number to adjust the frequency
        ax.plot(x, y, '*r')
        ax.set_title("Other")
        st.pyplot(fig)

    st.write('2. Decide about normalization')
    dataprep2 = st.selectbox("Select normalization", ["None", "StandardScaler", "Other?"], index=None)
#______________________________________________________________________________
# Step 5: Data normalization
    if 'dataprep2' in locals() and dataprep2 is not None:
        vals = normal(df, dataprep2)
        
        st.write('3. Select dimention reduction algorithm')
        dataprep3= st.selectbox("Select step", ["None", "UMAP", "Other?"], index=None)
        
#______________________________________________________________________________
# Step 6: Apply UMAP
        if 'dataprep3' in locals() and dataprep3 is not None:
            output = dim_reduction(vals, dataprep3)
            st.write('4. Select clustering algorithm')
            dataprep4 = st.selectbox("Select step", ["None", "HDBSCAN", "Other?"], index=None)
            
    #______________________________________________________________________________
            # Step 7: Apply HDBSCAN
            if 'dataprep4' in locals() and dataprep4 is not None:
                clustering(output, dataprep4, groups)
                    
                st.success('Analysis finished')

