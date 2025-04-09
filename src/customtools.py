#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:50:31 2024

@author: Daria Savvateeva
"""

from math import log
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN
#from matplotlib import cm
#import mplcursors
#import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

from . import visual as v


def file_format_verification(uploaded_file):
    # st.error("Unsupported file format.")
    if uploaded_file:
        return True
    else:
        return False

def file2df(uploaded_file):
    if uploaded_file is not None:
        try:
            # Display message based on file type
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, sep=st.session_state['sep'])
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".txt"):
                df = pd.read_csv(uploaded_file, delimiter='\t')
            else:
                pass
                
            return df
        except Exception as e:
            # Print the error type and message
            print(f"An error occurred: {type(e).__name__} - {e}")
            return None
    
def german_df_processing(df, exclude_columns):
    df_transformed = df.drop(columns=exclude_columns).applymap(
        lambda x: float(x.replace(",", ".")) if isinstance(x, str) else x
    )
    return pd.concat([df_transformed, df[exclude_columns]], axis=1)        
 
    
def process_splmp(df, col):
    """Handles the 'splmp' preprocessing step."""
    max_val = len(df)
    number = st.slider("Select a sample to display", min_value=1, max_value=max_val)

    if st.session_state['processed_splmp'] is None:
        df_data = treshold(df, col)
        df_norm = splmp(df_data)
        st.session_state['processed_treshold'] = df_data
        st.session_state['processed_splmp'] = df_norm
    else:
        df_data = st.session_state['processed_treshold']
        df_norm = st.session_state['processed_splmp']

    # Display the plot
    splmp_plot(df_data, df_norm, number)
    return df_norm, None

#  Plot `splmp` Results
def splmp_plot(df_data, df_norm, number):
    """Plots original vs modified data."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    x = list(df_data.columns)
    y_original = df_data.iloc[number, :].tolist()
    y_modified = df_norm.iloc[number, :].tolist()

    ax[0].plot(x, y_original)
    ax[0].set_title('Original')
    ax[0].set_xlabel('X-Axis Label')
    ax[0].set_ylabel('Y-Axis Label')

    ax[1].plot(x, y_modified, 'r')
    ax[1].set_title('Modified')
    ax[1].set_xlabel('X-Axis Label')
    ax[1].set_ylabel('Y-Axis Label')

    plt.tight_layout()
    st.pyplot(fig)

#  Handle "Other?" Option
def display_other_plot():
    """Displays a sample plot for the 'Other?' preprocessing option."""
    fig, ax = plt.subplots()
    x = np.arange(0, 10, 0.1)
    y = np.cos(x)
    ax.plot(x, y, '*r')
    ax.set_title("Other")
    st.pyplot(fig)
    
def treshold(df, col):
    
    # threshold above 0
    df = df.applymap(lambda x: 0 if (isinstance(x, (int, float)) and x < 0) else x)
    # drop column with labels
    df = df.drop(columns=col)
    return df
 
def splmp(df):
    """
    input: data above treshold
    applies splm formula
    returns dataframe
    """

    # tot_smpl_nmbr = len(df)
    # i_max = len(df.columns) # number of chemicals
    
    # data=[]
    # for chem in range(tot_smpl_nmbr):
    #     row=[]
    #     x_j = df.iloc[chem, :].sum()# sum of all elements in a row
    #     if x_j!=0:
    #         for col in range(i_max):
    #             col_vals=df.iloc[:, col]
    #             nmbr_valid_smpl = len([num for num in col_vals if num > 0])
    #             coef = log((1+tot_smpl_nmbr)/(1+nmbr_valid_smpl))
    #             x_ij = df.iloc[chem, col]      
    #             row+=[x_ij/x_j*coef]
    #             #print('x_j=', x_j, '  coef=', coef, '  nmbr_valid_smpl=', nmbr_valid_smpl)

    #     else:
    #         row=list(df.iloc[chem,:])
    #     data+=[row]
    
    # Precompute constants and data needed for vectorized operations
    tot_smpl_nmbr = len(df)
    #i_max = len(df.columns)  # Number of chemicals

    # Calculate x_j (sum of all elements in each row)
    x_j = df.sum(axis=1)

    # Calculate number of valid samples (elements > 0) for each column
    nmbr_valid_smpl = (df > 0).sum(axis=0)

    # Compute coefficient vector for all columns
    coef = np.log((1 + tot_smpl_nmbr) / (1 + nmbr_valid_smpl))

    # Create a mask for rows where x_j != 0
    mask_nonzero = x_j != 0

    # Initialize data with the same shape as df
    data = np.zeros_like(df, dtype=float)

    # For rows where x_j != 0, calculate the transformed values
    data[mask_nonzero] = (
        (df.loc[mask_nonzero].div(x_j[mask_nonzero], axis=0)).mul(coef, axis=1)
    ).values

    # For rows where x_j == 0, retain original values
    data[~mask_nonzero] = df.loc[~mask_nonzero].values
    
    out_df = pd.DataFrame(data, columns=df.columns)
    return out_df   


def normal(df, dataprep2):
    """
    normalization of the data

    """
    if dataprep2=="StandardScaler":
        vals = StandardScaler().fit_transform(df.values)
    else:
        vals = df.values
    return vals


def apply_umap(vals, neigh, min_dist, ncomp, m):
    reducer = umap.UMAP(n_neighbors=neigh, min_dist=min_dist,
                        n_components=ncomp, metric=m)
    embedding = reducer.fit_transform(vals)
    st.write(embedding)
    return embedding

def dim_reduction(vals): 
    dataprep = st.session_state["dimred"]
    if dataprep == 'UMAP':
        neigh = st.session_state["umap_neigh"]
        min_dist = st.session_state["umap_min_dist"]
        ncomp = st.session_state["umap_ncomp"]
        metric = st.session_state["umap_metric"]
        embedding = apply_umap(vals, neigh, min_dist, ncomp, metric)
        # figure
        v.dim_red_visual(embedding[:, 0], embedding[:, 1], 'Labels', 'UMAP')
        
        return embedding

def group_naming(groups):
    # input: list where each entry has letters and id number
    # output: list with only letters that are unique groups
    def remove_numbers(input_string):
        # Use a list comprehension to filter out numbers
        result = ''.join([char for char in input_string if not char.isdigit()])
        return result
    unique_groups = [remove_numbers(g) for g in groups]    
    return unique_groups

def clustering(vals, dataprep, groups):
    if dataprep == 'HDBSCAN':
        m_c_s = st.session_state["min_cluster_size"]
        eps = st.session_state["cluster_selection_epsilon"]

        hdb = HDBSCAN(cluster_selection_epsilon = eps, min_cluster_size = m_c_s).fit(vals) 
        labels = hdb.labels_ 
        st.session_state["class_categories"] = labels
        probabilities = hdb.probabilities_
        st.session_state["class_proba"] = probabilities
        
        v.clustering_visual(vals[:,0], vals[:,1], "HDBSCAN")
        
   
       



    
