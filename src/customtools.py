#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:50:31 2024

@author: daria
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


def apply_umap(vals):
    neigh = st.slider("Select the number of neighbors", min_value=1, max_value=100, value=15)
    min_dist = st.slider("Select the number of neighbors", min_value=0.0, max_value=1.0, value=0.5)
    reducer = umap.UMAP(n_neighbors=neigh, min_dist=min_dist,)
    embedding = reducer.fit_transform(vals)
    return embedding

def dim_reduction(vals, dataprep):
    if dataprep == 'UMAP':
        embedding = apply_umap(vals)
        # figure
        fig, ax = plt.subplots()
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            #c = [x for x in df['Type of wine'].map({"Red":'red', "white":'black',})]
            #c = [x for x in df['group'].map({'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'black', 'E': 'yellow'})]
        )
        ax.set_title('UMAP')
   
        st.pyplot(fig)
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
        min_cluster_size = st.slider("Select the minimum cluster size", \
                                     min_value=1, max_value=10)
        cluster_selection_epsilon = st.slider("Select cluster selection epsilon", \
                                     min_value=0.0, max_value=1.0, value=0.0)

        hdb = HDBSCAN(cluster_selection_epsilon = 0.0375, min_cluster_size = 3).fit(vals) 
        labels = hdb.labels_ 
        probabilities = hdb.probabilities_
        
        unique_labels = np.unique(labels)
        
        g_names = group_naming(groups)  # ids w/o digits
       
        
        
        # Generate a colormap for arbitrary group count
        colormap1 = cm.get_cmap('Pastel1', len(set(g_names)))
        group_colors_original = {group: colormap1(i) for i, group in enumerate(set(g_names))}  # Map each group to a color
        group_colors_original = {group: colors.to_hex(color) for group, color in group_colors_original.items()}       
        colors_original = [group_colors_original[group] for group in g_names]  # Map groups to colors original
        
        colormap2 = cm.get_cmap('tab20', len(unique_labels)) 
        group_colors_model = {group: colormap2(i) for i, group in enumerate(unique_labels)} 
        group_colors_model = {group: colors.to_hex(color) for group, color in group_colors_model.items()}      
        colors_model = [group_colors_model[group] for group in labels]  # 

        # Create scatter plots for each group
        traces = []
        for group_name, color in group_colors_model.items():
            group_mask = [g == group_name for g in labels]
            traces.append(
                go.Scatter(
                    x=vals[group_mask, 0],
                    y=vals[group_mask, 1],
                    mode='markers',
                    marker=dict(symbol='circle', size=14, color=color),
                    name="",  # Legend entry
                    #legendgroup=str(group_name),  # Group items in the legend
                    showlegend=True,  # Show legend for this trace
                    text="",  # Text to display on hover
                    hovertemplate=''  # Display only custom text
                )
            )
        print(len(groups))
        print(groups[1])
        for group_name, color in group_colors_original.items():
            group_mask = [g == group_name for g in g_names]
            for i, is_in_group in enumerate(group_mask):
                if is_in_group:  # Check if the point belongs to the current group
                    legend_sample = groups[i]  # Get the corresponding sample name
                    traces.append(
                        go.Scatter(
                            x=[vals[i, 0]],  # Single x-coordinate for the point
                            y=[vals[i, 1]],  # Single y-coordinate for the point
                            mode='markers',
                            marker=dict(symbol='cross', size=8, color=color),
                            name=legend_sample,  # Unique legend entry for this point
                            showlegend=False,  # Show legend for each point
                            text=[legend_sample],  # Text to display on hover
                            hovertemplate='%{text}'  # Display only custom text
                        )
                    )

        fig = go.Figure(data=traces)
        fig.update_layout(
            #xaxis_title='X-axis',
            #yaxis_title='Y-axis',
            yaxis=dict(showgrid=False)
        )
        
        # Display the plot in Streamlit
        st.plotly_chart(fig)
       



    
