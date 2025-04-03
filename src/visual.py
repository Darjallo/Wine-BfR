# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 19:41:14 2025

@author: Daria Savvateeva
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt 
from matplotlib.markers import MarkerStyle
from matplotlib.colors import to_hex
import plotly.graph_objects as go

def vis_params():
    """
    define all necessary params for figures
    """
    if "groups" in st.session_state:
        groups = st.session_state["groups"]
    else:
        groups = []
    if "class_categories" in st.session_state:  
        categories = st.session_state["class_categories"]
        unique_categories = np.unique(categories)
    else:
        categories = []
        unique_categories = []
 
    # GROUP is a known labes of the sample
    # CATEGORY is a class defined by a classification algorithm
    
    unique_groups = list(set(groups))  # Get unique group labels
    group_colors = {group: plt.cm.tab10(i) for i, group 
                    in enumerate(unique_groups)}  # Assign colors
    
    # Map each group label to its color
    colors = [group_colors[group] for group in groups]
    plotly_colors = [to_hex(group_colors[group]) for group in groups]
    
    # Get all valid marker symbols (excluding some non-usable ones)
    plt_markers = [m for m in MarkerStyle.markers.keys()
                   if isinstance(m, str) and len(m) == 1 and
                   m not in {' ', '', '.', ','}]
    
    plotly_markers = [
        "circle", 
        "square", 
        "diamond",
        "cross", 
        "x", 
        "triangle-up", 
        "triangle-down", 
        "triangle-left",
        "triangle-right",
        "pentagon", 
        "hexagon", 
        "hexagon2", 
        "star",
        "hexagram",
        "star-triangle-up", 
        "star-square", 
        "star-diamond", 
        "hourglass", 
        "bowtie", 
    ]
    
    
    # Assign a unique marker to each group
    category_markers = {
        category: plotly_markers[i % len(plotly_markers)]
        for i, category in enumerate(unique_categories)
    }
    markers = [category_markers[cat] for cat in categories]
    return colors, plotly_colors, unique_groups, group_colors, markers


def dim_red_visual(x, y, legend_title, fig_title):
    """
    Plot data after dimention reduction, 
    title is dimention reduction algorithm
    legends are known labels
    """
    colors, plotly_colors, unique_groups, group_colors, markers = vis_params()

    fig, ax = plt.subplots()
    ax.scatter(
        x, y,
        c=colors,
        edgecolors="none",
        alpha=0.6   
    )
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=group_colors[group], alpha=0.6, 
                          markersize=10) 
       for group in unique_groups]
   
    ax.legend(handles, unique_groups, title=legend_title)
    ax.set_title(fig_title)   
    st.pyplot(fig)
    
def clustering_visual(x, y, fig_title):
    colors, plotly_colors, unique_groups, group_colors, markers = vis_params()

    metas = [f"Point {i}" for i in range(len(x))]
    #data_colors = [colors.to_hex(c) for c in colors]
    fig = go.Figure()
    for x_i, y_i, meta, color, marker in zip(x, y, metas, plotly_colors, markers):
        fig.add_trace(go.Scatter(
            x=[x_i],
            y=[y_i],
            mode='markers',
            marker=dict(
                color=color,
                symbol=marker,
                size=20,
                opacity=0.6,
            ),
            name=meta,
            hovertext=meta,
            hoverinfo='text'
        ))
    fig.update_layout(
    title=fig_title,
    title_x=0.5,  # Center the title
    xaxis_title="",
    yaxis_title="",
    showlegend=False,
    autosize=True,  # Let Plotly auto size the plot
    width=900,  # You can adjust the size as needed
    height=900,  # Same width and height for a square shape
    plot_bgcolor="white",  # Background color of the plot area
    paper_bgcolor="white",  # Background color of the paper
    shapes=[dict(
        type="rect",  # Shape type is rectangle
        x0=0, x1=1, y0=0, y1=1,  # Coordinates for the full frame
        xref="paper", yref="paper",  # Reference to the entire figure
        line=dict(color="blue", width=2)  # Frame color and width
    )],
    )
    fig.update_xaxes(scaleanchor="y", showgrid=False, zeroline=False, showticklabels=False)  # Hide x-axis gridlines, zero line, and ticks
    fig.update_yaxes(scaleanchor="x", showgrid=False, zeroline=False, showticklabels=False)  # Hide y-axis gridlines, zero line, and ticks
    
    #__________________________________________________________________________
    # Adding alternative "legend-like" annotations with symbols
    
    anns = [{'y':1-0.5*i, 'text':'class '+str(i+1),} for i, m in enumerate(set(markers))]
    for ann in anns:
        fig.add_annotation(
            x=0,
            y=ann['y'],
            text=ann['text'],
            showarrow=False,
            font=dict(size=24, color="black"),
            align='left'
        )
    # fig.add_annotation(
    #     x=0.8, y=0.9,  # Position of the annotation (relative to the figure)
    #     text="Red: Circle\nBlue: Square\nGreen: Diamond\nOrange: Cross",  # Text to simulate a legend
    #     showarrow=False,
    #     font=dict(size=12, color="black"),
    #     align="left",
    #     borderpad=5,
    #     bgcolor="rgba(255, 255, 255, 0.6)",  # Background color of the annotation box
    #     bordercolor="black",  # Border color for the box
    #     borderwidth=2  # Border width for the annotation box
    # )

    # Adding symbols (e.g., circles, squares, diamonds, crosses) next to the annotation text
    symbol_annotations = [{"symbol": m, "x": -1, "y": 1-0.5*i, "color": "blue"} for i, m in enumerate(set(markers))]

    for annotation in symbol_annotations:
        fig.add_trace(go.Scatter(
            x=[annotation['x']], 
            y=[annotation['y']],
            mode='markers',
            marker=dict(
                color=annotation['color'],
                symbol=annotation['symbol'],
                size=24
            ),
            showlegend=False  # Hide from the main legend
        ))

    st.plotly_chart(fig)
    


