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


def dim_red_visual(x, y, legend_title, fig_title):
    """
    Plot data after dimention reduction, 
    title is dimention reduction algorithm
    legends are known labels
    """
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
                size=10
            ),
            name=meta,
            hovertext=meta,
            hoverinfo='text'
        ))
    fig.update_layout(
    title=fig_title,
    xaxis_title="",
    yaxis_title="",
    showlegend=False
)
    st.plotly_chart(fig)
    


