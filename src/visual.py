# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 19:41:14 2025

@author: Daria Savvateeva
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt 
from matplotlib.markers import MarkerStyle
import mplcursors

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

# Get all valid marker symbols (excluding some non-usable ones)
all_markers = [m for m in MarkerStyle.markers.keys()
               if isinstance(m, str) and len(m) == 1 and
               m not in {' ', '', '.', ','}]

# Assign a unique marker to each group
category_markers = {
    category: all_markers[i % len(all_markers)]
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
    
    fig, ax = plt.subplots()
    # for category in unique_categories:
    #     # Get indices of points in this category
    #     idx = [i for i, cat in enumerate(categories) if cat == category]
    #     ax.scatter(
    #         [x[i] for i in idx],
    #         [y[i] for i in idx],
    #         c=[colors[i] for i in idx],
    #         marker=category_markers[category],
    #         edgecolors="none",
    #         alpha=0.6,
    #         label=str(category)
    #     )
    
    #ax.legend(title="Category")
    metas = ['seal']  # Metadata for each point
    
    for x_i, y_i, c, m in zip(x, y, colors, markers):
        pl = ax.scatter(x_i, y_i, 
                   c=c, 
                   marker=m, 
                   edgecolors="none",
                   alpha=0.6,
                   )
    cursor = mplcursors.cursor(pl, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        index = sel.index  # Index of the selected point
        sel.annotation.set_text(metas[index])  # Show label
        
    ax.set_title(fig_title)
    st.pyplot(fig)


