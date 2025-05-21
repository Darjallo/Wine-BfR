# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 19:41:14 2025

@author: Daria Savvateeva
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt 
from matplotlib.markers import MarkerStyle
from matplotlib.colors import to_hex
import plotly.express as px
import plotly.graph_objects as go


def vis_params(label):
    """
    define all necessary params for figures
    """
    if "main_label" in st.session_state:
        d=st.session_state['df_raw']
        c=st.session_state["main_label"]
        
        if isinstance(c, list):
            c = c[0]     

        if label == '': 
            s = d[c]
        else:
            s = d[label]

        groups = s.to_list()

        
    else:
        groups = []
    if "class_categories" in st.session_state:  
        categories = st.session_state["class_categories"]
        #categories = categories.to_list()
        #categories = ['noise' if cat == -1 else cat for cat in categories]
        #unique_categories = np.unique(categories)
        unique_categories = list(set(categories))
        #unique_categories = unique_categories.tolist()
        # Replace -1 with 'noise'
        #unique_categories = ['noise' if cat == -1 else cat for cat in unique_categories]
        #st.write(unique_categories)
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
    # check that the first category is noise
    return colors, plotly_colors, unique_groups, group_colors, markers, categories

def square_fig(x, y):
    """
    Takes x and y values, defines the coordinates of a square frame around x and y
    """
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_mid = (x_min + x_max)/2
    y_mid = (y_min + y_max)/2
    
    x_half = x_mid - x_min
    y_half = y_mid - y_min
    bigger = max(x_half, y_half)*1.05
    
    
    x_range=[x_mid-bigger, x_mid+bigger]
    y_range=[y_mid-bigger, y_mid+bigger]
    
    y_delta = 0.1*(y_max-y_min)
    x_delta = 0.1*(x_max-x_min)
    return x_range, y_range, x_delta, y_delta, x_max, y_max

def dim_red_visual(x, y, legend_title, fig_title):
    """
    Plot data after dimention reduction, 
    title is dimention reduction algorithm
    legends are known labels
    """
    colors, plotly_colors, unique_groups, group_colors, markers, _ = vis_params('')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            color=plotly_colors,
            symbol='circle',
            size=20,
            opacity=0.6,
        ),
        #name=meta,
        #hovertext=meta,
        #hoverinfo='text'
    ))
        
    x_range, y_range, x_delta, y_delta, x_max, y_max = square_fig(x, y)

    fig.update_layout(width=700, height=600)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(scaleanchor="y", showgrid=False, zeroline=False, 
                     showticklabels=False, range=x_range, scaleratio=1) 
    fig.update_yaxes(scaleanchor="x", showgrid=False, zeroline=False, 
                     showticklabels=False, range=y_range, scaleratio=1) 




# LEGEND
    #Adding annotations for known labels
    anns = [{'y':y_max-y_delta*i, 'text':gr,} for i, gr in enumerate(unique_groups)]
    for ann in anns:
        fig.add_annotation(
            x=x_max + 2.5*x_delta,
            y=ann['y'],
            text=ann['text'],
            showarrow=False,
            font=dict(size=24, color="black"),
            align='left',
            xanchor='left' 
        )
        
    symbol_annotations = [{ "y": y_max-y_delta*i, "color": group_colors[gr]} for i, gr in enumerate(unique_groups)]
    #unique_groups, group_colors
    for annotation in symbol_annotations:
        fig.add_trace(go.Scatter(
            x=[x_max+2*x_delta], 
            y=[annotation['y']],
            #xref='paper', yref='paper',
            mode='markers',
            marker=dict(
                color=to_hex(annotation['color']),
                symbol='circle',
                size=20
            ),
            showlegend=False  # Hide from the main legend
        ))
    # Show plot in Streamlit
    st.plotly_chart(fig)


#____________
    # fig, ax = plt.subplots()
    # ax.scatter(
    #     x, y,
    #     c=colors,
    #     edgecolors="none",
    #     alpha=0.6   
    # )
    # handles = [plt.Line2D([0], [0], marker='o', color='w', 
    #                       markerfacecolor=group_colors[group], alpha=0.6, 
    #                       markersize=10) 
    #    for group in unique_groups]
   
    # ax.legend(handles, unique_groups, title=legend_title)
    # ax.set_title(fig_title)   
    # st.pyplot(fig)
    
def clustering_visual(x, y, fig_title, label):

    colors, plotly_colors, unique_groups, group_colors, markers, categories = vis_params(label)
    m_c = list(set(list(zip(markers, categories))))
    
    border_dict = dict(color='Black', width=2)
    borders = [border_dict if cat == -1 else {} for cat in categories]
    
    if st.session_state["hover_label"]=='':
        metas = [f"Point {i}" for i in range(len(x))]
    else:
        d=st.session_state['df_raw']
        c=st.session_state["hover_label"]
        metas = d[c]
    #data_colors = [colors.to_hex(c) for c in colors]
    fig = go.Figure()
    for x_i, y_i, meta, color, marker, b in zip(x, y, metas, plotly_colors, markers, borders):
        fig.add_trace(go.Scatter(
            x=[x_i],
            y=[y_i],
            mode='markers',
            marker=dict(
                color=color,
                symbol=marker,
                size=20,
                opacity=0.6,
                line=b,
            ),
            name=meta,
            hovertext=meta,
            hoverinfo='text'
        ))
      
    #__________________________________________________________________________
    # Adding alternative "legend-like" annotations with symbols
    

    # find min and max for x and y to define relative coordinates
    x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
    x_rel = x_min - 0.4 * (x_max-x_min)
    x_delta = 0.1*(x_max-x_min)
    y_rel = y_min + 1 * (y_max-y_min)
    y_delta = 0.1*(y_max-y_min)
    
    
    fig.update_layout(
    #title=fig_title,
    #title_x=0.5,  # Center the title
    xaxis_title="",
    yaxis_title="",
    showlegend=False,
    autosize=False,  # Let Plotly auto size the plot
    width=900,  # You can adjust the size as needed
    height=900,  # Same width and height for a square shape
    plot_bgcolor="white",  # Background color of the plot area
    paper_bgcolor="white",  # Background color of the paper
    # shapes=[dict(
    #     type="rect",  # Shape type is rectangle
    #     x0=x_min-x_delta, x1=1.1*x_max, y0=y_min-y_delta, y1=1.1*y_max,  # Coordinates for the full frame
    #     line=dict(color="blue", width=2)  # Frame color and width
    # )],
    )
    fig.update_xaxes(scaleanchor="y", showgrid=False, zeroline=False, showticklabels=False, scaleratio=1)  # Hide x-axis gridlines, zero line, and ticks
    fig.update_yaxes(scaleanchor="x", showgrid=False, zeroline=False, showticklabels=False, scaleratio=1)  # Hide y-axis gridlines, zero line, and ticks
    
    

    annotations = []
    for i, (m, cat) in enumerate(m_c):
        text = 'noise' if cat == -1 else f'class {i + 1}'

        annotations.append({"symbol": m, 
                            "symbol_x": x_rel, 
                            "symbol_y": y_rel-y_delta*i, 
                            "color": "grey",
                            'text_y': y_rel - y_delta * i, 
                            'text': text
                            })

    for ann in annotations:
        # Texts
        fig.add_annotation(
            x=x_rel + 0.5*x_delta,
            y=ann['text_y'],
            text=ann['text'],
            showarrow=False,
            font=dict(size=24, color="black"),
            align='left',
            xanchor='left' 
        )

        # Symbols:
        fig.add_trace(go.Scatter(
            x=[ann['symbol_x']], 
            y=[ann['symbol_y']],
            mode='markers',
            marker=dict(
                color=ann['color'],
                symbol=ann['symbol'],
                size=24,
            ),
            showlegend=False  # Hide from the main legend
        ))
    
    #__________________________________________________________________________
    #Adding annotations for known labels
    anns = [{'y':y_rel-y_delta*i, 'text':gr,} for i, gr in enumerate(unique_groups)]
    for ann in anns:
        fig.add_annotation(
            x=x_max + 2.5*x_delta,
            y=ann['y'],
            text=ann['text'],
            showarrow=False,
            font=dict(size=24, color="black"),
            align='left',
            xanchor='left' 
        )
        
    symbol_annotations = [{ "y": y_rel-y_delta*i, "color": group_colors[gr]} for i, gr in enumerate(unique_groups)]
    #unique_groups, group_colors
    for annotation in symbol_annotations:
        fig.add_trace(go.Scatter(
            x=[x_max+2*x_delta], 
            y=[annotation['y']],
            #xref='paper', yref='paper',
            mode='markers',
            marker=dict(
                color=to_hex(annotation['color']),
                symbol='circle',
                size=24
            ),
            showlegend=False  # Hide from the main legend
        ))

    st.plotly_chart(fig, use_container_width=True)
    

    


