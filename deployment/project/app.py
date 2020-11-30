import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (make_scorer,
                             roc_auc_score,
                             roc_curve,
                             f1_score,
                             recall_score,
                             precision_score,
                             precision_recall_curve,
                             classification_report,
                             plot_confusion_matrix,
                             plot_roc_curve,
                             confusion_matrix)

# import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

@st.cache
def getData():
    return pd.read_csv('/usr/src/app/data/test.csv')


_max_width_()

st.title('Vendor Quality Tracking')

df = getData()

st.write(df.head())

X_holdout = df.drop(columns='y')
y_holdout = df.y


clf = joblib.load('/usr/src/app/saved_models/random_forest_feature_selected.joblib')

c_report = classification_report(y_holdout, clf.predict(X_holdout))
st.text(c_report)

# fig = px.histogram(data['Start station'], x ='Start station')
# st.plotly_chart(fig)

cm = confusion_matrix(y_holdout, clf.predict(X_holdout))
cm = pd.DataFrame(cm, columns=['no', 'yes'], index=['no', 'yes'])
cm.columns.name = 'Predicted'
cm.index.name = 'True'

cm_norm = (cm.T/cm.T.sum()).T

cm_norm

visitors_array = cm_norm.round(3).to_numpy()[::-1, :]

#display(nparray)
Weekdays_list = ['no', 'yes'] 
Hours_list = ['yes', 'no']

layout_heatmap = go.Layout(
    title=('Confusion Matrix'),
    xaxis=dict(title='Predicted'), 
    yaxis=dict(title='True', dtick=1)
)

ff_fig = ff.create_annotated_heatmap(x= Weekdays_list, y=Hours_list, z=visitors_array, showscale = True, colorscale='Blues')
fig  = go.FigureWidget(ff_fig)
fig.layout=layout_heatmap
fig.layout.annotations = ff_fig.layout.annotations
fig.data[0].colorbar = dict(title='', titleside = 'right')
st.plotly_chart(fig)

# df = pd.read_csv('/usr/src/app/data/two_vendor_differences.csv')

# vendor = st.selectbox(
#     'Which Vendor would you like to inspect?',
#     ('Vendor 1', 'Vendor 2')
# )
# st.write('You selected:', vendor)

# fig = go.Figure()

# if vendor == 'Vendor 1':
#     for row in df.itertuples():
#         fig.add_scatter3d(
#             x=[row.x, row.x_vendor1],
#             y=[row.y, row.y_vendor1],
#             z=[row.z, row.z_vendor1],
#             showlegend=False,
#             marker=dict(
#                 size=1,
#                 color='cyan',
#             ),
#             line=dict(
#                 width=10,
#                 color='cyan',
#             ),
#         )

#     fig.add_scatter3d(
#         x=df.x_vendor1,
#         y=df.y_vendor1,
#         z=df.z_vendor1,
#         showlegend=False,
#         marker=dict(
#             size=((df.x_diff_vendor1**2 + df.y_diff_vendor1**2 + df.z_diff_vendor1**2)**.5),
#             color='red',
#         ),
#     )
# else:
#     for row in df.itertuples():
#         fig.add_scatter3d(
#             x=[row.x, row.x_vendor2],
#             y=[row.y, row.y_vendor2],
#             z=[row.z, row.z_vendor2],
#             showlegend=False,
#             marker=dict(
#                 size=1,
#                 color='cyan',
#             ),
#             line=dict(
#                 width=10,
#                 color='cyan',
#             ),
#         )

#     fig.add_scatter3d(
#         x=df.x_vendor2,
#         y=df.y_vendor2,
#         z=df.z_vendor2,
#         showlegend=False,
#         marker=dict(
#             size=((df.x_diff_vendor2**2 + df.y_diff_vendor2**2 + df.z_diff_vendor1**2)**.5),
#             color='red',
#         ),
#     )

# fig.add_scatter3d(
#     x=df.x,
#     y=df.y,
#     z=df.z,
#     showlegend=False,
#     marker=dict(
#         size=4,
#         color='midnightblue',
#     ),
# )

# fig.update_layout(
#     autosize=False,
#     width=1600,
#     height=800,
#     scene_camera_eye=dict(
#         x=3,
#         y=3,
#         z=3
#     )
# )

# st.plotly_chart(fig)

