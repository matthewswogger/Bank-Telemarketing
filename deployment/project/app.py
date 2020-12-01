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
    df = pd.read_csv('/usr/src/app/data/test.csv')
    X = df.drop(columns='y')
    y = df.y
    return X, y


def loadModel():
    return joblib.load('/usr/src/app/saved_models/random_forest_feature_selected.joblib')


def main():
    # _max_width_()
    choose_model = st.sidebar.selectbox("Choose the ML Model", ["NONE","Decision Tree", "Neural Network", "K-Nearest Neighbours"])

    st.title('Vendor Quality Tracking')

    X_holdout, y_holdout = getData()
    st.write(X_holdout.head())

    clf = loadModel()

    c_report = classification_report(y_holdout, clf.predict(X_holdout))
    st.text(c_report)

    # fig = px.histogram(data['Start station'], x ='Start station')
    # st.plotly_chart(fig)

    cm = confusion_matrix(y_holdout, clf.predict(X_holdout))
    cm = pd.DataFrame(cm, columns=['no', 'yes'], index=['no', 'yes'])
    cm.columns.name = 'Predicted'
    cm.index.name = 'True'

    cm_norm = (cm.T/cm.T.sum()).T

    visitors_array = cm_norm.round(3).to_numpy()[::-1, :]


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


if __name__ == "__main__":
    main()
