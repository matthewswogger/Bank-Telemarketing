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


def tp_tn_fn_fp(true_value, predicted_value):
    tp, tn, fn, fp = 0, 0, 0, 0
    for v, p in zip(true_value, predicted_value):
        if v == p:
            if v+p == 2: tp += 1
            else: tn += 1
        elif v == 1: fn += 1
        else: fp += 1
    return tp, tn, fn, fp

def lift_score(true_value, predicted_value):
    tp, tn, fn, fp = tp_tn_fn_fp(true_value, predicted_value)
    return ( tp/(tp+fp) ) / ( (tp+fn) / (tp+tn+fp+fn) )


def computeLift(res, arr, r):
    population = res.target.value_counts()
    population_ratio = population.loc[1.0]/population.sum()

    l = []
    for i in arr:
        d = res.loc[res.probability >= i,:]
        sample = d.target.value_counts()
        sample_ratio = sample.loc[1.0]/sample.sum()
        l.append([round(i, ndigits=r), sample_ratio/population_ratio, f'Y:{sample.loc[1.0]}, N:{sample.loc[0.0]}'])
    above_prob = pd.DataFrame(l, columns=['above_probability', 'lift', 'count'])
    return above_prob


@st.cache
def getData():
    df = pd.read_csv('/usr/src/app/data/test.csv')
    X = df.drop(columns='y')
    y = df.y
    return X, y


def loadModel():
    return joblib.load('/usr/src/app/saved_models/rand_forest_trees_20_depth_20.joblib')


def main():
    # _max_width_()

    x = st.sidebar.slider('x')
    st.write(x, 'squared is', x * x)

    add_slider = st.sidebar.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0)
    )

    st.title('Vendor Quality Tracking')

    X, y = getData()
    st.write(X.head())

    rand_forest = loadModel()

    y_predict = rand_forest.predict(X)
    y_proba = rand_forest.predict_proba(X)[:,1]

    auc = f'AUC Score: {roc_auc_score(y, y_proba)}'
    st.text(auc)

    c_report = classification_report(y, y_predict)
    st.text(c_report)

    lift = f'Lift for all holdout data: {lift_score(y, y_predict)}'
    st.text(lift)

    c_name = ['target', 'probability', 'prediction']

    tpp = np.column_stack((y.to_numpy(), y_proba, y_predict))
    tpp = pd.DataFrame(tpp, columns=c_name)

    thresholds = np.arange(0, 1, .1)
    round_prob = 1
    lift = computeLift(tpp, thresholds, round_prob)


    # fig = px.histogram(data['Start station'], x ='Start station')
    # st.plotly_chart(fig)

    # cm = confusion_matrix(y_holdout, clf.predict(X_holdout))
    # cm = pd.DataFrame(cm, columns=['no', 'yes'], index=['no', 'yes'])
    # cm.columns.name = 'Predicted'
    # cm.index.name = 'True'

    # cm_norm = (cm.T/cm.T.sum()).T

    # visitors_array = cm_norm.round(3).to_numpy()[::-1, :]


    # Weekdays_list = ['no', 'yes']
    # Hours_list = ['yes', 'no']

    # layout_heatmap = go.Layout(
    #     title=('Confusion Matrix'),
    #     xaxis=dict(title='Predicted'),
    #     yaxis=dict(title='True', dtick=1)
    # )

    # ff_fig = ff.create_annotated_heatmap(x= Weekdays_list, y=Hours_list, z=visitors_array, showscale = True, colorscale='Blues')
    # fig  = go.FigureWidget(ff_fig)
    # fig.layout=layout_heatmap
    # fig.layout.annotations = ff_fig.layout.annotations
    # fig.data[0].colorbar = dict(title='', titleside = 'right')
    # st.plotly_chart(fig)


if __name__ == "__main__":
    main()
