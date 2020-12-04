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
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.graph_objects as go
# import plotly.express as px
# import plotly.graph_objs as go
# import plotly.figure_factory as ff

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


def computeProfitCurve(res, arr, r):
    l = []
    for i in arr:
        exp, act = res.loc[res.probability >= i, ['expected_value', 'actual_value']].sum()
        l.append([round(i, ndigits=r), exp, act])
    above_prob = pd.DataFrame(l, columns=['above_probability', 'expected_profit', 'actual_profit'])
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

    cost_per_call = st.sidebar.slider('Cost Per Call')
    # st.write(x, 'squared is', x * x)

    profit_per_call = st.sidebar.slider(
        'Profit per Yes',
        # 0.0, 200.0, (25.0, 75.0)
        0.0, 200.0,
    )

    yes_profit = profit_per_call - cost_per_call

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

    thresholds = np.arange(0, 1, .1)
    round_prob = 1
    lift = computeLift(tpp, thresholds, round_prob)

    f, ax1 = plt.subplots(figsize=(20, 5))

    sns.barplot(data=lift, x='above_probability', y='lift', ax=ax1)
    for row in lift.itertuples():
        ax1.text(row.Index, row.lift, row.count, color='black', ha="center")
    ax1.hlines(1, -1,10)
    ax1.set_title('Holdout Set Lift')

    st.pyplot(f)

    thresholds = np.arange(0, 1, .01)
    round_prob = 2
    lift = computeLift(tpp, thresholds, round_prob)

    f, ax = plt.subplots(figsize=(20,5))
    sns.lineplot(data=lift, x='above_probability', y='lift', ax=ax)
    ax.set_title('Holdout Set Lift')

    st.pyplot(f)

    # cost_per_call = 5
    # yes_profit = 35 - cost_per_call

    func = lambda p: p*yes_profit - (1-p)*cost_per_call

    tpp['expected_value'] = tpp.probability.apply(func)
    tpp['actual_value'] = tpp.target.apply(func)

    # st.text('For Profit of > $0.00, target customers above this probablility:', tpp.loc[tpp.expected_value>0, 'probability'].min()*100)
    # st.text('For Profit of > $5.00, target customers above this probablility:', tpp.loc[tpp.expected_value>5, 'probability'].min()*100)
    # st.text('For Profit of > $10.00, target customers above this probablility:', tpp.loc[tpp.expected_value>10, 'probability'].min()*100)

    thresholds = np.arange(0, 1, .01)
    round_prob = 2
    profit = computeProfitCurve(tpp, thresholds, round_prob)

    f, ax1 = plt.subplots(figsize=(20,5))
    sns.lineplot(data=profit, x='above_probability', y='expected_profit', ax=ax1)
    sns.lineplot(data=profit, x='above_probability', y='actual_profit', ax=ax1)
    ax1.legend(['expected_profit', 'actual_profit'])
    ax1.set_title('Holdout Profit Curve')

    st.pyplot(f)


if __name__ == "__main__":
    main()
