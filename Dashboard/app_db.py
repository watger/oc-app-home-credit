import streamlit as st
from streamlit_shap import st_shap
import dvc.api

import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io

from io import BytesIO
import pickle
import requests

import shap
from shap import KernelExplainer
from shap import TreeExplainer
from shap import Explainer
import streamlit.components.v1 as components

from flask import Flask, render_template, jsonify
import json
import requests

from Model import HomeCreditRisk, HCRModel

# 2. Create app and model objects
model = HCRModel()

st.set_page_config(
    page_title="Dashboard Prêt à dépenser",
    #page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        #'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
def predict_risks(id_client):
    response = requests.get(
        "http://0.0.0.0:5000/predict?id_client="+str(id_client))
    content = json.loads(response.content.decode('utf-8'))
    
    if response.status_code != 200:
        return jsonify({
            'status': 'error',
            'message': 'La requête à l\'API n\'a pas fonctionné. Voici le message renvoyé par l\'API : {}'.format(content['message'])#
        }), 500
    return response,content

def load_data():
    #db_test = joblib.load('../data.csv')
    db_test = model.df
    #st.write(db_test.shape)

    db_test['YEARS_BIRTH']=(db_test['DAYS_BIRTH']/-365).apply(lambda x: int(x))
    db_test=db_test.reset_index(drop=True)
    
    db_test["age_cat"] = pd.cut(db_test.YEARS_BIRTH,
                                bins=[0, 20, 30, 40, 50, 60, 70, 80, 120],
                                labels=["0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
                               )
    
    db_test['YEARS_EMPLOYED']=(db_test['DAYS_EMPLOYED']/-365).apply(lambda x: float(x))
    db_test["employ_cat"] = pd.cut(db_test.YEARS_EMPLOYED,
                                bins=[-1, 1, 2, 5, 10, 15, 20, 30, 40, 120],
                                labels=["less than 1 year", "1-2 years", "2-5 years", "5-10 years", "11-15 years", "15-20 years", "20-30 years", "30-40 years", "more than 40 years"]
                               )
    
    db_test["employ_perc_cat"] = pd.cut(db_test.DAYS_EMPLOYED_PERC,
                                bins=[-0.1, 0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1],
                                labels=["less than 5%", "5-10%", "10-15%", "15-20%", "20-25%", "25-40%", "40-50%", "50-60%", "60-75%", "75-80%", "more than 80%"]
                               )
    
    
    db_test["income_cat"] = pd.cut(db_test.INCOME_PER_PERSON,
                                bins=[0, 10000, 50000, 75000, 110000, 9000000],
                                labels=["less than 10,000", "10,000-50,000", "50,000-75,000", "75,000-110,000", "more than 110,000"]
                               )
    
    #st.write(db_test.shape)
    #lgbm = joblib.load('../model_1.joblib')
    #lgbm = model.model
    
    col_to_drop = ["TARGET","SK_ID_CURR","YEARS_BIRTH","age_cat","YEARS_EMPLOYED","employ_cat","employ_perc_cat","income_cat"]
    
    X = db_test.drop(columns=col_to_drop)#
    #y = db_test.TARGET
    
    #st.write(X.shape)
    #lgbm.fit(X,y)
    
    
    
    #Calcul des shap_values
    shap_val_fp, expected_value_fp, shap_val_wf = shap_val(model.model, X)
       
    
    return db_test, X, col_to_drop, shap_val_fp, expected_value_fp, shap_val_wf#, y, lgbm

def shap_val(model, X):
    #Calcul des SHAP values
    #Waterfall
    explainer_wf = shap.Explainer(model, X)
    shap_values_wf = explainer_wf(X)
    
    #Forceplot
    explainer_fp = shap.TreeExplainer(model)
    shap_values_fp = explainer_fp.shap_values(X)[1]
    expected_value_fp = explainer_fp.expected_value[1]
    #shap_values = joblib.load('C:/Users/willi/Desktop/WILLIAM/E_LEARNING/OC_Projects/OC_Projects/Projet 7/shap_values_200.joblib')
    
    """
    #Force plot
    cls_explainer = TreeExplainer(model.model)
    cls_shap_values = cls_explainer.shap_values(X)[1]
    cls_expected_values = cls_explainer.expected_value[1]
    
    newrow = [0]*125
    X_test = np.vstack([X, newrow])
    
    #Waterfall
    explainer = Explainer(model.model,X_test)##data=None#Tree
    #shap_values = cls_explainer.shap_values(X)[0]
    #shap_values = explainer.shap_values(X_test)#[0]
    shap_values = explainer(X_test)#[0]
    #st.write(shap_values)
    #st.write(explainer.shap_values(X_test))
    expected_values = cls_explainer.expected_value[1]
    
    return cls_shap_values, cls_expected_values, shap_values, expected_values
    """
    return shap_values_fp, expected_value_fp, shap_values_wf
    


def color(pred):
    '''Définition de la couleur selon la prédiction'''
    if pred=='Approved':
        col='Green'
    else :
        col='Red'
    return col


data, X, col_to_drop, shap_val_fp, expected_value_fp, shap_val_wf = load_data()#, y, lgbm

customer_list = data.SK_ID_CURR

st.title('Dashboard Prêt à dépenser')


# Using "with" notation
with st.sidebar:
    option = st.selectbox(
        "Customer's identification number :",
        (customer_list))

    st.write('You selected:', option)
    
respond,content = predict_risks(option)



st.subheader('Prediction for the selected customer')

if content["prediction"] == "Approuved":
    color_pred = 'Green'
else :
    color_pred = 'Red'

pred_sentence = '<p>Prediction for Home Credit is <strong style="color:'+color_pred+';">'+ content["prediction"] + '</strong> </p>'
st.markdown(pred_sentence, unsafe_allow_html=True)
st.write('with a Probability of ', content["probability"])#

st.markdown('<p><i style="color:Orange;">The threshold is at 0,91 </i> </p>', unsafe_allow_html=True)
#st.write('tester',respond[2])

#st.write('Home Credit', pred)
#st.write('Probability', prob)

st.subheader('Data of the selected customer')

data_client = data[data.SK_ID_CURR == option]

st.dataframe(data_client)

X_clt = data_client.drop(columns=col_to_drop)

#shap_val_fp, exp_val_fp, shap_val_wf, exp_val_wf = shap_val(X_clt)


#def st_shap(plot, height=None):
#    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#    components.html(shap_html, height=height)


index = data.index[data.SK_ID_CURR == option][0]

#st.write(shap.plots.waterfall(shap_val))

#shap.initjs()

#col1, col2 = st.columns(2)

#with col1 :

st.subheader('Interpretabily of the prediction')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write(index)
st_shap(shap.force_plot(expected_value_fp, shap_val_fp[index,:], X.iloc[index,:]))
#st_shap(shap.force_plot(exp_val_fp, shap_val_fp[0,:], X_clt.iloc[0,:]))
#shap.initjs()

#st.write(shap_val_wf)
#st.pyplot(shap.waterfall_plot(shap_val_wf[0],20))
#st_shap(shap.waterfall_plot(shap_val_wf[0]))#,20#shap_val_wf[0,:]#, 1, show=False
#st.pyplot(shap.waterfall_plot(shap_val_wf[0], 15, show=False))#, matplotlib=True
#st.write(shap.plots.waterfall(shap_val_fp[0,:]))
#st_shap(shap.plots.waterfall(shap_val_wf[0]))
#shap_val_wf.values=shap_val_wf.values[:,:,1]
#shap_val_wf.base_values=shap_val_wf.base_values[:,1]
#st_shap(shap.plots._waterfall.waterfall_legacy(exp_val_wf, shap_val_wf[0]))
#shap_wf = shap.waterfall_plot(shap_val_wf[0], 15, show=False)
#st.write(np.array(shap_val_wf[0]))
#st.write(shap_val_wf[1])
#st.write(shap.plots.bar(np.array(shap_val_wf)[0],max_display=12))
#st.write(shap.bar_plot(np.array(shap_val_wf)[0]))
#st_shap(shap.bar_plot(np.array(shap_val_wf[0])))#
#st.pyplot(shap.plots.bar(shap_val_wf[0]))
#st_shap(shap.summary_plot(shap_val_wf[0,:], X_clt, plot_type="bar"))


st_shap(shap.plots.waterfall(shap_val_wf[0]),500,1000)#, width=1000, height=300

#buf = BytesIO()
#shap_wf.savefig(buf, format="png")
#st.image(buf)

#fig = plt.gcf() # gcf means "get current figure"
#fig.set_figheight(11)
#fig.set_figwidth(5)
#plt.rcParams['font.size'] = '7'
#plt.show()


#with col2 :
st.subheader('Interpretabily of global customers')

#st.markdown("<img style='text-align: right;' src='C:/Users/willi/Desktop/WILLIAM/E_LEARNING/OC_Projects/OC_Projects/Projet 7/shap_summary_plot.png' alt='Features importance for global customers'>", unsafe_allow_html=True)

st.image('shap_summary_plot.png', caption='Features importance for global customers')#, width=400


st.subheader('Comparison by caracteristics')

graph = st.selectbox(
    'By which caracteristic compare the customer ?',
    ('Age', 'Time employed', 'Income'))

if graph == 'Age' :
    #st.write('Graph Age')
    #st.write(data_client.age_cat)
    data_age = data[data.age_cat == data_client.age_cat.to_list()[0]]
    data_age_in = data_age.drop(columns=col_to_drop)
    #st.dataframe(data_age)
    data_age["y_prob"] = model.model.predict_proba(data_age_in)[:,0]#
    #st.dataframe(data_age)
    data_age["y_pred"] = np.where(data_age.y_prob<0.8, 1,0)
    
    
    #plt.figure(figsize=(2, 2))
    colors_age = sns.color_palette('pastel')[2:5]

    if data_age['y_pred'].iloc[0] == 0 :
        labels = ['Approuved', 'Rejected']
    else :
        labels = ['Rejected', 'Approuved']


    plt.pie(data_age.y_pred.value_counts(),
            labels = labels,
            colors = colors_age, 
            autopct='%0.0f%%')

    plt.legend(bbox_to_anchor=(0.95, 0.5), loc='lower left')
    plt.title("Home Credit by Age (" + data_client.age_cat.to_list()[0] + " years old)")
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf)
    
    #st.pyplot(plt,size=50)                  
    #st.write(data_age.shape)
    #st.dataframe(data_age)
    #st.write('You selected:', graph)
if graph == 'Time employed' :
    col1, col2 = st.columns(2)
    
    with col1:
        #st.write(data_client.employ_cat.to_list()[0])
        data_employ = data[data.employ_cat == data_client.employ_cat.to_list()[0]]
        data_employ_in = data_employ.drop(columns=col_to_drop)
        #st.dataframe(data_employ)
        data_employ["y_prob"] = model.model.predict_proba(data_employ_in)[:,0]
        data_employ["y_pred"] = np.where(data_employ.y_prob<0.8, 1,0)


        colors_employ = sns.color_palette('pastel')[4:7]

        if data_employ['y_pred'].iloc[0] == 0 :
            labels = ['Approuved', 'Rejected']
        else :
            labels = ['Rejected', 'Approuved']


        plt.pie(data_employ.y_pred.value_counts(),
                labels = labels,
                colors = colors_employ, 
                autopct='%0.0f%%')

        plt.legend(bbox_to_anchor=(0.95, 0.5), loc='lower left')
        plt.title("Home Credit by time employed (" + data_client.employ_cat.to_list()[0] + ")")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        st.image(buf)
    #st.pyplot(plt)
    
    #Clear the plot
    plt.clf()
    
    with col2 :

        data_employ_perc = data[data.employ_perc_cat == data_client.employ_perc_cat.to_list()[0]]
        data_employ_perc_in = data_employ_perc.drop(columns=col_to_drop)
        data_employ_perc["y_prob"] = model.model.predict_proba(data_employ_perc_in)[:,0]
        data_employ_perc["y_pred"] = np.where(data_employ_perc.y_prob<0.8, 1,0)


        colors_employ = sns.color_palette('pastel')[6:9]

        if data_employ_perc['y_pred'].iloc[0] == 0 :
            labels = ['Approuved', 'Rejected']
        else :
            labels = ['Rejected', 'Approuved']


        plt.pie(data_employ_perc.y_pred.value_counts(),
                labels = labels,
                colors = colors_employ, 
                autopct='%0.0f%%')

        plt.legend(bbox_to_anchor=(0.95, 0.5), loc='lower left')
        plt.title("Home Credit by pourcentage of time employed (" + data_client.employ_perc_cat.to_list()[0] + ")")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        st.image(buf)
    #st.pyplot(plt)
if graph == 'Income' :
    data_income = data[data.income_cat == data_client.income_cat.to_list()[0]]
    data_income_in = data_income.drop(columns=col_to_drop)
    data_income["y_prob"] = model.model.predict_proba(data_income_in)[:,0]
    data_income["y_pred"] = np.where(data_income.y_prob<0.8, 1,0)
    
    
    colors_income = sns.color_palette('pastel')[8:11]

    if data_income['y_pred'].iloc[0] == 0 :
        labels = ['Approuved', 'Rejected']
    else :
        labels = ['Rejected', 'Approuved']


    plt.pie(data_income.y_pred.value_counts(),
            labels = labels,
            colors = colors_income, 
            autopct='%0.0f%%')

    plt.legend(bbox_to_anchor=(0.95, 0.5), loc='lower left')
    plt.title("Home Credit by income per person (" + data_client.income_cat.to_list()[0] + ")")
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf)
    
    
default_time_emp = int(data_client.YEARS_EMPLOYED)
time_emp = st.slider('How many years of employment?', 0, 60, default_time_emp)