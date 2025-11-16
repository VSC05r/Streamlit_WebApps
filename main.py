#VSC
#https://www.datacamp.com/tutorial/streamlit?utm_aid=157156376311&utm_loc=9217992-&utm_mtd=-c&utm_kw=&gad_campaignid=19589720824
#streamlit run Streamlit_working main.py
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import streamlit as st
import pandas as pd
import numpy as np
import pyexpat 
import matplotlib.pyplot as plt
#import confuision_matrix_display
from sklearn.metrics import ConfusionMatrixDisplay
from pyexpat import model
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
#from sklearn.metrics import plot_confusion_matrix, RocCurveDisplay, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.datasets import make_classification


# def main():
#     st.title("Binary Classificatin Web App")

st.title("Binary Classificatin Web App")
st.text("Binary Classificatin Web App")
st.markdown("Are you possionous mushroom or not?")
st.sidebar.markdown("Are you possionous mushroom or not?")


@st.cache(persist=True)
def load_data():
    #data = pd.read_csv("D:/PythonEnv/Streamlit_working/Mushroom.csv")
    data = pd.read_csv("D:/PythonEnv/Streamlit_working/Mushroom.csv", header=0)
    labels = LabelEncoder()
    for col in data.columns:
        data[col] = labels.fit_transform(data[col])
    return data

@st.cache(persist=True)
def split(df):
    #y = df['ty']
    #y = df.ty
    y = df.iloc[:,0]
    x = df.drop(columns=["ty"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test

def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
        #plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()
        
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        RocCurveDisplay.from_estimator(model, x_test, y_test)
        #plot_roc_curve(model, x_test, y_test)
        st.pyplot()        
        
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
        #plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()
        
    #    st.write(confusion_matrix(y_test, model.predict(x_test)))
    # if "Accuracy" in metrics_list:
    #     st.subheader("Accuracy")
    #     st.write(accuracy_score(y_test, model.predict(x_test)))
    # if "Precision" in metrics_list:
    #     st.subheader("Precision")
    #     st.write(precision_score(y_test, model.predict(x_test)))
    # if "Recall" in metrics_list:
    #     st.subheader("Recall")
    #     st.write(recall_score(y_test, model.predict(x_test)))

df = load_data()
x_train, x_test, y_train, y_test = split(df)
class_names = ["edible", "poisonous"]
st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

#SVM

if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_SVM")
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
    gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")
    
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Support Vector Machine (SVM) Results")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(x_train, y_train)
        #y_pred = model.predict(x_test)
        #accuracy = accuracy_score(y_test, y_pred)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        #st.write("Accuracy: ", accuracy.round(2))
        st.write("Accuracy: ", accuracy)
        #st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
        #st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
        plot_metrics(metrics)
        #plot_metrics(["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"])


#Logistic Regression

if classifier == "Logistic Regression":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key="max_iter")
    #kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
    #gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")
    
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(x_train, y_train)
        #y_pred = model.predict(x_test)
        #accuracy = accuracy_score(y_test, y_pred)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        #st.write("Accuracy: ", accuracy.round(2))
        st.write("Accuracy: ", accuracy)
        #st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
        #st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
        plot_metrics(metrics)
        #plot_metrics(["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"])
        
        
#RandomForest

if classifier == "Random Forest":
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators = st.sidebar.number_input("Number of trees in the forest", 100, 1000, step=10, key="n_estimators")
    max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1, key="max_depth")
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
    #kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
    #gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")
    
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap== "True")
        model.fit(x_train, y_train)
        #y_pred = model.predict(x_test)
        #accuracy = accuracy_score(y_test, y_pred)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        #st.write("Accuracy: ", accuracy.round(2))
        st.write("Accuracy: ", accuracy)
        #st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
        #st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
        plot_metrics(metrics)
        #plot_metrics(["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"])

if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Mushroom Data Set (Classification)")
    st.write(df)
    