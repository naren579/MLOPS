import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
import streamlit as st
import pickle


import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
# Streamlit app

st.title("Upload the test data")

# User uploads a file
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
button_pressed=st.button("Predict")
# Check if a file is uploaded
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(uploaded_file)
        X,y=df.drop('Survived',axis=1),df['Survived']
        with open('/Titanic/model_titanic.pkl', 'rb') as f:
            clf = pickle.load(f)
        if button_pressed:
            pred_proba=clf.predict_proba(X)[:,1]
            y_pred=pd.DataFrame(clf.predict(X))
            df_pred=pd.concat([X,y_pred],axis=1)
            st.write('The AUC_ROC Score is:',roc_auc_score(y,pred_proba))
            st.table(df_pred)


    except pd.errors.EmptyDataError:
        st.warning("Uploaded file is empty. Please upload a valid CSV file.")
    except pd.errors.ParserError:
        st.warning("Error parsing the CSV file. Please check the file format.")

# with open('model_titanic.pkl', 'rb') as f:
#     clf = pickle.load(f)
