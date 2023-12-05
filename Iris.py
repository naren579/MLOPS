import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

st.set_page_config(layout="wide")
#st.set_page_config(page_title="Iris Species Prediction App")

df=sns.load_dataset('iris')
X=df.drop('species',axis=1)
y=df['species']

st.header('Please select the features below')
sepal_length=st.number_input('Sepal_Length:',min_value=0.0, max_value=10.0)
sepal_width=st.number_input('Sepal_Width:',min_value=0.0, max_value=10.0)
petal_length=st.number_input('Petal_Length:',min_value=0.0, max_value=10.0)
petal_width=st.number_input('Petal_Width:',min_value=0.0, max_value=10.0)

X_test=pd.DataFrame(data={'sepal_length':sepal_length,'sepal_width':sepal_width,'petal_length':petal_length,'petal_width':petal_width},index=[0])
st.header('Please select an Algorithm')
algo=st.selectbox('select algorithm',['','DecisionTreeClassifier','RandomForestClassifier','KNeighborsClassifier'],index=0)

param_grid = {'RandomForestClassifier':
    {'n_estimators': [5, 10, 15],
    'max_depth': [3, 5, 9],
    'min_samples_split': [2,4,6,8,10],
    'min_samples_leaf': [1, 2, 4]},

    'DecisionTreeClassifier':{
    'max_depth': [3, 5, 9],
    'min_samples_split': [2,4,6,8,10],
    'min_samples_leaf': [1, 2, 4]},

    'KNeighborsClassifier':{
        'n_neighbors': [3, 5, 7, 9],      
        'weights': ['uniform', 'distance'], 
        'p': [1, 2] 
    }
}

if algo == 'DecisionTreeClassifier':
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid[algo], cv=5, scoring='accuracy')
    grid_search.fit(X,y)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    prediction=best_model.predict(X_test)
    if prediction == 'versicolor':
        st.image('https://i.etsystatic.com/11300650/r/il/23fbbf/1171633447/il_794xN.1171633447_8s97.jpg',width=300,caption="Versicolor", use_column_width=False)
    elif prediction == 'setosa':
        st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/1280px-Irissetosa1.jpg',width=300,caption="setosa", use_column_width=False)
    else:
        st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/1024px-Iris_virginica_2.jpg',width=300,caption="virginica", use_column_width=False)
    st.write(f'The Predicted output is:{prediction}')
    
elif algo == 'RandomForestClassifier':
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid[algo], cv=5, scoring='accuracy')
    grid_search.fit(X,y)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    prediction=best_model.predict(X_test)
    if prediction == 'versicolor':
        st.image('https://i.etsystatic.com/11300650/r/il/23fbbf/1171633447/il_794xN.1171633447_8s97.jpg',width=300,caption="Versicolor", use_column_width=False)
    elif prediction == 'setosa':
        st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/1280px-Irissetosa1.jpg',width=300,caption="setosa", use_column_width=False)
    else:
        st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/1024px-Iris_virginica_2.jpg',width=300,caption="virginica", use_column_width=False)
    st.write(f'The Predicted output is:{prediction}')
else:
    grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid[algo], cv=5, scoring='accuracy')
    grid_search.fit(X,y)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    prediction=best_model.predict(X_test)
    if prediction == 'versicolor':
        st.image('https://i.etsystatic.com/11300650/r/il/23fbbf/1171633447/il_794xN.1171633447_8s97.jpg',width=300,caption="Versicolor", use_column_width=False)
    elif prediction == 'setosa':
        st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/1280px-Irissetosa1.jpg',width=300,caption="setosa", use_column_width=False)
    else:
        st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/1024px-Iris_virginica_2.jpg',width=300,caption="virginica", use_column_width=False)
    st.write(f'The Predicted output is:{prediction}')

# with open('iris_model.pkl', 'wb') as model_file:
#     pickle.dump(best_model, model_file)

# grid_search.fit(X,y)
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_
# st.write(best_params)