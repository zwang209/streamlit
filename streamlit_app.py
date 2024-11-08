import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("train.csv")

st.title("Titanic: binary classification project")
st.sidebar.title("Table of contents")
pages=['Exploration', "DataVisualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)

if page==pages[0]:
    st.write("### resentation of data")
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())
    if st.checkbox("Show NA"):
        st.dataframe(df.isna().sum())

if page==pages[1]:
    st.write("### Data Visualization")

    fig=plt.figure()
    sns.countplot(x='Survived', data=df)
    st.pyplot(fig)

    fig=plt.figure()
    sns.countplot(x='Sex', data=df)
    plt.title("Distribution of the passengers' gender")
    st.pyplot(fig)

    fig=plt.figure()
    sns.countplot(x='Pclass', data=df)
    plt.title("Distribution of the passenger' class")
    st.pyplot(fig)

    fig=sns.displot(x='Age', data=df)
    plt.title("Distribution of the passenger age")
    st.pyplot(fig)

    fig=plt.figure()
    sns.countplot(x='Survived', hue='Sex', data=df)
    st.pyplot(fig)

    fig=sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)

    fig=sns.lmplot(x='Age', y='Survived', hue='Pclass', data=df)
    st.pyplot(fig)

    fig,ax=plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.write(fig)

if page==pages[2]:
    st.write("### Modelling")

    df=df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
    y=df['Survived']
    x_cat=df[['Pclass','Sex','Embarked']]
    x_num=df[['Age','Fare','SibSp','Parch']]

    for col in x_cat.columns:
        x_cat[col]=x_cat[col].fillna(x_cat[col].mode()[0])
    for col in x_num.columns:
        x_num[col]=x_num[col].fillna(x_num[col].median())
        x_cat_scale=pd.get_dummies(x_cat,columns=x_cat.columns)
        x=pd.concat([x_cat_scale,x_num], axis=1)
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    def prediction(classifier):
        if classifier=='Random Forest':
            clf=RandomForestClassifier()
        elif classifier=='SVC':
            clf=SVC()
        elif classifier=='Logistic Regression':
            clf=LogisticRegression()
        clf.fit(x_train,y_train)
        return clf
    
    def scores(clf, choice):
        if choice=='Accuracy':
            return clf.score(x_test, y_test)
        elif choice=='Confusion matrix':
            return confusion_matrix(y_test,clf.predict(x_test))
    
    choice=['Random Forest','SVC','Logistic Regression']
    option=st.selectbox('Model choice', choice)
    st.write('The chosen model is: ', option)

    clf=prediction(option)
    display=st.radio('What do you want to show?',('Accuracy','Confusion matrix'))
    if display=='Accuracy':
        st.write(scores(clf,display))
    elif display=='Confusion matrix':
        st.dataframe(scores(clf,display))


