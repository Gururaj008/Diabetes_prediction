# Diabetes_prediction

Project Title	: Diabetes prediction using health indicators

Technologies : Python scripting, Data Collection, Outcome prediction

Domain : Health care

Problem Statement:
The problem statement is to create a Streamlit application that allows users to help answer questionnaire related to their own health and inturn use those indicators to predict whether the individual is non-diabetic, pre-diabetic and diabetic.

Steps involved in the project:

Set up a Streamlit app:
Created an intuitive UI using streamlit where users can answer questionnaire related to their current and past health conditions.

Selecting the impactful features for consideration:
Out of 22 features, using correlation, Mutual information, select k best and feature importance in random forest selected 12 most impactful features

Grid search to find out best parameters for the chosen models, given dataset:
Performed Grid search on the given dataset to find the best parameters for the models ( SVC, KNN classifier, Bagging classifier, Gradient boosting classifier, Random forest classifier ) 

Aggregating the result:
Put a voting classifier on top of the models to aggregate the final outcome

Packages used in the project:
Pandas, Streamlit, SKLearn

Hosted online at : https://gururaj008-diabetes-prediction-diabetes-pred-5sop4d.streamlit.app/
