import os, pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier,VotingClassifier
import warnings
warnings.filterwarnings('ignore')

def prediction(query1):
    df= pd.read_csv('Diabetes_new.csv')
    df = df.drop('Unnamed: 0', axis=1)
    df['BMI_coded'] = df['BMI_coded']/1.0
    df['PhysHlth_coded'] = df['PhysHlth_coded']/1.0
    df['MentHlth_coded'] = df['MentHlth_coded']/1.0
    y = df['Diabetes_012']
    X = df.drop('Diabetes_012', axis = 1)
    X = np.array(X)
    clf_1 = SVC(C=4, degree=2, kernel='poly')
    clf_1.fit(X,y)
    clf_2 = KNeighborsClassifier(algorithm= 'auto', n_neighbors= 9, weights= 'uniform')
    clf_2.fit(X,y)
    clf_3 = BaggingClassifier(max_features= 0.5, max_samples= 0.7, n_estimators= 30)
    clf_3.fit(X,y)
    clf_4 = GradientBoostingClassifier(criterion= 'friedman_mse', learning_rate= 0.15, loss='deviance', max_depth= 4, max_features= 'sqrt', min_impurity_decrease= 0.08,min_samples_split= 2, n_estimators= 20)
    clf_4.fit(X,y)
    clf_5 = RandomForestClassifier(criterion= 'gini', max_depth= 5, max_features= 'auto', min_impurity_decrease= 0.05, n_estimators= 50)
    clf_5.fit(X,y)
    vot_clf = VotingClassifier(estimators=[('svc',clf_1),('knn',clf_2),('bg',clf_3),('gb',clf_4),('rf',clf_5)], voting='hard',)
    vot_clf.fit(X,y)
    res = vot_clf.predict(query1)
    return res[0]
    

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title(':blue[Diabetes prediction using Health Indicators]')
    st.divider()
    st.subheader(":orange[About the project]")
    st.markdown('<div style="text-align: justify"> Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy. Your body breaks down most of the food you eat into sugar (glucose) and releases it into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin. Insulin acts like a key to let the blood sugar into your body’s cells for use as energy. With diabetes, your body doesn’t make enough insulin or can’t use it as well as it should. When there isn’t enough insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream. Over time, that can cause serious health problems, such as heart disease, vision loss, and kidney disease.</div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify"> The dataset used is the one uploaded by ALEX TEBOUL on Kaggle from BRFSS. The Behavioral Risk Factor Surveillance System (BRFSS) is a health-related telephone survey that is collected annually by the CDC. Each year, the survey collects responses from over 400,000 Americans on health-related risk behaviors, chronic health conditions, and the use of preventative services. It has been conducted every year since 1984. The features are either questions directly asked of participants, or calculated variables based on individual participant responses.  </div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify"> Out of 22 original features employing correlation, Select_Kbest and Mutual_Information from Sklearn , Feature importance from random forest selected 12 most important features for consideration. Using Grid search selected 5 most accurate models( SVC, KNN classifier, Bagging classifier, Gradient boosting classifier and Random forest classifier ),tuned the hyper parameters for increased accuracy. The final outcome is aggregated with a voting classifier.  </div>', unsafe_allow_html=True)
    st.divider()
    st.subheader(':orange[Please fill in the following details to predict your diabetes status]')
    col1, col2, col3 = st.columns(3)
    with col1:
        gen_hel = st.selectbox(':green[**On a scale 1-5 would you say that in general your health is:**]',
            ('1 : excellent', '2 : very good', '3 : good', '4 : fair', '5 : poor'),key=1)
    with col2:
        h_bp = st.selectbox(':green[**Are you suffering from High Blood Pressure**]',
            ('0 : No_high_BP ', '1 : High_BP'),key=2)
    with col3:
        bmi = st.selectbox(':green[**Select the range of your Body Mass Index**]',
            ('1 : 0 - 18.4 ', '2 : 18.5 - 24.9', '3 : 25 - 29.9','4 : >30'),key=3)
    st.write('')
    st.write('')
    col4, col5, col6 = st.columns(3)
    with col4:
        diff_w = st.selectbox(':green[**Do you have serious difficulty walking or climbing stairs?**]',
            ('0 : No  ', '1 : Yes'),key=4)
    with col5:
        age = st.selectbox(':green[**Select your age range**]',
            ('1 : 18 - 24','2 : 25 - 29','3 : 30 - 34','4 : 35 - 39','5 : 40 – 44','6 : 45 – 49','7 : 50 – 54','8 : 55 – 59','9 : 60 - 64 ','10 : 65 -69','11 : 70 – 74','12 : 75 - 80','13 : 80 or older'),key=5)
    with col6:
        he_di = st.selectbox(':green[**Do you have coronary heart disease (CHD) or myocardial infarction (MI)**]',
            ('0 : No ', '1 : Yes'),key=6)
    st.write('')
    st.write('')
    col7, col8, col9 = st.columns(3)
    with col7:
        ph_he = st.selectbox(':green[**For how many days during the past 30 days was your physical health not good?**]',
            ('1 : 0-10 days ', '2 : 11-20 days', '3: 20+ days'),key=7)
    with col8:
        ph_ac = st.selectbox(':green[**Physical activity in past 30 days - not including job**]',
            ('0 : No ', '1 : Yes'),key=8)
    with col9:
        str = st.selectbox(':green[**(Ever told) you had a stroke?**]',
            ('0 : No ', '1 : Yes'),key=9)
    st.write('')
    st.write('')
    col10, col11, col12 = st.columns(3)
    with col10:
        me_he = st.selectbox(':green[**For how many days during the past 30 days was your mental health not good?**]',
            ('1 : 0-10 days ', '2 : 11-20 days', '3: 20+ days'),key=10)
    with col11:
        chol = st.selectbox(':green[**Have you got your cholestorol levels checked in the past 5 years**]',
            ('0 : No ', '1 : Yes'),key=11)
    with col12:
        smok = st.selectbox(':green[**Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]**]',
            ('0 : No ', '1 : Yes'),key=12)
    
    

    res1 = gen_hel.split(':')
    val1 = int(res1[0])

    res2 = h_bp.split(':')
    val2 = int(res2[0])

    res3 = bmi.split(':')
    val3 = int(res3[0])

    res4 = diff_w.split(':')
    val4 = int(res4[0])

    res5 = age.split(':')
    val5 = int(res5[0])

    res6 = he_di.split(':')
    val6 = int(res6[0])

    res7 = ph_he.split(':')
    val7 = int(res7[0])

    res8 = ph_ac.split(':')
    val8 = int(res8[0])

    res9 = str.split(':')
    val9 = int(res9[0])

    res10 = me_he.split(':')
    val10 = int(res10[0])

    res11 = chol.split(':')
    val11 = int(res11[0])

    res12 = smok.split(':')
    val12 = int(res12[0])

    st.write('')
    st.write('')
    if st.button('Predict'):
        #st.write(val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12)
        query = np.array([val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12]).reshape(1,12)
        result = prediction(query)
        #st.markdown(result)
        if result == 0:
            st.subheader(':green[Congratulations! the prediction is : Free from diabetes]')
        elif result == 1:
            st.subheader(':orange[Warning! the prediction is : Pre-diabetic, please work on reducing your blood sugar levels]')
        elif result == 2:
            st.subheader(':red[Sorry! the prediction is : Diabetic, kindly consult a doctor]')

    st.divider()
    st.subheader(':orange[About the developer]')
    st.write('')
    st.markdown('<div style="text-align: justify">Gururaj H C is passionate about Machine Learning and fascinated by its myriad real world applications. Possesses work experience with both IT industry and academia. Currently pursuing “IIT-Madras Certified Advanced Programmer with Data Science Mastery Program” course as a part of his learning journey.  </div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('_An effort by_ :blue[**MAVERICK_GR**]')
    st.markdown(':green[**DEVELOPER CONTACT DETAILS**]')
    st.markdown(":orange[email id:] gururaj008@gmail.com")
    st.markdown(":orange[Personal webpage hosting other Datascience projects :] http://gururaj008.pythonanywhere.com/")
    st.markdown(":orange[LinkedIn profile :] https://www.linkedin.com/in/gururaj-hc-machine-learning-enthusiast/")




    
    


            