import streamlit as st
import joblib
import pandas as pd 


encode=joblib.load('Encoder.pkl')
ss=joblib.load('scaler.pkl')
model=joblib.load('Student_Performance.pkl')


st.title('Student Performance Prediction')
st.write('This app predicts student performance using data-driven insights for better learning outcomes!')
Hour_studied=st.number_input('How many hours did the student study per day?',value=0,min_value=0,max_value=24)
score=st.number_input("Enter the student's past academic scores",value=0)
sleep_hour=st.number_input('How many hours does the student sleep per day?',value=0,min_value=0,max_value=24)
sample_paper=st.number_input('How many sample question papers has the student practiced?',value=0)
eca=st.selectbox('Has the student been involved in any extracurricular activities?',('Yes',"No"))
eca=encode.transform([eca])

data=pd.DataFrame({'Hours Studied':Hour_studied, 'Previous Scores':score, 'Sleep Hours':sleep_hour, 'Sample Question Papers Practiced':sample_paper,'Encoded':eca})
scale=ss.transform(data)
Prediction=model.predict(scale)


if st.button('Predict Performance'):
    if Prediction[0]>60:
        st.success(f"Student's Performance Prediction is {round(Prediction[0],2)}")
    elif Prediction[0]>40:
        st.warning(f"Student's Performance Prediction is {round(Prediction[0],2)}")
    else :
        st.error(f"Student's Performance Prediction is {round(Prediction[0],2)}")
