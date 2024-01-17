import matplotlib as mlt 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score,f1_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import streamlit as st 
import base64
import pickle as pk




#configuring the page setup
st.set_page_config(page_title='Crime Prediction system',layout='centered')

#selection=option_menu(menu_title="Main Menu",options=["Single Prediction","Multi Prediction"],icons=["cast","book","cast"],menu_icon="house",default_index=0)
with st.sidebar:
    st.title("Home Page")
    selection=st.radio("select your option",options=["predict a Single Crime Case", "Predict for Multi-Criime cases"])


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your prediction</a>'
    return href


#single prediction function
def CrimePrediction(givendata):
    
    loaded_model=pk.load(open("saved_model.sav", "rb"))
    input_data_as_numpy_array = np.asarray(givendata)# changing the input_data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the array as we are predicting for one instance
    std_scaler_loaded=pk.load(open("saved_scaler.pkl", "rb"))
    std_X_resample=std_scaler_loaded.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_X_resample)
    if prediction==1 or prediction=="1":
      return "Crime Occurence detected"
    else:
      return "No Crime Occurence"
    
 



def main():
    st.header("Crime Prediction System")
    

    #getting user input
    Crime_Latitude= st.number_input("Enter the Latitude",format=None)
    st.write("the latitude is : ",Crime_Latitude )
    st.write("\n")
    st.write("\n")

    Crime_Longitude= st.number_input("Enter the Longitude",format=None)
    st.write("the Longitude is : ",Crime_Longitude )
    st.write("\n")
    st.write("\n")

    TypeCrime=st.selectbox("Crime_Type",["",'Assault', 'Burglary', 'Robbery', 'Vandalism'], key="type_crime")
    if TypeCrime=="Assault":
        Crime_Type=0
    elif TypeCrime== "Burglary":
        Crime_Type= 1
    elif TypeCrime== "Robbery":
        Crime_Type=2
    else:
        Crime_Type=3

    WeatherConditions=st.selectbox("Weather Condition",["",'Clear', 'Partly cloudy', 'Foggy', 'Rainy'],key="WeatherCondi")
    if WeatherConditions=='Clear':
        weather_conditions= 0
    elif WeatherConditions=='Rainy':
        weather_conditions=3
    elif WeatherConditions== 'Foggy':
        weather_conditions=2
    else:
        weather_conditions=1


    Demographic_GenderDistribution=st.slider("Demographic GenderDistribution",0,250,key="demogragenderdistri")

    
    Demographic_Income= st.selectbox("Demographic_IncomeLevels",["",'Low','Medium','High'],key="demo_income")
    if Demographic_Income=="Medium":
        Demographic_IncomeLevels=1
    elif Demographic_Income=="High":
        Demographic_IncomeLevels=2
    else:
        Demographic_IncomeLevels=0

    Demographic_Education= st.selectbox("Demographic_EducationLevels",["","No Education",'High school diploma','University degree','Advanced degree'],key="Demographic_Educat")
    if Demographic_Education=="University degree":
        Demographic_Education_Levels=2
    elif Demographic_Education=="High school diploma":
        Demographic_Education_Levels=1
    elif Demographic_Education=='Advanced degree':
        Demographic_Education_Levels=3
    else:
        Demographic_Education_Levels=0


    Demographic_PopulationDensity=st.slider("Demographic_PopulationDensity",0,1000,key="Demographic_Pop")
    st.write("\n")

    Demographic_Employment=st.selectbox("Demographic_EmploymentStatus",["",'Unemployed', 'Employed', 'Not in labor force'],key="employment")
    if Demographic_Employment=="Unemployed":
        Demographic_EmploymentStatus=0
    elif Demographic_Employment=="Employed":
        Demographic_EmploymentStatus=1
    else:
        Demographic_EmploymentStatus=2
        

    Demographic_Health=st.selectbox("Demographic Health Indicator",["","Chronic diseases","Good healthcare access","Healthy"],key="demographicHealth")
    if Demographic_Health== "Chronic diseases":
        Demographic_HealthIndicators=2
    
    elif Demographic_Health== "Good healthcare access":
        Demographic_HealthIndicators=1
    
    elif Demographic_Health== "Healthy":
        Demographic_HealthIndicators=0

 
    
    Economic_UnemploymentRate = st.number_input("What is the Economic_UnemploymentRate",format=None,key="ecounemply")
    st.write("the Economic_UnemploymentRate is : ",Economic_UnemploymentRate )
    st.write("\n")
    st.write("\n")


    Economic_PovertyRate = st.number_input("What is the Economic_PovertyRate",format=None,key="ecopovert")
    st.write("the Economic_PovertyRate is : ",Economic_PovertyRate )
    st.write("\n")
    st.write("\n")

    Economic_EconomicGrowth=st.number_input("What is the Economic_EconomicGrowth",format=None,key="eco_eco_grwth")
    st.write("the Economic_EconomicGrowth is : ", Economic_EconomicGrowth )
    st.write("\n")
    st.write("\n")


    PolicePres=st.selectbox("Police Presence",["","Yes","No"],key="polcpresence")
    if PolicePres=="Yes":
        PolicePresence=1
    else:
        PolicePresence=0

    months=st.selectbox("select the month",["","January","February","March","April","May","June","July","August","September","October","November","December"],key="allmnth")
    if months== "January":
        the_months=1
    elif months== "February":
        the_months=2
    
    elif months== "March":
        the_months=3
    
    elif months== "April":
        the_months=4

    
    elif months== "May":
        the_months=5

    elif months== "June":
        the_months=6

    elif months== "July":
        the_months=7

    elif months== "August":
        the_months=8

    elif months== "September":
        the_months=9

    elif months== "October":
        the_months=10

    elif months== "November":
        the_months=11

    else:
        the_months=12

    

    dates=st.slider("Date",1,31,key="allDates")


    detectionResult = ''#for displaying result
    
    # creating a button for Prediction
    if (Crime_Latitude!="" or Crime_Latitude !=0) and (Crime_Longitude!="" or Crime_Longitude !=0) and TypeCrime!="" and WeatherConditions !="" and Demographic_GenderDistribution>0 and Demographic_Income!="" and Demographic_Education!="" and Demographic_PopulationDensity >= 0 and Demographic_Employment!="" and Demographic_Health !="" and Economic_UnemploymentRate>=0 and Economic_PovertyRate>=0 and Economic_EconomicGrowth>=0 and PolicePres != "" and months!="" and dates>=1 and st.button("Predict"):
        detectionResult = CrimePrediction([Crime_Latitude,Crime_Longitude,Crime_Type,weather_conditions,Demographic_GenderDistribution,Demographic_IncomeLevels,Demographic_Education_Levels,Demographic_PopulationDensity,Demographic_EmploymentStatus,Demographic_HealthIndicators,Economic_UnemploymentRate,Economic_PovertyRate,Economic_EconomicGrowth,PolicePresence,the_months,dates])
        st.success(detectionResult)


def multi(input_data):
    loaded_model=pk.load(open("saved_model.sav", "rb"))
    dfinput = pd.read_csv(input_data)
    # dfinput=dfinput.iloc[1:]
    dfinput=dfinput.reset_index(drop=True)
    dfinput.drop(dfinput.columns[1],inplace=True,axis=1)

    st.header('A view of the uploaded dataset')
    st.markdown('')
    st.dataframe(dfinput)

    dfinput=dfinput.values
    std_scaler_loaded=pk.load(open("saved_scaler.pkl", "rb"))
    std_dfinput=std_scaler_loaded.transform(dfinput)
    
    
    predict=st.button("predict")


    if predict:
        prediction = loaded_model.predict(std_dfinput)
        interchange=[]
        for i in prediction:
            if i==1:
                newi="Crime case Detected"
                interchange.append(newi)
            elif i==0:
                newi="No Crime"
                interchange.append(newi)
            
        st.subheader('Here is your prediction')
        prediction_output = pd.Series(interchange, name='Crime Prediction  results')
        prediction_id = pd.Series(np.arange(len(interchange)),name="Patient_ID")
        dfresult = pd.concat([prediction_id, prediction_output], axis=1)
        st.dataframe(dfresult)
        st.markdown(filedownload(dfresult), unsafe_allow_html=True)
        

if selection =="predict a Single Crime Case":
    main()

if selection == "Predict for Multi-Criime cases":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #---------------------------------#
    # Prediction
    #--------------------------------
    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    st.header('Upload your csv file here')
    uploaded_file = st.file_uploader("", type=["csv"])
    #--------------Visualization-------------------#
    # Main panel
    
    # Displays the dataset
    if uploaded_file is not None:
        #load_data = pd.read_table(uploaded_file).
        multi(uploaded_file)
    else:
        st.info('Upload your dataset !!')
    