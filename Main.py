import streamlit as st
import joblib
from utils import process_new
import numpy as np

model_RF=joblib.load('forest_model.pkl')
model_KNN=joblib.load('KNN.model.pkl')
model_LogReg=joblib.load('LogReg_model.pkl')
model_NB=joblib.load('NB_model.pkl')
model_SVM=joblib.load('SVM_model.pkl')
model_DT=joblib.load('decision_tree.pkl')

def satisfaction_classification():
    
    st.set_page_config(
        layout='wide',
        page_title='Mohamed shams Project',
        page_icon='toy_airplane_travel_plane_aircraft_transportation_icon_258728.ico'
    )
    
    page = st.sidebar.radio('Select Page', ['Overview', 'Describtive Statistics', 'Charts'])
    if page == 'Overview':
        
        
        st.title('Satisfaction Classification Predicition....')
        st.markdown('<hr>', unsafe_allow_html=True)
         ## Choose Model
        model_type = st.selectbox('Choose the Model', options=['RandomForest','KNN' ,'Naive-Baise','SVM','Logistic','DecisionTree'])
        
        col1,space, col2 = st.columns([3, 2, 3])
        
        with col1:
            with st.container():
    
                Gender = st.selectbox('Gender', options=['Male', 'Female'])
                Customer_Type=st.selectbox('Customer Type ', options=['Loyal Customer', 'disloyal Customer'])
                Age=st.number_input('Age', value=18, step=1)
                Type_of_Travel=st.selectbox('Type of Travel ', options=['Business travel', 'Personal Travel'])
                Class=st.selectbox('Class ', options=['Business', 'Eco','Eco Plus'])
                Flight_Distance=st.number_input('Flight_Distance', value=100,)
                Inflight_wifi_service=st.selectbox('Inflight_wifi_service ', options=[0,1,2,3,4,5])
                Departure_Arrival_time_convenient=st.selectbox('Departure/Arrival time convenient', options=[0,1,2,3,4,5])
                Ease_of_Online_booking=st.selectbox('Ease of Online booking ', options=[0,1,2,3,4,5])
                Gate_location=st.selectbox('Gate location ', options=[0,1,2,3,4,5])
                Food_and_drinke=st.selectbox('Food and drink ', options=[0,1,2,3,4,5])
            
        with col2:
            with st.container():
                spac, col, spac2 = st.columns([3, 2, 3])
                Online_boarding=st.selectbox('Online boarding ', options=[0,1,2,3,4,5])
                Seat_comfort=st.selectbox('Seat comfort',options=[0,1,2,3,4,5])    
                
                Inflight_entertainment=st.selectbox('Inflight entertainment ', options=[0,1,2,3,4,5])
                On_board_service=st.selectbox('On-board service ', options=[0,1,2,3,4,5])
                Leg_room_service=st.selectbox('Leg room service', options=[0,1,2,3,4,5])
                Baggage_handling=st.selectbox('Baggage handling ', options=[0,1,2,3,4,5])
                Checkin_service=st.selectbox('Checkin service ', options=[0,1,2,3,4,5])
                Inflight_service=st.selectbox('Inflight service ', options=[0,1,2,3,4,5])
                Cleanliness=st.selectbox('Cleanliness ',options=[0,1,2,3,4,5])
                Departure_Delay_in_Minutes=st.number_input('Departure Delay in Minutes', value=5)
                Arrival_Delay_in_Minutes=st.number_input('Arrival Delay in Minutes', value=5)
    
        if st.button('Predict Satisfaction ...'):
            ## Concatenate the users data
            new_data = np.array([Gender,
                                 Customer_Type,Age, Type_of_Travel,Class,
                                 Flight_Distance,Inflight_wifi_service,Departure_Arrival_time_convenient,Ease_of_Online_booking,
                                 Gate_location,Food_and_drinke,Online_boarding,Seat_comfort,Inflight_entertainment,On_board_service,
                                 Leg_room_service,Baggage_handling,Checkin_service,Inflight_service,Cleanliness,Departure_Delay_in_Minutes,
                                Arrival_Delay_in_Minutes])
            ## Call the function from utils.py to apply the pipeline
            
            
            X_processed = process_new(X_new=new_data)
            
            ## Predict using Model
            if model_type == 'RandomForest':
                y_pred =model_RF.predict(X_processed)[0]
                
            elif model_type == 'Logistic':
                y_pred = model_LogReg.predict(X_processed)[0]
            
            elif model_type == 'KNN':
                y_pred = model_KNN.predict(X_processed)[0]
                
            elif model_type == 'Naive-Baise':
                y_pred = model_NB.predict(X_processed)[0] 
                
            elif model_type == 'SVM':
                y_pred = model_SVM.predict(X_processed)[0]       
            
            elif model_type == 'DecisionTree':
                y_pred = model_DT.predict(X_processed)[0]
            
            #y_pred=model.predict(X_processed)[0]
                
            y_pred = bool(y_pred)

        
            st.success(f'Satisfaction Prediction is ... {y_pred},--------------> Note that the false meaning the client Not_satisfying')

if __name__== '__main__' :
    
    satisfaction_classification()
    