import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from PIL import Image
import numpy as np

# Load the model with pickle
with open('Price.pkl', 'rb') as file:
    model = pickle.load(file)

# Create MinMaxScaler
mm = MinMaxScaler()

def preprocess_input(input_df):
    # Create a copy of the input DataFrame to avoid modifying the original
    processed_df = input_df.copy()

    # Label Encode 'Area'
    label_encoder = LabelEncoder()
    processed_df['Area'] = label_encoder.fit_transform(processed_df['Area'])
    processed_df['Serviced'] = label_encoder.fit_transform(processed_df['Serviced'])    
    processed_df['Newly Built'] = label_encoder.fit_transform(processed_df['Newly Built'])
    processed_df['Furnished'] = label_encoder.fit_transform(processed_df['Furnished'])
    # MinMax scaling for numerical features
    numerical_features = ['Bedrooms', 'Bathrooms', 'Toilets', 'Area']

    # Fit MinMaxScaler on numerical features
    mm.partial_fit(processed_df[numerical_features])

    # Transform using the fitted MinMaxScaler
    processed_df[numerical_features] = mm.transform(processed_df[numerical_features])

    return processed_df




# Main function to create web app interface
def main():
    st.title('Rental Price Prediction App')
    st.write('This App recommends rental prices of landed properties within Lagos based on inputted property features and location. ')
    img = Image.open('rent.jpg')
    st.image(img, width=500)
    
    input_data = {}  # Dictionary to store input data
    col1, col2 = st.columns(2)

    with col1:
        # Collect user data
        input_data['Serviced'] = st.radio('Serviced', ['Yes', 'No'])
        input_data['Newly Built'] = st.radio('Newly Built', ['Yes', 'No'])
        input_data['Furnished'] = st.radio('Furnished', ['Yes', 'No'])

    with col2:
        input_data['Bedrooms'] = st.number_input('Number of Bedrooms', min_value=0, max_value=10, step=1)
        input_data['Bathrooms'] = st.number_input("Number of Bathrooms", min_value=0, max_value=10, step=1)
        input_data['Toilets'] = st.number_input("Number of Toilet", min_value=0, max_value=10, step=1)
        input_data['Area'] = st.selectbox("Area",['Agungi', 'VGC', 'Osapa London', 'Ologolo', 'Ikoyi', 'Chevron',
                                                   'Ajah', 'Admiralty way', 'Ikate', 'VI', 'Onike', 'Lekki Phase 1',
                                                   'Abule Oja', 'Ikota', 'Orchid', 'Jakande', 'Estate', 'Idado',
                                                   'Akoka', 'Opebi', 'Adekunle', 'Egbe', 'Kusenla', 'Lekki Phase 2',
                                                   'Surulere', 'Chevron Drive', 'Fola Agoro', 'Gbagada',
                                                   'Ebute metta', 'Isolo', 'Ikate Elegushi', 'Ojo', 'Okota', 'Ayobo',
                                                   'Sangotedo', 'Ogba', 'Iba', 'Igbo Efon', 'Obanikoro', 'Ilupeju',
                                                   'Amuwo Odofin', 'Alimosho', 'Aguda', 'kosofe', 'Iwaya', 'Baruwa',
                                                   'Alagomeji', 'Alapere', 'Marwa', 'Ikorodu', 'Onipanu',
                                                   'Second toll gate', 'Peace estate', 'Abule Egba', 'Aboru', 'Salem',
                                                   'Agege', 'Alagbado', 'Isheri', 'Ifako Gbagada', 'bawala', 'Ikotun',
                                                   '2nd toll gate', 'Adeniyi Jones', 'Maryland', 'Lekki right',
                                                   'Ilasan', 'Ogudu GRA', 'LBS', 'New road', 'Jibowu', 'command',
                                                   'Chevy View Estate', 'Oregun', 'Ikeja GRA', 'Freedom Way',
                                                   'Anthony village', 'bajulaiye', 'Cement', 'shangisha', 'Tejuosho',
                                                   'Ketu', 'Palmgroove', 'Sholuyi', 'Mushin', 'Omole Phase 1',
                                                   'Ijesha', 'Oshodi', 'Ikeja Along', 'Awoyaya', 'Orile', 'Mile 12',
                                                   'Alausa', 'Ejigbo', 'Allen avenue', 'Magodo Phase 1',
                                                   'Chisco Ikate', 'Ibeju Lekki', 'Ilaje', 'Oniru', 'Ikosi Ketu',
                                                   'Akin Adesola VI', 'Nike Art Gallery', 'Deeper life',
                                                   'Bode Thomas', 'Makoko', 'Iyana paja', 'Apapa', 'Omole Phase2',
                                                   'Lakowe', 'Lagos island', 'Berger', 'coker', 'Ado', 'Abesan',
                                                   'Ota', 'Magodo phase 2', 'Olowora', 'Alaguntan', 'Mile 2', 'Waec',
                                                   'Morroco', 'Eleko', 'Badagry', 'Kudirat Abiola way', 'Obawole',
                                                   'Alaba', 'Magodo shangisha', 'Opic', 'Ebute', 'Elf', 'Ogombo',
                                                   'Ogudu Ori Oke', 'Banana Island', 'Gbagada phase 2', 'Owode',
                                                   'Alhaja', 'Epe', 'Okokomaiko', 'Allen ikeja', 'Festac town',
                                                   'Agboyi', 'Obalende', 'Majek', 'Pipeline', 'Ifako Ijaiye',
                                                   'Oke Ira', 'Ogunlana drive', 'lagos Mainland'], key='Area')
    
    input_df = pd.DataFrame([input_data])
    st.write(input_df)  # Display collected data
    
    if st.button('PREDICT'):
        final_df = preprocess_input(input_df)
        st.write(final_df)
        
        # Extract the numpy array from the DataFrame for prediction
        input_for_prediction = final_df.to_numpy()
        
        print("Input for prediction:", input_for_prediction)
        prediction = model.predict(input_for_prediction)
        print("Raw prediction:", prediction)
        # Assuming prediction is a NumPy array or a scalar
        prediction = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
        
        st.write(f'The predicted price based on your inputs is {prediction}')

# Run the main function when this script is executed directly
if __name__ == '__main__':
    main()
