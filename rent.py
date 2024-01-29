

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from joblib import load
import streamlit as st
from PIL import Image

# Import Model
model = load('Price.joblib')

columns_to_scale = ['Serviced','Newly Built','Furnished','Area']



# Function to preprocess input data
def preprocess_data(df):
    processed_df = df.copy()

    # Perform label encoding for categorical columns
    categorical_cols = ['Serviced','Newly Built','Furnished','Area']
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        if col in processed_df.columns:
            processed_df[col] = label_encoder.fit_transform(processed_df[col])

    # Perform Min-Max scaling for all columns
    mm_scaler = MinMaxScaler()
    processed_df[processed_df.columns] = mm_scaler.fit_transform(processed_df)

    return processed_df
def preprocessor(input_df):
    # Preprocess categorical and numerical columns separately
    input_df = preprocess_data(input_df)
    return input_df


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
    
    input_df = pd.DataFrame([input_data])  # Convert collected data into a DataFrame

    if st.button('Predict'):  # When the 'Predict' button is clicked
        final_df = preprocessor(input_df)  # Preprocess the collected data
        prediction = model.predict(final_df)[0]  # Use the model to predict the outcome

        # Display the prediction result
        st.write("The predicted price of the rental properties is:")
        st.write(prediction)


if __name__ == '__main__':
    main()