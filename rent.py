

import pandas as pd
from joblib import load
import streamlit as st
from PIL import Image

# Import Model
model = load('Price.joblib')

columns_to_scale = ['Serviced','Newly Built','Furnished','Area']



# Function to preprocess input data
  

def replace_values_with_dict(df, column_dict):
    for column, value_dict in column_dict.items():
        df[column] = df[column].replace(value_dict)
    return df   

        


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
    
      # Convert collected data into a DataFrame
    st.write(pd.DataFrame([input_data]))

    if st.button('Predict'):
        df = pd.DataFrame ([input_data])
        column_dicts = {
            'Serviced': {'Yes': 1, 'No': 0},
            'Newly Built': {'Yes': 1, 'No': 0},
            'Furnished': {'Yes': 1, 'No': 0},
            'Area': {'Agungi': 12, 'VGC': 139, 'Osapa London': 126, 'Ologolo': 114, 'Ikoyi': 70, 'Chevron': 37, 'Ajah': 13, 'Admiralty way': 7, 'Ikate': 62, 'VI': 140, 'Onike': 118, 'Lekki Phase 1': 86, 'Abule Oja': 4, 'Ikota': 68, 'Orchid': 123, 'Jakande': 78, 'Estate': 49, 'Idado': 57, 'Akoka': 15, 'Opebi': 121, 'Adekunle': 5, 'Egbe': 44, 'Kusenla': 82, 'Lekki Phase 2': 87, 'Surulere': 137, 'Chevron Drive': 38, 'Fola Agoro': 51, 'Gbagada': 53, 'Ebute metta': 43, 'Isolo': 75, 'Ikate Elegushi': 63, 'Ojo': 110, 'Okota': 113, 'Ayobo': 30, 'Sangotedo': 134, 'Ogba': 105, 'Iba': 55, 'Igbo Efon': 60, 'Obanikoro': 103, 'Ilupeju': 73, 'Amuwo Odofin': 26, 'Alimosho': 23, 'Aguda': 11, 'kosofe': 146, 'Iwaya': 76, 'Baruwa': 33, 'Alagomeji': 18, 'Alapere': 20, 'Marwa': 94, 'Ikorodu': 66, 'Onipanu': 119, 'Second toll gate': 135, 'Peace estate': 131, 'Abule Egba': 3, 'Aboru': 2, 'Salem': 133, 'Agege': 10, 'Alagbado': 17, 'Isheri': 74, 'Ifako Gbagada': 58, 'bawala': 143, 'Ikotun': 69, '2nd toll gate': 0, 'Adeniyi Jones': 6, 'Maryland': 95, 'Lekki right': 88, 'Ilasan': 72, 'Ogudu GRA': 107, 'LBS': 83, 'New road': 100, 'Jibowu': 79, 'command': 145, 'Chevy View Estate': 39, 'Oregun': 124, 'Ikeja GRA': 65, 'Freedom Way': 52, 'Anthony village': 27, 'bajulaiye': 142, 'Cement': 36, 'shangisha': 148, 'Tejuosho': 138, 'Ketu': 80, 'Palmgroove': 130, 'Sholuyi': 136, 'Mushin': 99, 'Omole Phase 1': 116, 'Ijesha': 61, 'Oshodi': 127, 'Ikeja Along': 64, 'Awoyaya': 29, 'Orile': 125, 'Mile 12': 96, 'Alausa': 21, 'Ejigbo': 45, 'Allen avenue': 24, 'Magodo Phase 1': 89, 'Chisco Ikate': 40, 'Ibeju Lekki': 56, 'Ilaje': 71, 'Oniru': 120, 'Ikosi Ketu': 67, 'Akin Adesola VI': 14, 'Nike Art Gallery': 101, 'Deeper life': 41, 'Bode Thomas': 35, 'Makoko': 93, 'Iyana paja': 77, 'Apapa': 28, 'Omole Phase2': 117, 'Lakowe': 85, 'Lagos island': 84, 'Berger': 34, 'coker': 144, 'Ado': 8, 'Abesan': 1, 'Ota': 128, 'Magodo phase 2': 90, 'Olowora': 115, 'Alaguntan': 19, 'Mile 2': 97, 'Waec': 141, 'Morroco': 98, 'Eleko': 46, 'Badagry': 31, 'Kudirat Abiola way': 81, 'Obawole': 104, 'Alaba': 16, 'Magodo shangisha': 91, 'Opic': 122, 'Ebute': 42, 'Elf': 47, 'Ogombo': 106, 'Ogudu Ori Oke': 108, 'Banana Island': 32, 'Gbagada phase 2': 54, 'Owode': 129, 'Alhaja': 22, 'Epe': 48, 'Okokomaiko': 112, 'Allen ikeja': 25, 'Festac town': 50, 'Agboyi': 9, 'Obalende': 102, 'Majek': 92, 'Pipeline': 132, 'Ifako Ijaiye': 59, 'Oke Ira': 111, 'Ogunlana drive': 109, 'lagos Mainland': 147}
            
            }
        
        input_data_processed = replace_values_with_dict(df, column_dicts)
        #input_df = pd.DataFrame([input_data])
        st.write(input_data_processed)
        
        st.write(input_data_processed)
        #final_df = preprocess_data(input_data_processed)
        
        prediction = model.predict(input_data_processed)[0]  # Use the model to predict the outcome
        # Display the prediction result
        st.write("The predicted price of the rental properties is:",round(prediction))
        


if __name__ == '__main__':
    main()