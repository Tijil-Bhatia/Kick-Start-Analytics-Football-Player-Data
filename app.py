import streamlit as st
import pandas as pd
import joblib
import base64

# Load data and model
df = pd.read_csv("C:/Users/LENOVO/Downloads/minor/fifa_cleaned (1).csv")  # Adjust the file path as needed
model_filename = 'linear_regression_model.sav'
model = joblib.load(model_filename)

# Define features for prediction
features = ['age', 'potential', 'height_cm', 'weight_kgs',
            'crossing', 'finishing', 'dribbling',
            'club_rating']

# Streamlit app
st.set_page_config(layout="wide")

st.title("Football Player Rating Predictor")

# Function to predict future overall rating for a given player
def predict_player_rating(player_name):
    # Find the player data based on the player name
    player_data = df[df['name'] == player_name]
    
    if player_data.empty:
        st.write(f"No data found for player: {player_name}")
        return None
    
    # Extract the features for the player
    player_features = player_data[features].values
    
    # Predict the overall rating using the trained model
    predicted_rating = model.predict(player_features)
    
    # Return the predicted overall rating
    return round(predicted_rating[0], 2)

# Create input box for player name
player_name = st.text_input("Enter the player name:")

# Create a button to trigger prediction
if st.button("Predict"):
    if player_name:
        predicted_rating = predict_player_rating(player_name)
        if predicted_rating is not None:
            st.write(f"The predicted future performance rating for {player_name} is: {predicted_rating}")
