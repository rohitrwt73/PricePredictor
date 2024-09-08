import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
# Replace this with the actual path to your dataset file
df=pd.read_csv('NIFTY50_all.csv') # Assuming you have a CSV with the data

# Encode the 'Symbol' (Company) column
le = LabelEncoder()
df['Symbol_encoded'] = le.fit_transform(df['Symbol'])

# Select features and target
X = df[['Symbol_encoded', 'Open', 'High', 'Low', 'Volume']]  # Input features
y = df['Close']  # Target (Close price)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model (optional, displays RMSE)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Streamlit UI
st.title('Stock Market Price Prediction')

# Display RMSE for reference
st.write(f"Model RMSE: {rmse:.2f}")

# Dropdown for selecting a company (symbol)
symbol = st.selectbox('Select Company Symbol', df['Symbol'].unique())

# Input fields for other stock data (open, high, low, volume)
open_price = st.number_input('Open Price', min_value=0.0, step=0.1, format="%.2f")
high_price = st.number_input('High Price', min_value=0.0, step=0.1, format="%.2f")
low_price = st.number_input('Low Price', min_value=0.0, step=0.1, format="%.2f")
volume = st.number_input('Volume', min_value=0, step=1)

# Prediction button
if st.button('Predict'):
    # Encode the selected symbol (company)
    symbol_encoded = le.transform([symbol])[0]
    
    # Prepare input for prediction
    input_data = np.array([[symbol_encoded, open_price, high_price, low_price, volume]])
    
    # Predict the close price using the trained model
    predicted_price = model.predict(input_data)[0]
    
    # Display the predicted price
    st.success(f'Predicted Close Price for {symbol}: ${predicted_price:.2f}')
