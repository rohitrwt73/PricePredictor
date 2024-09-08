
# Stock Market Price Prediction using Machine Learning

This project is a **Stock Market Price Prediction** tool built using **Machine Learning**. It leverages historical stock data to predict the closing price of stocks for various companies. The model takes into account key features such as **Open Price**, **High Price**, **Low Price**, **Volume**, and the **Company Symbol** to generate predictions. The web-based application is built using **Streamlit**, making it easy for users to interact with the model.

## Features

- 📈 **Multi-Company Support**: Predict stock prices for different companies by selecting their symbol from the dataset.
- 🔧 **Interactive User Interface**: Powered by **Streamlit**, the app provides a user-friendly interface where users can input stock data and instantly get predictions.
- ⚡ **Machine Learning Model**: The app uses a **Linear Regression** model to predict stock prices, trained on historical data.
- 📊 **RMSE Evaluation**: The model is evaluated using **Root Mean Squared Error (RMSE)** to measure prediction accuracy.
- 🏷 **Company Symbol Encoding**: The model supports predictions across multiple companies by encoding the company symbols as features.

## How It Works

1. **Data Preprocessing**: The dataset includes several key stock indicators such as `Open`, `High`, `Low`, `Volume`, and `Close` prices, as well as the **Company Symbol**. The symbols are label encoded to be used as a feature in the model.
2. **Model Training**: A **Linear Regression** model is trained on the historical data using the features (`Open`, `High`, `Low`, `Volume`, and encoded Symbol) to predict the `Close` price.
3. **User Inputs**: The app provides a clean interface for users to input stock data (Open Price, High Price, Low Price, Volume) and select the company symbol.
4. **Prediction**: Once the user submits the data, the app returns the predicted closing price for the stock.

## Dataset

The dataset used for training the model includes the following features:

- `Symbol`: Represents the company.
- `Open`: The opening price of the stock.
- `High`: The highest price during the day.
- `Low`: The lowest price during the day.
- `Close`: The closing price of the stock (target variable).
- `Volume`: The number of shares traded.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rohitrwt73/PricePredictor.git
   cd PricePredictor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run stockMarket.py
   ```

## Usage

1. After running the application, you will be able to:
   - Select a company symbol from the dropdown list.
   - Input stock data such as `Open`, `High`, `Low`, and `Volume`.
   - Press the **Predict** button to get the predicted closing price for the selected company.

2. The model will return a predicted stock price based on the input values and display it on the screen.

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For building the interactive web-based frontend.
- **Scikit-Learn**: For training the Linear Regression model.
- **Pandas & Numpy**: For data manipulation and preprocessing.

## Future Enhancements

- ✅ Add more sophisticated machine learning algorithms such as Random Forests or Neural Networks to improve the accuracy of the predictions.
- ✅ Include more financial indicators/features for more accurate predictions.
- ✅ Add data visualizations for historical stock trends and comparison between predicted and actual prices.
- ✅ Improve UI/UX design for better user experience.

