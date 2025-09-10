import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ta.momentum import StochasticOscillator
import plotly.express as px # This is the library you need to install.

st.set_page_config(page_title="Stock Market Prediction", layout="wide")

st.title("Stock Market Movement Prediction 📈")

# --- Sidebar for User Input ---
st.sidebar.header("User Input")
ticker_symbol = st.sidebar.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
st.sidebar.markdown("---")

# --- Functions ---

@st.cache_data
def load_data(ticker, start, end):
    """Fetches historical stock data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error(f"No data found for ticker: {ticker}. Please check the ticker symbol and date range.")
            return None
        
        # Check if columns are a multi-index and flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        data = data.rename(columns={
            "Adj Close": "Close",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        })
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return None

def feature_engineering(df):
    """
    Creates technical indicators and features from the stock data.
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    # Create target variable: 1 if next day's close price is higher, else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Basic features
    df["Daily_Return"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()
    df["Vol_MA_5"] = df["Volume"].rolling(5).mean()
    df["Vol_MA_20"] = df["Volume"].rolling(20).mean()

    # The ta library expects a 1-dimensional Series for its calculations.
    close_prices = df["Close"].squeeze()
    highs = df["High"].squeeze()
    lows = df["Low"].squeeze()
    volumes = df["Volume"].squeeze()

    # RSI
    df["RSI_14"] = ta.momentum.RSIIndicator(close_prices, window=14).rsi()

    # MACD
    macd = ta.trend.MACD(close_prices)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close_prices)
    df["BB_Middle"] = bollinger.bollinger_mavg()
    df["BB_Upper"] = bollinger.bollinger_hband()
    df["BB_Lower"] = bollinger.bollinger_lband()
    df["BB_Width"] = bollinger.bollinger_wband()

    # Stochastic Oscillator
    stoch = StochasticOscillator(high=highs, low=lows, close=close_prices)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # ADX
    adx_indicator = ta.trend.ADXIndicator(highs, lows, close_prices)
    df["ADX"] = adx_indicator.adx()
    
    # OBV
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close_prices, volumes).on_balance_volume()

    # CCI
    df["CCI"] = ta.trend.CCIIndicator(highs, lows, close_prices).cci()

    # Lagged features
    lags = [1, 2, 3, 4]
    for feature in ['Daily_Return', 'Volume', 'RSI_14', 'MACD', 'ADX']:
        for lag in lags:
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

    df = df.dropna()
    return df

@st.cache_data
def train_and_evaluate_models(data_with_features):
    """
    Trains and evaluates multiple machine learning models.
    """
    if data_with_features is None or data_with_features.empty:
        return None, None, None, None

    # Separate features (X) and target (y)
    feature_cols = [
        "Daily_Return", "MA_5", "MA_20", "MA_50", "Volatility_20",
        "RSI_14", "Vol_MA_5", "Vol_MA_20", "MACD", "MACD_signal",
        "MACD_hist", "BB_Middle", "BB_Upper", "BB_Lower", "BB_Width",
        "Stoch_K", "Stoch_D", "ADX", "OBV", "CCI",
        'Daily_Return_lag1', 'Daily_Return_lag2', 'Daily_Return_lag3', 'Daily_Return_lag4',
        'Volume_lag1', 'Volume_lag2', 'Volume_lag3', 'Volume_lag4',
        'RSI_14_lag1', 'RSI_14_lag2', 'RSI_14_lag3', 'RSI_14_lag4',
        'MACD_lag1', 'MACD_lag2', 'MACD_lag3', 'MACD_lag4',
        'ADX_lag1', 'ADX_lag2', 'ADX_lag3', 'ADX_lag4'
    ]
    
    # Filter for available columns
    feature_cols = [col for col in data_with_features.columns if col in data_with_features.columns]
    X = data_with_features[feature_cols]
    y = data_with_features["Target"]

    # Split into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(probability=True, random_state=42),
    }

    results = {}
    best_model_name = ""
    best_accuracy = 0

    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                "accuracy": accuracy,
                "report": classification_report(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "y_test": y_test,
                "y_pred": y_pred
            }
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
        except Exception as e:
            st.warning(f"Failed to train {name}: {e}")
            continue

    best_model = None
    if best_model_name:
        best_model = models[best_model_name]
        best_model.fit(X_train_scaled, y_train)

    return results, best_model_name, best_model, scaler, feature_cols

# --- Main App Logic ---
if __name__ == "__main__":
    df = load_data(ticker_symbol, start_date, end_date)
    if df is not None:
        df_features = feature_engineering(df)
        if df_features is not None:
            st.header(f"Historical Data for {ticker_symbol.upper()}")
            st.write(df_features.tail())

            # Plotting the Close Price
            fig, ax = plt.subplots()
            ax.plot(df["Close"])
            ax.set_title(f"{ticker_symbol.upper()} Close Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            st.pyplot(fig)

            # Training and evaluation
            with st.spinner("Training models..."):
                results, best_model_name, best_model, scaler, feature_cols = train_and_evaluate_models(df_features)

            if results:
                st.header("Model Performance")
                for name, res in results.items():
                    accuracy = res["accuracy"]
                    st.subheader(f"{name} Accuracy: {accuracy:.4f}")
                    st.write("Classification Report:")
                    st.text(res["report"])
                    st.write("Confusion Matrix:")
                    st.write(res["confusion_matrix"])
                    st.markdown("---")

                if best_model_name:
                    st.success(f"The best performing model is **{best_model_name}** with an accuracy of {results[best_model_name]['accuracy']:.4f}")

                    # --- Prediction for Next Day ---
                    st.header("Next Day Prediction")
                    try:
                        last_row = df_features[feature_cols].iloc[[-1]]
                        last_row_scaled = scaler.transform(last_row)
                        
                        prediction = best_model.predict(last_row_scaled)[0]
                        probability = best_model.predict_proba(last_row_scaled)[0]
                        
                        st.subheader(f"Prediction for **{ticker_symbol.upper()}** tomorrow:")
                        if prediction == 1:
                            st.success(f"The model predicts the stock will go **UP** tomorrow.📈")
                            st.write(f"Probability: {probability[1]*100:.2f}%")
                        else:
                            st.error(f"The model predicts the stock will go **DOWN** tomorrow.📉")
                            st.write(f"Probability: {probability[0]*100:.2f}%")
                    except Exception as e:
                        st.error(f"Could not make a prediction for the next day: {e}")

                    # --- Real vs. Predicted Plot ---
                    st.header("Real vs. Predicted Stock Movement (Test Set)")
                    try:
                        best_model_results = results[best_model_name]
                        y_test = best_model_results["y_test"]
                        y_pred = best_model_results["y_pred"]
                        
                        comparison_df = pd.DataFrame({
                            'Actual': y_test.values,
                            'Predicted': y_pred
                        }, index=y_test.index)
                        
                        # Create a figure using Plotly for interactive visualization
                        fig_compare = px.line(
                            df_features.loc[y_test.index],
                            y="Close",
                            title=f"Actual vs. Predicted Movement for {ticker_symbol.upper()}"
                        )
                        
                        # Add markers for correct and incorrect predictions
                        correct_ups = comparison_df[(comparison_df['Actual'] == 1) & (comparison_df['Predicted'] == 1)]
                        incorrect_ups = comparison_df[(comparison_df['Actual'] == 0) & (comparison_df['Predicted'] == 1)]
                        correct_downs = comparison_df[(comparison_df['Actual'] == 0) & (comparison_df['Predicted'] == 0)]
                        incorrect_downs = comparison_df[(comparison_df['Actual'] == 1) & (comparison_df['Predicted'] == 0)]

                        if not correct_ups.empty:
                            fig_compare.add_scatter(x=correct_ups.index, y=df_features.loc[correct_ups.index, "Close"], mode='markers', name='Correct Up', marker_symbol='triangle-up', marker=dict(color='green', size=10))
                        if not incorrect_ups.empty:
                            fig_compare.add_scatter(x=incorrect_ups.index, y=df_features.loc[incorrect_ups.index, "Close"], mode='markers', name='Incorrect Up', marker_symbol='triangle-up', marker=dict(color='red', size=10))
                        if not correct_downs.empty:
                            fig_compare.add_scatter(x=correct_downs.index, y=df_features.loc[correct_downs.index, "Close"], mode='markers', name='Correct Down', marker_symbol='triangle-down', marker=dict(color='green', size=10))
                        if not incorrect_downs.empty:
                            fig_compare.add_scatter(x=incorrect_downs.index, y=df_features.loc[incorrect_downs.index, "Close"], mode='markers', name='Incorrect Down', marker_symbol='triangle-down', marker=dict(color='red', size=10))

                        st.plotly_chart(fig_compare)

                    except Exception as e:
                        st.error(f"Could not generate Real vs. Predicted plot: {e}")
            else:
                st.error("No models were successfully trained.")
