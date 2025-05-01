"""
Synthetic trading data generator for model training.
Generates realistic OHLCV data with target labels for ML training.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def generate_synthetic_data(
    num_days: int = 500,
    initial_price: float = 50.0,
    volatility: float = 0.015,
    volume_base: int = 1000000,
    volume_volatility: float = 0.3,
    target_days_ahead: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV trading data with future return targets.
    
    Args:
        num_days: Number of trading days to generate
        initial_price: Starting price of the asset
        volatility: Daily price volatility (standard deviation of returns)
        volume_base: Base trading volume
        volume_volatility: Volume volatility
        target_days_ahead: How many days ahead to calculate returns for target
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with date, open, high, low, close, volume, target columns
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate dates for trading days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=num_days * 2)  # Extra days for business day filtering
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    dates = date_range[:num_days]  # Take exact number of trading days
    
    # Initial price and lists to store values
    price = initial_price
    opens, highs, lows, closes, volumes = [], [], [], [], []
    
    # Generate OHLC and volume data
    for i in range(len(dates)):
        # Daily change percent
        daily_change = np.random.normal(0, volatility)
        
        # Open price (previous close or initial price)
        open_price = price if i == 0 else closes[i-1]
        
        # Close price
        close_price = open_price * (1 + daily_change)
        
        # High and low with randomized ranges
        high_range = abs(np.random.normal(0, volatility))
        low_range = abs(np.random.normal(0, volatility))
        
        high_price = max(open_price, close_price) * (1 + high_range)
        low_price = min(open_price, close_price) * (1 - low_range)
        
        # Volume with some randomness
        volume = int(volume_base * (1 + np.random.normal(0, volume_volatility)))
        volume = max(100, volume)  # Ensure volume is positive
        
        # Store values
        opens.append(round(open_price, 2))
        highs.append(round(high_price, 2))
        lows.append(round(low_price, 2))
        closes.append(round(close_price, 2))
        volumes.append(volume)
        
        # Update price for next iteration
        price = close_price
    
    # Generate target variable (future price movement)
    targets = []
    for i in range(len(dates)):
        if i < len(dates) - target_days_ahead:  
            future_return = (closes[i+target_days_ahead] / closes[i]) - 1
        else:
            # For the last few days, use the average return of similar period
            future_return = np.mean([
                (closes[-1] / closes[max(0, -target_days_ahead-5)]) - 1
            ])
        
        targets.append(round(future_return, 4))
    
    # Create the DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'target': targets
    })
    
    return df

def save_to_csv(df: pd.DataFrame, filename: str = 'training_data.csv') -> str:
    """
    Save the generated DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    df.to_csv(filename, index=False)
    return filename

def display_in_streamlit(df: pd.DataFrame):
    """
    Display the generated data in Streamlit UI.
    
    Args:
        df: DataFrame to display
    """
    st.subheader("Generated Synthetic Trading Data")
    st.dataframe(df.head(10))
    
    # Data statistics
    st.subheader("Data Statistics")
    stats = {
        "Total Rows": len(df),
        "Date Range": f"{df['date'].min()} to {df['date'].max()}",
        "Price Range": f"${df['low'].min():.2f} - ${df['high'].max():.2f}",
        "Avg. Volume": f"{df['volume'].mean():.0f}",
        "Target Range": f"{df['target'].min():.2%} to {df['target'].max():.2%}"
    }
    st.json(stats)
    
    # Add download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="synthetic_trading_data.csv",
        mime="text/csv",
    )

def add_to_model_training_ui():
    """Add synthetic data generation UI to model_training page"""
    st.sidebar.subheader("🔄 Generate Synthetic Data")
    
    with st.sidebar.expander("Synthetic Data Parameters"):
        num_days = st.slider("Number of Days", 100, 1000, 500)
        initial_price = st.slider("Initial Price ($)", 10.0, 200.0, 50.0)
        volatility = st.slider("Price Volatility", 0.005, 0.05, 0.015, 0.001)
        target_days = st.slider("Target Days Ahead", 1, 20, 5)
    
    if st.sidebar.button("Generate Sample Data"):
        data = generate_synthetic_data(
            num_days=num_days, 
            initial_price=initial_price,
            volatility=volatility,
            target_days_ahead=target_days
        )
        display_in_streamlit(data)

# If the script is run directly, generate and save data
if __name__ == "__main__":
    data = generate_synthetic_data()
    filename = save_to_csv(data)
    print(f"Sample data saved to {filename}")
    print(data.head())