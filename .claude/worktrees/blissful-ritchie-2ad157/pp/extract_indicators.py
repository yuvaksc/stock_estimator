import pandas as pd
import pandas_ta as ta
import os
import matplotlib.pyplot as plt

input_dir = "stock_data/price/raw" 
output_dir = "pp/indicators"

os.makedirs(output_dir, exist_ok=True)

csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

for file in csv_files:
    file_path = os.path.join(input_dir, file)

    if __name__ == "__main__":
        df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
        
        # Moving Averages
        df.ta.sma(length=14, append=True)  # Simple Moving Average
        df.ta.ema(length=14, append=True)  # Exponential Moving Average
        
        # Momentum Indicators
        df.ta.rsi(length=14, append=True)  # Relative Strength Index
        df.ta.stoch(append=True)  # Stochastic Oscillator
        df.ta.adx(append=True)  # Average Directional Index

        # Trend Indicators
        df.ta.macd(append=True)  # MACD Indicator
        df.ta.bbands(append=True)  # Bollinger Bands

        # Volume Indicators
        df.ta.obv(append=True)  # On-Balance Volume

        # Statistical Indicators
        df.ta.stdev(length=14, append=True)  # Standard Deviation

        drop_cols = ['open', 'high', 'low', 'close', 'volume']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
        df = df.loc["2014-01-01":"2016-03-31"]

        output_path = os.path.join(output_dir, f"indicators_{file}")
        df.to_csv(output_path)
        print(f"Saved: {output_path}")

