import streamlit as st
import pandas as pd
from VDIIM import load_and_process_data

st.set_page_config(page_title="Nifty Volatility Dashboard", layout="wide")

st.title("📊 Nifty Volatility Analysis")

# Load the data using your existing function
try:
    df, daily_df = load_and_process_data()
    
    st.subheader("Current Market Volatility (Intraday)")
    st.line_chart(df['rolling_vol_15'].tail(100))

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Recent Price Data")
        st.dataframe(df[['open', 'high', 'low', 'close', 'range']].tail(10))
    
    with col2:
        st.write("### Daily Summary")
        st.dataframe(daily_df.tail(10))

except Exception as e:
    st.error(f"Error loading data: {e}")