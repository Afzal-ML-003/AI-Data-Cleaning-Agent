import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re

st.set_page_config(page_title="AI Data Cleaner", layout="wide")
st.title("AI-Based Data Cleaner (50,000+ Rows Optimized)")

# Caching data loading for big files
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# Clean unwanted symbols from text columns
def clean_column(col):
    if col.dtype == object:
        return col.apply(lambda x: re.sub(r'[₹,–-]', '', str(x)).strip())
    return col

# Clean & encode full DataFrame
@st.cache_data
def clean_and_encode(df):
    # Clean symbols
    df_cleaned = df.apply(clean_column)

    # Drop duplicates
    df_cleaned.drop_duplicates(inplace=True)

    # Fill missing/null values
    df_cleaned.fillna("Unknown", inplace=True)

    # Encode object columns
    label_encoders = {}
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == object:
            le = LabelEncoder()
            try:
                df_cleaned[col] = le.fit_transform(df_cleaned[col])
                label_encoders[col] = le
            except:
                pass
    return df_cleaned

uploaded_file = st.file_uploader("Upload your CSV file (Supports 50,000+ rows)", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Reading and cleaning data..."):
        df = load_csv(uploaded_file)
        st.success(f"File loaded successfully! Total rows: {len(df)}")

        df_cleaned = clean_and_encode(df)

        st.subheader("🔍 Cleaned & Encoded Preview (First 100 rows)")
        st.write(df_cleaned.head(100))

        # Download button
        csv = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv'
        ) 