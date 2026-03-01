import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.title("Lab 1 - Data Exploration and Visualization")

step1, step2, step3, step4, step5, step6 = st.tabs(["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"])

with step1:
    st.header("Load and Explore the Dataset")
    # Load the dataset using pandas
    df = pd.read_csv("oct25-2024.csv")

    # Display the first few rows
    st.write("First 5 rows:")
    st.write(df.head())

    # Print basic dataset information (columns, data types, missing values)
    st.write("Dataset columns:")
    st.write(df.columns.tolist())
    st.write("Data types:")
    st.write(df.dtypes)
    st.write("Missing values:")
    st.write(df.isna().sum())

with step2:
    st.header("Descriptive Statistics and EDA")
    # Compute summary statistics for the dataset
    st.write("Descriptive statistics:")
    st.write(df.describe())

    # What do you observe? Are there any unusual values?
    float_cols = []
    for col in df.columns:
        if df[col].dtype == np.float64:
            float_cols.append(col)
    for col in float_cols:
        st.write(f"unusual values in : {col}")
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        IQR = q3 - q1

        lower = q1 - 1.5 * IQR
        upper = q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        st.write(outliers)

with step3:
    st.header("Covariance and Correlation")
    # Compute the covariance matrix
    st.write("Covariance matrix:")
    st.write(df[float_cols].cov())

    # Compute the correlation matrix
    st.write("Correlation matrix:")
    st.write(df[float_cols].corr())

    # Which variables have the highest positive and negative correlation?
    corr = df[float_cols].corr()
    for c in corr.columns:
        corr.loc[c, c] = 0
    max_val = corr.stack().max()
    max_pairs = corr.stack().idxmax()
    min_val = corr.stack().min()
    min_pairs = corr.stack().idxmin()


    st.write(f"highest positive correlation: {max_pairs} with value: {max_val}")
    st.write(f"highest negative correlation: {min_pairs} with value: {min_val}")

with step4:
    st.header("Outlier Detection and Treatment")
    for col in float_cols:
        st.write(f"Outlier detection for column: {col}")
        # Compute Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[col].quantile(0.25)
        st.write(f"Q1 = {Q1}")
        Q3 = df[col].quantile(0.75)
        st.write(f"Q3 = {Q3}")

        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1
        st.write(f"IQR = {IQR}")

        # Identify potential outliers
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        st.write(f"number of outliers: {len(outliers)}")

        # How many outliers do you find? Should they be removed or corrected?
        df_no_outliers = df.copy()
        df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower) & (df_no_outliers[col] <= upper)]
        st.write(f"dataset after removing outliers: ")
        st.dataframe(df_no_outliers[col])
        # df[col] = df_no_outliers[col]
        st.divider()

with step5:
    st.header("Data Visualization using Plotly Express")
    # Create a scatter plot of Salinity vs. Temperature
    fig1 = px.scatter(df, x="Sal psu", y="Temp °C", title="Salinity vs. Temperature")
    st.plotly_chart(fig1)

    # Create a histogram of pH levels
    fig2 = px.histogram(df, x="pH", title="pH Distribution")
    st.plotly_chart(fig2)

    # What trends or anomalies do you observe in the visualizations?
    st.write("anomalies: The two graphs show two extreme values in their lower left corners.")

with step6:
    st.header("Creating a Streamlit App")
    # Import Streamlit and load the dataset
    import streamlit as st

    # Display dataset preview (checkbox to show/hide raw data)
    st.subheader("Dataset preview:")
    show_raw = st.checkbox("Show raw data")
    if show_raw:
        st.dataframe(df)

    # Display correlation matrix and key statistics
    st.subheader("Key statistics:")
    st.write(df.describe())
    st.subheader("Correlation matrix:")
    corr = df[float_cols].corr()
    st.dataframe(corr)

    # Create a simple visualization (scatter plot, histogram, etc.)
    st.subheader("Visualizations :")
    feature1 = st.selectbox("Select the first feature", df.columns.tolist())
    feature2 = st.selectbox("Select the second feature", df.columns.tolist(), index=1)
    fig3 = px.scatter(df, x=feature1, y=feature2, title=f"{feature1} vs {feature2}")
    st.plotly_chart(fig3)
    fig4 = px.histogram(df, x=feature1, title=f"{feature1} Distribution")
    st.plotly_chart(fig4)
    fig5 = px.histogram(df, x=feature2, title=f"{feature2} Distribution")
    st.plotly_chart(fig5)

    # Run the Streamlit app and explore the interactive features
    # type in the terminal: streamlit run ".\Lab 1 Data Exploration Lab.py"

    st.subheader("Map")

    feature_selected_map = st.selectbox("Select a feature for map", df.columns.tolist())

    fig6 = px.scatter_mapbox(df,
                             lat="Latitude",
                             lon="Longitude",
                             zoom=17,
                             mapbox_style="open-street-map",
                             hover_data=df,
                             title="Map of Biscayne Bay Water Quality",
                             color=feature_selected_map, )
    st.plotly_chart(fig6)
