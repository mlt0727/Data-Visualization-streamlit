import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

BASE = Path(__file__).resolve().parent
# __file__：Python 自动给你的一个变量，值是当前正在运行的这个脚本文件的路径（比如 .../Data Visualization.py）。
# Path(__file__)：把这个路径变成 pathlib.Path 对象（更好用、更跨平台）。
# .resolve()：把路径转换成绝对路径，并“规范化”（消掉 ..、符号链接等）。
# .parent：取这个路径的上一级目录（也就是文件所在文件夹）。

st.title("Water Quality Dashboard")
st.header("Data Visualization")

st.divider()


#SIDEBAR

st.sidebar.header("Load Datasets")

file_uploaded = st.sidebar.file_uploader("Upload a file", type=["csv"])

if file_uploaded is not None:
    st.sidebar.write("File uploaded successfully")
    df = pd.read_csv(file_uploaded)
    st.sidebar.success(
        f"Your file has {len(df)} rows and {len(df.columns)} columns. And the features are: {df.columns.tolist()}")
else:
    st.sidebar.warning("Please upload a file! If no files are uploaded, a sample dataset will be used instead.")
    df = pd.read_csv(BASE / "biscayne_bay_water_quality2.csv")

st.sidebar.divider()

st.sidebar.header("User Feedback")
feedback_stars = st.sidebar.feedback("stars")
feedback_thumbs = st.sidebar.feedback("thumbs")
if feedback_stars or feedback_thumbs:
    st.sidebar.success("Thank you for your feedback!")


#MAIN PAGE

tab1, tab2, tab3 = st.tabs(["Descriptive Statistics", "Charts", "Maps"])

with tab1:
    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())

with tab2:
    st.subheader("Line Chart")
    col1, col2 = st.columns([4, 1])
    with col1:
        feature_selected = st.selectbox("Select a feature", df.columns.tolist())

    with col2:
        color_selected = st.color_picker("Select a color", "#6495DD")

    fig1 = px.line(df,
                   x="Time",
                   y=feature_selected,
                   title=f"{feature_selected} over Time")
    fig1.update_traces(line_color=color_selected)
    st.plotly_chart(fig1)

    st.divider()

    feature_selected2 = st.selectbox("Select a second feature", df.columns.tolist())
    fig2 = px.scatter(df,
                      x=feature_selected,
                      y=feature_selected2,
                      title=f"{feature_selected} vs {feature_selected2}")
    st.plotly_chart(fig2)

    st.divider()

    fig3 = px.scatter_3d(df,
                         x="Longitude",
                         y="Latitude",
                         z="Total Water Column (m)",
                         color = feature_selected,
                         title=f"3D Scatter Plot of Total Water Column vs. Latitude and Longitude")
    fig3.update_scenes(zaxis_autorange="reversed")
    st.plotly_chart(fig3)

    st.divider()

    fig4 = px.box(df,
                  x=feature_selected,
                  title=f"{feature_selected} Box Plot")
    st.plotly_chart(fig4)

    st.divider()

    fig5 = px.imshow(df.select_dtypes(include=["number"]).corr(),
                     aspect="auto",
                     title="Correlation Heatmap",
                     text_auto=".2f")
    st.plotly_chart(fig5)

with tab3:
    st.subheader("Map")

    feature_selected_map = st.selectbox("Select a feature for map", df.columns.tolist())

    fig6 = px.scatter_mapbox(df,
                             lat="Latitude",
                             lon="Longitude",
                             zoom=17,
                             mapbox_style="open-street-map",
                             hover_data=df,
                             title="Map of Biscayne Bay Water Quality",
                             color=feature_selected_map,)
    st.plotly_chart(fig6)