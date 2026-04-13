import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="What Stories Do Data Tell?", layout="wide")
st.title("What Stories Do Data Tell?")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "1. Data Acquisition",
    "2. Preliminary Analysis",
    "3. EDA",
    "4. Visualization",
    "5. Hypotheses",
    "6. T-Test",
    "7. Interpretation",
    "8. Machine Learning",
])

with tab1:
    st.header("1. Data Acquisition")

    st.write("Download dataset and import it")
    df = pd.read_csv("gapminder.csv")

    st.subheader("Verify the dataset’s integrity")
    st.write("encoding: UTF-8")

with tab2:
    st.header("2. Preliminary Analysis")
    st.write(f"number of rows: {df.shape[0]} \n\n number of columns: {df.shape[1]}")

    st.subheader("Data Types")
    dtype_df = df.dtypes.reset_index()
    dtype_df.columns = ["Column", "Data Type"]
    st.dataframe(dtype_df)

    st.write("Since the datatype is object, we need to convert it to numeric")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    dtype_df = df.dtypes.reset_index()
    dtype_df.columns = ["Column", "Data Type"]
    st.dataframe(dtype_df)

    st.subheader("Variable Information")
    variable_info = pd.DataFrame({
        "Variable": df.columns,
        "Description": [
            "Country name",
            "Income per person",
            "Alcohol consumption",
            "Armed forces rate",
            "Breast cancer rate per 100,000",
            "CO2 emissions",
            "Female employment rate",
            "HIV rate",
            "Internet use rate",
            "Life expectancy",
            "Oil use per person",
            "Polity score",
            "Residential electricity use per person",
            "Suicide rate per 100,000",
            "Employment rate",
            "Urban population rate"
        ]
    })
    st.dataframe(variable_info)

    st.subheader("Missing Values")
    st.write("number of missing values:", df.isnull().sum())

    for col in df.columns[1:]:
        df[col] = df[col].fillna(df[col].median(), inplace=True)
    st.write("After filling the missing values with median:", df.isnull().sum())

    st.subheader("Outliers")
    before_outliers = {}
    after_outliers = {}
    for col in df.columns[1:]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before_outliers[col] = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        after_outliers[col] = ((df[col] < lower) | (df[col] > upper)).sum()
    st.dataframe({"Before Outliers": before_outliers, "After Outliers": after_outliers})

    st.subheader("Cleaned Data Preview")
    st.dataframe(df[df["country"].isin(["China", "India", "United States", "Germany"])])

numeric_cols = df.columns[1:]

with tab3:
    st.header("3. Exploratory Data Analysis")
    st.subheader("Summary Statistics")
    means = {}
    medians = {}
    modes = {}
    stds = {}

    for col in numeric_cols:
        means[col] = df[col].mean()
        medians[col] = df[col].median()
        modes[col] = df[col].mode()[0]
        stds[col] = df[col].std()
    st.dataframe({"Mean": means, "Median": medians, "Mode": modes, "Standard Deviation": stds})

    st.subheader("Histograms")
    hist_col = st.selectbox("Select a variable for histogram", numeric_cols, key="hist")
    fig1, ax = plt.subplots()
    ax.hist(df[hist_col], bins=20)
    ax.set_title(f"Histogram of {hist_col}")
    ax.set_xlabel(hist_col)
    ax.set_ylabel("Frequency")
    st.pyplot(fig1)

    st.subheader("Density Plot")
    density_col = st.selectbox("Select a variable for density plot", numeric_cols, key="density")
    fig2, ax = plt.subplots()
    df[density_col].plot(kind="density", ax=ax)
    ax.set_title(f"Density Plot of {density_col}")
    ax.set_xlabel(density_col)
    st.pyplot(fig2)

    st.subheader("Correlation Matrix")
    corr_matrix = df[numeric_cols].corr()
    fig3, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig3)

    st.subheader("Scatter Plot")
    x_col = st.selectbox("Select X variable", numeric_cols, key="x")
    y_col = st.selectbox("Select Y variable", numeric_cols, key="y")

    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col])
    ax.set_title(f"{y_col} vs {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

    st.subheader("Interesting Trends")
    st.write(f"Highest life expectancy: {df.loc[df['lifeexpectancy'].idxmax(), 'country']}")

with tab4:
    st.header("4. Visualization")

    st.subheader("Bar Chart")
    selected_countries = st.multiselect(
        "Select Countries",
        options=sorted(df["country"].unique()),
        default=["China", "India", "United States"]  # 可改可删
    )
    y_col = st.selectbox("Select Y variable", numeric_cols)
    filtered_df = df[df["country"].isin(selected_countries)]
    fig_bar = px.bar(
        filtered_df,
        x="country",
        y=y_col,
        title=f"Bar chart of selected countries vs {y_col}",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Scatter Plot")
    x_col = st.selectbox("Select X variable", numeric_cols, key="viz_x")
    y_col = st.selectbox("Select Y variable", numeric_cols, key="viz_y")
    fig_scatter = px.scatter(
        df,
        x=x_col,
        y=y_col,
        hover_name="country",
        title=f"{y_col} vs {x_col}"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Histogram")
    hist_col = st.selectbox("Select a variable for histogram", numeric_cols, key="viz_hist")
    fig_hist = px.histogram(
        df,
        x=hist_col,
        nbins=20,
        title=f"Distribution of {hist_col}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Box Plot")
    box_col = st.selectbox("Select a variable for box plot", numeric_cols, key="viz_box")
    fig_box = px.box(
        df,
        y=box_col,
        points="outliers",
        title=f"Box Plot of {box_col}"
    )
    st.plotly_chart(fig_box, use_container_width=True)

with tab5:
    st.header("5. Formulate Hypotheses")

    st.subheader("Research Question")
    st.write("Is there a significant difference in life expectancy between countries with high internet use rates and countries with low internet use rates?")

    st.subheader("Null Hypothesis (H0)")
    st.write("There is no significant difference in mean life expectancy between countries with high internet use rates and countries with low internet use rates.")

    st.subheader("Alternative Hypothesis (Ha)")
    st.write("There is a significant difference in mean life expectancy between countries with high internet use rates and countries with low internet use rates.")

with tab6:
    st.header("6. Hypothesis Testing (t-test)")

    median_internet = df["internetuserate"].median()
    high_group = df[df["internetuserate"] > median_internet]["lifeexpectancy"]
    low_group = df[df["internetuserate"] <= median_internet]["lifeexpectancy"]

    st.subheader("Group Information")
    st.write("Median internet use rate:", round(median_internet, 2))
    st.write("High internet use group size:", len(high_group))
    st.write("Low internet use group size:", len(low_group))

    st.subheader("Independence")
    st.write("The two groups are independent because each country belongs to only one group.")

    st.subheader("Normality Test")
    shapiro_high = stats.shapiro(high_group)
    shapiro_low = stats.shapiro(low_group)

    normality_df = pd.DataFrame({
        "Group": ["High Internet Use", "Low Internet Use"],
        "Shapiro Statistic": [shapiro_high.statistic, shapiro_low.statistic],
        "p-value": [shapiro_high.pvalue, shapiro_low.pvalue]
    })
    st.dataframe(normality_df)

    st.subheader("Homogeneity of Variance")
    levene_test = stats.levene(high_group, low_group)
    st.write("Levene Statistic:", levene_test.statistic)
    st.write("p-value:", levene_test.pvalue)

    st.subheader("Independent Two-Sample t-test")
    equal_var = True if levene_test.pvalue > 0.05 else False
    ttest_result = stats.ttest_ind(high_group, low_group, equal_var=equal_var)
    st.write(ttest_result.statistic)

    st.subheader("Degree of Freedom")
    n1 = len(high_group)
    n2 = len(low_group)
    dfree = n1 + n2 - 2
    st.write("dfree:", dfree)

    st.subheader("Results")
    results_df = pd.DataFrame({
        "Metric": ["Test Statistic", "Degrees of Freedom", "p-value"],
        "Value": [ttest_result.statistic, dfree, ttest_result.pvalue]
    })
    st.dataframe(results_df)

with tab7:
    st.header("7. Interpret Hypothesis Test Results")
    p_value = ttest_result.pvalue
    alpha = 0.05
    st.subheader("Decision")
    st.write("Significance level (alpha):", alpha)
    st.write("p-value:", p_value)
    if p_value < alpha:
        st.write("Since the p-value is less than 0.05, we reject the null hypothesis.")
        st.write("This means there is a significant difference in mean life expectancy between countries with high internet use rates and countries with low internet use rates.")
    else:
        st.write("Since the p-value is greater than or equal to 0.05, we fail to reject the null hypothesis.")
        st.write("This means there is not enough evidence to say that the mean life expectancy is significantly different between the two groups.")

    st.subheader("Real-World Meaning")
    st.write("This result suggests that internet use rate may be related to life expectancy.")
    st.write("Countries with higher internet use may also have better healthcare, education, and living conditions.")
    st.write("This can be useful for understanding differences in development between countries.")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

with tab8:
    st.header("8. Machine Learning")
    X = df.drop(columns=["lifeexpectancy", "country", "femaleemployrate", "hivrate", "suicideper100th", "employrate"])
    y = df["lifeexpectancy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict((X_test))

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    f1 = 2 * (r2 * y_test.shape[0]) / (y_test.shape[0] + y_pred.shape[0])

    st.subheader("Model Information")
    st.write("Model: Linear Regression")
    st.write("Target Variable(X): lifeexpectancy")
    st.write(f"Features(y): {df.drop(columns=["lifeexpectancy", "country", "femaleemployrate", "hivrate", "suicideper100th", "employrate"]).columns.tolist()},(Select from the more correlate in heatmap")
    st.write("Training set size:", len(X_train))
    st.write("Testing set size:", len(X_test))

    st.subheader("Model Performance")
    st.write("R-squared:", r2)
    st.write("Accuracy:", round(r2, 2) * 100, "%")
    st.write("Mean Squared Error:", mse)
    st.write("Root Mean Squared Error:", rmse)


    st.subheader("Actual vs Predicted")
    results_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    st.dataframe(results_df)
    fig = px.scatter(
        results_df,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Life Expectancy"
    )
    st.plotly_chart(fig, use_container_width=True)

    models = {
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5

        st.subheader(name)
        st.write("R²:", r2)
        st.write("RMSE:", rmse)



