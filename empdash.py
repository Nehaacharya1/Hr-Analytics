#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px


# Page Configuration


st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #F5F7FA;
}

[data-testid="stSidebar"] {
    background-color: #E8F0FE;
}

h1, h2, h3 {
    color: #2C3E50;
}

div[data-testid="metric-container"] {
    background-color: white;
    border: 2px solid #4F46E5;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# Load Dataset


df = pd.read_csv("C:/Users/nehau/Downloads/Palo Alto Networks.csv"
)

# Convert Attrition if needed

if df["Attrition"].dtype == "object":
    df["Attrition"] = df["Attrition"].replace({
        "Yes": 1,
        "No": 0
    })
df["Attrition Label"] = df["Attrition"].map({
    0: "Stayed",
    1: "Left"
})

# Sidebar Filter


st.sidebar.header("Filters")

if "Department" in df.columns:
    department = st.sidebar.selectbox(
        "Select Department",
        ["All"] + list(df["Department"].unique())
    )

    if department != "All":
        df = df[df["Department"] == department]

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=100
)

st.sidebar.title("HR Analytics")


# Title


st.markdown("""
<h1 style='text-align:center;
color:#4F46E5;
font-size:42px;'>
📊 Employee Attrition Prediction & Risk Scoring System
</h1>
""", unsafe_allow_html=True)

# Tabs


tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Risk Analysis",
    "Department Analysis",
    "Model Performance"
])


# DASHBOARD TAB

with tab1:

    st.subheader("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Employees",
        len(df)
    )



    col2.metric(
        "Employees Left",
        int(df["Attrition"].sum())
    )


    col3.metric(
        "Attrition Rate",
        f"{df['Attrition'].mean()*100:.2f}%"
    )

    if "MonthlyIncome" in df.columns:
        col4.metric(
            "Avg Income",
            f"${df['MonthlyIncome'].mean():,.0f}"
        )

    st.subheader("Attrition Distribution")

    fig = px.pie(
        df,
        names="Attrition",
        color="Attrition", 
        color_discrete_map={
            "Stayed": "#2ECC71", 
            "Left": "#E74C3C"
        },
        hole=0.4,
        title="Employee Attrition Distribution"
   )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Employee Data")

    st.dataframe(df.head(20))



# RISK ANALYSIS TAB


with tab2:

    st.subheader("Employee Risk Categories")

    risk_df = pd.DataFrame()

    risk_df["Risk Category"] = df["Attrition"].map({
        0: "Low Risk",
        1: "High Risk",

    })

    st.bar_chart(
        risk_df["Risk Category"].value_counts(),

    )

    st.info(
        "Currently using Attrition labels as risk categories. "
        "Replace with Random Forest prediction probabilities later."
    )



# DEPARTMENT ANALYSIS TAB


with tab3:

    if "Department" in df.columns:

        st.subheader("Department Distribution")

        dept_count = (
            df["Department"]
            .value_counts()
            .reset_index()
        )

        dept_count.columns = [
            "Department",
            "Employees"
        ]

        fig = px.bar(
            dept_count,
            x="Department",
            y="Employees",
            color="Employees",
            color_continuous_scale="Blues",
            title="Employees by Department"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if (
        "Department" in df.columns and
        "MonthlyIncome" in df.columns
    ):

        st.subheader(
            "Average Monthly Income by Department"
        )

        dept_income = (
            df.groupby("Department")
            ["MonthlyIncome"]

            .mean()
            .reset_index()
        )

        fig = px.bar(
            dept_income,
            x="Department",
            y="MonthlyIncome",
            color="MonthlyIncome",
            color_continuous_scale="Viridis"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )



# MODEL PERFORMANCE TAB


with tab4:

    st.subheader("Random Forest Model Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Accuracy",
            "84.01%"
        )

        st.metric(
            "Precision",
            "50%"
        )

    with col2:
        st.metric(
            "Recall",
            "27.66%"
        )

        st.metric(
            "ROC-AUC",
            "79.40%"
        )

    st.write(
        """
        ### Interpretation

        - Accuracy is good (84%).
        - ROC-AUC shows good discrimination ability.
        - Recall is low, meaning many employees who actually leave are missed.
        - Further tuning, feature engineering, and threshold optimization can improve attrition detection.
        """
    )

