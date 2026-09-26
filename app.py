import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("fraudTrain_sample.csv")
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"]
    )
    return df

df = load_data()

st.title("💳 Credit Card Fraud Detection")
st.subheader("Exploratory Data Analysis Dashboard")
st.write(
    "Interactive analysis of credit card transactions "
    "to identify patterns in fraudulent transactions."
)

total_transactions = len(df)
fraud_transactions = df["is_fraud"].sum()
non_fraud_transactions = total_transactions - fraud_transactions
fraud_percentage = fraud_transactions / total_transactions * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", f"{total_transactions:,}")

with col2:
    st.metric("Fraud Transactions", f"{fraud_transactions:,}")

with col3:
    st.metric("Non-Fraud Transactions", f"{non_fraud_transactions:,}")

with col4:
    st.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")

st.divider()

st.header("1. Fraud vs Non-Fraud Transactions")

fraud_counts = df["is_fraud"].value_counts().reset_index()
fraud_counts.columns = ["is_fraud", "count"]

fraud_counts["transaction_type"] = fraud_counts["is_fraud"].map({
    0: "Non-Fraud",
    1: "Fraud"
})

fraud_counts["percentage"] = (
    fraud_counts["count"] /
    fraud_counts["count"].sum() * 100
)

fig = px.bar(
    fraud_counts,
    x="transaction_type",
    y="percentage",
    text="percentage",
    title="Fraud vs Non-Fraud Transactions (%)",
    labels={
        "transaction_type": "Transaction Type",
        "percentage": "Percentage of Transactions"
    }
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig.update_yaxes(
    range=[0, 105],
    ticksuffix="%"
)

st.plotly_chart(fig, width="stretch")

st.divider()

st.header("2. Transaction Amount Analysis")

amount_limit = df["amt"].quantile(0.99)
amount_data = df[df["amt"] <= amount_limit]

fig = px.histogram(
    amount_data,
    x="amt",
    nbins=50,
    title="Transaction Amount Distribution",
    labels={"amt": "Transaction Amount"}
)

st.plotly_chart(fig, width="stretch")

st.caption(
    f"Showing transaction amounts up to the 99th percentile "
    f"(${amount_limit:,.2f}) for better readability."
)

st.subheader("Fraud vs Transaction Amount")

box_data = df[df["amt"] <= amount_limit].copy()

box_data["fraud_status"] = box_data["is_fraud"].map({
    0: "Non-Fraud",
    1: "Fraud"
})

fig = px.box(
    box_data,
    x="fraud_status",
    y="amt",
    points=False,
    title="Fraud vs Transaction Amount",
    labels={
        "fraud_status": "Fraud Status",
        "amt": "Transaction Amount"
    }
)

st.plotly_chart(fig, width="stretch")

st.caption(
    "The chart uses transactions up to the 99th percentile "
    "to make the comparison easier to read."
)

st.divider()

st.header("3. Fraud Transactions by Category")

fraud_category = (
    df[df["is_fraud"] == 1]
    .groupby("category")
    .size()
    .reset_index(name="fraud_count")
    .sort_values("fraud_count", ascending=False)
)

fig = px.bar(
    fraud_category,
    x="category",
    y="fraud_count",
    title="Fraud Transactions by Category",
    labels={
        "category": "Category",
        "fraud_count": "Fraud Transactions"
    }
)

st.plotly_chart(fig, width="stretch")

st.divider()

st.header("4. Fraud Transactions by Hour")

df["hour"] = df["trans_date_trans_time"].dt.hour

fraud_hour = (
    df[df["is_fraud"] == 1]
    .groupby("hour")
    .size()
    .reset_index(name="fraud_count")
)

fig = px.bar(
    fraud_hour,
    x="hour",
    y="fraud_count",
    title="Fraud Transactions by Hour",
    labels={
        "hour": "Hour of Day",
        "fraud_count": "Fraud Transactions"
    }
)

fig.update_xaxes(
    tickmode="linear",
    dtick=1
)

st.plotly_chart(fig, width="stretch")

st.divider()

st.header("5. Correlation Analysis")

numeric_columns = [
    "amt",
    "zip",
    "lat",
    "long",
    "city_pop",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
    "hour"
]

correlation = df[numeric_columns].corr()

fig = px.imshow(
    correlation,
    text_auto=".2f",
    aspect="auto",
    title="Correlation Heatmap"
)

st.plotly_chart(fig, width="stretch")

st.divider()

st.header("6. Dataset Preview")

st.write(
    f"Showing the first 10 rows of the dataset "
    f"containing {len(df):,} transactions."
)

st.dataframe(
    df.head(10),
    width="stretch"
)