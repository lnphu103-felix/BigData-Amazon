import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

st.set_page_config(page_title="Amazon Delivery Prediction", page_icon="🚚", layout="wide")

st.title("🚚 Amazon Delivery Time Prediction & Big Data Analytics")

# 1. Tối ưu Spark Session cho 1GB RAM trên Streamlit Cloud
@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("AmazonDeliveryLight") \
        .config("spark.driver.memory", "512m") \
        .config("spark.executor.memory", "512m") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

spark = get_spark_session()

st.sidebar.header("📁 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải lên file amazon_delivery.csv", type=["csv"])

if uploaded_file is None:
    st.info("👈 Vui lòng tải file `amazon_delivery.csv` ở thanh bên trái để bắt đầu.")
    st.stop()

# 2. Đọc & Xử lý Dữ liệu
pdf_raw = pd.read_csv(uploaded_file)
pdf_raw.columns = [c.strip() for c in pdf_raw.columns]

numeric_candidates = ["Delivery_Time", "Agent_Age", "Agent_Rating", "Distance"]
for col in numeric_candidates:
    if col in pdf_raw.columns:
        pdf_raw[col] = pd.to_numeric(pdf_raw[col], errors='coerce')

pdf_raw = pdf_raw.dropna()

# Giảm tải dữ liệu nếu file quá lớn (>10.000 dòng)
if len(pdf_raw) > 10000:
    pdf_sample = pdf_raw.sample(n=10000, random_state=42)
else:
    pdf_sample = pdf_raw.copy()

df = spark.createDataFrame(pdf_sample)

tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Huấn luyện Mô hình", "📈 Feature Importance"])

sns.set_theme(style="whitegrid")

# ============================================================
# TAB 1: EDA
# ============================================================
with tab1:
    st.subheader("📋 Dữ liệu xem trước")
    st.dataframe(pdf_sample.head(5))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(pdf_sample["Delivery_Time"], kde=True, color="teal", bins=25, ax=ax)
        ax.set_title("Distribution of Delivery Time", fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

        if "Traffic" in pdf_sample.columns:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(data=pdf_sample.groupby("Traffic")["Delivery_Time"].mean().reset_index(), x="Traffic", y="Delivery_Time", palette="Reds_r", ax=ax)
            ax.set_title("Avg Delivery Time by Traffic", fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.heatmap(pdf_sample.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix", fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

        if "Weather" in pdf_sample.columns:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(data=pdf_sample.groupby("Weather")["Delivery_Time"].mean().reset_index(), x="Weather", y="Delivery_Time", palette="Blues_r", ax=ax)
            ax.set_title("Avg Delivery Time by Weather", fontweight='bold')
            plt.xticks(rotation=20)
            st.pyplot(fig)
            plt.close(fig)

# ============================================================
# TAB 2: HUẤN LUYỆN MÔ HÌNH (TỐI ƯU RAM)
# ============================================================
with tab2:
    st.subheader("🤖 Huấn luyện Mô hình")

    model_choice = st.selectbox(
        "Chọn mô hình muốn huấn luyện:",
        ["Chạy tất cả (Siêu nhẹ)", "Linear Regression", "Random Forest", "Gradient-Boosted Trees"]
    )

    categorical_cols = [c for c in ["Weather", "Traffic", "Vehicle", "Area", "Category"] if c in df.columns]
    numerical_cols = [c for c in ["Agent_Age", "Agent_Rating", "Distance"] if c in df.columns]

    indexed_cols = [f"{c}_index" for c in categorical_cols]
    encoded_cols = [f"{c}_vec" for c in categorical_cols]

    indexer = StringIndexer(inputCols=categorical_cols, outputCols=indexed_cols, handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols)
    assembler = VectorAssembler(inputCols=encoded_cols + numerical_cols, outputCol="features")

    pipeline = Pipeline(stages=[indexer, encoder, assembler])
    prepared_df = pipeline.fit(df).transform(df)

    train_data, test_data = prepared_df.randomSplit([0.8, 0.2], seed=42)

    if st.button("🚀 Bắt đầu huấn luyện"):
        with st.spinner("Đang xử lý..."):
            models = {}
            if model_choice in ["Chạy tất cả (Siêu nhẹ)", "Linear Regression"]:
                models["Linear Regression"] = LinearRegression(featuresCol="features", labelCol="Delivery_Time")
            if model_choice in ["Chạy tất cả (Siêu nhẹ)", "Random Forest"]:
                models["Random Forest"] = RandomForestRegressor(featuresCol="features", labelCol="Delivery_Time", numTrees=15, maxDepth=5, seed=42)
            if model_choice in ["Chạy tất cả (Siêu nhẹ)", "Gradient-Boosted Trees"]:
                models["GBT Regressor"] = GBTRegressor(featuresCol="features", labelCol="Delivery_Time", maxIter=10, maxDepth=4, seed=42)

            eval_rmse = RegressionEvaluator(labelCol="Delivery_Time", predictionCol="prediction", metricName="rmse")
            eval_r2 = RegressionEvaluator(labelCol="Delivery_Time", predictionCol="prediction", metricName="r2")
            eval_mae = RegressionEvaluator(labelCol="Delivery_Time", predictionCol="prediction", metricName="mae")

            res = []
            for name, model in models.items():
                m = model.fit(train_data)
                preds = m.transform(test_data)
                res.append({
                    "Model": name,
                    "RMSE": eval_rmse.evaluate(preds),
                    "R2 Score": eval_r2.evaluate(preds),
                    "MAE": eval_mae.evaluate(preds)
                })

            res_df = pd.DataFrame(res)
            st.dataframe(res_df.style.highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'))

            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(data=res_df, x="Model", y="RMSE", palette="viridis", ax=ax)
            ax.set_title("RMSE Comparison", fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

            gc.collect()

# ============================================================
# TAB 3: FEATURE IMPORTANCE
# ============================================================
with tab3:
    st.subheader("📌 Feature Importance (Random Forest)")
    if st.button("Phân tích Feature Importance"):
        with st.spinner("Đang tính toán..."):
            rf = RandomForestRegressor(featuresCol="features", labelCol="Delivery_Time", numTrees=15, maxDepth=5, seed=42)
            rf_model = rf.fit(train_data)
            importances = rf_model.featureImportances.toArray()

            feat_names = [f"Feature_{i}" for i in range(len(importances))]
            try:
                attrs = train_data.schema["features"].metadata["ml_attr"]["attrs"]
                feat_list = []
                for k in ["numeric", "binary", "nominal"]:
                    if k in attrs:
                        for item in attrs[k]:
                            feat_list.append((item["idx"], item["name"]))
                feat_list.sort(key=lambda x: x[0])
                if len(feat_list) == len(importances):
                    feat_names = [x[1] for x in feat_list]
            except Exception:
                pass

            fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=fi_df.head(10), x='Importance', y='Feature', palette='magma', ax=ax)
            ax.set_title("Top 10 Feature Importance", fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

            gc.collect()
