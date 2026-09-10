import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# ------------------------------------------------------------
# 1. Cấu hình Trang & Spark Session
# ------------------------------------------------------------
st.set_page_config(
    page_title="Amazon Delivery Prediction",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Amazon Delivery Time Prediction & Big Data Analytics")
st.write("Ứng dụng phân tích dữ liệu lớn và dự báo thời gian giao hàng sử dụng **PySpark MLlib** và **Streamlit**.")

@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("Amazon_Delivery_Time_Prediction") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

spark = get_spark_session()

# ------------------------------------------------------------
# 2. Sidebar - Tải Dữ liệu
# ------------------------------------------------------------
st.sidebar.header("📁 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải lên file amazon_delivery.csv", type=["csv"])

if uploaded_file is None:
    st.info("👈 Vui lòng tải file `amazon_delivery.csv` ở thanh bên trái để bắt đầu phân tích.")
    st.stop()

# ------------------------------------------------------------
# 3. Đọc & Làm sạch Dữ liệu
# ------------------------------------------------------------
pdf_raw = pd.read_csv(uploaded_file)

# Chuẩn hóa tên cột
pdf_raw.columns = [c.strip() for c in pdf_raw.columns]

# Ép kiểu dữ liệu định lượng
numeric_candidates = ["Delivery_Time", "Agent_Age", "Agent_Rating", "Distance"]
for col in numeric_candidates:
    if col in pdf_raw.columns:
        pdf_raw[col] = pd.to_numeric(pdf_raw[col], errors='coerce')

pdf_raw = pdf_raw.dropna()

# Chuyển đổi sang PySpark DataFrame
df = spark.createDataFrame(pdf_raw)

# ------------------------------------------------------------
# 4. Giao diện Tabs
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 EDA & Visualizations", "🤖 Huấn luyện Mô hình", "📈 Feature Importance & Dự báo"])

sns.set_theme(style="whitegrid")

# ============================================================
# TAB 1: EDA & TRỰC QUAN HÓA DỮ LIỆU
# ============================================================
with tab1:
    st.subheader("📋 Dữ liệu xem trước (5 dòng đầu)")
    st.dataframe(pdf_raw.head(5))
    st.write(f"**Tổng số bản ghi sạch:** {len(pdf_raw):,} dòng")

    st.markdown("---")
    st.subheader("📊 Trực quan hóa Khám phá Dữ liệu (EDA)")

    col1, col2 = st.columns(2)

    with col1:
        # 1. Distribution of Delivery Time
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(pdf_raw["Delivery_Time"], kde=True, color="teal", bins=30, ax=ax)
        ax.set_title("1. Distribution of Delivery Time", fontweight='bold')
        ax.set_xlabel("Delivery Time (mins)")
        st.pyplot(fig)
        plt.close(fig)

        # 3. Traffic Condition
        if "Traffic" in pdf_raw.columns:
            traffic_df = pdf_raw.groupby("Traffic")["Delivery_Time"].mean().reset_index().sort_values(by="Delivery_Time", ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=traffic_df, x="Traffic", y="Delivery_Time", palette="Reds_r", ax=ax)
            ax.set_title("3. Average Delivery Time by Traffic Condition", fontweight='bold')
            ax.set_ylabel("Avg Delivery Time (mins)")
            st.pyplot(fig)
            plt.close(fig)

        # 5. Area Distribution
        if "Area" in pdf_raw.columns:
            area_df = pdf_raw["Area"].value_counts().reset_index()
            area_df.columns = ["Area", "count"]
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=area_df, x="Area", y="count", palette="Greens_r", ax=ax)
            ax.set_title("5. Total Number of Deliveries by Area", fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

        # 7. Boxplot Traffic
        if "Traffic" in pdf_raw.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(data=pdf_raw, x="Traffic", y="Delivery_Time", palette="Set2", ax=ax)
            ax.set_title("7. Delivery Time Distribution by Traffic", fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        # 2. Correlation Matrix
        numeric_cols = pdf_raw.select_dtypes(include=np.number).columns
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(pdf_raw[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title("2. Correlation Matrix of Numeric Features", fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

        # 4. Weather Condition
        if "Weather" in pdf_raw.columns:
            weather_df = pdf_raw.groupby("Weather")["Delivery_Time"].mean().reset_index().sort_values(by="Delivery_Time", ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=weather_df, x="Weather", y="Delivery_Time", palette="Blues_r", ax=ax)
            ax.set_title("4. Average Delivery Time by Weather", fontweight='bold')
            plt.xticks(rotation=25)
            st.pyplot(fig)
            plt.close(fig)

        # 6. Vehicle Type
        if "Vehicle" in pdf_raw.columns:
            vehicle_df = pdf_raw.groupby("Vehicle")["Delivery_Time"].mean().reset_index().sort_values(by="Delivery_Time", ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=vehicle_df, x="Vehicle", y="Delivery_Time", palette="Purples_r", ax=ax)
            ax.set_title("6. Average Delivery Time by Vehicle Type", fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

        # 8. Boxplot Weather
        if "Weather" in pdf_raw.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(data=pdf_raw, x="Weather", y="Delivery_Time", palette="Set3", ax=ax)
            ax.set_title("8. Delivery Time Distribution by Weather", fontweight='bold')
            plt.xticks(rotation=25)
            st.pyplot(fig)
            plt.close(fig)

    # 9. Agent Rating Scatter
    if "Agent_Rating" in pdf_raw.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(data=pdf_raw, x="Agent_Rating", y="Delivery_Time", alpha=0.5, color="darkorange", ax=ax)
        ax.set_title("9. Agent Rating vs. Delivery Time", fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# TAB 2: PIPELINE & HUẤN LUYỆN MÔ HÌNH
# ============================================================
with tab2:
    st.subheader("⚙️ Feature Engineering & Training Pipeline")

    categorical_cols = [c for c in ["Weather", "Traffic", "Vehicle", "Area", "Category"] if c in df.columns]
    numerical_cols = [c for c in ["Agent_Age", "Agent_Rating", "Distance"] if c in df.columns]

    indexed_cols = [f"{c}_index" for c in categorical_cols]
    encoded_cols = [f"{c}_vec" for c in categorical_cols]

    indexer = StringIndexer(inputCols=categorical_cols, outputCols=indexed_cols, handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols)
    assembler_inputs = encoded_cols + numerical_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    pipeline = Pipeline(stages=[indexer, encoder, assembler])
    pipeline_model = pipeline.fit(df)
    prepared_df = pipeline_model.transform(df)

    train_data, test_data = prepared_df.randomSplit([0.8, 0.2], seed=42)

    st.success(f"Dữ liệu phân chia hoàn tất: **Train ({train_data.count()} mẫu)** | **Test ({test_data.count()} mẫu)**")

    if st.button("🚀 Bắt đầu huấn luyện các mô hình"):
        with st.spinner("Đang tiến hành huấn luyện Linear Regression, Random Forest và GBT..."):
            models = {
                "Linear Regression": LinearRegression(featuresCol="features", labelCol="Delivery_Time"),
                "Random Forest": RandomForestRegressor(featuresCol="features", labelCol="Delivery_Time", numTrees=50, seed=42),
                "Gradient-Boosted Trees": GBTRegressor(featuresCol="features", labelCol="Delivery_Time", maxIter=30, seed=42)
            }

            evaluator_rmse = RegressionEvaluator(labelCol="Delivery_Time", predictionCol="prediction", metricName="rmse")
            evaluator_r2 = RegressionEvaluator(labelCol="Delivery_Time", predictionCol="prediction", metricName="r2")
            evaluator_mae = RegressionEvaluator(labelCol="Delivery_Time", predictionCol="prediction", metricName="mae")

            results = []
            for name, model in models.items():
                trained_model = model.fit(train_data)
                predictions = trained_model.transform(test_data)

                results.append({
                    "Model": name,
                    "RMSE": evaluator_rmse.evaluate(predictions),
                    "R2 Score": evaluator_r2.evaluate(predictions),
                    "MAE": evaluator_mae.evaluate(predictions)
                })

            results_df = pd.DataFrame(results)
            st.session_state['results_df'] = results_df

    if 'results_df' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Bảng so sánh kết quả mô hình")
        st.dataframe(st.session_state['results_df'].style.highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'))

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=st.session_state['results_df'], x="Model", y="RMSE", palette="viridis", ax=ax)
        ax.set_title("Model Comparison - Root Mean Squared Error (RMSE)", fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# TAB 3: FEATURE IMPORTANCE & DỰ BÁO
# ============================================================
with tab3:
    st.subheader("📌 Feature Importance & Trực quan dự báo")

    rf = RandomForestRegressor(featuresCol="features", labelCol="Delivery_Time", numTrees=50, seed=42)
    rf_model = rf.fit(train_data)
    importances = rf_model.featureImportances.toArray()

    try:
        attrs = train_data.schema["features"].metadata["ml_attr"]["attrs"]
        feature_list = []
        for key in ["numeric", "binary", "nominal"]:
            if key in attrs:
                for item in attrs[key]:
                    feature_list.append((item["idx"], item["name"]))
        feature_list.sort(key=lambda x: x[0])
        feature_names = [x[1] for x in feature_list]
    except Exception:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    if len(feature_names) != len(importances):
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    col_fi1, col_fi2 = st.columns(2)

    with col_fi1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=feature_importance_df.head(15), x='Importance', y='Feature', palette='magma', ax=ax)
        ax.set_title("Top Feature Importance Analysis (Random Forest)", fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

    with col_fi2:
        predictions = rf_model.transform(test_data)
        pred_pd = predictions.select("Delivery_Time", "prediction").limit(200).toPandas()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=pred_pd, x="Delivery_Time", y="prediction", alpha=0.7, color="crimson", ax=ax)
        ax.plot([pred_pd["Delivery_Time"].min(), pred_pd["Delivery_Time"].max()],
                [pred_pd["Delivery_Time"].min(), pred_pd["Delivery_Time"].max()], 'k--', lw=2)
        ax.set_title("Actual vs Predicted Delivery Time (Random Forest)", fontweight='bold')
        ax.set_xlabel("Actual Delivery Time (mins)")
        ax.set_ylabel("Predicted Delivery Time (mins)")
        st.pyplot(fig)
        plt.close(fig)
