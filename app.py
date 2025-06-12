import streamlit as st
import pandas as pd
import os
import joblib

# ==== Cấu hình giao diện chung ====
st.set_page_config(
    page_title="IoT Fault Detection Demo", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== Style bổ sung ====
st.markdown("""
    <style>
    .stButton button {
        background-color: #1a73e8 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px;
        padding: 8px 24px;
        margin-top: 8px;
    }
    .st-expanderHeader {
        font-weight: 600;
        color: #2563eb;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.12rem;
        padding: 0.3rem 1.2rem;
        color: #1a73e8;
    }
    .stTextInput > div > div > input, .stNumberInput input {
        background: #f1f5f9 !important;
        border-radius: 7px;
    }
    .stApp {
        background: #f8fafc;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='color:#1a73e8; font-size:2.3rem; font-weight:700; margin-bottom:10px;'>🛠️ Dự đoán lỗi thiết bị IoT (XGBoost & LightGBM)</h1>", 
    unsafe_allow_html=True
)

MODEL_DIR = 'evaluation/pipline'
xgb_model_path = os.path.join(MODEL_DIR, 'xgb_final_model.pkl')
lgbm_model_path = os.path.join(MODEL_DIR, 'lgbm_final_model.pkl')

# ==== Kiểm tra file model ====
if not (os.path.exists(xgb_model_path) and os.path.exists(lgbm_model_path)):
    st.error("❌ Không tìm thấy file model! Hãy kiểm tra lại đường dẫn.")
    st.stop()

try:
    xgb_model = joblib.load(xgb_model_path)
    lgbm_model = joblib.load(lgbm_model_path)
except Exception as e:
    st.error(f"❌ Lỗi khi load model: {e}")
    st.stop()

feature_list = [
    'Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current',
    'FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score'
]

# ===== Tabs giao diện chính =====
tab1, tab2, tab3 = st.tabs([
    "📌 Dự đoán mẫu lẻ", 
    "📂 Dự đoán file CSV", 
    "❓ Hướng dẫn sử dụng"
])

# ==== TAB 1: Dự đoán mẫu lẻ ====
with tab1:
    st.markdown("### 📝 Nhập thông số cảm biến & chọn mô hình để dự đoán lỗi")
    st.write(
        """
        <span style='font-size: 1.02rem; color:#373d44;'>
        Điền đủ giá trị các trường dưới đây, chọn mô hình muốn sử dụng, và nhấn <b>Dự đoán</b>.<br>
        Kết quả dự đoán và xác suất lỗi sẽ hiển thị ngay lập tức.
        </span>
        """, unsafe_allow_html=True
    )
    with st.container():
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown("#### 1. Chọn mô hình")
            model_name = st.radio("Chọn model", ("XGBoost", "LightGBM"), horizontal=True, index=0)

            st.markdown("#### 2. Nhập thông số cảm biến")
            inputs = {}
            for feat in feature_list:
                inputs[feat] = st.number_input(feat, value=0.0, format="%.3f", step=0.01)

            if st.button("🔍 Dự đoán", use_container_width=True):
                input_df = pd.DataFrame([inputs])
                model = xgb_model if model_name == "XGBoost" else lgbm_model
                try:
                    pred = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0]
                    label = {
                        0: "✅ Bình thường",
                        1: "⚠️ Lỗi loại 1",
                        2: "❗ Lỗi loại 2",
                        3: "🔥 Lỗi loại 3"
                    }.get(pred, str(pred))
                    st.success(f"**Kết quả dự đoán Fault_Type:** `{label}`")
                    st.markdown("##### Xác suất từng lớp lỗi:")
                    df_prob = pd.DataFrame({'Xác suất (%)': np.round(proba*100, 2)}, 
                                          index=[f'Fault_Type {i}' for i in range(len(proba))])
                    st.dataframe(df_prob, use_container_width=True, height=180)
                    st.bar_chart(df_prob)
                except Exception as e:
                    st.error(f"❌ Lỗi khi dự đoán: {e}")

        with col2:
            st.markdown("""
            <div style='background:#f0f6ff; padding: 18px 16px; border-radius:14px; font-size: 1.10rem;'>
                <b>Hướng dẫn:</b> <br>
                <ul>
                  <li>Chọn mô hình: <b>XGBoost</b> hoặc <b>LightGBM</b></li>
                  <li>Điền giá trị cảm biến cho từng trường</li>
                  <li>Nhấn <b>Dự đoán</b> để xem kết quả</li>
                  <li><b>Fault_Type:</b> 0 = Bình thường, 1/2/3 = các loại lỗi</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ==== TAB 2: Dự đoán file CSV ====
with tab2:
    st.markdown("### 📂 Upload file CSV để dự đoán hàng loạt")
    uploaded_file = st.file_uploader("Tải lên file CSV dữ liệu", type=['csv'])
    model_batch = st.radio("Chọn mô hình", ("XGBoost", "LightGBM"), horizontal=True, index=0)
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            if not all([col in df_batch.columns for col in feature_list]):
                st.error("⚠️ File thiếu cột, cần đủ: " + ", ".join(feature_list))
            else:
                model = xgb_model if model_batch == "XGBoost" else lgbm_model
                batch_pred = model.predict(df_batch[feature_list])
                label_map = {
                    0: "Bình thường",
                    1: "Lỗi loại 1",
                    2: "Lỗi loại 2",
                    3: "Lỗi loại 3"
                }
                pred_label = [label_map.get(x, x) for x in batch_pred]
                st.success("✅ Dự đoán thành công! Kết quả bên dưới:")
                st.dataframe(df_batch.assign(Fault_Prediction=pred_label), use_container_width=True)
                csv_out = df_batch.assign(Fault_Prediction=pred_label).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Tải về kết quả CSV",
                    data=csv_out,
                    file_name='prediction_results.csv',
                    mime='text/csv'
                )
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file hoặc dự đoán batch: {e}")

# ==== TAB 3: Hướng dẫn sử dụng ====
with tab3:
    st.markdown("### ℹ️ Hướng dẫn sử dụng hệ thống dự đoán lỗi thiết bị IoT")
    st.markdown("""
    <div style='background:#e8f3fe; border-radius: 12px; padding: 16px; font-size: 1.10rem;'>
    <ul>
        <li><b>Tab 1:</b> Dự đoán nhanh từng mẫu bằng cách nhập số liệu cảm biến và chọn mô hình.</li>
        <li><b>Tab 2:</b> Upload file CSV (có đầy đủ các cột cảm biến) để dự đoán hàng loạt.</li>
        <li>Kết quả trả về gồm nhãn lỗi (<b>Fault_Type</b>) và xác suất từng loại lỗi.</li>
        <li>Mô hình đã huấn luyện gồm <b>XGBoost</b> và <b>LightGBM</b>. Có thể chọn nhanh giữa 2 loại này.</li>
        <li>0 = Bình thường, 1/2/3 = các loại lỗi khác nhau.</li>
        <li>File kết quả dự đoán CSV có thể tải về ngay.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
