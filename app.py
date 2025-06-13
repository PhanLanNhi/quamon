import streamlit as st
import pandas as pd
import os
import joblib

st.set_page_config(
    page_title="IoT Fault Detection Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== CSS nâng cấp hiện đại ====
st.markdown("""
    <style>
    .stApp {background-color: #f4f7fd;}
    .main-title {
        color: #273c75; 
        font-size: 2.45rem; 
        font-weight: 800; 
        letter-spacing: -1.5px;
        margin-bottom: -3px;
    }
    .subtitle {
        color: #6c757d; 
        font-size: 1.07rem; 
        margin-bottom: 15px;
    }
    .card {
        background: linear-gradient(120deg,#f1f5ff 70%,#e0ecff 100%);
        border-radius: 18px; 
        box-shadow: 0 3px 24px #b8cbfa33;
        padding: 28px 24px 18px 24px; 
        margin-bottom: 24px;
    }
    .side-card {
        background: linear-gradient(135deg,#e0f7fa 70%,#e3f0fc 100%);
        border-radius: 13px; 
        box-shadow: 0 2px 12px #b8cbfa20;
        padding: 14px 18px 8px 18px; 
        margin-bottom: 14px;
    }
    .stButton button {
        background: linear-gradient(90deg,#5f6dfa 0,#3750fa 100%);
        color: white !important;
        font-weight: bold !important;
        border-radius: 12px;
        padding: 10px 34px;
        margin-top: 8px;
        font-size: 1.13rem;
        box-shadow: 0 2px 16px #b8cbfa20;
        transition: 0.18s;
    }
    .stButton button:hover {background: linear-gradient(90deg,#4f46e5 0,#4338ca 100%)!important;}
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.13rem;
        padding: 0.38rem 1.35rem;
        color: #3a43d6;
        font-weight: 700;
    }
    .stTextInput > div > div > input, .stNumberInput input {
        background: #eaf2fd !important;
        border-radius: 7px;
        border: 1.4px solid #dbeafe;
        font-size: 1.08rem;
    }
    .stDataFrame {border-radius: 8px !important;}
    .success-card {
        background: #e8faf0;
        border-radius: 10px;
        padding: 12px 16px;
        color: #088c52;
        margin-top: 10px;
        font-size: 1.08rem;
        border-left: 6px solid #38bdf8;
    }
    .mytag {
        display: inline-block;
        background: #e0e7ff;
        border-radius: 5px;
        color: #4338ca;
        font-size: 0.97rem;
        padding: 2.5px 10px;
        margin: 2.5px 7px 2.5px 0;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)

# ==== Header ====
st.markdown("<div class='main-title'>🛰️ Dự đoán lỗi thiết bị IoT</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>So sánh mô hình <b>XGBoost</b> & <b>LightGBM</b>. Kết quả trực quan, thao tác hiện đại.</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🧑‍🔬 Dự đoán mẫu lẻ",
    "📄 Dự đoán file CSV",
    "📘 Hướng dẫn"
])

# ==== Load model ====
MODEL_DIR = r'D:\dow\project\evaluation\pipline'
xgb_model_path = os.path.join(MODEL_DIR, 'xgb_final_model.pkl')
lgbm_model_path = os.path.join(MODEL_DIR, 'lgbm_final_model.pkl')
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
    'Normalized_Temp', 'Normalized_Vibration', 'Normalized_Pressure',
    'Normalized_Voltage', 'Normalized_Current',
    'FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score'
]

# ==== TAB 1: Dự đoán mẫu lẻ ====
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### ✍️ Nhập thông số cảm biến & chọn mô hình")
    model_name = st.selectbox("Chọn mô hình", ("XGBoost", "LightGBM"), key="model_select_1")
    st.markdown("##### Nhập giá trị cảm biến:")

    cols = st.columns(4)
    inputs = {}
    for i, feat in enumerate(feature_list):
        with cols[i % 4]:
            inputs[feat] = st.number_input(feat, value=0.0, step=0.01, format="%.3f", key=f"{feat}_input")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔮 Dự đoán", use_container_width=True):
        input_df = pd.DataFrame([inputs])
        model = xgb_model if model_name == "XGBoost" else lgbm_model
        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            result_color = "#e86a1a" if pred == 1 else "#2563eb"
            result_txt = "Lỗi (1)" if pred == 1 else "Bình thường (0)"
            st.markdown(
                f"<div style='background: #f7f8fd; border-radius: 9px; padding: 16px 12px 7px 12px; margin-bottom:15px; "
                f"font-size:1.3rem; color:{result_color};'><b>Kết quả dự đoán:</b> <span style='color:{result_color};'><b>{result_txt}</b></span></div>",
                unsafe_allow_html=True
            )
            st.markdown("##### Xác suất từng lớp:")
            df_prob = pd.DataFrame({'Xác suất': proba}, index=['Bình thường (0)', 'Lỗi (1)'])
            st.dataframe(df_prob, use_container_width=True, height=110)
            st.bar_chart(df_prob)
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")

    st.markdown(
        "<div class='side-card'><b>Hướng dẫn:</b><br>"
        "<ul style='margin-bottom:0;'>"
        "<li>Chọn mô hình <b>XGBoost</b> hoặc <b>LightGBM</b>.</li>"
        "<li>Nhập đủ 8 thông số cảm biến.</li>"
        "<li>Nhấn <b>Dự đoán</b> để xem kết quả.</li>"
        "<li><b>0 = Bình thường, 1 = Lỗi</b>.</li>"
        "</ul></div>",
        unsafe_allow_html=True
    )

# ==== TAB 2: Dự đoán file CSV ====
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 📤 Tải lên file CSV để dự đoán hàng loạt")
    uploaded_file = st.file_uploader("Tải lên file CSV dữ liệu", type=['csv'])
    model_batch = st.selectbox("Chọn mô hình batch", ("XGBoost", "LightGBM"), key="model_select_2")
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            if not all([col in df_batch.columns for col in feature_list]):
                st.error("❌ File thiếu cột, cần đủ: " + ", ".join(feature_list))
            else:
                model = xgb_model if model_batch == "XGBoost" else lgbm_model
                batch_pred = model.predict(df_batch[feature_list])
                st.success("✅ Dự đoán thành công! Kết quả bên dưới:")
                st.dataframe(df_batch.assign(Fault_Prediction=batch_pred), use_container_width=True)
                csv_out = df_batch.assign(Fault_Prediction=batch_pred).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Tải về kết quả CSV",
                    data=csv_out,
                    file_name='prediction_results.csv',
                    mime='text/csv'
                )
        except Exception as e:
            st.error(f"Lỗi khi đọc file hoặc dự đoán batch: {e}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='side-card'><b>Mẫu cột hợp lệ:</b> " +
        "".join(f"<span class='mytag'>{f}</span>" for f in feature_list) + "</div>",
        unsafe_allow_html=True
    )

# ==== TAB 3: Hướng dẫn sử dụng ====
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🚦 Hướng dẫn sử dụng")
    st.markdown("""
    <ol>
        <li><b>Tab <span style="color:#2563eb">Dự đoán mẫu lẻ</span>:</b> Nhập từng thông số cảm biến, chọn mô hình và nhấn <b>Dự đoán</b>.</li>
        <li><b>Tab <span style="color:#2563eb">Dự đoán file CSV</span>:</b> Upload file chứa dữ liệu cảm biến với đúng 8 cột đầu vào.</li>
        <li><b>Kết quả:</b> <span style="color:#2563eb"><b>0 = Bình thường, 1 = Lỗi</b></span>. Có thể tải kết quả dự đoán hàng loạt.</li>
        <li>File CSV cần đúng các cột:
            <br>""" + "".join(f"<span class='mytag'>{f}</span>" for f in feature_list) + """
        </li>
    </ol>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
