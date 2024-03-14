import shap
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as mlp
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from library.preprocessing import FeatureSelect, LimitOutlier, FeatureEngineer

if "random" not in st.session_state:
    st.session_state.random = np.random.default_rng(seed=6)

# 네비게이션 구성 설정
st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

# 데이터 캐싱
@st.cache_data
def get_model():
    model = joblib.load("model/baseline_model.pkl")
    explainer = shap.TreeExplainer(model[-1], feature_perturbation="tree_path_dependent")
    return model, explainer

def choice_random(data):
    random_data = data.loc[st.session_state.random.choice(data.index), :]
    st.session_state.loan_amnt = random_data.loan_amnt
    st.session_state.term = random_data.term
    st.session_state.int_rate = random_data.int_rate
    st.session_state.dti = random_data.dti
    st.session_state.fico_range_low = random_data.fico_range_low
    st.session_state.fico_range_high = random_data.fico_range_high
    st.session_state.last_fico_range_high = random_data.last_fico_range_high
    st.session_state.last_fico_range_low = random_data.last_fico_range_low
    st.session_state.open_acc = random_data.open_acc
    st.session_state.revol_util = random_data.revol_util
    st.session_state.total_acc = random_data.total_acc
    st.session_state.verification_status = random_data.verification_status
    st.session_state.avg_cur_bal = random_data.avg_cur_bal
    st.session_state.mo_sin_old_rev_tl_op = random_data.mo_sin_old_rev_tl_op
    st.session_state.pct_tl_nvr_dlq = random_data.pct_tl_nvr_dlq

model, explainer = get_model()

# main페이지 변수 할당
if 'data' in st.session_state:
    numric_features = st.session_state.numric_features
    categorical = st.session_state.categorical
    target = st.session_state.target
    data = st.session_state.data
    colors = st.session_state.colors
    
else:
    st.switch_page("main.py")


# CSS로 버튼 center
################################################################################
# CSS 스타일을 포함한 HTML 문자열
centered_button_style = """
<style>
    .stButton>button {
        display: block;
        margin: auto;
        border: None;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }
    .stButton>button:hover {
        background-color: whiteblue !important;
        color: red !important;
        border: 0.2px solid #E5E5E5 !important;
        box-shadow: None !important;
        font-weight: bold;
        transition: box-shadow 0.3s ease;
    }
    
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5{
         position: fixed;
         top: 2.9em;
         width: 92%;
         z-index: 9999;
         background-color: white;
    }
    
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div:nth-child(3) > div:nth-child(1) > div > div > div > div > div > button{
        width: 8em;
        margin-left: 0em;
    }
    
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div:nth-child(3) > div:nth-child(2) > div > div > div > div > div > button{
        width: 75%;
        margin-right: 0em;
        position: absolute;
        right: 0em;
    }
    
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div:nth-child(3) > div:nth-child(3) > div > div > div > div > div > button{
        width: 75%;
        margin-right: 0em;
</style>
"""


# CSS를 사용하여 버튼을 가운데로 정렬
st.markdown(centered_button_style, unsafe_allow_html=True)
################################################################################




# 페이지 구성
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Home"):
        st.switch_page("main-Copy1.py")
with col2:
    if st.button("Lending Club"):
        st.switch_page("pages/page_1-Copy1.py")
with col3:
    if st.button("Investor"):
        st.switch_page("pages/page_2-Copy1.py")

####
# st.divider()
####


st.markdown('## Risk Management')
st.button("Random", help="데이터에서 랜덤하게 1행을 뽑습니다.", on_click=choice_random, args=(data, ))

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Privacy','Revol','Credit','Loan','Result'])
with tab1:
    st.selectbox("verification_status", options=["Not Verified", "Source Verified", "Verified"], key="verification_status")
    st.slider("last_fico_range_high", min_value=data.last_fico_range_high.min(), max_value=data.last_fico_range_high.max(), key="last_fico_range_high")
    st.slider("last_fico_range_low", min_value=data.last_fico_range_low.min(), max_value=data.last_fico_range_low.max(), key="last_fico_range_low")
    st.slider("dti", min_value=data.dti.min(), max_value=data.dti.max(), key="dti")
    st.slider("fico_range_low", min_value=data.fico_range_low.min(), max_value=data.fico_range_low.max(), key="fico_range_low")
    st.slider("fico_range_high", min_value=data.fico_range_high.min(), max_value=data.fico_range_high.max(), key="fico_range_high")
with tab2:
    st.slider("revol_util", min_value=data.revol_util.min(), max_value=data.revol_util.max(), key="revol_util")
    st.slider("mo_sin_old_rev_tl_op", min_value=data.mo_sin_old_rev_tl_op.min(), max_value=data.mo_sin_old_rev_tl_op.max(), key="mo_sin_old_rev_tl_op")
with tab3:
    st.slider("open_acc", min_value=data.open_acc.min(), max_value=data.open_acc.max(), key="open_acc")
    st.slider("total_acc", min_value=data.total_acc.min(), max_value=data.total_acc.max(), key="total_acc")
    st.slider("avg_cur_bal", min_value=data.avg_cur_bal.min(), max_value=data.avg_cur_bal.max(), key="avg_cur_bal")
    st.slider("pct_tl_nvr_dlq", min_value=data.pct_tl_nvr_dlq.min(), max_value=data.pct_tl_nvr_dlq.max(), key="pct_tl_nvr_dlq")
with tab4:
    st.selectbox("term", help="대출기간", options=[" 36 months", " 60 months"], key="term")
    st.slider("loan_amnt", min_value=data.loan_amnt.min(), max_value=data.loan_amnt.max(), key="loan_amnt", help="대출금액")
    st.slider("int_rate", min_value=data.int_rate.min(), max_value=data.int_rate.max(), key="int_rate", help="이자율")
with tab5:
    predict_data = pd.DataFrame(data=[[st.session_state.loan_amnt, 
                                   st.session_state.term, 
                                   st.session_state.int_rate, 
                                   np.nan, 
                                   np.nan, 
                                   st.session_state.verification_status, 
                                   np.nan, 
                                   st.session_state.dti, 
                                   st.session_state.fico_range_low, 
                                   st.session_state.fico_range_high, 
                                   st.session_state.open_acc, 
                                   st.session_state.revol_util, 
                                   st.session_state.total_acc, 
                                   st.session_state.last_fico_range_high, 
                                   st.session_state.last_fico_range_low, 
                                   st.session_state.avg_cur_bal, 
                                   st.session_state.mo_sin_old_rev_tl_op, 
                                   np.nan, 
                                   st.session_state.pct_tl_nvr_dlq]], 
                            columns=["loan_amnt", "term", "int_rate", "sub_grade", "emp_length", 
                                     "verification_status", "addr_state", "dti", "fico_range_low", "fico_range_high", 
                                     "open_acc", "revol_util", "total_acc", "last_fico_range_high", "last_fico_range_low", 
                                     "avg_cur_bal", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "pct_tl_nvr_dlq"])
    
    predict_data_preprocessed = model[:4].transform(predict_data)
    shap_values = explainer.shap_values(predict_data_preprocessed)
    
    predict = model.predict(predict_data)
    st.markdown("<center><h3>repayment forecast results</h3></center>", unsafe_allow_html=True)
    # st.title("Charged Off" if predict else "Fully Paid", )
    st.markdown("<center><h1>Charged Off</h1></center>" if predict else "<center><h1>Fully Paid</h1></center>", unsafe_allow_html=True)
    if not predict:
        st.balloons()
    with st.expander('Waterfall Plot'):
        st_shap(shap.waterfall_plot(shap.Explanation(values=shap_values[1][0, :], base_values=explainer.expected_value[1], feature_names=predict_data_preprocessed.columns.tolist())), height=400, width=1300)