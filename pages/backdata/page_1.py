import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import folium
from streamlit_folium import st_folium, folium_static
import requests



# 네비게이션 구성 설정
st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

# 색깔 가져오기


# main페이지 변수 할당
if 'data' in st.session_state:
    numric_features = st.session_state.numric_features
    categorical = st.session_state.categorical
    target = st.session_state.target
    data = st.session_state.data
    colors = st.session_state.colors
    map_data = st.session_state.map_data
    state_geo = st.session_state.state_geo
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
    
    # #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5 > div:nth-child(1) > div > div > div > div > div{
    #      margin-left: 1em;
    #      # width: 5em;
    #      # height: 5em;
    #      # border-radius: 50%;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5 > div:nth-child(1) > div > div > div > div > div > button{
        width: 8em;
        margin-left: 0em;
    }
    
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5 > div:nth-child(2) > div > div > div > div > div > button{
        width: 75%;
        margin-right: 0em;
        position: absolute;
        right: 0em;
    }
    
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5 > div:nth-child(3) > div > div > div > div > div > button{
        width: 75%;
        margin-right: 0em;
    }
</style>
"""


# CSS를 사용하여 버튼을 가운데로 정렬
st.markdown(centered_button_style, unsafe_allow_html=True)
################################################################################




# 페이지 구성
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Home"):
        st.switch_page("main.py")
with col2:
    if st.button("Lending Club"):
        st.switch_page("pages/page_1.py")
with col3:
    if st.button("Investor"):
        st.switch_page("pages/page_2.py")


# ####
# st.divider()
# ####
def tab1_num():
    st.session_state.tab1_cat = None

def tab1_cat():
    st.session_state.tab1_num = None



st.markdown('## Derive Insights')
tab1, tab2, tab3 = st.tabs(["Single", "Multiple", "Map"])
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        single_select_feature_num = st.selectbox('특성 선택', numric_features, label_visibility='collapsed', index=None, placeholder="수치형 특성 선택", key='tab1_num', on_change=tab1_num)
            
    with col2:
        single_select_feature_cat = st.selectbox('특성 선택', categorical, label_visibility='collapsed', index=None, placeholder="범주형 특성 선택", key='tab1_cat', on_change=tab1_cat)
    with col3:
        bins = st.slider('bins', min_value=0, max_value=100, value=0, label_visibility='collapsed', disabled=False if single_select_feature_num else True)
        col3_1, col3_2 = st.columns(2)
        with col3_1:
            hue_flag = st.checkbox('hue_flag', value=False if single_select_feature_cat else True, key='tab1_hue')
        with col3_2:
            kde_flag = st.checkbox('kde_flag', value=False if single_select_feature_cat else True, disabled=False if single_select_feature_num else True)
            
    col1, col2 = st.columns([0.95,0.05])
    with col1:
        if single_select_feature_num != None:
            fig = plt.figure(figsize=(15,4))
            sns.histplot(x=single_select_feature_num, hue=target if hue_flag else None, multiple='stack',bins="auto" if not bins else bins,kde=kde_flag, data=data,palette=colors)
            st.pyplot(fig)
            fig = None
        if single_select_feature_cat != None:
            fig = plt.figure(figsize=(15,4))
            if single_select_feature_cat == 'addr_state' or single_select_feature_cat == 'sub_grade':
                plt.yticks(fontsize=5)  # 폰트 크기 조절
            sns.histplot(y=single_select_feature_cat, hue=target if hue_flag else None, multiple='stack', shrink=0.5, data=data,palette=colors)
            st.pyplot(fig)
            fig = None
        with col2:
            pass

# # 선택된 특성을 업데이트
# previous_select_num = single_select_feature_num
# previous_select_cat = single_select_feature_cat
        
with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        select_feature_num1 = st.selectbox('특성 선택', numric_features, label_visibility='collapsed', index=None, placeholder="수치형 특성 선택", key='tab2_num1')
            
    with col2:
        select_feature_num2 = st.selectbox('특성 선택', numric_features, label_visibility='collapsed', index=None, placeholder="수치형 특성 선택", key='tab2_num2')
    with col3:
        if (select_feature_num1 != None) & (select_feature_num2 != None):
            st.metric('Coefficient',f'{round(data[[select_feature_num1,select_feature_num2]].corr().loc[select_feature_num1,select_feature_num2],3):,}')
        else:
            st.metric('Coefficient','0.0')
        hue_flag = st.checkbox('hue_flag', value=False, key='tab2_hue')

    col1, col2 = st.columns([0.95,0.05])
    with col1:
        if (select_feature_num1 != None) & (select_feature_num2 != None):
            fig = plt.figure(figsize=(15,4))
            sns.scatterplot(x=select_feature_num1,y=select_feature_num2, data=data,palette=colors,alpha=0.3, hue=target if hue_flag else True)
            st.pyplot(fig)
            fig = None
    with col2:
        pass
            
# last_fico_range_high
with tab3:
    state_geo = requests.get("https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json").json() # US states

    df = data[["addr_state", "loan_status"]].groupby("addr_state").value_counts().reset_index()
    df = df.rename(columns={0 : 'count'})
    df = pd.DataFrame(data=[[id, df[(df.addr_state == id) & (df.loan_status == "Charged Off")]["count"].values[0] / (df[(df.addr_state == id) & (df.loan_status == "Fully Paid")]["count"].values[0] + df[(df.addr_state == id) & (df.loan_status == "Charged Off")]["count"].values[0]) * 100] for id in data.addr_state.sort_values().unique()], columns=["state", "Charged Off rate"]).astype({"Charged Off rate": np.float64})
    
    m = folium.Map([48, -102], zoom_start=3,heigh='75%')
    cp = folium.Choropleth(geo_data=state_geo, 
                           data=df, 
                           columns=df.columns, 
                           key_on="feature.id", 
                           bins=df["Charged Off rate"].quantile([0, 0.25, 0.5, 0.75, 1]).to_list(), 
                           fill_color="YlGn", 
                           line_weight=0.7, 
                           line_opacity=0.4, 
                           name="US Cholopleth Map", 
                           legend_name="Charged Off Rate (%)").add_to(m)
    for data in cp.geojson.data.get("features"):
        data.get("properties")["id"] = f"{data.get('properties').get('name')} ({data.get('id')})"
        data.get("properties")["rate"] = df[df.state == data.get("id")]["Charged Off rate"].values.round(2)[0]
    folium.GeoJsonTooltip(fields=["id", "rate"], aliases=["State Name", "Charge Off rate (%)"], style="font-size: 12px").add_to(cp.geojson)
    col1, col2 = st.columns([0.99,0.01])
    with col1:
        folium_static(m)
    with col2:
        pass
    ####
    st.divider()
    ####