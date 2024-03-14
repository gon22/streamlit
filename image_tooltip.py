import base64
import streamlit as st

with open("img/model.png", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

html_content = f"""
<style>
.tooltip {{
  position: relative;
  display: inline-block;
  cursor: pointer;
}}

.tooltip .tooltiptext {{
  visibility: hidden;
  width: 120px;
  background-color: black;
  color: white;
  text-align: center;
  border-radius: 6px;
  padding: 5px 0;
  position: absolute;
  z-index: 1;
  top: 100%;
  left: 50%;
  margin-left: -60px;
  opacity: 0;
  transition: opacity 0.3s;
}}

.tooltip:hover .tooltiptext {{
  visibility: visible;
  opacity: 1;
}}
</style>

<div class="tooltip">
  <p>ℹ️</p>
  <span class="tooltiptext">
                            <img src="data:image/png;base64,{image_base64}" style="width:1000px;height:1000px;">
  </span>
</div>
"""

st.markdown(html_content, unsafe_allow_html=True)