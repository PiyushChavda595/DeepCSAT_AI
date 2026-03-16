import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="DeepCSAT AI",
    page_icon="🤖",
    layout="wide"
)

# --------------------------
# Custom CSS (Gradient + Glass UI)
# --------------------------
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

.card {
background: rgba(255,255,255,0.08);
padding:25px;
border-radius:20px;
backdrop-filter: blur(10px);
box-shadow: 0 8px 32px rgba(0,0,0,0.3);
margin-bottom:20px;
}

.title {
font-size:40px;
font-weight:700;
text-align:center;
padding-bottom:10px;
}

.subtitle {
text-align:center;
color:#cccccc;
margin-bottom:30px;
}

.stButton>button {
background: linear-gradient(45deg,#00c6ff,#0072ff);
color:white;
border-radius:10px;
height:3em;
width:100%;
font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# Load Model
# --------------------------
model = tf.keras.models.load_model("csat_ann_model.keras")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# --------------------------
# Header
# --------------------------
st.markdown('<div class="title">DeepCSAT AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Customer Satisfaction Prediction Platform</div>', unsafe_allow_html=True)

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("Customer Interaction Data")

response_time = st.sidebar.slider("Response Time (minutes)",0,500,50)
hour = st.sidebar.slider("Issue Hour",0,23,12)
day = st.sidebar.slider("Day of Week",0,6,3)
weekend = st.sidebar.selectbox("Weekend",[0,1])
channel = st.sidebar.slider("Channel Code",0,10,2)
category = st.sidebar.slider("Issue Category",0,10,3)
tenure = st.sidebar.slider("Customer Tenure",0,5,2)
shift = st.sidebar.slider("Agent Shift",0,3,1)

# --------------------------
# Create Input Data
# --------------------------
data = pd.DataFrame([{
    "Response_Time":response_time,
    "Hour":hour,
    "Day":day,
    "Weekend":weekend,
    "channel_name":channel,
    "category":category,
    "Tenure Bucket":tenure,
    "Agent Shift":shift
}])

for col in features:
    if col not in data.columns:
        data[col] = 0

data = data[features]

scaled = scaler.transform(data)

# --------------------------
# Predict
# --------------------------
if st.button("Predict Satisfaction"):

    pred = model.predict(scaled)[0][0]
    probability = float(pred)

    if probability > 0.5:
        label = "Satisfied"
        color = "#00ff9f"
    else:
        label = "Unsatisfied"
        color = "#ff4b4b"

    col1,col2 = st.columns(2)

    # --------------------------
    # Result Card
    # --------------------------
    with col1:
        st.markdown(f"""
        <div class="card">
        <h2>Prediction</h2>
        <h1 style="color:{color};">{label}</h1>
        <p>Probability: {probability:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------
    # Gauge Chart
    # --------------------------
    with col2:

        fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        title={'text':"Satisfaction Score"},
        gauge={
        'axis':{'range':[0,100]},
        'bar':{'color':"#00c6ff"},
        'steps':[
        {'range':[0,40],'color':"#ff4b4b"},
        {'range':[40,70],'color':"#ffa500"},
        {'range':[70,100],'color':"#00ff9f"}
        ]
        }
        ))

        st.plotly_chart(fig,use_container_width=True)

    # --------------------------
    # Probability Chart
    # --------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    fig2 = go.Figure(data=[
        go.Bar(
            x=["Unsatisfied","Satisfied"],
            y=[1-probability, probability],
            marker_color=["#ff4b4b","#00ff9f"]
        )
    ])

    fig2.update_layout(title="Prediction Confidence")

    st.plotly_chart(fig2,use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# Footer
# --------------------------
st.markdown("""
<br><br>
<center>
DeepCSAT AI Platform  
Developed by **Piyush Chavda**
</center>
""", unsafe_allow_html=True)