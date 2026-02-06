import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import time
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION (Hacker/Cyber Theme)
# ------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="SentinAI | Cyber Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "Command Center" look
st.markdown("""
<style>
    /* Main Background - Dark Navy/Black */
    .stApp { background-color: #0b0f19; color: #e6edf3; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    
    /* Input Fields */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1f2937 !important; color: #00ff41 !important; 
        border: 1px solid #374151 !important; font-family: 'Courier New', monospace;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #059669, #10b981); color: white; border: none; 
        font-weight: bold; text-transform: uppercase; letter-spacing: 2px; transition: 0.3s;
        border-radius: 2px;
    }
    .stButton>button:hover { box-shadow: 0 0 15px #10b981; transform: scale(1.01); }
    
    /* Headers */
    h1, h2, h3, h4 { font-family: 'Courier New', monospace; color: #00ff41 !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------
# 2. CORE SYSTEM: LOAD MODEL & CONFIGURE INPUTS
# ------------------------------------------------------------------------------------------------
@st.cache_resource
def load_system():
    # A. Load the Trained Model
    try:
        with open('dt_rf.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("🚨 CRITICAL: 'dt_rf.pkl' not found.")
        return None, None, None

    # B. Load Data Reference (Only for input options, not training)
    try:
        df_ref = pd.read_csv('Test_data.csv')
    except FileNotFoundError:
        st.error("🚨 CRITICAL: 'Test_data.csv' not found.")
        return None, None, None

    # C. SYNC MODEL & DATA
    # We ask the model exactly what features it was trained on.
    # This ignores any columns in the CSV that the model doesn't use.
    if hasattr(model, "feature_names_in_"):
        required_features = model.feature_names_in_
    else:
        # Fallback if model doesn't save feature names (older sklearn versions)
        # We try to exclude known non-feature columns
        exclude = ['class', 'target', 'attack', 'label']
        required_features = [c for c in df_ref.columns if c not in exclude]

    return model, df_ref, required_features

model, df_ref, required_features = load_system()

# ------------------------------------------------------------------------------------------------
# 3. DYNAMIC SIDEBAR: GENERATE INPUTS BASED ON PKL FILE
# ------------------------------------------------------------------------------------------------
st.sidebar.title("🎛️ PACKET CONFIG")
st.sidebar.markdown("`INJECT TRAFFIC PARAMETERS`")

user_input = {}

if model is not None and df_ref is not None:
    
    # Identify Categorical vs Numerical features required by the PKL
    # We look at the CSV reference to know which is which
    cat_cols = [c for c in required_features if c in df_ref.columns and df_ref[c].dtype == 'object']
    num_cols = [c for c in required_features if c in df_ref.columns and df_ref[c].dtype != 'object']

    # --- 1. CATEGORICAL INPUTS (Protocol, Service, Flag) ---
    for col in cat_cols:
        options = df_ref[col].unique()
        user_input[col] = st.sidebar.selectbox(f"Select {col}", options)

    st.sidebar.markdown("---")

    # --- 2. NUMERICAL INPUTS ---
    # We prioritize "Key Metrics" for better UI, hide others in expander
    priority_metrics = ['src_bytes', 'dst_bytes', 'count', 'srv_count', 'hot', 'duration']
    
    for col in num_cols:
        if col in priority_metrics:
            # Set ranges based on reference data
            avg_val = float(df_ref[col].mean())
            max_val = float(df_ref[col].max())
            
            if max_val > 1000: # Use number input for large values
                user_input[col] = st.sidebar.number_input(f"{col}", value=avg_val, min_value=0.0)
            else:
                user_input[col] = st.sidebar.slider(f"{col}", 0.0, max_val, avg_val)

    # Hidden inputs for less important features required by the model
    remaining_cols = [c for c in num_cols if c not in priority_metrics]
    if remaining_cols:
        with st.sidebar.expander("ADVANCED PARAMETERS"):
            for col in remaining_cols:
                avg_val = float(df_ref[col].mean())
                user_input[col] = st.number_input(f"{col}", value=avg_val)

# ------------------------------------------------------------------------------------------------
# 4. MAIN DASHBOARD
# ------------------------------------------------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🛡️ SENTINAI DEFENSE GRID")
    st.caption("AI-POWERED NETWORK INTRUSION DETECTION")
with col2:
    st.markdown("""
        <div style='padding: 10px; border-radius: 5px; border: 1px solid #374151; background-color: #1f2937; text-align: center;'>
            <h3 style='margin:0; color:#10b981 !important;'>ONLINE</h3>
            <small style='color:#9ca3af'>SYSTEM ACTIVE</small>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------------------------------------
# 5. EXECUTION ENGINE
# ------------------------------------------------------------------------------------------------
if st.button("🚀 INITIATE TRAFFIC SCAN", use_container_width=True):
    if model is not None:
        
        # A. Processing Animation
        with st.status("🔍 SCANNING PACKET HEADER...", expanded=True) as status:
            time.sleep(0.5)
            st.write("Aligning feature vectors...")
            time.sleep(0.3)
            st.write("Querying Random Forest Model...")
            status.update(label="SCAN COMPLETE", state="complete", expanded=False)

        # B. Data Preprocessing (Crucial Step)
        # 1. Create a DataFrame from User Input
        input_df = pd.DataFrame([user_input])
        
        # 2. Encode Categorical Variables
        # We must encode the user input (e.g., 'tcp') into the number the model expects.
        # We use the reference data to 'learn' the encoding on the fly.
        
        # Combine Reference Data + User Input to ensure LabelEncoder covers all categories
        # Filter reference data to only include the columns the model needs
        ref_subset = df_ref[required_features]
        combined_df = pd.concat([ref_subset, input_df], axis=0)
        
        # Apply Label Encoder to all Object columns
        for col in combined_df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            combined_df[col] = le.fit_transform(combined_df[col].astype(str))
        
        # 3. Extract the final processed input (the last row)
        final_input_vector = combined_df.tail(1)

        # C. Prediction
        prediction = model.predict(final_input_vector)[0]
        
        # Try to get confidence probability
        try:
            probs = model.predict_proba(final_input_vector)[0]
            # Assuming Class 1 is Attack/Anomaly
            attack_prob = probs[1]
        except:
            # Fallback if model doesn't support probability
            attack_prob = 1.0 if prediction == 1 or prediction == 'anomaly' else 0.0

        # ----------------------------------------------------------------
        # 6. RESULTS VISUALIZATION
        # ----------------------------------------------------------------
        r1, r2 = st.columns([1, 2])
        
        with r1:
            # GAUGE CHART
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = attack_prob * 100,
                title = {'text': "THREAT LEVEL"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#ef4444" if attack_prob > 0.5 else "#10b981"},
                    'bgcolor': "#1f2937",
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(16, 185, 129, 0.2)"},
                        {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                    ],
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Courier New"})
            st.plotly_chart(fig, use_container_width=True)
            
        with r2:
            st.subheader("ANALYSIS REPORT")
            if attack_prob > 0.5:
                st.error(f"🚨 ALERT: MALICIOUS TRAFFIC DETECTED")
                st.markdown(f"""
                <div style='background: #450a0a; padding: 15px; border-left: 5px solid #ef4444;'>
                    [cite_start]<strong style='color: #ef4444'>DIAGNOSIS:</strong> Intrusion Pattern Match [cite: 19]<br>
                    <strong style='color: #ef4444'>CONFIDENCE:</strong> {attack_prob*100:.2f}%<br>
                    <strong style='color: #ef4444'>ACTION:</strong> TERMINATE CONNECTION
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"✅ STATUS: NORMAL TRAFFIC")
                st.markdown(f"""
                <div style='background: #064e3b; padding: 15px; border-left: 5px solid #10b981;'>
                    [cite_start]<strong style='color: #10b981'>DIAGNOSIS:</strong> Safe Signature [cite: 9]<br>
                    <strong style='color: #10b981'>CONFIDENCE:</strong> {(1-attack_prob)*100:.2f}%<br>
                    <strong style='color: #10b981'>ACTION:</strong> ALLOW TRAFFIC
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        
        # RADAR CHART (Fingerprint)
        st.subheader("📡 TRAFFIC FINGERPRINT")
        # Visualize key metrics compared to average
        radar_feats = ['count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
        valid_radar = [f for f in radar_feats if f in user_input]
        
        if valid_radar:
            vals = []
            for f in valid_radar:
                mx = float(df_ref[f].max())
                val = float(user_input[f])
                vals.append(val/mx if mx > 0 else 0)

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=vals,
                theta=valid_radar,
                fill='toself',
                name='Scan Fingerprint',
                line_color='#ef4444' if attack_prob > 0.5 else '#00ff41'
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='#1f2937',
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

else:
    st.info("👈 CONFIGURE PARAMETERS IN THE SIDEBAR AND CLICK 'INITIATE TRAFFIC SCAN'")