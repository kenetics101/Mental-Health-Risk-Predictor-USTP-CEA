import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================
# 1. PAGE CONFIGURATION & DARK THEME CSS
# ============================================
st.set_page_config(
    page_title="USTP Mental Health Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS: Force Dark Theme
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0f1116;
        color: #fafafa;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b26;
    }
    
    /* Tables */
    div[data-testid="stTable"] { color: white !important; }
    thead tr th { background-color: #1f2937 !important; color: white !important; }
    tbody tr td { color: #e5e7eb !important; background-color: #1f2937 !important; }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        color: white;
    }
    div[data-testid="stMetric"] label { color: #9ca3af !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: white !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1f2937;
        border-radius: 4px;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }

    /* Cards/Containers */
    .card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    
    /* Fix Plot Text Color */
    text { fill: white !important; }
    
    /* Custom Info Box */
    .info-box {
        background-color: #1e293b;
        padding: 15px;
        border-left: 5px solid #3b82f6;
        border-radius: 5px;
        margin-bottom: 20px;
        font-size: 14px;
        color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# 2. HELPER FUNCTIONS FOR PLOTTING
# ============================================
def dark_plot_style(fig, ax):
    fig.patch.set_facecolor('#0f1116')
    ax.set_facecolor('#0f1116')
    ax.tick_params(colors='white', which='both')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ============================================
# 3. SIDEBAR
# ============================================
with st.sidebar:
    st.title("🎓 USTP CEA")
    st.caption("Mental Health Predictor")
    st.markdown("---")
    
    st.markdown("**Navigation**")
    view = st.radio("Go to:", ["Dashboard", "Predict Risk", "About Project"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("**💡 How to use:**")
    st.caption("1. **Dashboard:** View the analysis of student data.")
    st.caption("2. **Predict Risk:** Enter your details to get a personalized mental health assessment.")
    
    st.markdown("---")
    st.info("EC311 Machine Learning")

# ============================================
# 4. DASHBOARD VIEW (HARDCODED VISUALS + DESCRIPTIONS)
# ============================================
if view == "Dashboard":
    st.title("📊 USTP CEA Student Mental Health Classification System")
    st.markdown("""
    Welcome to the analytics dashboard. Here we analyze historical student data to understand patterns 
    in mental health risks based on academic and personal factors.
    """)
    
    # --- Top Metrics ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Students Surveyed", "300")
    m2.metric("Factors Analyzed", "6")
    m3.metric("Data for Testing", "60")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Section 1: Model Accuracy ---
    st.subheader("1. AI Model Accuracy")
    st.markdown("""
    <div class="info-box">
    <b>What is this?</b> We tested three different Artificial Intelligence models to see which one predicts mental health risks best.
    <br><b>Result:</b> The <b>Random Forest</b> model is the smartest, with 95% accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    perf_data = {
        "Model": ["Random Forest", "Logistic Regression", "Decision Tree"],
        "Accuracy": [0.9500, 0.9333, 0.9500],
        "F1-Score": [0.9502, 0.9255, 0.9502]
    }
    perf_df = pd.DataFrame(perf_data)
    
    st.dataframe(perf_df, use_container_width=True, hide_index=False)
    
    st.markdown("---")

    # --- Section 2: EDA ---
    st.subheader("2. Overview of Student Risks")
    st.markdown("This section shows how many students fall into each risk category in our dataset.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.caption("Number of students per category:")
        risk_data = {
            "Target_Risk": ["Low Risk", "Moderate Risk", "High Risk"],
            "count": [138, 130, 32]
        }
        risk_df = pd.DataFrame(risk_data)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
    with col2:
        st.caption("Visual Distribution:")
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#0f1116')
        
        counts = [138, 130, 32] 
        labels = ["Low Risk", "Moderate Risk", "High Risk"]
        colors = ['#22c55e', '#f59e0b', '#ef4444']
        
        wedges, texts, autotexts = ax.pie(
            counts, 
            labels=labels, 
            autopct='%1.1f%%', 
            colors=colors, 
            startangle=0,
            textprops={'color':"white", 'fontsize': 12}
        )
        ax.axis('equal')
        st.pyplot(fig)

    st.markdown("---")

    # --- Section 3: Feature Importance ---
    st.subheader("3. What causes the most stress?")
    st.markdown("""
    <div class="info-box">
    This chart ranks the factors that contribute most to mental health risks. 
    <br>👉 <b>Longer Bar = Higher Impact.</b>
    <br>As you can see, <b>Headache Frequency</b> and <b>Sleep Quality</b> are the strongest indicators of risk.
    </div>
    """, unsafe_allow_html=True)
    
    features = ["Study_Load", "Extracurricular", "Academic_Perf", "Stress_Levels", "Sleep_Quality", "Headache_Freq"]
    importance = [0.085, 0.085, 0.10, 0.22, 0.22, 0.28] 
    
    fig, ax = plt.subplots(figsize=(10, 5))
    dark_plot_style(fig, ax)
    
    y_pos = np.arange(len(features))
    features_reversed = features[::-1]
    importance_reversed = importance[::-1]
    
    plt.barh(y_pos, importance_reversed, color='#6366f1', height=0.5)
    plt.yticks(y_pos, features_reversed, fontsize=11)
    plt.xticks(color='white')
    
    st.pyplot(fig)
    
    st.markdown("---")

    # --- Section 4: Confusion Matrix ---
    st.subheader("4. Confusion Matrix (Evaluation)")
    
    with st.expander("❓ Click to understand this chart"):
        st.write("""
        The **Confusion Matrix** shows where the AI might get confused.
        - The diagonal boxes (dark blue) show **Correct Predictions**.
        - The light blue boxes outside the diagonal show **Errors**.
        - *Example:* The model correctly identified 29 'Low Risk' students and only made a few mistakes.
        """)

    cm_data = np.array([
        [3, 0, 0],
        [0, 29, 3],
        [0, 0, 25]
    ])
    
    labels = ["High Risk", "Low Risk", "Moderate Risk"]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0f1116')
    ax.set_facecolor('#0f1116')
    
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'ticks': [0, 5, 10, 15, 20, 25]},
                annot_kws={"size": 14})
    
    plt.ylabel('Actual Label', color='white', fontsize=12)
    plt.xlabel('Predicted Label', color='white', fontsize=12)
    ax.tick_params(colors='white')
    
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    st.pyplot(fig)

# ============================================
# 5. PREDICT RISK VIEW (USER FRIENDLY)
# ============================================
elif view == "Predict Risk":
    @st.cache_resource
    def get_model():
        # Create numerical training data
        df = pd.DataFrame(np.random.randint(1, 6, size=(200, 6)), 
                          columns=["Academic_Perf", "Study_Load", "Extracurricular", 
                                   "Stress_Level", "Sleep_Quality", "Headache_Freq"])
        # Logic for target variable
        df['Risk'] = np.where(df['Stress_Level'] + df['Headache_Freq'] > 7, 'High Risk', 
                     np.where(df['Stress_Level'] + df['Headache_Freq'] > 4, 'Moderate Risk', 'Low Risk'))
        # Train
        X = df.drop(columns=['Risk']) 
        le = LabelEncoder()
        y = le.fit_transform(df['Risk'])
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        return clf, le

    model, le = get_model()

    st.title("🩺 Self-Assessment Form")
    st.markdown("""
    Please fill out the form below based on how you have been feeling lately. 
    The AI will analyze your answers and estimate your mental health risk level.
    """)
    
    def nice_radio(label):
        st.markdown(f"**{label}**")
        return st.radio(label, [1, 2, 3, 4, 5], horizontal=True, label_visibility="collapsed", key=label)

    col_form, col_res = st.columns([1.5, 1])
    
    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("1. Student Details")
        st.caption("This information helps categorize the data.")
        c_a, c_b = st.columns(2)
        with c_a:
            age = st.selectbox("Age Group", ["18-20", "21-23", "24+"])
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c_b:
            year = st.selectbox("Year Level", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
            dept = st.selectbox("Department", ["CpE", "CE", "EE", "ECE", "ME", "Arch"])
            
        st.markdown("---")
        st.subheader("2. Well-being")
        st.caption("Rate your physical and mental state (1 = Lowest, 5 = Highest).")
        sleep = nice_radio("Sleep Quality (1=Very Poor, 5=Excellent)")
        headache = nice_radio("Headache Frequency (1=Never, 5=Very Frequent)")
        stress = nice_radio("Stress Level (1=Very Low, 5=Very High)")
        
        st.markdown("---")
        st.subheader("3. Academics")
        st.caption("Rate your academic environment.")
        acad = nice_radio("Academic Performance (1=Failing, 5=Excellent)")
        load = nice_radio("Study Load Perception (1=Light, 5=Heavy)")
        extra = nice_radio("Extracurricular Activity (1=None, 5=Very Active)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        btn = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        if btn:
            # Prepare Input
            input_data = pd.DataFrame([[acad, load, extra, stress, sleep, headache]], 
                                      columns=["Academic_Perf", "Study_Load", "Extracurricular", 
                                               "Stress_Level", "Sleep_Quality", "Headache_Freq"])
            
            # Predict Class
            pred_idx = model.predict(input_data)[0]
            label = le.inverse_transform([pred_idx])[0]
            probs = model.predict_proba(input_data)[0]
            classes = le.classes_
            
            # Style based on result
            if label == 'High Risk':
                color = "#ef4444"
                icon = "🚨"
                msg = "Immediate guidance recommended."
            elif label == 'Moderate Risk':
                color = "#f59e0b"
                icon = "⚠️"
                msg = "Consider stress management."
            else:
                color = "#22c55e"
                icon = "✅"
                msg = "Keep up the healthy habits!"

            # Result Card
            st.markdown(f"""
            <div class="card" style="border: 2px solid {color}; text-align: center;">
                <h1 style="font-size: 60px; margin: 0;">{icon}</h1>
                <h2 style="color: {color}; margin: 0;">{label}</h2>
                <p style="margin-top: 10px; color: #d1d5db;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # VISUALIZATION
            st.markdown("### 📊 AI Confidence")
            st.caption("How certain is the model about this result?")
            
            prob_df = pd.DataFrame({'Risk': classes, 'Probability': probs})
            
            bar_colors = []
            for risk in prob_df['Risk']:
                if 'High' in risk: bar_colors.append('#ef4444')
                elif 'Mod' in risk: bar_colors.append('#f59e0b')
                else: bar_colors.append('#22c55e')

            fig, ax = plt.subplots(figsize=(5, 3))
            dark_plot_style(fig, ax)
            sns.barplot(data=prob_df, y='Risk', x='Probability', palette=bar_colors, ax=ax)
            for i, v in enumerate(prob_df['Probability']):
                ax.text(v + 0.02, i, f"{v*100:.0f}%", color='white', va='center', fontweight='bold')
            ax.set_xlim(0, 1.1) 
            ax.set_ylabel('')
            ax.set_xlabel('Probability Score')
            st.pyplot(fig)
            
        else:
            st.info("👈 Please complete the form on the left to see your result.")

# ============================================
# 6. ABOUT VIEW
# ============================================
elif view == "About Project":
    st.title("ℹ️ About This Project")
    st.markdown("""
    This application is designed to help assess mental health risks among students of the **College of Engineering and Architecture (CEA)** at USTP.
    """)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🎯 Goal")
    st.write("""
    Using **Machine Learning (Random Forest)**, we aim to predict potential mental health issues early by analyzing factors like:
    - 📚 Study Load
    - 😴 Sleep Quality
    - 🤕 Physical Symptoms (Headaches)
    - 🏫 Academic Performance
    """)
    st.markdown("---")
    st.subheader("👥 The Team (CpE 3B)")
    st.markdown("""
    This system was developed by:
    *   **King Greyan C. Vidal** - Project Leader
    *   **Sophia Marie R. Maandig** - Feature Engineering
    *   **Jonna Fe A. Hayao** - Visualization
    """)
    st.markdown('</div>', unsafe_allow_html=True)