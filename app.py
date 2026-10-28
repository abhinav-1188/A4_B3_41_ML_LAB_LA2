import os
import re
import math
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity

# Page setup
st.set_page_config(
    page_title="Resume Screening & Job Predictor",
    page_icon="📄",
    layout="wide"
)

# Custom Clean Styling
st.markdown("""
<style>
    .main-title {
        color: #1e3a8a;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .sub-title {
        color: #475569;
        font-size: 1rem;
        margin-bottom: 20px;
    }
    .card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .badge-role {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e3a8a;
        background: #e0e7ff;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-block;
    }
    .badge-prob {
        font-size: 1.25rem;
        font-weight: 700;
        color: #065f46;
        background: #d1fae5;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-block;
    }
    .tag {
        display: inline-block;
        background: #e2e8f0;
        color: #1e293b;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .tag-missing {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
TEST_RESUMES_DIR = os.path.join(BASE_DIR, 'test_sample_resumes')

# Simplified Company Presets
COMPANIES = {
    "Google": {
        "role": "Data Scientist / ML Engineer",
        "skills": "Python, SQL, Machine Learning, TensorFlow, Deep Learning, Cloud, Statistics",
        "penalty": 0.12
    },
    "Amazon": {
        "role": "DevOps & Cloud Engineer",
        "skills": "AWS, Docker, Kubernetes, Linux, CI/CD, Python, Terraform",
        "penalty": 0.12
    },
    "Microsoft": {
        "role": "Software Engineer",
        "skills": "Java, C#, Azure, Microservices, SQL, Docker, APIs",
        "penalty": 0.12
    },
    "TCS / Infosys": {
        "role": "Software Developer",
        "skills": "Java, Python, SQL, HTML, CSS, JavaScript, SDLC",
        "penalty": 0.05
    },
    "Custom Company": {
        "role": "Software Engineer",
        "skills": "Python, Java, SQL, Git, Machine Learning",
        "penalty": 0.08
    }
}


@st.cache_resource
def load_resources():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))
    all_models = joblib.load(os.path.join(MODELS_DIR, 'all_models.joblib'))
    metrics_df = pd.read_csv(os.path.join(MODELS_DIR, 'metrics_summary.csv'))
    
    # Filter strictly to the 4 practical models
    practical_models = [
        'Support Vector Machine (Linear SVM)',
        'Random Forest Classifier',
        'Decision Tree Classifier',
        'K-Nearest Neighbors (KNN)'
    ]
    filtered_df = metrics_df[metrics_df['Model'].isin(practical_models)].copy()
    filtered_df['Model'] = filtered_df['Model'].replace({
        'Support Vector Machine (Linear SVM)': 'Support Vector Machine (SVM)'
    })
    return vectorizer, label_encoder, all_models, filtered_df


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'#\S+|@\S+', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    text = re.sub(r'\b\d{10,12}\b', ' ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def extract_pdf_text(file_bytes) -> str:
    extracted = ""
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                extracted += t + "\n"
    return extracted


vectorizer, label_encoder, all_models, metrics_df = load_resources()

# SIDEBAR: Minimal & Clean
with st.sidebar:
    st.markdown("### 📁 Quick Test Resumes")
    sample_files = [f for f in os.listdir(TEST_RESUMES_DIR) if f.endswith('.pdf')]
    chosen_sample = st.selectbox("Load Sample PDF:", ["None"] + sample_files)

# HEADER
st.markdown('<div class="main-title">Automated Resume & Job Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning Laboratory Project — Benchmarking 4 Practical Models with PDF Extraction</div>', unsafe_allow_html=True)

# 2 CLEAN TABS
tab1, tab2 = st.tabs([
    "📄 Resume & Company Predictor",
    "📊 4 Models Comparison (From Practicals)"
])

# TAB 1: RESUME & COMPANY PREDICTOR
with tab1:
    st.subheader("1. Input Candidate Resume & Target Company")
    
    col_input1, col_input2 = st.columns([1, 1])
    resume_text = ""

    with col_input1:
        st.markdown("**Upload Resume (PDF):**")
        uploaded = st.file_uploader("Choose PDF File", type=["pdf", "txt"], label_visibility="collapsed")
        
        if uploaded is not None:
            if uploaded.name.endswith(".pdf"):
                with st.spinner("Extracting text via pdfplumber..."):
                    resume_text = extract_pdf_text(uploaded)
            else:
                resume_text = uploaded.read().decode("utf-8")
        elif chosen_sample != "None":
            with open(os.path.join(TEST_RESUMES_DIR, chosen_sample), "rb") as f:
                resume_text = extract_pdf_text(f)
            st.success(f"Loaded: `{chosen_sample}`")

    with col_input2:
        st.markdown("**Or Paste Resume Text Directly:**")
        manual_txt = st.text_area("Paste Text", value=resume_text if resume_text else "", height=140, label_visibility="collapsed", placeholder="Paste resume text here...")
        if manual_txt.strip():
            resume_text = manual_txt

    st.markdown("---")
    st.subheader("2. Target Company & Role Selection")
    
    comp_col1, comp_col2 = st.columns([1, 1.2])
    with comp_col1:
        selected_company = st.selectbox("Select Target Company:", list(COMPANIES.keys()))
        preset_info = COMPANIES[selected_company]
        
        if selected_company == "Custom Company":
            company_name = st.text_input("Enter Company Name:", value="Tech Innovations")
            target_role = st.text_input("Target Role:", value="Software Engineer")
            required_skills_input = st.text_input("Required Skills (comma separated):", value="Python, SQL, Machine Learning, Docker")
        else:
            company_name = selected_company
            target_role = preset_info["role"]
            required_skills_input = preset_info["skills"]
            st.caption(f"Default Role: **{target_role}**")
            st.caption(f"Required Skills: `{required_skills_input}`")

    with comp_col2:
        st.write("")
        st.write("")
        predict_button = st.button("🚀 Predict Role & Hiring Probability", type="primary", use_container_width=True)

    if predict_button:
        if not resume_text.strip():
            st.error("Please upload a PDF or paste resume text first!")
        else:
            with st.spinner("Running ML model inference..."):
                cleaned_resume = clean_text(resume_text)
                features = vectorizer.transform([cleaned_resume])
                
                # Best model from practicals (SVM / Random Forest)
                best_model = all_models['Support Vector Machine (Linear SVM)']
                pred_encoded = best_model.predict(features)[0]
                predicted_role = label_encoder.inverse_transform([pred_encoded])[0]

                # Calculate Company Hiring Probability
                req_skills = [s.strip().lower() for s in required_skills_input.split(",") if s.strip()]
                matched_skills = [s for s in req_skills if s in cleaned_resume]
                missing_skills = [s for s in req_skills if s not in matched_skills]
                skill_ratio = len(matched_skills) / max(1, len(req_skills))

                # Semantic alignment via TF-IDF cosine similarity
                jd_clean = clean_text(f"{target_role} {required_skills_input}")
                jd_vec = vectorizer.transform([jd_clean])
                sim = cosine_similarity(features, jd_vec)[0][0]

                # Hiring Probability calibrated formula
                penalty = preset_info["penalty"]
                logit = -1.0 + (3.0 * sim) + (3.2 * skill_ratio) - (2.0 * penalty)
                hiring_prob = round(min(98.0, max(15.0, (1.0 / (1.0 + math.exp(-logit))) * 100)), 1)

                # Linear regression suitability score (from Practical 3)
                suitability_score = round(40.0 + (35.0 * sim) + (25.0 * skill_ratio), 1)

                st.markdown("---")
                st.subheader("3. Machine Learning Output")

                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    st.markdown(f"""
                    <div class="card">
                        <span style="color:#64748b; font-size:0.9rem; font-weight:600;">PREDICTED DOMAIN / ROLE:</span><br>
                        <div class="badge-role">💼 {predicted_role}</div>
                        <p style="margin-top:10px; color:#475569; font-size:0.9rem;">
                            Classified using <b>Support Vector Machine (SVM)</b> & <b>TF-IDF Vectorizer</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with res_c2:
                    st.markdown(f"""
                    <div class="card">
                        <span style="color:#64748b; font-size:0.9rem; font-weight:600;">HIRING PROBABILITY AT {company_name.upper()}:</span><br>
                        <div class="badge-prob">🎯 {hiring_prob}% Probability</div>
                        <p style="margin-top:10px; color:#475569; font-size:0.9rem;">
                            Suitability Score: <b>{suitability_score} / 100</b> (Linear Regression: w₀ + w₁x).
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Skills Comparison
                sk_col1, sk_col2 = st.columns(2)
                with sk_col1:
                    st.markdown(f"**✅ Matched Skills in Your Resume ({len(matched_skills)}/{len(req_skills)}):**")
                    if matched_skills:
                        for s in matched_skills:
                            st.markdown(f'<span class="tag">✓ {s}</span>', unsafe_allow_html=True)
                    else:
                        st.info("No exact matching skills detected.")

                with sk_col2:
                    st.markdown(f"**⚠️ Missing Skills for {company_name}:**")
                    if missing_skills:
                        for s in missing_skills:
                            st.markdown(f'<span class="tag-missing">✕ {s}</span>', unsafe_allow_html=True)
                    else:
                        st.success("All required skills found in your resume!")


# TAB 2: 4 MODELS COMPARISON (FROM PRACTICALS)
with tab2:
    st.subheader("📊 State-of-the-Art Model Benchmark (The 4 Models from Practicals)")
    st.markdown("Quantitative comparison across the **4 machine learning classifiers** directly implemented in your lab practicals:")

    # Clean Table
    display_df = metrics_df[['Model', 'Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)']].copy()
    st.dataframe(
        display_df.style.highlight_max(subset=['Accuracy', 'F1-Score (Weighted)'], color='#d1fae5'),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Model Performance Analytics")

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown("**1. Accuracy, Precision & F1-Score Comparison:**")
        metric_fig = os.path.join(FIGURES_DIR, 'model_comparison_metrics.png')
        if os.path.exists(metric_fig):
            st.image(metric_fig, use_container_width=True)

    with g_col2:
        st.markdown("**2. Confusion Matrix Heatmap (SVM Classifier):**")
        cm_path = os.path.join(FIGURES_DIR, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)

    g_col3, g_col4 = st.columns(2)
    with g_col3:
        st.markdown("**3. Top Predictive Keywords (Random Forest Feature Importance):**")
        kw_fig = os.path.join(FIGURES_DIR, 'top_keywords_feature_importance.png')
        if os.path.exists(kw_fig):
            st.image(kw_fig, use_container_width=True)

    with g_col4:
        st.markdown("**4. Job Category Distribution in Dataset:**")
        dist_fig = os.path.join(FIGURES_DIR, 'category_distribution.png')
        if os.path.exists(dist_fig):
            st.image(dist_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**5. Decision Tree Architecture (Practical 5 Entropy Splits):**")
    tree_fig = os.path.join(FIGURES_DIR, 'decision_tree_structure.png')
    if os.path.exists(tree_fig):
        st.image(tree_fig, caption="Decision Tree Information Gain splits showing how keywords like 'developer', 'java', 'devops', and 'python' branch into job roles.", use_container_width=True)



