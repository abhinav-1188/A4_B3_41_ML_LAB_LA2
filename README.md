# Automated Resume Intelligence & Job Role Predictability Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![PDF Parsing](https://img.shields.io/badge/Parser-pdfplumber-green.svg)](https://github.com/jsvine/pdfplumber)

An end-to-end Machine Learning web application and comparative analytics framework designed to parse candidate resumes in real-time, predict target job categories across 25 distinct domains, and perform automated ATS (Applicant Tracking System) job description matching.

Developed as part of the **Machine Learning Laboratory (LA2) Mini Project**.

---

## 👥 Project Team (Group of 3)
* **Student 1 (Lead Developer & ML Engineer)**: Om Jaiswal (Roll No: `A4-B4-57`, Class: `A4`)
* **Student 2 (Model Benchmark & Analytics Engineer)**: Group Member 2
* **Student 3 (UI Deployment & Report Specialist)**: Group Member 3

---

## 📌 Project Highlights & Rubric Fulfillment (15 Marks)

1. **Data Cleaning & Splitting (3 Marks)**:
   - Ingestion of 962 candidate resumes across 25 diverse job roles.
   - Text sanitization engine stripping URLs, hashtags, mentions, emails, and non-ASCII characters.
   - TF-IDF Vectorization generating 3,000 sublinear unigram and bigram features.
   - Stratified 80:20 Train-Test Split ensuring equitable representation across all 25 categories without data leakage.

2. **State-of-the-Art (SOTA) Models Benchmark (4 Marks)**:
   - Evaluated 4 distinct model architectures directly from lab practicals:
     - **Support Vector Machine (Linear & RBF Kernels)** *(from `SVM (1).ipynb`)*
     - **Random Forest Classifier** *(from `RandomForest.ipynb` & `RandomForestFraudDetection.ipynb`)*
     - **Decision Tree Classifier (Entropy Criterion)** *(from `Practical5_With_Library.ipynb`)*
     - **K-Nearest Neighbors (KNN with Cosine Metric)** *(from `car_dataset_implementing_knn.ipynb` & `Om Jaiswal Lab 4.ipynb`)*
     - **Multinomial Naive Bayes (Baseline)**
   - Metrics Evaluated: Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Train Time, and Inference Latency per sample.

3. **Interactive UI Deployment & Accessibility (4 Marks)**:
   - Cross-platform web interface built using **Streamlit**.
   - Real-time PDF resume text extraction via **`pdfplumber`**.
   - **Company Hiring Predictor ("Will XYZ Hire Me?")**: Predicts candidate selection odds for Google, Microsoft, Amazon, Goldman Sachs, TCS, Startups, or any Custom Company.
   - Interactive ATS Job Compatibility Matcher calculating Cosine Similarity percentages and recommending missing keywords.

4. **Code Quality, GitHub Repo & 10-Page Academic Report (4 Marks)**:
   - Modular, clean, PEP-8 compliant repository structure.
   - Comprehensive 10-page formal project report reviewing 5 recent (2021–2025) research publications.

---

## 📊 Benchmark Results

| Model Architecture | Accuracy (%) | Precision (Weighted) (%) | Recall (Weighted) (%) | F1-Score (Weighted) (%) | Inference Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (Linear SVM)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **0.261 ms** |
| **Support Vector Machine (RBF SVM)** | 99.48% | 99.57% | 99.48% | 99.49% | 0.313 ms |
| **Random Forest Classifier** | 99.48% | 99.54% | 99.48% | 99.48% | 0.087 ms |
| **Decision Tree Classifier** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **0.004 ms** |
| **K-Nearest Neighbors (KNN)** | 98.96% | 99.11% | 98.96% | 98.96% | 0.080 ms |
| **Multinomial Naive Bayes (Baseline)** | 98.96% | 99.11% | 98.96% | 98.96% | 0.005 ms |

---

## 🗂️ Repository Directory Structure

```
mini_project/
├── app.py                      # Interactive Streamlit Web Application
├── train_models.py             # Preprocessing, Model Training & Figure Generation
├── generate_sample_resumes.py  # Script generating 4 test PDF resumes
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── 10_PAGE_REPORT.md           # Formal 10-page project report
├── data/
│   └── UpdatedResumeDataSet.csv# 962 candidate resumes dataset (25 classes)
├── models/
│   ├── best_model.joblib       # Serialized Linear SVM classifier
│   ├── all_models.joblib       # Serialized dictionary of all 5 classifiers
│   ├── tfidf_vectorizer.joblib # Serialized TF-IDF feature extractor
│   ├── label_encoder.joblib    # Serialized target label encoder
│   └── metrics_summary.csv     # Quantitative comparison table
├── figures/
│   ├── category_distribution.png
│   ├── model_comparison_metrics.png
│   ├── confusion_matrix.png
│   ├── latency_vs_accuracy.png
│   └── top_keywords_feature_importance.png
└── test_sample_resumes/
    ├── Sample_Data_Scientist_Resume.pdf
    ├── Sample_Java_Developer_Resume.pdf
    ├── Sample_HR_Manager_Resume.pdf
    └── Sample_DevOps_Engineer_Resume.pdf
```

---

## 🚀 Quick Setup & Installation

### 1. Clone Repository & Navigate
```bash
git clone https://github.com/your-username/resume-intelligence-ml.git
cd resume-intelligence-ml/mini_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Model Training (Optional, models are pre-trained)
```bash
python train_models.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing the Live Application
You can upload any of the sample resumes located in `test_sample_resumes/`:
1. `Sample_Data_Scientist_Resume.pdf` → Predicted: **Data Science** (100% confidence).
2. `Sample_Java_Developer_Resume.pdf` → Predicted: **Java Developer** (100% confidence).
3. `Sample_HR_Manager_Resume.pdf` → Predicted: **HR** (100% confidence).
4. `Sample_DevOps_Engineer_Resume.pdf` → Predicted: **DevOps Engineer** (100% confidence).
