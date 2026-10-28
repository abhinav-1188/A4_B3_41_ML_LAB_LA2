import os
import re
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 4 SOTA Models from student's lab practicals + Baseline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Set styles for academic-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'UpdatedResumeDataSet.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def clean_resume_text(text: str) -> str:
    """
    Comprehensive regex-based text cleaning:
    - Removes URLs and hyperlinks
    - Removes emails and mentions (@user)
    - Removes hashtags (#topic)
    - Removes special symbols and non-ASCII characters
    - Normalizes multiple spaces into a single space
    - Converts to lowercase
    """
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
    # Remove RT and cc
    text = re.sub(r'\b(RT|cc)\b', ' ', text)
    # Remove hashtags and mentions
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    # Remove phone numbers
    text = re.sub(r'\b\d{10,12}\b', ' ', text)
    # Remove punctuations and special characters
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    # Normalize excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def main():
    print("=" * 70)
    print("STEP 1: Loading and Inspecting Dataset")
    print("=" * 70)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Loaded Successfully! Shape: {df.shape}")
    print(f"Total Unique Job Roles / Categories: {df['Category'].nunique()}")
    print("\nClass Distribution:")
    print(df['Category'].value_counts())

    plt.figure(figsize=(12, 6))
    order = df['Category'].value_counts().index
    sns.countplot(data=df, y='Category', order=order, hue='Category', palette='crest', legend=False)
    plt.title('Job Category Distribution in Dataset (25 Target Roles)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Number of Candidate Resumes', fontsize=12)
    plt.ylabel('Job Role Category', fontsize=12)
    plt.tight_layout()
    cat_dist_path = os.path.join(FIGURES_DIR, 'category_distribution.png')
    plt.savefig(cat_dist_path, dpi=300)
    plt.close()
    print(f"Saved: {cat_dist_path}")

    print("\n" + "=" * 70)
    print("STEP 2: Data Cleaning & Preprocessing")
    print("=" * 70)
    print("Cleaning raw text (URLs, symbols, emails, whitespace)...")
    start_time = time.time()
    df['Cleaned_Resume'] = df['Resume'].apply(clean_resume_text)
    print(f"Cleaning completed in {time.time() - start_time:.2f} seconds.")

    # Encode target labels
    label_encoder = LabelEncoder()
    df['Category_Encoded'] = label_encoder.fit_transform(df['Category'])
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, 'label_encoder.joblib'))
    print(f"Target classes encoded into {len(label_encoder.classes_)} unique numeric identifiers.")

    print("\n" + "=" * 70)
    print("STEP 3: Stratified Data Splitting & Feature Extraction")
    print("=" * 70)
    # Stratified 80:20 split matching lab work
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['Cleaned_Resume'],
        df['Category_Encoded'],
        test_size=0.20,
        random_state=42,
        stratify=df['Category_Encoded']
    )
    print(f"Training Samples: {len(X_train_raw)} (80%) | Testing Samples: {len(X_test_raw)} (20%)")

    # TF-IDF Vectorizer with unigrams + bigrams
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 2)
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
    print(f"TF-IDF Feature Matrix Dimensions: {X_train.shape} features generated.")

    print("\n" + "=" * 70)
    print("STEP 4: Training & Benchmarking 4 SOTA Models + Baseline")
    print("=" * 70)

    # Dictionary of models corresponding directly to student's lab assignments
    models = {
        'Support Vector Machine (Linear SVM)': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
        'Support Vector Machine (RBF SVM)': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        'Decision Tree Classifier': DecisionTreeClassifier(criterion='entropy', max_depth=25, random_state=42),
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        'Multinomial Naive Bayes (Baseline)': MultinomialNB(alpha=0.1)
    }

    results = []
    saved_models = {}
    best_f1 = -1.0
    best_model_name = None

    for name, model in models.items():
        print(f"\n--- Training: {name} ---")
        t_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t_start

        # Measure inference latency on test set
        t_pred_start = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - t_pred_start) / len(y_test) * 1000  # ms per sample

        acc = accuracy_score(y_test, y_pred)
        prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results.append({
            'Model': name,
            'Accuracy': round(acc * 100, 2),
            'Precision (Weighted)': round(prec_weighted * 100, 2),
            'Recall (Weighted)': round(rec_weighted * 100, 2),
            'F1-Score (Weighted)': round(f1_weighted * 100, 2),
            'F1-Score (Macro)': round(f1_macro * 100, 2),
            'Train Time (s)': round(train_time, 3),
            'Inference Latency (ms/sample)': round(inference_time, 3)
        })

        saved_models[name] = model

        print(f"Accuracy: {acc*100:.2f}% | F1-Score: {f1_weighted*100:.2f}% | Latency: {inference_time:.3f} ms")

        if f1_weighted > best_f1:
            best_f1 = f1_weighted
            best_model_name = name

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_csv_path = os.path.join(MODELS_DIR, 'metrics_summary.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved metrics summary to: {results_csv_path}")

    # Save the Best Model and All Models
    best_model = saved_models[best_model_name]
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.joblib'))
    joblib.dump(saved_models, os.path.join(MODELS_DIR, 'all_models.joblib'))
    print(f"[BEST MODEL SELECTED]: {best_model_name} (Saved to best_model.joblib)")

    print("\n" + "=" * 70)
    print("STEP 5: Generating Publication-Grade Figures for 10-Page Report")
    print("=" * 70)

    # 1. Bar Chart of Model Comparison (Only 4 Lab Practical Models)
    practical_models_list = [
        'Support Vector Machine (Linear SVM)',
        'Random Forest Classifier',
        'Decision Tree Classifier',
        'K-Nearest Neighbors (KNN)'
    ]
    plot_df = results_df[results_df['Model'].isin(practical_models_list)].copy()
    plot_df['Model'] = plot_df['Model'].replace({'Support Vector Machine (Linear SVM)': 'SVM (Linear)'})

    plt.figure(figsize=(10, 5))
    bar_width = 0.25
    x_indices = np.arange(len(plot_df))
    plt.bar(x_indices - bar_width, plot_df['Accuracy'], width=bar_width, label='Accuracy (%)', color='#2b5c8f')
    plt.bar(x_indices, plot_df['Precision (Weighted)'], width=bar_width, label='Precision (%)', color='#4682b4')
    plt.bar(x_indices + bar_width, plot_df['F1-Score (Weighted)'], width=bar_width, label='F1-Score (%)', color='#20b2aa')
    plt.xticks(x_indices, plot_df['Model'], rotation=15, ha='right', fontsize=10, fontweight='medium')
    plt.ylabel('Score (%)', fontsize=11, fontweight='bold')
    plt.ylim(95, 101)
    plt.title('Performance Metrics Comparison Across 4 Lab Practical Models', fontsize=12, fontweight='bold', pad=12)
    plt.legend(frameon=True, facecolor='white', loc='lower right')
    plt.tight_layout()
    metrics_fig_path = os.path.join(FIGURES_DIR, 'model_comparison_metrics.png')
    plt.savefig(metrics_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {metrics_fig_path}")

    # 2. Confusion Matrix Heatmap for Best Model
    y_best_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_best_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cbar_kws={'label': 'Number of Classified Resumes'}
    )
    plt.title(f'Confusion Matrix Heatmap: {best_model_name} (Test Set)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Job Category', fontsize=12, fontweight='bold')
    plt.ylabel('True Job Category', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_fig_path = os.path.join(FIGURES_DIR, 'confusion_matrix.png')
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_fig_path}")

    # 3. Latency vs Accuracy Tradeoff Scatter Plot
    plt.figure(figsize=(8, 5))
    for _, row in results_df.iterrows():
        plt.scatter(row['Inference Latency (ms/sample)'], row['F1-Score (Weighted)'], s=140, alpha=0.85, edgecolors='black')
        plt.annotate(
            row['Model'].split('(')[0].strip(),
            (row['Inference Latency (ms/sample)'], row['F1-Score (Weighted)']),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9
        )
    plt.title('Accuracy vs Inference Latency Efficiency Frontier', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Inference Latency (ms per sample)', fontsize=11, fontweight='bold')
    plt.ylabel('Weighted F1-Score (%)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    latency_fig_path = os.path.join(FIGURES_DIR, 'latency_vs_accuracy.png')
    plt.savefig(latency_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {latency_fig_path}")

    # 4. Feature Importance from Random Forest
    rf_model = saved_models['Random Forest Classifier']
    feature_names = np.array(vectorizer.get_feature_names_out())
    importances = rf_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    top_features = feature_names[top_indices]
    top_importances = importances[top_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, hue=top_features, palette='mako', legend=False)
    plt.title('Top 20 Most Predictive Keywords Extracted by Random Forest', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Gini Feature Importance', fontsize=11, fontweight='bold')
    plt.ylabel('Token / Keyword', fontsize=11, fontweight='bold')
    plt.tight_layout()
    kw_fig_path = os.path.join(FIGURES_DIR, 'top_keywords_feature_importance.png')
    plt.savefig(kw_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {kw_fig_path}")

    # 5. Decision Tree Structure Visualization (from Practical 5)
    dt_model = saved_models['Decision Tree Classifier']
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_model,
        max_depth=3,  # Top 3 decision levels for clean presentation
        feature_names=feature_names,
        class_names=label_encoder.classes_,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Decision Tree Architecture (Top Decision Splits from Practical 5)", fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    tree_fig_path = os.path.join(FIGURES_DIR, 'decision_tree_structure.png')
    plt.savefig(tree_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {tree_fig_path}")

    print("\n" + "=" * 70)
    print("ALL TRAINING & VISUALIZATION ARTIFACTS GENERATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    main()
