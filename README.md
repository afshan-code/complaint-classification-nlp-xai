# complaint-classification-nlp-xai
Hybrid NLP and ML framework for consumer complaint classification 
and monetary relief prediction using BERT, FinBERT, XGBoost and XAI
# Consumer Complaint Classification Using Hybrid NLP, Transformers & Explainable AI

## Overview

This project develops and evaluates an explainable hybrid Natural Language 
Processing (NLP) framework for analysing consumer financial complaints using 
the Consumer Financial Protection Bureau (CFPB) dataset. It systematically 
compares classical machine learning models, transformer-based architectures, 
and a novel hybrid model that integrates contextual text embeddings with 
structured complaint metadata. Model decisions are interpreted using 
Explainable AI (XAI) techniques to ensure transparency and regulatory 
accountability.

This work was submitted as an MSc Dissertation in Data Analytics at 
CCT College Dublin (2025).

## Objectives

- Classify CFPB complaints into product categories by comparing classical 
  ML and transformer-based NLP models
- Predict monetary relief (restitution) outcomes using a hybrid modelling 
  framework combining text embeddings with structured complaint attributes
- Interpret model predictions using SHAP, LIME, and attention visualisation 
  to ensure transparency in high-stakes regulatory contexts

## Tools & Technologies

- Python (Pandas, NumPy, Scikit-learn)
- Hugging Face Transformers (BERT, FinBERT)
- XGBoost, Logistic Regression, LinearSVC
- SHAP, LIME, Attention Visualisation (XAI)
- PyTorch / Transformers pipeline
- CFPB Consumer Complaint Database (2020–2025)

## Methodology

- EDA on CFPB dataset covering complaint volume, product distribution, 
  issue types, and restitution outcomes
- Feature engineering combining TF-IDF representations with BERT-derived 
  contextual embeddings
- Two classification tasks:
  - Binary: monetary relief prediction (restitution vs no restitution)
  - Multiclass: product category classification (10+ categories)
- Hybrid MLP model integrating BERT text embeddings with structured 
  metadata features
- Class imbalance handling using stratified splits and macro F1 evaluation
- XAI layer applied across all model architectures using SHAP (global + 
  local), LIME (instance-level), and transformer attention visualisation

## Key Results

- Hybrid framework achieved the highest macro F1-score (0.6561) for binary 
  monetary relief prediction, outperforming all standalone models
- LinearSVC achieved the highest macro F1 (0.6595) for multiclass product 
  classification due to robust linear decision boundaries
- Transformer models (BERT, FinBERT) achieved the highest balanced accuracy 
  (~0.77) and ROC-AUC scores above 0.95 across product classification
- FinBERT's domain-specific pretraining provided enhanced minority-class 
  precision, confirming the value of financial-domain semantics
- SHAP and LIME confirmed predictions are driven by meaningful semantic 
  patterns, with transaction-related language strongly influencing 
  monetary relief outcomes

## Dataset

The CFPB Consumer Complaint Database is publicly available at:
https://www.consumerfinance.gov/data-research/consumer-complaints/

A preprocessed sample used in this study is available at:
https://drive.google.com/file/d/1njbqYd641H2w9Yw2jcdw6SHNKOQM1F4o/view

