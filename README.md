# arHateDetector: Multi-Dialectal Arabic Hate Speech Detection

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Accuracy](https://img.shields.io/badge/accuracy-93%25-orange)
![Framework](https://img.shields.io/badge/framework-AraBERT%20%7C%20Tensorflow-red)

## 📌 Project Overview
**arHateDetector** is a comprehensive framework designed to detect hate speech in Arabic social media content. Unlike traditional models that focus solely on Modern Standard Arabic (MSA), this project provides a unified approach to identifying offensive language across a wide range of regional dialects (Levantine, Gulf, Egyptian, and Maghrebi).

By leveraging state-of-the-art Transformer-based models and a massive, curated dataset, **arHateDetector** achieves a benchmark accuracy of **93%**, significantly outperforming baseline machine learning classifiers.

## 🚀 Key Features
*   **Multi-Dialectal Support:** Specialized processing for MSA and diverse Arabic dialects.
*   **Advanced NLP Pipeline:** Integrated cleaning, normalization, and lemmatization using the **Farasa** tool.
*   **Hybrid Modeling:** Comparison between traditional ML (Linear SVC) and Deep Learning (CNN, AraBERT).
*   **Real-time Prediction:** A Flask-based web application for instant hate speech classification.

## 📊 The arHateDataset
The framework is trained on a consolidated corpus of **34,107 Tweets**, curated from multiple public sources to ensure dialectal diversity.
*   **Hate Speech:** 32% (10,914 tweets)
*   **Normal Speech:** 68% (23,193 tweets)
*   **Dialects included:** Algerian, Saudi, Gulf, Levantine, and MSA.

## 🛠️ Technical Methodology
### 1. Preprocessing Pipeline
We implement a rigorous cleaning process to reduce noise and improve model generalization:
*   **Noise Removal:** Filtering of hashtags, URLs, mentions, and non-Arabic characters.
*   **Normalization:** Unifying character variations (e.g., Alif, Ya, Ta-Marbuta).
*   **Lemmatization:** Reducing words to their linguistic roots via the **Farasa** library.

### 2. Model Architectures
We evaluated 11 different models to find the optimal balance between speed and precision:
*   **Transformer:** AraBERT v0.2-Twitter-base (State-of-the-art)
*   **Deep Learning:** 1D-Convolutional Neural Network (CNN)
*   **Machine Learning:** Linear SVC, Multinomial Naive Bayes, SGD, and Logistic Regression.

## 📈 Performance Results
| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| **AraBERT** | **93.0%** | **0.93** |
| **Linear SVC** | 89.0% | 0.89 |
| **CNN** | 88.0% | 0.88 |

## 💻 Installation & Usage

### Prerequisites
*   Python 3.8+
*   Tensorflow / PyTorch
*   Transformers (HuggingFace)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/arHateDetector.git

# Install dependencies
pip install -r requirements.txt

# Run the Web App
python app.py
```

## 🌐 Web Application
The project includes a user-friendly interface built with **Flask, HTML5, and CSS3**. Users can input any Arabic sentence to receive a real-time classification of "Hate" or "Normal."

## 📚 References & Inspiration
This project is inspired by the research paper: 
> *Khezzar, R., Moursi, A., & Al Aghbari, Z. (2023). arHateDetector: detection of hate speech from standard and dialectal Arabic Tweets. Discover Internet of Things.*

