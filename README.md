# arHateDetector Evolution: Multi-Dialectal & Moroccan-Specialized Detection

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Dataset Size](https://img.shields.io/badge/total_data-92%2C127_tweets-blueviolet)
![Framework](https://img.shields.io/badge/framework-AraBERT%20%7C%20MorrBERT-red)

## 📌 Project Overview
**arHateDetector Evolution** is an advanced NLP framework designed to identify hate speech across the diverse linguistic landscape of the Arab world. While the original research focused on a unified model for 34K tweets, this project represents a significant evolution:
1.  **Scaling:** Expanding the general training set to **71,725 tweets**.
2.  **Specialization:** Introducing a dedicated pipeline for **Moroccan Darija** using a specialized **20,402-tweet** dataset.
3.  **Dual-Model Architecture:** Leveraging **AraBERT** for general dialects and **MorrBERT** for the specific complexities of Maghrebi code-switching and slang.

## 🚀 Key Features
*   **Massive Scalability:** Trained on a consolidated corpus of over 92,000 samples.
*   **Regional Expertise:** Specifically optimized for Moroccan Darija using the `otmangi/MorrBERT` architecture.
*   **Linguistic Preprocessing:** Advanced pipeline involving `Farasa` lemmatization and dialect-specific cleaning.
*   **Dual-Tier Classification:** Real-time detection that distinguishes between standard Arabic hate speech and localized Moroccan offensive language.

## 📊 The Evolution Datasets
This framework utilizes two distinct high-volume datasets to ensure comprehensive coverage:

| Dataset | Count (Rows) | Focus |
| :--- | :--- | :--- |
| **arHateDataset.csv** | 71,725 | MSA, Gulf, Levantine, Egyptian dialects |
| **Moroccan_Darija_Offensive.csv** | 20,402 | Moroccan Darija slang & code-switching |
| **Total Consolidated Data** | **92,127** | **Total Research Corpus** |

*   **Original Research Comparison:** This implementation provides a **170% increase** in data volume compared to the 34,107-tweet baseline.

## 🛠️ Technical Methodology

### 1. Preprocessing Pipeline
To handle the morphological richness of Arabic and the unique orthography of Darija, we use:
*   **Text Cleaning:** Filtering mentions, URLs, and non-Arabic characters while preserving dialect-specific emojis.
*   **Normalization:** Unifying Alif, Ya, and Ta-Marbuta across all samples.
*   **Lemmatization:** Reducing Eastern and Standard Arabic words to their roots via **Farasa** to improve feature matching.

### 2. Model Architectures
We transitioned from basic ML classifiers to a dual Transformer-based strategy:
*   **General Model:** `aubmindlab/bert-base-arabertv02-twitter` fine-tuned on the 71K dataset for broad regional coverage.
*   **Specialist Model:** `otmangi/MorrBERT` fine-tuned on the 20K Moroccan dataset to capture nuances that general models typically miss.

## 📈 Performance Results (Target Metrics)
| Model | Target Accuracy | Primary Dataset | Scope |
| :--- | :--- | :--- | :--- |
| **AraBERT v0.2** | **93.5%** | arHateDataset (71K) | General Arabic |
| **MorrBERT** | **92.0%** | Moroccan_Darija (20K) | Moroccan Specific |
| **Linear SVC** | 89.0% | Mixed Corpus | Baseline |

## 💻 Installation & Usage

### Prerequisites
*   Python 3.8+
*   PyTorch / Transformers (HuggingFace)
*   FarasaPy & PyArabic

### Setup
```bash
# Clone the repository
git clone https://github.com/SaidIbenariba/AraHateSpeech_Detection_Twitter_Master2.git

# Install dependencies
pip install -r requirements.txt

# run app 
python app.py
```

## 🌐 Web Application
The framework includes a **Flask-based** interface where users can input text to see how the different models classify the content. The interface highlights whether a tweet is flagged as "Hate" or "Normal" based on both general and Moroccan-specific detection tiers.

## 📚 References & Inspiration
*   **Reference Paper:** *Khezzar, R., et al. (2023). arHateDetector: detection of hate speech from standard and dialectal Arabic Tweets.*
*   **Moroccan Architecture:** *otmangi/MorrBERT* (Moussaoui & El Younoussi, 2023).
*   **Darija Dataset:** *Ibrahimi & Mourhir (2023). Moroccan Darija Offensive Language Detection Dataset.* 
