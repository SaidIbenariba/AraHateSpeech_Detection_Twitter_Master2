from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocess import preprocess, load_stopwords

DATASET_PATH = "datasets/arHateDataset.csv"
TEXT_COL = "Tweet"
LABEL_COL = "Class"

def main():
    # 1) Load data
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    df = df[[TEXT_COL, LABEL_COL]].dropna()

    # ensure label is int (0/1)
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # 2) Load stopwords
    stopwords = load_stopwords("src/stop_words_arabic.csv")

    # 3) Preprocess texts
    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(lambda t: preprocess(t, stopwords=stopwords, keep_latin=False))

    X = df[TEXT_COL].tolist()
    y = df[LABEL_COL].tolist()

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 5) TF-IDF
    vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=0.0001,
    max_df=0.95,
    max_features=50000
)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 6) Model
    model = LinearSVC(class_weight="balanced")
    model.fit(X_train_vec, y_train)

    # 7) Evaluate
    y_pred = model.predict(X_test_vec)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    # 8) Save
    Path("models").mkdir(exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    joblib.dump(model, "models/linearsvc_model.joblib")
    print("\n Saved to models/tfidf_vectorizer.joblib and models/linearsvc_model.joblib")

if __name__ == "__main__":
    main()
