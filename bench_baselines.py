from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocess import preprocess, load_stopwords

# =========================
ARHATE_PATH = "datasets/arHateDataset.csv"  # columns: Tweet,Class
MDOLLD_PATH = "datasets/Moroccan_Darija_Offensive_Language_Detection_Dataset.csv"  # columns: text,label

STOP_AR = "src/stop_words_arabic.csv"
STOP_DARIJA_LATIN = "src/stop_words_darija_latin.csv"  
# =========================
# TF-IDF params (Plan)
# =========================
TFIDF_MIN_DF = 0.0001
TFIDF_MAX_DF = 0.95
MAX_FEATURES = 50000
NGRAM_RANGE = (1, 2)

# =========================
# Split params (Plan)
# =========================
TEST_SIZE = 0.25
RANDOM_STATE = 42

# keep_latin:

KEEP_LATIN_ARHATE = False
KEEP_LATIN_MDOLLD = True


def ensure_dirs():
    Path("reports").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)


def load_and_standardize(csv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[[text_col, label_col]].dropna()
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].astype(int)
    return df


def safe_stopwords():
    return load_stopwords(STOP_AR, STOP_DARIJA_LATIN)


def save_report(dataset_name: str, model_name: str, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=4)

    out_path = f"reports/{dataset_name}_{model_name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"=== Dataset: {dataset_name} | Model: {model_name} ===\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(rep)
        f.write("\n")

    print(f"✅ Saved report: {out_path}")
    return cm, rep


def run_models(dataset_name: str, df: pd.DataFrame, keep_latin: bool):
    print(f"\n==================== {dataset_name} ====================")

    stopwords = safe_stopwords()

    # Preprocess text
    df["clean_text"] = df["text"].astype(str).apply(
        lambda t: preprocess(t, stopwords=stopwords, keep_latin=keep_latin)
    )

    X = df["clean_text"].tolist()
    y = df["label"].tolist()

    # Split 75/25 stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        max_features=MAX_FEATURES
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Save TF-IDF
    joblib.dump(vectorizer, f"models/{dataset_name}_tfidf.joblib")

    # -----------------
    # Model 1: LinearSVC
    # -----------------
    svc = LinearSVC()
    svc.fit(X_train_vec, y_train)
    pred_svc = svc.predict(X_test_vec)

    save_report(dataset_name, "LinearSVC", y_test, pred_svc)
    joblib.dump(svc, f"models/{dataset_name}_linearsvc.joblib")

    # -----------------
    # Model 2: SGDClassifier (SVM-like)
    # -----------------
    sgd = SGDClassifier(
        loss="hinge",      # linear SVM style
        max_iter=3000,
        tol=1e-3,
        random_state=RANDOM_STATE
    )
    sgd.fit(X_train_vec, y_train)
    pred_sgd = sgd.predict(X_test_vec)

    save_report(dataset_name, "SGDClassifier", y_test, pred_sgd)
    joblib.dump(sgd, f"models/{dataset_name}_sgd.joblib")

    print(f"✅ Saved models: models/{dataset_name}_*.joblib")


def main():
    ensure_dirs()

    # 1) arHateDataset
    df_arhate = load_and_standardize(ARHATE_PATH, text_col="Tweet", label_col="Class")
    run_models("arHate", df_arhate, keep_latin=KEEP_LATIN_ARHATE)

    # 2) MDOLLD (Moroccan Darija Offensive Language Dataset)
    df_mdolld = load_and_standardize(MDOLLD_PATH, text_col="text", label_col="label")
    run_models("MDOLLD", df_mdolld, keep_latin=KEEP_LATIN_MDOLLD)

    print("\n🎉 Done. Check folders: reports/ and models/")


if __name__ == "__main__":
    main()
