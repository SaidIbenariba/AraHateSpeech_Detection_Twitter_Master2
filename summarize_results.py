# summarize_results.py
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")  # ✅ No GUI backend (fixes Tkinter error)

import matplotlib.pyplot as plt


REPORTS_DIR = Path("reports")
OUT_CSV = REPORTS_DIR / "summary_metrics.csv"
OUT_ACC_PNG = REPORTS_DIR / "accuracy_bar.png"
OUT_F1_PNG = REPORTS_DIR / "f1_class1_bar.png"


def parse_report_filename(filename: str):
    """
    Expected: <Dataset>_<Model>.txt
    Examples:
      arHate_LinearSVC.txt
      MDOLLD_SGDClassifier.txt
    """
    base = filename.replace(".txt", "")
    if "_" not in base:
        return None, None
    dataset, model = base.split("_", 1)
    return dataset, model


def extract_accuracy(text: str):
    """
    Extract accuracy from sklearn classification report text.
    Example line: 'accuracy                           0.8939      8435'
    """
    m = re.search(r"accuracy\s+([0-9]*\.[0-9]+)\s+\d+", text)
    if m:
        return float(m.group(1))
    return None


def extract_f1_class1(text: str):
    """
    Extract F1-score for class '1' row from sklearn classification report.
    Example row:
        1       0.8321    0.8433    0.8377      2738
    """
    # match start of line with optional spaces then "1" then columns
    m = re.search(r"^\s*1\s+([0-9]*\.[0-9]+)\s+([0-9]*\.[0-9]+)\s+([0-9]*\.[0-9]+)\s+\d+\s*$",
                  text, flags=re.MULTILINE)
    if m:
        f1 = float(m.group(3))
        return f1
    return None


def load_reports():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    txt_files = sorted([p for p in REPORTS_DIR.glob("*.txt")])

    if not txt_files:
        print("❌ No .txt reports found in 'reports/'")
        print("   Run: python bench_baselines.py first.")
        return pd.DataFrame(columns=["Dataset", "Model", "Accuracy", "F1_Class1"])

    for path in txt_files:
        dataset, model = parse_report_filename(path.name)
        if not dataset or not model:
            # ignore files that don't follow naming convention
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")

        acc = extract_accuracy(text)
        f1c1 = extract_f1_class1(text)

        rows.append({
            "Dataset": dataset,
            "Model": model,
            "Accuracy": acc,
            "F1_Class1": f1c1
        })

    df = pd.DataFrame(rows)
    # clean & sort
    df = df.dropna(subset=["Accuracy", "F1_Class1"], how="any")
    df = df.sort_values(["Dataset", "Model"]).reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame):
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"✅ Saved summary to {OUT_CSV.as_posix()}")
    print(df)


def plot_bars(df: pd.DataFrame):
    if df.empty:
        print("❌ Empty summary dataframe. Nothing to plot.")
        return

    # Accuracy plot
    pivot_acc = df.pivot(index="Dataset", columns="Model", values="Accuracy")
    ax = pivot_acc.plot(kind="bar", rot=0)
    ax.set_title("Accuracy by Dataset and Model")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(OUT_ACC_PNG, dpi=200)
    plt.close()
    print(f"✅ Saved plot to {OUT_ACC_PNG.as_posix()}")

    # F1 Class 1 plot
    pivot_f1 = df.pivot(index="Dataset", columns="Model", values="F1_Class1")
    ax = pivot_f1.plot(kind="bar", rot=0)
    ax.set_title("F1 (Class 1) by Dataset and Model")
    ax.set_ylabel("F1 Class 1")
    plt.tight_layout()
    plt.savefig(OUT_F1_PNG, dpi=200)
    plt.close()
    print(f"✅ Saved plot to {OUT_F1_PNG.as_posix()}")


def main():
    df = load_reports()
    save_csv(df)
    plot_bars(df)


if __name__ == "__main__":
    main()
