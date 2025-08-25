from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def build_dataset() -> tuple[list[str], list[int]]:
    texts = [
        "I loved this movie, it was fantastic and uplifting",
        "An excellent performance; absolutely brilliant",
        "Terrible plot and bad acting",
        "I hated it; a waste of time",
        "A wonderful and charming story",
        "Awful, dull, and boring",
        "Best film of the year!",
        "Not my taste, very disappointing",
        "Heartwarming and beautifully shot",
        "Poorly written and unpleasant",
    ]
    labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative
    return texts, labels


def main() -> None:
    texts, labels = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1000),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, target_names=["neg", "pos"]))


if __name__ == "__main__":
    main()


