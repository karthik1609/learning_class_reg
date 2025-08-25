### Learning: scikit-learn demos (classification, regression, clustering, NLP, CV)

This repo demonstrates simple examples using scikit-learn for:
- Classification
- Regression
- Clustering
- Basic NLP (text vectorization + simple classifier)
- Basic CV (simple image ops + clustering)

It uses `uv` for Python package management.

Quick start

1) Install dependencies

```bash
uv sync
```

2) Run demos

```bash
uv run python demos/classification_demo.py
uv run python demos/regression_demo.py
uv run python demos/clustering_demo.py
uv run python demos/nlp_demo.py
uv run python demos/cv_demo.py
```

Notes

- Demos run headless and print metrics to the console.
- CV demo writes two images to the project root: `demos_outputs_cv_original.png`, `demos_outputs_cv_kmeans.png`.
- If classification demo warns about `multi_class` deprecation, it's safe to ignore.
