# AI-Powered Resume Matching with Big Data Processing

A distributed resume matching system using Apache PySpark and TF-IDF.

## Dataset
- **Source:** [Resume Dataset - Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- **Size:** 2,484 resumes across 24 job categories

## Results

### Speed-up
| Partitions | Time (s) | Speed-up |
|------------|----------|----------|
| 1          | 15.30    | 1.00x    |
| 2          | 6.30     | 2.43x    |
| 4          | 6.30     | 2.43x    |
| 8          | 6.43     | 2.38x    |

### Metrics
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 47.58% |
| Precision | 47.26% |
| Recall    | 47.58% |
| F1-Score  | 47.22% |

## How to Run
```bash
pip install pyspark scikit-learn pandas matplotlib numpy
python src/main.py
```
