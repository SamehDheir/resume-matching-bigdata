# AI-Powered Resume Matching with Big Data Processing

A distributed resume matching system using Apache PySpark and TF-IDF, built for the Big Data course at the Islamic University of Gaza.

## Dataset
- **Source:** [Resume Dataset - Kaggle](https://www.kaggle.com/datasets/snehaanbhauw/resume-dataset)
- **Size:** 24,840 resumes across 24 job categories (10x augmented)

## Results

### Speed-up (Parallel Processing)
| Partitions | Time (s) | Speed-up |
|------------|----------|----------|
| 1          | 11.57    | 1.00x    |
| 2          | 4.79     | 2.42x    |
| 4          | 4.42     | 2.62x    |
| 8          | 7.66     | 1.51x    |

### Evaluation Metrics (on 2,484 sample)
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 77.70% |
| Precision | 77.85% |
| Recall    | 77.70% |
| F1-Score  | 77.74% |

## How to Run
```bash
pip install pyspark scikit-learn pandas matplotlib numpy
python src/main.py
```

## Tech Stack
- Apache PySpark 4.0
- scikit-learn
- pandas / numpy
- matplotlib
