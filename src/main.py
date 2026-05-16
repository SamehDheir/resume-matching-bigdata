import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score

spark = SparkSession.builder \
    .appName("ResumeMatching") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = pd.read_csv("data/Resume.csv")
sdf = spark.createDataFrame(df[["ID", "Resume_str", "Category"]])
sdf_clean = sdf.withColumn("text",
    trim(regexp_replace(lower(col("Resume_str")), "[^a-z\\s]", "")))

def run_pipeline(num_partitions):
    sdf_part = sdf_clean.repartition(num_partitions)
    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="text", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="filtered"),
        HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=5000),
        IDF(inputCol="rawFeatures", outputCol="features")
    ])
    start = time.time()
    model = pipeline.fit(sdf_part)
    result = model.transform(sdf_part)
    result.count()
    elapsed = round(time.time() - start, 2)
    print(f"Partitions: {num_partitions} | Time: {elapsed}s")
    return elapsed, result

times = {}
for p in [1, 2, 4, 8]:
    times[p], _ = run_pipeline(p)

_, result = run_pipeline(8)
result_pd = result.select("Category", "features").toPandas()

def sparse_to_array(v):
    arr = np.zeros(5000)
    for i, val in zip(v.indices, v.values):
        arr[i] = val
    return arr

X = normalize(np.array([sparse_to_array(r) for r in result_pd["features"]]))
y = result_pd["Category"].values
sim_matrix = cosine_similarity(X)
y_true, y_pred = [], []

for i in range(len(y)):
    sims = sim_matrix[i].copy()
    sims[i] = -1
    y_pred.append(y[np.argmax(sims)])
    y_true.append(y[i])

accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0)*100:.2f}%")
print(f"Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0)*100:.2f}%")
print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted', zero_division=0)*100:.2f}%")
spark.stop()
