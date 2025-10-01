# Bitcoin ML Pipeline

This repository provides a complete workflow for working with cryptocurrency OHLCV (Open, High, Low, Close, Volume) datasets. It covers data loading, exploratory data analysis (EDA), feature engineering, dataset preparation for machine learning, model training, and evaluation.

**Supported DataFrame Libraries:**  
You can load and process your data using any of the following frameworks:
- **Pandas** (for traditional in-memory analysis)
- **Polars** (for fast, multi-threaded DataFrame operations)
- **PySpark** (for distributed processing and large-scale data)

This flexibility allows you to choose the best tool for your dataset

---

## Dataset

This project uses the [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data) from Kaggle.

- **Format:** CSV, OHLCV (Open, High, Low, Close, Volume) at 1-minute intervals
- **Size:** ~350MB (over 2 million rows)
- **Columns:** Timestamp (epoch seconds), Open, High, Low, Close, Volume

You can download the dataset directly from Kaggle and place it in your project directory.  
For instructions on loading the dataset into Pandas, Polars, or PySpark, refer to the [DataFrameLoader](#1-dataframeloader) section below and see usage examples in the notebook: [Bitcoin_DataAnalysis.ipynb](./Bitcoin_DataAnalysis.ipynb)

---

## Installation and Setup

Install all required Python packages:

```bash
pip install -r requirements.txt
```

Or, manually install the main dependencies:

```bash
pip install pandas polars pyspark pyspark-connect matplotlib mplfinance seaborn scikit-learn
```

For PySpark with Hadoop 3 support (if needed):

```bash
PYSPARK_HADOOP_VERSION=3 pip install pyspark
```

Start a Spark session in your notebook or script:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Bitcoin_DataAnalysis").getOrCreate()
```
