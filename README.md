# 時間序列預測
南華大學 跨領域-人工智慧期末報告

11024101 黃右孝、11024244 黃威志

---

本教學是使用 TensorFlow 進行時間序列預測的簡介。它建構了幾種不同樣式的模型，包括卷積神經網路 (CNN) 和循環神經網路 (RNN)。

本教學包括兩個主要部分，每個部分包含若干小節：

- 預測單一時間步驟：
  * 單一特徵。
  + 所有特徵。
- 預測多個時間步驟：
  * 單次：一次做出所有預測。
  + 自迴歸：一次做出一個預測，並將輸出饋送回模型。

## 安裝
```pthon
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
```
```
2023-11-08 00:53:03.106712: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2023-11-08 00:53:03.106761: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2023-11-08 00:53:03.108388: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
```
## 天氣資料集
本教學使用[馬克斯普朗克生物地球化學研究所](https://www.bgc-jena.mpg.de/wetter/)記錄的[天氣時間序列資料集](https://www.bgc-jena.mpg.de/)。

此資料集包含了14個不同特徵，例如氣溫、氣壓和濕度。自2003年起，這些數據每10分鐘就會被收集一次。為了提高效率，您將只使用2009至2016年之間收集的資料。資料集的這一部分由François Chollet為他的[Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)一書所準備。
```pthon
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
```
```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip
13568290/13568290 [==============================] - 0s 0us/step
```
本教學僅處理**每小時預測**，因此先從10分鐘間隔到1小時對資料進行下採樣：
```python
df = pd.read_csv(csv_path)
# Slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
```
讓我們看一下數據。下面是前幾行：
```
df.head()
```

