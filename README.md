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
