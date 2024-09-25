import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress


# 自相关计算函数
def autocor(x):
    xm = np.mean(x)
    cor = 0
    xsd = 0
    for i in range(len(x) - 1):
        cor += (x[i] - xm) * (x[i + 1] - xm)
        xsd += (x[i] - xm) ** 2
    cor = cor / xsd + 1 / len(x)
    return cor


# Sen's slope 计算函数
def sen_slope(A):
    Q = []
    for i in range(len(A) - 1):
        for j in range(i + 1, len(A)):
            Q.append((A[j] - A[i]) / (j - i))
    return np.median(Q)


# WS2001 方法实现
def ws2001_method(data):
    n = len(data)
    c0 = autocor(data)
    x = data.copy()

    # 如果自相关系数较大，进行预白化
    if c0 >= 0.05:
        for i in range(n - 1):
            data[i] = (data[i + 1] - c0 * data[i]) / (1 - c0)

    # 计算线性趋势
    slope, intercept, _, _, _ = linregress(range(n), data)

    return slope, intercept


# 从txt文件读取数据
def load_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        data = np.loadtxt(file)
    return np.arange(len(data)), data


# 主程序
file_path = r"E:\2024yan1\qitong-lesson3\第3次课附件-WS2001\input.txt" # 替换成你的文件路径
years, data = load_data_from_txt(file_path)

# 计算趋势
slope, intercept = ws2001_method(data)

# 输出结果
print(f"线性趋势斜率: {slope}")

# 绘制趋势图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.plot(years, data, 'k-o', label='原始数据')
plt.plot(years, intercept + slope * years, 'r--', label='线性趋势')
plt.xlabel('序号')
plt.ylabel('数据')
plt.legend()
plt.show()

