"""
古典SVM（線形カーネル）によるIrisデータの分類と決定境界の可視化

【実験設定の補足】
1. 可視化を目的とするため、Irisデータの先頭2特徴量のみを使用。
2. 量子SVM（2クラス分類）との比較のため、ターゲットクラスを2種に限定。
3. 決定境界の確信度を可視化するため、probability=Trueを設定し等高線を描画。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# --- 1. データ準備 (Irisデータ) ---
iris = datasets.load_iris()
X = iris.data[:, :2]  # 特徴量を2つに絞る（可視化用）
y = iris.target

# 2クラス分類にするため、ラベル2 (Virginica) を除外して 0と1 だけにします
X = X[y != 2]
y = y[y != 2]

# --- 2. 古典SVM (線形カーネル) の学習 ---
print("Irisデータに対して古典SVM(Linear)を学習中...")
# probability=True にすることで、決定境界の「色の濃淡（確率）」を計算できるようにします
model = SVC(kernel='linear', probability=True)
model.fit(X, y)

# --- 3. 描画用グリッドの作成 ---
# グラフの端っこに少し余裕を持たせます
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
h = 0.02 # グリッドの細かさ
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# --- 4. グリッド点に対する予測（確率計算） ---
# predict_proba で「青色クラスになる確率」を計算します
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# --- 5. グラフの描画 ---
plt.figure(figsize=(8, 6))

# 背景を確率に応じて塗りつぶす (RdBu: 赤〜青 のグラデーション)
# alpha=0.6 で少し透けさせて、データ点が見やすいようにします
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# 実際のデータ点をプロット
plt.scatter(X[y==0, 0], X[y==0, 1], c='r', marker='o', edgecolors='k', label='Class 0 (Setosa)')
plt.scatter(X[y==1, 0], X[y==1, 1], c='b', marker='s', edgecolors='k', label='Class 1 (Versicolor)')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Classical SVM Decision Boundary on Iris Data')
plt.legend()
plt.show()
