"""
古典SVM（RBFカーネル）によるAd Hocデータの分類と決定境界の可視化

【実験の意図】
1. 量子的相関を持つAd Hocデータセットに対し、強力な古典非線形カーネル（RBF）が
   どのような決定境界を形成するかを可視化する。
2. 古典的な高次元写像（無限次元ヒルベルト空間への写像）であっても、
   特定の量子的な規則性を持つデータの識別には限界があることを実証する。
3. 量子SVM（QSVC）が100%の精度を出せる一方で、古典SVMが低精度に留まる視覚的な理由を提示。
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from qiskit_machine_learning.datasets import ad_hoc_data

# --- 1. Ad Hocデータの生成 ---
feature_dim = 2
train_size = 20
test_size = 10

X_train, y_train, X_test, y_test = ad_hoc_data(
    training_size=train_size,
    test_size=test_size,
    n=feature_dim,
    gap=0.3
)

# ---ラベルの形式を変換 (One-hot -> 1D) ---
y_train_1d = np.argmax(y_train, axis=1)
y_test_1d = np.argmax(y_test, axis=1)

# --- 古典SVM (RBFカーネル) の学習 ---
print("古典SVM (RBFカーネル) で学習中...")
c_svc = SVC(kernel='rbf', probability=True)
c_svc.fit(X_train, y_train_1d)

# --- 決定境界の描画準備 ---
# グラフの表示範囲設定
x_min, x_max = X_train[:, 0].min() - 0.2, X_train[:, 0].max() + 0.2
y_min, y_max = X_train[:, 1].min() - 0.2, X_train[:, 1].max() + 0.2

# グリッドの作成（解像度）
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# --- グリッド点に対する予測 ---
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = c_svc.predict_proba(grid_points)[:, 1]
Z = Z.reshape(xx.shape)

# ---グラフの描画 ---
plt.figure(figsize=(8, 6))

# 決定境界（背景色）
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# 実際のデータ点（散布図）
plt.scatter(X_train[y_train_1d == 0, 0], X_train[y_train_1d == 0, 1],
            c='r', marker='o', edgecolors='k', label='Class A (0)')
plt.scatter(X_train[y_train_1d == 1, 0], X_train[y_train_1d == 1, 1],
            c='b', marker='s', edgecolors='k', label='Class B (1)')

plt.title("Classical SVM Decision Boundary on Ad Hoc Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
