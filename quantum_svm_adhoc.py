"""
量子SVM (QSVC) による Ad Hoc データセットの分類と決定境界の可視化

【プログラムの解説】
1. 特徴量写像: ZZFeatureMap (reps=2) を使用し、非線形な量子もつれを含む
   量子特徴空間へデータをマッピング。
2. カーネル計算: Qiskit Primitives V2 (StatevectorSampler) を基盤とした
   FidelityQuantumKernel を採用。
3. 可視化: 量子計算の負荷を考慮し、グリッド解像度を調整した上で、
   量子SVMが形成する複雑な決定境界を等高線として描画。
4. 目的: 古典SVMでは識別困難な Ad Hoc データに対し、量子機械学習が
   100%に近い正解率で完璧な分離を行う様子を視覚化する。
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Qiskit関連の最新インポート
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute

# --- Ad Hocデータの生成 ---
feature_dim = 2
train_size = 20
test_size = 10
X_train, y_train, X_test, y_test = ad_hoc_data(
    training_size=train_size,
    test_size=test_size,
    n=feature_dim,
    gap=0.3
)

# ラベルを2次元(One-hot)から1次元(0 or 1)に変換
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

# --- 量子SVM (QSVC) の学習 ---
print("量子SVM (QSVC) で学習中（1〜2分かかります）...")
# 2量子ビットを用い、量子もつれを生成する特徴量マップ
f_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear')
sampler = StatevectorSampler()
fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(feature_map=f_map, fidelity=fidelity)
q_svc = QSVC(quantum_kernel=kernel)

start = time.time()
q_svc.fit(X_train, y_train)
q_score = q_svc.score(X_test, y_test)
train_time = time.time() - start

# --- 決定境界の描画準備 ---
print("\n決定境界の計算を開始します（さらに数分かかります）...")
start_plot = time.time()

# グラフの表示範囲設定
x_min, x_max = X_train[:, 0].min() - 0.2, X_train[:, 0].max() + 0.2
y_min, y_max = X_train[:, 1].min() - 0.2, X_train[:, 1].max() + 0.2

# グリッドの作成（解像度）
# ※量子計算が重いため、少し粗めのグリッド(間隔0.15)に設定しています
h = 0.15
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# --- グリッド点に対する予測 ---
# 量子SVMで全グリッド点のクラスを予測します
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = q_svc.predict(grid_points)
Z = Z.reshape(xx.shape)
end_plot = time.time()
print(f"描画用計算完了: {end_plot - start_plot:.1f} 秒")

# --- グラフの描画 ---
plt.figure(figsize=(8, 6))

# 決定境界（背景色）
# 予測されたクラス（0または1）に応じて色を塗ります
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

# 実際のデータ点（散布図）
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
            c='r', marker='o', edgecolors='k', label='Class A (0)')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
            c='b', marker='s', edgecolors='k', label='Class B (1)')

plt.title(f"Quantum SVM (QSVC) Decision Boundary\nAccuracy: {q_score*100:.1f}%")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
