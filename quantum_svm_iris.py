"""
量子SVM（QSVC）によるIrisデータの分類と決定境界の可視化

【量子実験としてのポイント】
1. 特徴量マップ: ZZFeatureMapを採用し、データの非線形な量子エンコーディングを実施。
2. スケーリング: 量子ビットの回転角に合わせ、MinMaxScalerで[0, π]の範囲に写像。
3. カーネル計算: Qiskit Primitives V2規格のStatevectorSamplerを用い、
   ComputeUncomputeアルゴリズムによって忠実度（Fidelity）を算出。
4. 描画負荷: 量子シミュレーションの計算量を抑えるため、決定境界のグリッド解像度を調整。
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute

# --- 1. データ準備とモデル学習 ---
iris = datasets.load_iris()
X = iris.data[iris.target != 2, :2] 
y = iris.target[iris.target != 2]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# モデル定義
f_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')
sampler = StatevectorSampler()
fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(feature_map=f_map, fidelity=fidelity)
qsvc = QSVC(quantum_kernel=kernel)

print("学習を開始しています（約30秒）...")
start_train = time.time()
qsvc.fit(X_scaled, y) 
end_train = time.time()

# --- 正解率の計算 ---
accuracy = qsvc.score(X_scaled, y)
print(f"--- 実験結果 ---")
print(f"学習時間: {end_train - start_train:.4f} 秒")
print(f"正解率  : {accuracy * 100:.2f} %")
print(f"----------------")

# ---描画用のグリッドデータを作成 ---
print("\n決定境界の計算を開始します。しばらくお待ちください...")
start_plot = time.time()

h = 0.8 # グリッドの細かさ（計算負荷削減のため粗めに設定）
x_min, x_max = X_scaled[:, 0].min() - 0.2, X_scaled[:, 0].max() + 0.2
y_min, y_max = X_scaled[:, 1].min() - 0.2, X_scaled[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = qsvc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
end_plot = time.time()
print(f"描画用計算完了: {end_plot - start_plot:.1f} 秒")

# --- グラフの描画 ---
plt.figure(figsize=(8, 6))

# 背景を予測結果で塗りつぶす
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)

# 実際のデータ点を散布図で重ねる
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors='k')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Scaled Feature 1 (Length)')
plt.ylabel('Scaled Feature 2 (Width)')
plt.title(f'Quantum SVM Decision Boundary (Accuracy: {accuracy*100:.1f}%)')
plt.show()
