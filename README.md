# 🪙 Bitcoin Price Prediction: Random Forest vs Linear Regression (cuML)

This project compares two GPU-accelerated models—**Random Forest Regressor** and **Linear Regression**—for predicting Bitcoin prices using NVIDIA’s RAPIDS cuML library. It includes GPU profiling and performance metrics to assess each model’s accuracy and efficiency.

---

## 📘 Models Overview

| Feature                     | Random Forest Regressor                       | Linear Regression                            |
|----------------------------|-----------------------------------------------|----------------------------------------------|
| Library                    | `cuml.ensemble.RandomForestRegressor`         | `cuml.linear_model.LinearRegression`         |
| Dataset                    | Bitcoin historical prices                     | BTCUSD 1-min data via Google Drive           |
| Input Preprocessing        | MinMax Scaling                                | Standard Scaling                              |
| Hardware                   | NVIDIA A100 (CUDA 12.4)                        | NVIDIA A100 (CUDA 12.4)                     |

---

## ⚙️ Model Characteristics

| Aspect                     | Random Forest                                | Linear Regression                            |
|---------------------------|----------------------------------------------|----------------------------------------------|
| Type                      | Ensemble / Tree-based                        | Linear model                                 |
| Captures Non-linearity    | ✅                                            | ❌ (Assumes linearity)                        |
| Interpretability          | Moderate                                     | High                                         |
| Training Speed            | Slower due to ensemble trees                 | Very fast                                    |
| Overfitting Risk          | Low to Moderate                              | Higher (without regularization)             |

---

## 📊 Performance Metrics

### ✅ Random Forest (cuML)
| Metric                | Value          |
|----------------------|----------------|
| Mean Squared Error   | 71,194,427.06  |
| R-squared (R²)       | 0.8472         |
| Mean Absolute Error  | 4,153.05       |

### ✅ Linear Regression (cuML)
> Performance metrics for Linear Regression are not directly extracted from the notebook. Run the evaluation using:
```python
from cuml.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

## 🔬 GPU Profiling (Random Forest Only)

| Task            | Result            |
|-----------------|-------------------|
| Training Time   | 14.2s (wall time) |
| Inference Time  | 49.3ms (wall time)|
| GPU Utilization | 0% (idle on infer)|
| VRAM Usage      | 3.4GB / 40GB      |


---

## 💻 Code Snippets

### 📌 Random Forest Training
```python
from cuml.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=101)
rf_regressor.fit(X_train, y_train)
```

### 📌 Linear Regression Training
```python
from cuml.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

---

## 🚀 Setup & Usage

### 🔧 Requirements
```bash
pip install cuml-cu12 pandas numpy
```

### ▶️ Running the Models
1. Apply appropriate scaling: MinMax for Random Forest, StandardScaler for Linear Regression.
2. Train and evaluate using the provided code snippets.

---

## 📜 License

This project may include third-party dependencies and is subject to their respective licenses.


---

## 📝 Notes

- Random Forest is generally more robust to non-linear patterns in Bitcoin prices.
- Linear Regression is faster and simpler, but may underperform on complex datasets.
- Consider using both models as benchmarks for more advanced architectures.

