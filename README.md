# 🚀 Green AI Benchmark: Attention vs Matrix Multiplication

## 📌 Overview

This project explores the environmental and computational impact of Transformer-style attention mechanisms compared to multithreaded matrix multiplication.

The benchmark evaluates:

- ⚡ Execution Time
- 🌍 CO₂ Emissions (kg CO2eq)
- 📈 Scalability with matrix size
- 🧠 Transformer Attention complexity

using:

- PyTorch
- CodeCarbon
- Matplotlib 3D

---

## 🎯 Objective

The goal is to study the trade-off between:

- Algorithmic performance
- Energy consumption
- Environmental impact

in AI numerical computations.

This project is part of a broader research direction on:

> Green AI, Sustainable Computing, and Eco-Responsible Machine Learning.

---

## 🧪 Compared Methods

### A — Scaled Dot-Product Attention

Inspired by Transformer architectures:

\[
Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Operations:
- Matrix multiplication
- Transposition
- Softmax
- Attention weighting

---

### B — Multithreaded Matrix Multiplication

PyTorch optimized matrix multiplication using CPU multithreading.

Operations:
- Matrix multiplication
- Matrix transposition
- Parallel computation

---

## 📊 Metrics

For multiple matrix sizes:

```python
[16, 24, 32, 48, 64, 96, 128]
```

the benchmark measures:

- ⏱ Mean execution time (s/op)
- 🌱 Mean CO₂ emissions (kg CO2eq)
- 📉 Standard deviation
- 🔬 Statistical comparison

Each experiment is repeated:

```python
k = 15
```

times for statistical reliability.

---

## 📈 3D Visualization

The project generates a 3D comparative graph:

- X → Matrix size (N)
- Y → Mean execution time
- Z → Mean CO₂ emissions

Including:
- Attention vs MatMul curves
- 3D error bars
- Statistical variability visualization

Example output:

```text
ges_3d.png
```

---

## 🛠 Technologies Used

| Technology | Purpose |
|---|---|
| PyTorch | Matrix operations |
| CodeCarbon | CO₂ tracking |
| NumPy | Statistics |
| Matplotlib | 3D visualization |
| Python | Benchmark framework |

---

## 📂 Project Structure

```text
green-ai-benchmark/
│
├── src/
│   ├── attention.py
│   ├── concurrent_mat_mul.py
│   └── benchmark_attention_matmul_3d.py
│
├── images/
│   └── ges_3d.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/green-ai-benchmark.git
cd green-ai-benchmark
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Benchmark

```bash
python benchmark_attention_matmul_3d.py
```

---

## 📌 Example Console Output

```text
Attention :
temps = 0.003214 ± 0.000421 s/op
GES = 0.000000012341 ± 0.000000001241 kg CO2eq

MatMul :
temps = 0.001842 ± 0.000210 s/op
GES = 0.000000008742 ± 0.000000000934 kg CO2eq
```

---

## 🌍 Research Context

This work is related to:

- Green Computing
- Sustainable AI
- Transformer Optimization
- Energy-Aware Deep Learning
- Eco-Responsible Numerical Computing

---

## 🔮 Future Improvements

- 🔥 FlashAttention integration
- ⚙️ GPU benchmarking
- ☁️ Cloud energy comparison
- 📊 Real-time monitoring dashboard
- 🧠 Large-scale Transformer profiling

---

## 👨‍💻 Author

Ahmed Mohamed Yislim  
Master's Student — IoT & Intelligent Systems

---

## ⭐ If you like this project

- Give it a ⭐ on GitHub
- Fork the repository
- Contribute with new optimization ideas

---

## 📜 License

MIT License
