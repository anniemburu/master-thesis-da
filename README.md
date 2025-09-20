# Comparative Analysis of Deep Neural Network and Tree-Based Models for Regression Tasks: An Empirical Investigation

This repository contains the source code and experiments conducted for my master's thesis project. This project is based on [TabSurvey](https://github.com/kathrinse/TabSurvey) by @kathrinse.  
I adapted and extended their work to suit my thesis. The research investigates the comparative performance of **Deep Neural Networks (DNNs)** and **Tree-Based Models** across a wide range of regression tasks on tabular data.

## 📑 Thesis Summary

This project systematically evaluates the performance of modern deep learning approaches and traditional tree-based algorithms on **14 diverse regression datasets**.  

Key contributions:
- Implementation of baseline **tree-based models** (XGBoost, Random Forest).  
- Implementation and tuning of **deep learning models**, including feed-forward DNNs and transformer-based architectures (SAINT).  
- Use of **Optuna** for hyperparameter optimization with **K-Fold and Nested Cross-Validation** strategies.  
- Comparative analysis of performance trade-offs between interpretability, scalability, and predictive accuracy.  

**Findings:**  
- Tree-based models remain strong baselines for tabular regression.  
- Properly tuned DNNs can achieve competitive results, particularly with advanced regularization and encoding strategies.  
- Transformer-based models (SAINT) show promise but require careful tuning and larger datasets to consistently outperform tree ensembles.  

---

## Installation

Clone the repository:
```bash
git clone https://github.com/anniemburu/master-thesis-da.git
