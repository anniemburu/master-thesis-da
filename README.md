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

1. Clone the repository:
```bash
git clone https://github.com/anniemburu/master-thesis-da.git
```
2. Create and activate a virtual environment:
```bash
conda create -n myenv python=3.10  #name your env
conda activate myenv
```
3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
cd empirical_experiments/
```

4: Source the the datasets: The datasets is used for this work is a benchmark datasets form [AutoML Benchmark Regression](https://www.openml.org/search?type=benchmark&sort=tasks_included&study_type=task&id=269) in OpenML.

5. Run the train script:

In order to run one-to-one experiment as in one datasets against one models run as:
```bash
python train.py --config/<config-file of the dataset>.yml --model_name <Name of the Model>
```
Refer to the different parameters in ```bash empirical_experiments/utils/parser.py```. The most important ones to configure are as follows:
* ```bash --config``` : .yaml file for the target dataset
* ```bash --dataset ``` : The dataset
* ```bash --objective``` : The learning objectives regression, probabilistic_regression
* ```bash --optimize_hyperparameters``` : Set True for hyperparameter tuning.
* ```bash --binning```: Binning method to use i.e 'uniform', 'quantile' or 'kmeans'
* ```bash --strategy```: Binning strategy to use i,e 'sturges' or 'freedmans'
* ```bash --class_comp``` : Set True for Regression-Classification conversion.


## 🙏 Acknowledgements

Special thanks to my supervisor and the broader open-source research community.
