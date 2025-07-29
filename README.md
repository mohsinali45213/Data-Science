# 🧠 Types of Machine Learning

Machine Learning (ML) is a subset of Artificial Intelligence (AI) that enables systems to learn patterns from data to make predictions or decisions. ML algorithms can be categorized into:

1. Supervised Learning  
2. Unsupervised Learning  
3. Reinforcement Learning  
4. Semi-Supervised Learning

---

## 📘 1. Supervised Learning

### ✅ Definition:
Learning from **labeled data** — both inputs and correct outputs are provided.

### 🎯 Goal:
To **learn a function** that maps input `X` to output `Y`.

---

### 📊 Supervised Task Types

| Task Type     | Description                                    | Common Algorithms                                                |
|---------------|------------------------------------------------|------------------------------------------------------------------|
| **Classification** | Predict **discrete** class labels (e.g., spam or not) | ✅ Logistic Regression  <br> ✅ Decision Tree  <br> ✅ Random Forest <br> ✅ Support Vector Machine (SVM) <br> ✅ K-Nearest Neighbors (KNN) <br> ✅ Naive Bayes <br> ✅ Neural Networks |
| **Regression**     | Predict **continuous** values (e.g., house prices)     | ✅ Linear Regression <br> ✅ Ridge/Lasso Regression <br> ✅ Decision Tree Regressor <br> ✅ Random Forest Regressor <br> ✅ SVR (Support Vector Regression) <br> ✅ XGBoost Regressor <br> ✅ Neural Networks (e.g., MLPRegressor) |

### 🧪 Use Cases:
- Classification: Email spam detection, disease diagnosis  
- Regression: Stock price prediction, temperature forecasting

### 📏 Evaluation Metrics:
| Classification Metrics          | Regression Metrics                |
|----------------------------------|------------------------------------|
| Accuracy, Confusion Matrix, Precision, Recall, F1  | MSE, RMSE, MAE, R² Score, Adjusted R² Score |

---

## 📙 2. Unsupervised Learning

### ✅ Definition:
Learning from **unlabeled data** — only input data `X` is provided.

### 🎯 Goal:
To **discover patterns, clusters, or structure** within data.

---

### 📊 Unsupervised Task Types

| Task Type               | Description                                         | Common Algorithms                                               |
|-------------------------|-----------------------------------------------------|------------------------------------------------------------------|
| **Clustering**          | Group similar data points into clusters             | ✅ K-Means <br> ✅ DBSCAN <br> ✅ Agglomerative Clustering <br> ✅ Gaussian Mixture Models (GMM) <br> ✅ Mean Shift |
| **Dimensionality Reduction** | Reduce number of input features while preserving structure | ✅ PCA (Principal Component Analysis) <br> ✅ t-SNE <br> ✅ LLE (Locally Linear Embedding) <br> ✅ Autoencoders <br> ✅ UMAP |

### 🧪 Use Cases:
- Clustering: Customer segmentation, anomaly detection  
- Dimensionality Reduction: Feature compression, data visualization

### 📏 Evaluation Metrics:
| Clustering Metrics                          | Dimensionality Reduction |
|---------------------------------------------|---------------------------|
| Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz | Explained Variance, Reconstruction Error |

---

## 📕 3. Reinforcement Learning

### ✅ Definition:
Involves an **agent** learning through **trial and error** in an **environment**, guided by **rewards** and **penalties**.

### 🎯 Goal:
To **maximize cumulative reward** over time through learning optimal actions.

---

### 📊 Reinforcement Learning Categories

| Category              | Description                                  | Common Algorithms                                 |
|------------------------|----------------------------------------------|---------------------------------------------------|
| **Model-Free Methods** | Learn directly from experience              | ✅ Q-Learning <br> ✅ SARSA <br> ✅ DQN (Deep Q-Networks) |
| **Policy Optimization**| Directly learn the action policy             | ✅ REINFORCE <br> ✅ Actor-Critic <br> ✅ PPO (Proximal Policy Optimization) <br> ✅ A3C |

### 🧪 Use Cases:
- Self-driving cars  
- Game playing (e.g., AlphaGo, Atari)  
- Robotic control  
- Automated trading systems

### 📏 Evaluation Metrics:
- Cumulative Reward  
- Average Return per Episode  
- Learning Speed (Convergence Rate)

---

## 📗 4. Semi-Supervised Learning

### ✅ Definition:
Uses **a small amount of labeled data** with **a large amount of unlabeled data** to improve learning accuracy.

### 🎯 Goal:
To combine **efficiency of unlabeled data** with **accuracy from labeled data**.

---

### 📊 Semi-Supervised Learning Techniques

| Approach                | Description                                                | Common Algorithms                                             |
|-------------------------|------------------------------------------------------------|---------------------------------------------------------------|
| **Self-Training**       | Model iteratively labels unlabeled data                    | ✅ Any classifier with confidence thresholds (e.g., SVM, RF)  |
| **Co-Training**         | Two models train on different views/features               | ✅ Logistic Regression + Decision Tree                        |
| **Label Propagation**   | Spreads labels in a graph-based structure                  | ✅ Label Propagation <br> ✅ Label Spreading                  |
| **Semi-Supervised SVM** | SVMs adapted to exploit unlabeled data                     | ✅ Transductive SVM (TSVM)                                    |

### 🧪 Use Cases:
- Image recognition (few labeled images)
- Medical diagnostics (where labeled data is expensive)
- Web page classification

### 📏 Evaluation Metrics:
- Same as Supervised Learning (Accuracy, F1 Score, etc.)
- **Label Efficiency**: Performance improvement per number of labeled examples

---

## 📌 Summary Table

| ML Type                | Task                     | Algorithm Examples                                | Use-Cases                        | Metrics                      |
|------------------------|--------------------------|---------------------------------------------------|----------------------------------|------------------------------|
| **Supervised**         | Classification           | SVM, RF, KNN, Logistic Regression, Neural Net     | Spam detection, disease diagnosis| Accuracy, F1, ROC-AUC        |
|                        | Regression               | Linear Regression, SVR, RF Regressor, XGBoost     | House price prediction           | MSE, RMSE, R² Score          |
| **Unsupervised**       | Clustering               | K-Means, DBSCAN, GMM, Agglomerative               | Market segmentation              | Silhouette Score, DB Index   |
|                        | Dimensionality Reduction | PCA, t-SNE, Autoencoders                          | Feature reduction, visualization | Explained Variance           |
| **Reinforcement**      | Sequential Decision-Making| Q-Learning, DQN, PPO, Actor-Critic               | Robotics, Games, Trading         | Total Reward, Avg. Return    |
| **Semi-Supervised**    | Mixed (Few Labels)       | Self-training, Co-training, Label Propagation     | Image classification, NLP        | Accuracy, Label Efficiency   |

---

## 📚 References

- [DeepLearning.ai](https://www.deeplearning.ai/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*
- Hastie, Tibshirani, and Friedman (2009). *The Elements of Statistical Learning

> ✍️ *Feel free to fork and modify this document. Add practical code examples using Scikit-learn, PyTorch, or TensorFlow to create a fully functional ML guide.*


| Algorithm                    | Import Statement                                  | ML Type(s)              | Learning Type |
|-----------------------------|-------------------------------------------------|------------------------|---------------|
| LinearRegression            | from sklearn.linear_model import LinearRegression | Regression             | Supervised    |
| LogisticRegression          | from sklearn.linear_model import LogisticRegression | Classification         | Supervised    |
| Ridge                      | from sklearn.linear_model import Ridge           | Regression             | Supervised    |
| Lasso                      | from sklearn.linear_model import Lasso           | Regression             | Supervised    |
| ElasticNet                 | from sklearn.linear_model import ElasticNet      | Regression             | Supervised    |
| KNeighborsClassifier       | from sklearn.neighbors import KNeighborsClassifier | Classification         | Supervised    |
| KNeighborsRegressor        | from sklearn.neighbors import KNeighborsRegressor | Regression             | Supervised    |
| SVC (Support Vector Classifier) | from sklearn.svm import SVC                   | Classification         | Supervised    |
| SVR (Support Vector Regressor)  | from sklearn.svm import SVR                   | Regression             | Supervised    |
| LinearSVC                  | from sklearn.svm import LinearSVC                 | Classification         | Supervised    |
| LinearSVR                  | from sklearn.svm import LinearSVR                 | Regression             | Supervised    |
| DecisionTreeClassifier     | from sklearn.tree import DecisionTreeClassifier   | Classification         | Supervised    |
| DecisionTreeRegressor      | from sklearn.tree import DecisionTreeRegressor    | Regression             | Supervised    |
| RandomForestClassifier     | from sklearn.ensemble import RandomForestClassifier | Classification         | Supervised    |
| RandomForestRegressor      | from sklearn.ensemble import RandomForestRegressor | Regression             | Supervised    |
| GradientBoostingClassifier | from sklearn.ensemble import GradientBoostingClassifier | Classification     | Supervised    |
| GradientBoostingRegressor  | from sklearn.ensemble import GradientBoostingRegressor | Regression           | Supervised    |
| AdaBoostClassifier         | from sklearn.ensemble import AdaBoostClassifier    | Classification         | Supervised    |
| AdaBoostRegressor          | from sklearn.ensemble import AdaBoostRegressor     | Regression             | Supervised    |
| GaussianNB                 | from sklearn.naive_bayes import GaussianNB         | Classification         | Supervised    |
| BernoulliNB                | from sklearn.naive_bayes import BernoulliNB        | Classification         | Supervised    |
| MultinomialNB              | from sklearn.naive_bayes import MultinomialNB      | Classification         | Supervised    |
| MLPClassifier (Neural Network) | from sklearn.neural_network import MLPClassifier | Classification         | Supervised    |
| MLPRegressor (Neural Network)  | from sklearn.neural_network import MLPRegressor  | Regression             | Supervised    |
| PCA (Principal Component Analysis) | from sklearn.decomposition import PCA        | Dimensionality Reduction | Unsupervised  |
| KMeans                     | from sklearn.cluster import KMeans                 | Clustering             | Unsupervised  |
| DBSCAN                     | from sklearn.cluster import DBSCAN                 | Clustering             | Unsupervised  |
| AgglomerativeClustering    | from sklearn.cluster import AgglomerativeClustering | Clustering           | Unsupervised  |







| Algorithm                    | When to Use                                                   | When NOT to Use                                                                             |
|------------------------------|---------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| LinearRegression             | Predicting continuous values with linear relationships.        | Avoid if the relationship is highly non-linear or there are many outliers.                  |
| LogisticRegression           | Binary/multiclass classification with linearly separable data. | Avoid if classes are not linearly separable or for very large, complex feature spaces without kernel tricks. |
| Ridge                        | Regularized linear regression to prevent overfitting.          | Not ideal if feature selection or sparsity is needed (use Lasso instead).                   |
| Lasso                        | Linear regression with feature selection by coefficient shrinkage. | Can be unstable if features are highly correlated; might drop important correlated features. |
| ElasticNet                   | Balances Ridge and Lasso for linear regression with regularization. | Avoid if interpretability is primary goal or if non-linear relationships dominate.           |
| KNeighborsClassifier         | Simple classification when data is small and well-separated.   | Avoid with very large or high-dimensional datasets (curse of dimensionality) or noisy data.  |
| KNeighborsRegressor          | Non-linear regression with instance-based approach.            | Not suitable for large datasets or very noisy data; prediction can be slow.                  |
| SVC (Support Vector Classifier) | Classification with clear class separation in high-dimensional data. | Avoid on very large datasets due to high computational cost; struggles with noisy, overlapping classes. |
| SVR (Support Vector Regressor)  | Regression in high-dimensional spaces with non-linear functions. | Avoid for very large datasets; sensitive to hyperparameters and scaling.                     |
| LinearSVC                    | Fast linear classification.                                   | Poor choice if data is not linearly separable or complex nonlinear decision boundaries exist.|
| LinearSVR                    | Linear regression requiring fast computation.                 | Avoid when data relationships are highly non-linear or contain outliers.                     |
| DecisionTreeClassifier       | Interpretable classification, handling non-linearity and categorical features. | Avoid deep trees on small datasets due to overfitting; unstable with small data variations.  |
| DecisionTreeRegressor        | Regression with interpretability and non-linear relationships. | Avoid deep trees on limited data to prevent overfitting; sensitive to noisy data.            |
| RandomForestClassifier       | Robust classification with complex relationships.              | Avoid models requiring high interpretability; can be slow for real-time predictions.         |
| RandomForestRegressor        | Non-linear regression handling interactions well.              | Same as classifier; not suitable if model transparency is critical or if very fast predictions needed. |
| GradientBoostingClassifier   | Highly accurate classification through boosting.               | Avoid if training speed is a concern; sensitive to noisy data and outliers.                  |
| GradientBoostingRegressor    | Powerful regression improving by sequentially fitting residuals. | Avoid if data is very noisy or if tuning is not feasible due to complexity.                  |
| AdaBoostClassifier           | Boosting weak classifiers for noisy datasets.                  | Poor performance if base learners are too complex; sensitive to outliers.                    |
| AdaBoostRegressor            | Boosted regression to handle bias-prone tasks.                 | Avoid if data is very noisy or weak learners are inappropriate.                              |
| GaussianNB                   | Classification with Gaussian-distributed numeric features.     | Avoid if feature independence assumption is violated or distributions are not Gaussian-like. |
| BernoulliNB                  | Binary/boolean features classification.                        | Ineffective for continuous or highly correlated features.                                    |
| MultinomialNB                | Text or count-based classification (e.g., document classification). | Not suitable for continuous features or non-count data.                                       |
| MLPClassifier (Neural Network)| Complex classification with non-linear decision boundaries.   | Avoid if dataset is small (overfitting risk); requires tuning and longer training time.      |
| MLPRegressor (Neural Network)| Non-linear regression with neural networks.                   | Avoid on small datasets or when interpretability is needed; sensitive to feature scaling.    |
| PCA (Principal Component Analysis) | Dimensionality reduction for visualization or preprocessing. | Not effective if variance is not meaningful or if components lack interpretability.           |
| KMeans                       | Partitioning data into k spherical clusters of similar size.   | Avoid with clusters of varying density/shape; sensitive to outliers and requires k preset.   |
| DBSCAN                       | Density-based clustering for arbitrary shapes and noise.       | Not suitable for varying density clusters or very high-dimensional spaces without preprocessing. |
| AgglomerativeClustering      | Hierarchical clustering to understand nested data structures.  | Poor scalability with very large datasets; sensitive to noise and requires distance metric choice. |

