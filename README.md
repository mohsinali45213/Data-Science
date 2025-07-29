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
