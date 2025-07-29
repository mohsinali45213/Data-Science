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
- Hastie, Tibshirani, and Friedman (2009). *The Elements of Statistical Learning*

---



Certainly! Here is a sample **README.md** file that documents both tables you referenced — the table with Algorithm, Import Statement, ML Type(s), Learning Type, and the table with When NOT to Use guidance. The README is formatted to clearly explain the content and usage of these tables for scikit-learn algorithms.

```markdown
# Scikit-Learn Algorithms Reference

This repository provides a concise reference guide for commonly used **scikit-learn** algorithms, their import statements, machine learning types, learning types (supervised or unsupervised), and practical recommendations for when NOT to use each algorithm.

## Tables Included

### 1. Algorithm Overview Table

This table lists popular scikit-learn algorithms with four key details:

| Field              | Description                                                         |
|--------------------|---------------------------------------------------------------------|
| **Algorithm**      | Name of the scikit-learn algorithm or utility                       |
| **Import Statement**| Python import statement to use the algorithm                        |
| **ML Type(s)**     | The type of machine learning task it performs (Classification, Regression, Clustering, Dimensionality Reduction) |
| **Learning Type**  | Whether the algorithm is used for Supervised or Unsupervised learning |

This overview helps users quickly identify and import the appropriate scikit-learn class based on their machine learning problem type.

### 2. When NOT to Use Table

This table complements the overview by providing scenarios and limitations when each algorithm is **not recommended**. It helps avoid common pitfalls and guides users to choose algorithms better suited to their specific data characteristics and goals.

| Field              | Description                                                         |
|--------------------|---------------------------------------------------------------------|
| **Algorithm**      | Name of the algorithm                                               |
| **When NOT to Use** | Typical scenarios or limitations where the algorithm might perform poorly or be inappropriate |

---

## How to Use This Reference

1. **Algorithm Selection:**
   - Use the first table to find algorithms by ML task (e.g., Classification or Regression).
   - Check if the algorithm is supervised or unsupervised.
   - Copy the import statement directly into your Python project.

2. **Avoiding Pitfalls:**
   - Consult the "When NOT to Use" table to understand limitations.
   - Prevent applying unsuitable algorithms to your data.

3. **Extending the Reference:**
   - This reference is not exhaustive. Feel free to add more algorithms or expand with additional details.
   - Complement with scikit-learn official documentation for newest releases and advanced configurations.

---

## Example Excerpt

| Algorithm                | Import Statement                                   | ML Type(s)                | Learning Type |
|--------------------------|--------------------------------------------------|---------------------------|---------------|
| LinearRegression         | from sklearn.linear_model import LinearRegression | Regression               | Supervised    |
| KMeans                  | from sklearn.cluster import KMeans                  | Clustering              | Unsupervised  |

**When NOT to Use - Linear Regression**

- Avoid if the relationship between features and target is highly non-linear.
- Not suitable if there are many outliers.

**When NOT to Use - KMeans**

- Avoid with clusters of varying density or shape.
- Not robust to outliers and noise.

---

## License

This reference guide is provided for educational and practical use. Please refer to the [scikit-learn](https://scikit-learn.org) license for the underlying library usage.

---

## Contact

For questions or suggestions, please open an issue or contact the maintainer.

---

*Generated on July 29, 2025*
```

If you'd like, I can prepare the README content as a downloadable file or in other formats as well. Let me know!

> ✍️ *Feel free to fork and modify this document. Add practical code examples using Scikit-learn, PyTorch, or TensorFlow to create a fully functional ML guide.*
