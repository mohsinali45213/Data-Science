# 🧠 Types of Machine Learning

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables systems to learn patterns from data and make predictions or decisions without being explicitly programmed. Based on the nature of the training signal and feedback available, ML is categorized into the following types:

---

## 📘 1. Supervised Learning

### ✅ Definition:
Supervised Learning is based on **labeled datasets**, meaning the algorithm is trained on input-output pairs. The model learns to map inputs (features) to the correct output (target).

### 🎯 Goal:
To **predict an output** (label) for new, unseen data.

### 🔧 Common Algorithms:
| Algorithm              | Description                                           |
|------------------------|-------------------------------------------------------|
| Linear Regression      | Predicts a continuous numerical value.               |
| Logistic Regression    | Classifies data into binary or multi-class labels.  |
| Decision Tree          | Tree-based model for classification/regression.      |
| Random Forest          | Ensemble of decision trees for robust predictions.   |
| Support Vector Machine (SVM) | Maximizes margin between classes.               |
| K-Nearest Neighbors (KNN) | Predicts label based on nearest data points.     |
| Neural Networks        | Mimic human brain with layers and neurons.           |

### 🧪 Use Cases:
- Email spam classification
- Disease diagnosis
- Credit scoring
- Predicting stock prices
- Image classification

### 📊 Evaluation Metrics:
| Task Type     | Common Metrics                          |
|---------------|------------------------------------------|
| Classification| Accuracy, Precision, Recall, F1 Score, ROC-AUC |
| Regression    | MSE (Mean Squared Error), RMSE, MAE, R² Score |

---

## 📙 2. Unsupervised Learning

### ✅ Definition:
Unsupervised Learning deals with **unlabeled data**. The algorithm tries to identify hidden patterns or intrinsic structures in the data.

### 🎯 Goal:
To **discover underlying patterns**, groupings, or structures without explicit guidance.

### 🔧 Common Algorithms:
| Algorithm         | Description                                               |
|-------------------|-----------------------------------------------------------|
| K-Means Clustering| Partitions data into K clusters based on distance.       |
| Hierarchical Clustering | Builds a tree of clusters.                        |
| DBSCAN             | Density-based clustering to identify core samples.      |
| PCA (Principal Component Analysis) | Dimensionality reduction technique.    |
| Autoencoders       | Neural networks for feature learning and compression.   |
| t-SNE              | Visualizes high-dimensional data in 2D/3D.              |

### 🧪 Use Cases:
- Customer segmentation
- Anomaly detection
- Market basket analysis
- Topic modeling (e.g., in NLP)
- Data compression

### 📊 Evaluation Metrics:
| Task Type   | Common Metrics                            |
|-------------|--------------------------------------------|
| Clustering  | Silhouette Score, Davies–Bouldin Index, Calinski-Harabasz Index |
| Dimensionality Reduction | Variance Explained, Reconstruction Error |

---

## 📕 3. Reinforcement Learning

### ✅ Definition:
Reinforcement Learning is about training agents to make sequences of decisions by **interacting with an environment**. It learns via **trial and error**, receiving **rewards** or **penalties**.

### 🎯 Goal:
To **maximize cumulative reward** through a sequence of actions.

### 🔧 Common Algorithms:
| Algorithm          | Description                                               |
|--------------------|-----------------------------------------------------------|
| Q-Learning         | Model-free method to learn optimal policy.               |
| Deep Q Networks (DQN)| Combines Q-learning with deep neural networks.         |
| SARSA              | Similar to Q-learning but considers current policy.       |
| Policy Gradient    | Directly optimizes the policy function.                  |
| Actor-Critic       | Combines value and policy-based methods.                 |

### 🧪 Use Cases:
- Game playing (e.g., AlphaGo, Atari)
- Robotics (motion control)
- Self-driving cars
- Dynamic pricing
- Portfolio optimization

### 📊 Evaluation Metrics:
| Metric               | Description                             |
|----------------------|------------------------------------------|
| Cumulative Reward    | Total reward collected by agent.         |
| Average Return       | Mean reward over multiple episodes.      |
| Convergence Rate     | Speed of learning to optimal policy.     |

---

## 📗 4. Semi-Supervised Learning

### ✅ Definition:
Semi-Supervised Learning combines a **small amount of labeled data** with a **large amount of unlabeled data**. It aims to leverage the unlabeled data to improve learning accuracy.

### 🎯 Goal:
To improve prediction by utilizing both labeled and unlabeled data.

### 🔧 Common Algorithms:
| Algorithm                | Description                                               |
|--------------------------|-----------------------------------------------------------|
| Self-Training            | Uses model’s own predictions to label unlabeled data.     |
| Co-Training              | Trains two classifiers on different views of the data.    |
| Label Propagation        | Propagates labels from labeled to unlabeled data.         |
| Semi-Supervised SVM      | Extends SVMs to unlabeled data.                          |
| Graph-Based Methods      | Utilizes graphs to find label spreading over nodes.       |

### 🧪 Use Cases:
- Medical image classification (where labeling is expensive)
- Speech recognition
- Web content classification
- Document categorization

### 📊 Evaluation Metrics:
Uses the same metrics as supervised learning (Accuracy, F1, etc.) but often includes:
- **Learning Curve Analysis**
- **Label Efficiency** (performance with limited labeled data)

---

## 📌 Summary Table

| Type                | Data Type        | Algorithms                            | Use-Cases                            | Metrics                             |
|---------------------|------------------|----------------------------------------|--------------------------------------|--------------------------------------|
| **Supervised**      | Labeled           | LR, SVM, DT, RF, NN                   | Spam detection, house price prediction | Accuracy, F1, MSE, R²                |
| **Unsupervised**    | Unlabeled         | K-Means, DBSCAN, PCA, t-SNE           | Clustering, anomaly detection         | Silhouette Score, CH Index          |
| **Reinforcement**   | Agent-based       | Q-Learning, DQN, SARSA, Actor-Critic  | Robotics, gaming, self-driving cars   | Reward, Convergence, Return         |
| **Semi-Supervised** | Few labeled + unlabeled | Self-Training, Co-Training        | Medical, web categorization           | Accuracy, Label Efficiency          |

---

## 📚 References

- [DeepLearning.ai ML Specialization](https://www.deeplearning.ai/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.

---

> ✍️ *Feel free to fork this repository and enhance the examples with real-world datasets using Scikit-learn, TensorFlow, or PyTorch!*
