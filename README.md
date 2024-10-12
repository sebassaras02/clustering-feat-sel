# 📊 FeatureClus: Feature Selection for Clustering Models

Welcome to **FeatureClus**, a Python library designed to simplify **feature selection** for **clustering models**. This tool helps you select the most relevant features that enhance clustering performance, ensuring you avoid the "curse of dimensionality" and make your clustering algorithms more efficient and interpretable. 🧠

## 🔍 How It Works

The feature selection process is driven by evaluating how each feature impacts the clustering results. **FeatureClus** uses an isolated data shift for each feature to assess its importance. The process follows these steps:

1. **MinMaxScaler**: First, we scale the features using MinMaxScaler to normalize the data.
2. **PCA (80% variance)**: Next, we apply Principal Component Analysis (PCA) to reduce dimensionality, retaining 80% of the variance.
3. **DBSCAN Clustering**: After reducing the dimensionality, DBSCAN is used to perform clustering.
4. **Silhouette Score Calculation**: For each feature, we calculate the silhouette score to evaluate the quality of the clusters. The silhouette score represents how similar an object is to its own cluster compared to other clusters.
5. **Data Shift and Feature Importance**: By applying isolated shifts to each feature and recalculating the silhouette score, we measure how the score changes. The absolute difference in the silhouette score after shifting each feature is used to rank the features by importance.

This method ensures that the features are evaluated for their individual contribution to the clustering process, allowing you to focus on the most impactful features.

## 🚀 Key Features
- 🔍 **Feature Ranking**: Ranks features based on the absolute change in silhouette score after applying isolated shifts to each feature.
- 📈 **Cluster Evaluation Metrics**: Calculates the silhouette score to assess the clustering quality and the influence of each feature.
- 💻 **Easy-to-Use API**: A simple, intuitive API that can be easily integrated into your machine learning pipeline.


## 📦 Installation

To install the library, run the following command:

```bash
pip install featureclus
```

## 📊 Example

Here is a quick example of how to use **FeatureClus** with a clustering algorithm (e.g., KMeans):

```python
from sklearn.cluster import KMeans
from featureclus import FeatureSelection

# Sample DataFrame
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [5, 6, 7, 8, 9],
    'C': [9, 8, 7, 6, 5]
})

# Initialize the FeatureSelection
fs = FeatureSelection(data)

# Select features for clustering
top_features = fs.select_top_features(k=2)

# Perform clustering with the selected features
model = KMeans(n_clusters=2)
model.fit(top_features)
```

## 🛠️ Methods

### `get_metrics()`
Returns metrics that assess how each feature contributes to clustering.

### `select_top_features(k)`
Selects the top `k` features based on their importance to clustering results.

## 📋 Requirements
- Python 3.7+
- Pandas 1.2+
- NumPy 1.19+
- Scikit-learn 0.24+ (for clustering algorithms)


## ☕ Support the Project

If you find this inventory optimization tool helpful and would like to support its continued development, consider buying me a coffee. Your support helps maintain and improve this project!

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.paypal.com/paypalme/sebassarasti)

### Other Ways to Support
- ⭐ Star this repository
- 🍴 Fork it and contribute
- 📢 Share it with others who might find it useful
- 🐛 Report issues or suggest new features

Your support, in any form, is greatly appreciated! 🙏

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Happy clustering! 🎉
