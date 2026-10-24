import pandas as pd
import numpy as np

import matplotlib
#matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print("First 5 Rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

print("\nDataset Information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

#Data Quality Check

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())

# STEP 5 — Exploratory Data Analysis (EDA)

# 1. Gender Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x="Gender")
plt.title("Customer Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.show()


# 2. Age Distribution
plt.figure(figsize=(7, 4))
sns.histplot(data=data, x="Age", bins=20, kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()


# 3. Annual Income Distribution
plt.figure(figsize=(7, 4))
sns.histplot(data=data, x="Annual Income (k$)", bins=20, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of Customers")
plt.show()


# 4. Spending Score Distribution
plt.figure(figsize=(7, 4))
sns.histplot(data=data, x="Spending Score (1-100)", bins=20, kde=True)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")
plt.show()


# 5. Income vs Spending Score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="Annual Income (k$)",
    y="Spending Score (1-100)"
)
plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.show()


# STEP 6 — Select Features for Clustering

X = data[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

print("\nSelected Features:")
print(X.head())

print("\nFeature Shape:")
print(X.shape)

# STEP 7 — Feature Scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Features:")
print(X_scaled[:5])


# STEP 8 — Elbow Method

inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

print("\nInertia Values:")
print(inertia)

# STEP 9 — Plot Elbow Method

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.xticks(range(1, 11))
plt.grid(True)

plt.show()


# STEP 10 — K-Means Clustering

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

# Train the model
kmeans.fit(X_scaled)

# Get cluster labels
data["Cluster"] = kmeans.labels_

print("\nCluster Labels:")
print(data["Cluster"].value_counts().sort_index())

# STEP 11 — Visualize Customer Clusters

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=data,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")

plt.show()

print("\nCustomer Cluster Results:")
print(data[[
    "CustomerID",
    "Annual Income (k$)",
    "Spending Score (1-100)",
    "Cluster"
]].head(10))

print("\nNumber of Customers in Each Cluster:")
print(data["Cluster"].value_counts().sort_index())

# STEP 12 — Analyze Customer Segments

cluster_summary = data.groupby("Cluster")[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]].mean().round(2)

print("\nCluster Summary:")
print(cluster_summary)

# STEP 13 — Save Model and Scaler

joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!")