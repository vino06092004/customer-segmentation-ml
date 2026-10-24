# 🛒 Customer Segmentation & Behavioral Analytics

An end-to-end unsupervised machine learning web application using **K-Means Clustering** to segment retail customers based on **Annual Income** and **Spending Score**.

## 🚀 Features
- **Live Predictor**: Real-time customer classification using dynamic inputs and interactive cluster visualizations.
- **Cluster Analytics**: Business persona breakdown with actionable marketing insights.
- **Data Explorer**: Interactive exploration of customer records and assigned segments.

## 🛠️ Tech Stack
- **Language**: Python
- **Machine Learning**: scikit-learn (K-Means, StandardScaler)
- **Web App**: Streamlit
- **Visualizations**: Plotly, Matplotlib, Seaborn

## 💻 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/vino06092004/customer-segmentation-ml.git](https://github.com/vino06092004/customer-segmentation-ml.git)
   cd customer-segmentation-ml

STEP 02 - Create and Activate Virtual Environment
Bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate

STEP 03 - Install Dependencies
Bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

STEP 04 - Configure Database Connection
Set up your local MySQL database:

SQL
CREATE DATABASE marketing_analytics;

Update your database configuration inside config.py (or .env):

Python
DB_USER = "root"
DB_PASSWORD = "your_password"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "marketing_analytics"

STEP 05 - Run Data Cleaning & Ingestion
Run the pipeline to clean raw records, generate RFM scores, and load the cleaned dataset into MySQL:

Bash
python data_pipeline.py

STEP 06 - Run Customer Segmentation & Clustering
Execute the clustering algorithm to generate customer personas and cluster labels:

Bash
python segmentation_model.py
STEP 07 - View Dashboards
Open the provided reports/campaign_dashboard.pbix file in Power BI Desktop to view the live KPIs, conversion funnels, and customer cluster breakdowns.

📦 requirements.txt 🔧
Plaintext
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
mysql-connector-python>=8.0.0
SQLAlchemy>=2.0.0
PyMySQL>=1.0.0
python-dotenv>=1.0.0
jupyterlab>=4.0.0
📈 Key Outcomes & Insights
Identified 4 distinct customer segments (High-Value Loyalists, At-Risk Customers, Occasional Buyers, and Bargain Hunters).

Highlighted underperforming campaign channels, enabling an estimated 15–20% reallocation of marketing budget to high-conversion channels.

Built automated SQL pipelines to reduce data preparation and transformation time.

### How to Add This to Your Repo

1. Open your project folder in **VS Code**.
2. Right-click in the file explorer on the left, click **New File**, and name it **`README.md`**.
3. Paste the markdown text above into the file and save (`Ctrl + S`).
4. In your terminal, run:
   ```powershell
   git add README.md
   git commit -m "docs: add comprehensive project README"
   git push origin main

