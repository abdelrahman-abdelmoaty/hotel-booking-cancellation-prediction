# hotel-booking-cancellation-prediction

# Hotel Booking Cancellation Prediction - Big Data Analytics Project

## Project Overview

This project predicts hotel booking cancellations using machine learning algorithms and big data technologies. The project implements a complete data pipeline from data ingestion to model evaluation, utilizing MongoDB for NoSQL storage and Apache Spark for distributed processing.

## Dataset

**Source**: [Hotel Booking Demand Dataset - Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

- **Records**: 119,390 hotel bookings
- **Features**: 32 columns
- **Target**: `is_canceled` (binary classification)
- **Task**: Predict if a booking will be cancelled

## Project Structure

```
hotel-booking-prediction/
├── notebooks/
│   ├── 01_data_ingestion_mongodb.ipynb    # Load data into MongoDB
│   ├── 02_eda_analysis.ipynb              # Exploratory Data Analysis
│   ├── 03_spark_preprocessing.ipynb       # PySpark data processing
│   ├── 04_ml_models.ipynb                 # Train multiple ML models
│   └── 05_evaluation_visualization.ipynb  # Results and visualizations
├── src/
│   ├── data_loader.py                     # MongoDB connection utilities
│   ├── preprocessing.py                   # Feature engineering functions
│   └── models.py                          # ML model implementations
├── data/
│   └── hotel_bookings.csv                 # Raw dataset (download from Kaggle)
├── reports/
│   └── figures/                           # Saved visualizations
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd hotel-booking-prediction
```

### 2. Install Dependencies

For local development:

```bash
pip install -r requirements.txt
```

For Google Colab, dependencies are installed in each notebook.

### 3. MongoDB Atlas Setup

1. Create a free MongoDB Atlas account at https://www.mongodb.com/cloud/atlas
2. Create a new cluster (free tier M0)
3. Create a database user with read/write permissions
4. Whitelist your IP address (or use 0.0.0.0/0 for Colab)
5. Get your connection string
6. Create a `.env` file in the project root:
   ```
   MONGODB_URI=your_connection_string_here
   MONGODB_DB=hotel_bookings
   MONGODB_COLLECTION=bookings
   ```

### 4. Download Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) and place it in the `data/` directory as `hotel_bookings.csv`.

## Usage

### Running in Google Colab

1. Upload all notebooks to Google Colab
2. Upload the dataset to Colab or mount Google Drive
3. Set up MongoDB Atlas connection string in the first notebook
4. Run notebooks sequentially (01 → 05)

### Running Locally

1. Ensure all dependencies are installed
2. Set up MongoDB connection in `.env` file
3. Run notebooks using Jupyter:
   ```bash
   jupyter notebook
   ```

## Notebooks Overview

1. **01_data_ingestion_mongodb.ipynb**:

   - Loads CSV data into MongoDB Atlas
   - Creates indexes for efficient querying
   - Demonstrates MongoDB aggregation queries

2. **02_eda_analysis.ipynb**:

   - Comprehensive exploratory data analysis
   - Univariate and bivariate analysis
   - Data quality assessment
   - Key insights extraction

3. **03_spark_preprocessing.ipynb**:

   - PySpark setup and data loading
   - Feature engineering pipeline
   - Data preprocessing and transformation
   - Train/test split

4. **04_ml_models.ipynb**:

   - Trains multiple ML models:
     - Logistic Regression
     - Naive Bayes
     - Decision Tree
     - Random Forest
   - Model training and prediction

5. **05_evaluation_visualization.ipynb**:
   - Model evaluation metrics
   - Model comparison
   - Advanced visualizations
   - Results summary

## Machine Learning Models

- **Logistic Regression**: Baseline model with interpretable coefficients
- **Naive Bayes**: Fast probabilistic classifier
- **Decision Tree**: Interpretable rule-based model
- **Random Forest**: Ensemble method for improved accuracy

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

## Technologies Used

- **Python 3.x**: Primary programming language
- **PySpark**: Distributed data processing and ML
- **MongoDB Atlas**: Cloud NoSQL database
- **PyMongo**: MongoDB Python driver
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn/Plotly**: Data visualization
- **Google Colab**: Development environment

## Team Contributions

Team members should commit code with clear commit messages indicating their contributions. Use GitHub's commit history to track individual work.

## Report Structure

The final report should include:

- **Title**: Hotel Booking Cancellation Prediction
- **Description of Approach**: Methodology, algorithms, and pipeline
- **Output and Results**: Model performance, visualizations, and insights
- **Screenshots**: Key outputs from notebooks and MongoDB queries

## License

This project is for educational purposes as part of Big Data Analytics course.
