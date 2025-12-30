# Hotel Booking Cancellation Prediction - Final Report

## Title

Hotel Booking Cancellation Prediction Using Big Data Analytics

## Team Members

- [Team Member 1 Name] - [Student ID]
- [Team Member 2 Name] - [Student ID]
- [Team Member 3 Name] - [Student ID]
- [Team Member 4 Name] - [Student ID] (if applicable)

## Project Structure

```
Hotel Booking Cancellation Prediction/
├── data/
│   ├── hotel_bookings.csv                    # Original dataset
│   ├── hotel_bookings_from_mongodb.csv       # Exported from MongoDB
│   ├── model_metrics.csv                     # Model performance metrics
│   └── processed_data/                       # Spark processed data (Parquet)
│       ├── train_data.parquet/
│       └── test_data.parquet/
├── notebooks/
│   ├── 01_data_ingestion_mongodb.ipynb      # MongoDB integration
│   ├── 02_eda_analysis.ipynb                 # Exploratory data analysis
│   ├── 03_spark_preprocessing.ipynb         # Spark feature engineering
│   ├── 04_ml_models.ipynb                   # Model training
│   └── 05_evaluation_visualization.ipynb    # Evaluation and visualization
├── reports/
│   ├── figures/                              # Generated visualizations
│   │   ├── model_comparison_1.png
│   │   ├── model_comparison_2.png
│   │   ├── interactive_metrics_comparison.html
│   │   └── [other figures when available]
│   └── model_metrics_final.csv               # Final metrics for report
├── models/                                   # Saved ML models (created after training)
│   ├── naive_bayes_model/
│   ├── decision_tree_model/
│   └── naive_bayes_scaler/
├── app/                                      # Streamlit web application
│   ├── streamlit_app.py                      # Interactive web interface
│   └── README.md
├── src/                                      # Source code modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── models.py
└── requirements.txt                          # Python dependencies
```

## Description of Approach

### 1. Dataset Overview

- **Dataset**: Hotel Booking Demand Dataset from Kaggle
- **Size**: 119,390 hotel bookings
- **Features**: 32 columns
- **Target Variable**: `is_canceled` (binary classification)
- **Task**: Predict if a hotel booking will be cancelled

### 2. Data Preprocessing

- **MongoDB Integration** (Notebook: `01_data_ingestion_mongodb.ipynb`):
  - Loaded dataset into MongoDB Atlas (cloud NoSQL database)
  - Created indexes for efficient querying
  - Demonstrated aggregation pipelines for data analysis
  - Exported data to CSV for Spark processing
- **Exploratory Data Analysis** (Notebook: `02_eda_analysis.ipynb`):
  - Analyzed data distribution and quality
  - Identified missing values (only 4 missing values in 'children' column)
  - Performed univariate and bivariate analysis
  - Discovered key factors influencing cancellations
  - Created comprehensive visualizations

### 3. Feature Engineering (Notebook: `03_spark_preprocessing.ipynb`)

- **Spark Data Processing**:
  - Loaded data from CSV into Spark DataFrame
  - Used `nullValue="NA"` parameter to handle 'NA' strings as null values
  - Created new features:
    - `total_nights`: Sum of weekend and weekday nights
    - `total_guests`: Sum of adults, children, and babies
    - `total_booking_value`: ADR × total nights
    - `lead_time_category`: Categorized lead time into bins (Very Short, Short, Medium, Long, Very Long)
    - `previous_cancellation_ratio`: Ratio of previous cancellations
  - Handled missing values:
    - Filled missing values in numerical columns with column means
    - Replaced missing 'country' values with "Unknown"
    - Cleaned 'NA' strings in children, agent, and company columns
  - Encoded categorical variables using StringIndexer (10 categorical features)
  - Assembled 28 total features (18 numerical + 10 categorical) into feature vectors
  - Split data into training (80%) and test (20%) sets
  - Saved processed data as Parquet files for efficient loading

### 4. Machine Learning Model Training (Notebook: `04_ml_models.ipynb`)

- **Model Training Pipeline**:
  - Loaded preprocessed data from Parquet files
  - Trained models using PySpark MLlib
  - Implemented proper data scaling for Naive Bayes (MinMaxScaler to [0, 1] range)
  - Evaluated models using BinaryClassificationEvaluator and MulticlassClassificationEvaluator
  - Saved model metrics to CSV for reporting
  - Saved trained models to `models/` directory for later use (Naive Bayes, Decision Tree, and scaler)

### 5. Machine Learning Models

Trained and compared two different models using PySpark MLlib:

1. **Naive Bayes**

   - Probabilistic classifier
   - Fast and efficient
   - Works well with categorical features
   - Features scaled to [0, 1] range (required for Naive Bayes)

2. **Decision Tree**
   - Interpretable rule-based model
   - Handles non-linear relationships
   - Provides feature importance
   - Configured with maxDepth=10 and maxBins=256

### 6. Evaluation and Visualization (Notebook: `05_evaluation_visualization.ipynb`)

- **Evaluation Metrics**:
  - **Accuracy**: Overall correctness of predictions
  - **Weighted Precision**: Proportion of positive predictions that are correct (weighted by class support)
  - **Weighted Recall**: Proportion of actual positives correctly identified (weighted by class support)
  - **F1-Score**: Harmonic mean of precision and recall
  - **ROC-AUC**: Area under the ROC curve
- **Visualizations Generated**:
  - Model performance comparison bar charts
  - Confusion matrices (when predictions available)
  - ROC curves comparison (when predictions available)
  - Feature importance analysis
  - Interactive Plotly visualizations
  - All figures saved to `reports/figures/` directory

### 7. Interactive Web Interface (Streamlit App: `app/streamlit_app.py`)

- **Web Application Features**:
  - **Model Performance Dashboard**: View detailed metrics, comparisons, and visualizations
  - **Prediction Interface**: Test models with custom booking inputs or select from example bookings
  - **Example Bookings**: Pre-configured examples including:
    - High Cancellation Risk (long lead time, previous cancellations)
    - Low Cancellation Risk (short lead time, returning guest)
    - Corporate Booking
    - Family Vacation
  - **Real-time Predictions**: Heuristic-based cancellation risk assessment with factor analysis
  - User-friendly interface with intuitive navigation

## Output and Results

### Model Performance Comparison

| Model         | Accuracy | Weighted Precision | Weighted Recall | F1-Score | AUC    |
| ------------- | -------- | ------------------ | --------------- | -------- | ------ |
| Naive Bayes   | 0.7709   | 0.8257             | 0.7709          | 0.7382   | 0.4942 |
| Decision Tree | 0.8390   | 0.8390             | 0.8390          | 0.8390   | 0.8585 |

**Note**: All metrics are calculated using PySpark MLlib evaluators on the test set.

### Key Findings

1. **Best Performing Model**: Decision Tree

   - Achieved accuracy of 0.8390 (83.90%)
   - Achieved AUC-ROC of 0.8585
   - Superior performance across all metrics compared to Naive Bayes
   - Reasons for superior performance:
     - Better handling of non-linear relationships in the data
     - Ability to capture complex feature interactions
     - No assumptions about feature distributions (unlike Naive Bayes)

2. **Key Factors Influencing Cancellation** (from Decision Tree feature importance):

   - Feature 25: Highest importance (0.4414)
   - Feature 0: Second highest (0.1045)
   - Feature 20: Third highest (0.0859)
   - _Note: Feature names should be mapped from indices to actual feature names_

3. **Insights from Model Comparison**:
   - Decision Tree significantly outperforms Naive Bayes in AUC-ROC (0.8585 vs 0.4942)
   - Naive Bayes shows poor discrimination ability (AUC < 0.5 suggests worse than random)
   - Decision Tree provides balanced precision and recall
   - Both models show similar accuracy, but Decision Tree is more reliable overall

### Screenshots

#### 1. MongoDB Data Ingestion

[Insert screenshot of MongoDB collection with data and indexes]

#### 2. Exploratory Data Analysis

[Insert screenshots of:

- Target variable distribution
- Cancellation rate by hotel type
- Correlation heatmap
- Other key visualizations]

#### 3. Spark Processing (`03_spark_preprocessing.ipynb`)

[Insert screenshots of:

- Spark session initialization with version info
- Data loading confirmation (119,390 records)
- Feature engineering results (new features created)
- Missing values analysis
- Data preprocessing pipeline completion
- Train/test split results (80/20 split)]

#### 4. Model Training (`04_ml_models.ipynb`)

[Insert screenshots of:

- Model training progress for Naive Bayes and Decision Tree
- Sample predictions from both models
- Model comparison table with all metrics
- Feature importance from Decision Tree (top 10 features)]

#### 5. Evaluation Visualizations

[Insert screenshots from `reports/figures/`:

- `model_comparison_1.png`: Accuracy and AUC comparison charts
- `model_comparison_2.png`: Precision, Recall, and F1-Score comparison charts
- `confusion_matrices.png`: Confusion matrices for both models (if available)
- `roc_curves.png`: ROC curves comparison (if available)
- `feature_importance.png`: Decision Tree feature importance plot (if available)
- `interactive_metrics_comparison.html`: Interactive Plotly visualization (open in browser)]

#### 6. Streamlit Web Interface

[Insert screenshots of:

- Streamlit app home page
- Model Performance page showing metrics and visualizations
- Make Prediction page with example bookings dropdown
- Prediction results showing cancellation risk and contributing factors]

### MongoDB Queries Demonstrated

[Include screenshots/examples of MongoDB aggregation queries:

- Cancellation rate by hotel type
- Cancellation rate by deposit type
- Top countries by booking count
- Average lead time by market segment]

### Spark Scalability Discussion

[Discuss how Spark enables:

- Distributed processing of large datasets (119,390 records processed)
- Scalability to handle millions of records (data partitioned across 5 partitions)
- Efficient feature engineering at scale (28 features assembled into vectors)
- Parallel model training (MLlib's distributed algorithms)
- Memory-efficient processing (caching for repeated operations)
- Parquet format for fast data loading and storage]

## Bonus Components Implemented

1. **Multiple Big Data Components**: Both MongoDB and Spark integration
2. **Comprehensive Evaluation**: Detailed model comparison with multiple metrics
3. **Advanced Visualizations**:
   - Interactive Plotly charts (saved as HTML)
   - Model comparison bar charts (saved as PNG)
   - ROC curves comparison (when available)
   - Feature importance analysis (when available)
   - All visualizations automatically saved to `reports/figures/`
4. **Interactive Web Interface (Streamlit)**:
   - User-friendly web application for exploring model performance
   - Real-time prediction interface with example bookings
   - Factor-based cancellation risk assessment
   - Responsive design with multiple pages
5. **Performance Optimization**:
   - Spark caching for faster access
   - MongoDB indexing for efficient queries
   - Parquet format for efficient data storage and loading
   - Model persistence for reuse
6. **Data Pipeline**: End-to-end pipeline from MongoDB → CSV → Spark → ML → Evaluation → Web Interface
7. **Feature Engineering**: 5 engineered features + 28 total features (18 numerical + 10 categorical)
8. **Robust Data Handling**:
   - Proper handling of 'NA' strings in CSV
   - Missing value imputation with column means
   - Safe type casting using try_cast
9. **Model Persistence**: Trained models saved for deployment and reuse

## Challenges and Solutions

### Challenge 1: [Description]

**Solution**: [How you solved it]

### Challenge 2: [Description]

**Solution**: [How you solved it]

## Conclusion

[Summary of project outcomes, key learnings, and potential future improvements]

## Technical Stack

- **Python 3.13**: Programming language
- **MongoDB Atlas**: Cloud NoSQL database for data storage
- **Apache Spark 4.1.0**: Distributed data processing framework
- **PySpark MLlib**: Machine learning library for Spark
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework for interactive interfaces
- **Jupyter Notebooks**: Interactive development environment

## References

- Hotel Booking Demand Dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- MongoDB Atlas: https://www.mongodb.com/cloud/atlas
- Apache Spark Documentation: https://spark.apache.org/docs/latest/
- PySpark MLlib: https://spark.apache.org/docs/latest/ml-guide.html
- PySpark API Reference: https://spark.apache.org/docs/latest/api/python/

## GitHub Repository

[Link to your GitHub repository]

## Team Contributions

- **[Member 1]**: [Specific contributions, e.g., "MongoDB setup, EDA notebook"]
- **[Member 2]**: [Specific contributions, e.g., "Spark preprocessing, ML models"]
- **[Member 3]**: [Specific contributions, e.g., "Evaluation, visualizations, report"]
- **[Member 4]**: [If applicable]

---

## Important Notes

1. **Notebook Execution Order**: Run notebooks in sequence (01 → 02 → 03 → 04 → 05) to ensure all dependencies are available. For best results, run notebooks 04 and 05 in the same session so that predictions are available for visualization.

2. **Figure Generation**: Some figures (confusion matrices, ROC curves, feature importance) require predictions from the ML models notebook. Make sure to run `04_ml_models.ipynb` before `05_evaluation_visualization.ipynb` in the same session, or the predictions won't be available.

3. **File Paths**: All paths in the notebooks use relative paths from the `notebooks/` directory. Make sure to run notebooks from the correct directory.

4. **Colab References Removed**: All Colab-specific paths and references have been removed. The project now uses local paths.

5. **Streamlit App**: To run the interactive web interface:
   ```bash
   source venv/bin/activate
   streamlit run app/streamlit_app.py
   ```
   The app will open at `http://localhost:8501`

---

**Note**: Replace all [placeholders] with actual values and add screenshots as indicated.
