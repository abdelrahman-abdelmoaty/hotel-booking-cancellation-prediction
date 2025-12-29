# Hotel Booking Cancellation Prediction - Final Report

## Title
Hotel Booking Cancellation Prediction Using Big Data Analytics

## Team Members
- [Team Member 1 Name] - [Student ID]
- [Team Member 2 Name] - [Student ID]
- [Team Member 3 Name] - [Student ID]
- [Team Member 4 Name] - [Student ID] (if applicable)

## Description of Approach

### 1. Dataset Overview
- **Dataset**: Hotel Booking Demand Dataset from Kaggle
- **Size**: 119,390 hotel bookings
- **Features**: 32 columns
- **Target Variable**: `is_canceled` (binary classification)
- **Task**: Predict if a hotel booking will be cancelled

### 2. Data Preprocessing
- **MongoDB Integration**: 
  - Loaded dataset into MongoDB Atlas (cloud NoSQL database)
  - Created indexes for efficient querying
  - Demonstrated aggregation pipelines for data analysis
  
- **Exploratory Data Analysis**:
  - Analyzed data distribution and quality
  - Identified missing values and handled them appropriately
  - Performed univariate and bivariate analysis
  - Discovered key factors influencing cancellations

### 3. Feature Engineering
- Created new features:
  - `total_nights`: Sum of weekend and weekday nights
  - `total_guests`: Sum of adults, children, and babies
  - `total_booking_value`: ADR × total nights
  - `lead_time_category`: Categorized lead time into bins
  - `previous_cancellation_ratio`: Ratio of previous cancellations
- Handled missing values using appropriate strategies
- Encoded categorical variables using StringIndexer

### 4. Big Data Processing with Spark
- Used Apache Spark (PySpark) for distributed data processing
- Implemented feature engineering pipeline using Spark transformations
- Assembled features into vectors for ML models
- Split data into training (80%) and test (20%) sets

### 5. Machine Learning Models
Trained and compared four different models:

1. **Logistic Regression**
   - Baseline model with interpretable coefficients
   - Fast training and prediction
   - Good for linear relationships

2. **Naive Bayes**
   - Probabilistic classifier
   - Fast and efficient
   - Works well with categorical features

3. **Decision Tree**
   - Interpretable rule-based model
   - Handles non-linear relationships
   - Provides feature importance

4. **Random Forest** (Bonus)
   - Ensemble method combining multiple trees
   - Improved accuracy through voting
   - Robust to overfitting

### 6. Evaluation Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Output and Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | [Value] | [Value] | [Value] | [Value] | [Value] |
| Naive Bayes | [Value] | [Value] | [Value] | [Value] | [Value] |
| Decision Tree | [Value] | [Value] | [Value] | [Value] | [Value] |
| Random Forest | [Value] | [Value] | [Value] | [Value] | [Value] |

### Key Findings

1. **Best Performing Model**: [Model Name]
   - Achieved [metric] of [value]
   - Reasons for superior performance: [explanation]

2. **Key Factors Influencing Cancellation**:
   - [Factor 1]: [Explanation]
   - [Factor 2]: [Explanation]
   - [Factor 3]: [Explanation]

3. **Insights from EDA**:
   - [Insight 1]
   - [Insight 2]
   - [Insight 3]

### Screenshots

#### 1. MongoDB Data Ingestion
[Insert screenshot of MongoDB collection with data and indexes]

#### 2. Exploratory Data Analysis
[Insert screenshots of:
- Target variable distribution
- Cancellation rate by hotel type
- Correlation heatmap
- Other key visualizations]

#### 3. Spark Processing
[Insert screenshots of:
- Spark session initialization
- Feature engineering results
- Data preprocessing pipeline]

#### 4. Model Training
[Insert screenshots of:
- Model training progress
- Sample predictions
- Model comparison table]

#### 5. Evaluation Visualizations
[Insert screenshots of:
- Confusion matrices for all models
- ROC curves comparison
- Feature importance plots
- Performance comparison charts]

### MongoDB Queries Demonstrated
[Include screenshots/examples of MongoDB aggregation queries:
- Cancellation rate by hotel type
- Cancellation rate by deposit type
- Top countries by booking count
- Average lead time by market segment]

### Spark Scalability Discussion
[Discuss how Spark enables:
- Distributed processing of large datasets
- Scalability to handle millions of records
- Efficient feature engineering at scale
- Parallel model training]

## Bonus Components Implemented

1. **Multiple Big Data Components**: Both MongoDB and Spark integration
2. **Additional Models**: 4 models with comprehensive comparison
3. **Advanced Visualizations**: 
   - Interactive Plotly charts
   - ROC curves comparison
   - Feature importance analysis
4. **Performance Optimization**:
   - Spark caching for faster access
   - MongoDB indexing for efficient queries
5. **Data Pipeline**: End-to-end pipeline from MongoDB → Spark → ML → Evaluation
6. **Feature Engineering**: 10+ engineered features

## Challenges and Solutions

### Challenge 1: [Description]
**Solution**: [How you solved it]

### Challenge 2: [Description]
**Solution**: [How you solved it]

## Conclusion

[Summary of project outcomes, key learnings, and potential future improvements]

## References

- Hotel Booking Demand Dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- MongoDB Atlas: https://www.mongodb.com/cloud/atlas
- Apache Spark Documentation: https://spark.apache.org/docs/latest/
- PySpark MLlib: https://spark.apache.org/docs/latest/ml-guide.html

## GitHub Repository
[Link to your GitHub repository]

## Team Contributions

- **[Member 1]**: [Specific contributions, e.g., "MongoDB setup, EDA notebook"]
- **[Member 2]**: [Specific contributions, e.g., "Spark preprocessing, ML models"]
- **[Member 3]**: [Specific contributions, e.g., "Evaluation, visualizations, report"]
- **[Member 4]**: [If applicable]

---

**Note**: Replace all [placeholders] with actual values and add screenshots as indicated.

