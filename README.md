# Hotel Booking Cancellation Prediction

Big Data Analytics project predicting hotel booking cancellations using machine learning and distributed processing.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name=big-data --display-name="Python (big-data)"
```

### 2. Download Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) and place in `data/hotel_bookings.csv`

### 3. Run the Project

```bash
# Activate environment
source venv/bin/activate

# Start Jupyter
jupyter notebook
```

**In Jupyter:**

1. Select kernel: **Python (big-data)** (top-right dropdown)
2. Run notebooks in order:
   - `01_data_ingestion_mongodb.ipynb` (optional - skip if not using MongoDB)
   - `02_eda_analysis.ipynb`
   - `03_spark_preprocessing.ipynb`
   - `04_ml_models.ipynb`
   - `05_evaluation_visualization.ipynb`

## MongoDB Setup (Optional)

If you want to use MongoDB (notebook 01):

1. Create account at https://www.mongodb.com/cloud/atlas
2. Create cluster (free tier M0)
3. Create database user
4. Whitelist IP (or use `0.0.0.0/0` for all IPs)
5. Get connection string: Cluster → Connect → Connect your application → Python
6. Create `.env` file in project root:
   ```bash
   MONGODB_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   MONGODB_DB=hotel_bookings
   MONGODB_COLLECTION=bookings
   ```

**Note**: You can skip notebook 01 entirely - notebooks 02-05 work with CSV files directly.

## Project Structure

```
big-data/
├── notebooks/          # Jupyter notebooks (run in order 01-05)
├── src/               # Python utility modules
├── app/               # Streamlit web application
│   └── streamlit_app.py
├── data/              # Dataset and processed data
├── models/            # Saved ML models (created after running 04_ml_models.ipynb)
├── reports/           # Generated visualizations and reports
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Notebooks

1. **01_data_ingestion_mongodb.ipynb** - Load data into MongoDB (optional)
2. **02_eda_analysis.ipynb** - Exploratory data analysis and visualizations
3. **03_spark_preprocessing.ipynb** - Feature engineering with PySpark
4. **04_ml_models.ipynb** - Train Naive Bayes and Decision Tree models
5. **05_evaluation_visualization.ipynb** - Model evaluation and comparison

## Models

- **Naive Bayes** - Probabilistic classifier
- **Decision Tree** - Rule-based classifier

## Streamlit Web Interface

An interactive web interface is available to explore model performance and make predictions.

### Running the Streamlit App

```bash
# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run Streamlit app
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Features

- **📊 Model Performance**: View detailed metrics and comparisons
- **🔮 Make Prediction**: Test the models with custom booking inputs
- **📈 Data Exploration**: Explore the dataset and statistics
- **ℹ️ About**: Project information and documentation

### Note

For full prediction functionality:

1. Run `04_ml_models.ipynb` to train and save the models
2. Models will be saved to `models/` directory
3. The Streamlit app will load these models for predictions

## Technologies

- Python 3.x, PySpark, MongoDB Atlas, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, Streamlit

## Troubleshooting

**"ModuleNotFoundError"**: Activate venv and install requirements  
**"FileNotFoundError: hotel_bookings.csv"**: Download dataset to `data/` folder  
**"Spark session failed"**: Install Java if needed: `brew install openjdk` (Mac)  
**"Kernel not found"**: Run `python -m ipykernel install --user --name=big-data`  
**"MongoDB DNS error"**: Check connection string in `.env` file - must use actual cluster name, not placeholder

## License

Educational project for Big Data Analytics course.
