# Housing Price Prediction Project

## Overview
This project aims to predict median housing prices in California using machine learning techniques. The dataset is sourced from the GitHub repository of Aurélien Géron and contains various features such as geographical location, number of rooms, population, and median income. The model is built using a RandomForestRegressor and optimized through GridSearchCV.

## Dataset
- The dataset is automatically downloaded and extracted if not found in the local directory.
- Each row in the dataset represents a district in California.
- The dataset contains features like `longitude`, `latitude`, `median_income`, `total_rooms`, `population`, and `ocean_proximity`.

## Steps in the Project
### 1. Data Preprocessing
- Handling missing values using the median imputation strategy.
- Converting categorical features (`ocean_proximity`) using OneHotEncoding.
- Creating new feature combinations such as `rooms_per_household`, `bedrooms_ratio`, and `people_per_house`.
- Standardizing numerical features using StandardScaler.

### 2. Data Visualization & Analysis
- Histograms and scatter plots are used to explore data distributions.
- Correlation matrices and scatter matrix plots analyze feature relationships.

### 3. Splitting the Data
- Stratified sampling is used based on `median_income` to ensure representative training and test datasets.
- The training set is separated from labels (`median_house_value`).

### 4. Model Training & Evaluation
- A RandomForestRegressor is trained on the processed dataset.
- Model performance is evaluated using RMSE (Root Mean Squared Error).
- Predictions are made on the test dataset.

### 5. Hyperparameter Tuning
- GridSearchCV is used to find the best hyperparameters (`n_estimators` and `max_depth`).
- The best model is saved using joblib for future use.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Usage
1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd housing-price-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the script:
   ```sh
   python housing_price_prediction.py
   ```
4. The trained model is saved as `final_model.pkl`.

## Results
- The optimized RandomForestRegressor achieves a lower RMSE on the test dataset.
- The model can be loaded and used for further predictions.

## Future Improvements
- Experiment with other machine learning models like Gradient Boosting.
- Add feature engineering techniques for better predictions.
- Deploy the model as a web API for real-time predictions.

