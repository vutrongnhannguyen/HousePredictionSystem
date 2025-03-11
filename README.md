# 🏡 Kaggle Housing Price Prediction - Top 2.86% Achievement

https://www.kaggle.com/competitions/home-data-for-ml-course

## 👥 Team
- 👨‍💻 Nguyen Vu Trong Nhan
- 👨‍💻 Duong Hoang Anh Khoa
- 👩‍💻 Seokyung Kim
- 👩‍💻 Shirin Shujaa
- 👨‍💻 Ngo Ngoc Thinh

## 📌 Introduction

This project is a submission for the Kaggle Housing Prices competition, where our team secured a top **2.86% ranking globally**. The goal of the competition is to predict house sale prices based on **81** explanatory variables describing various aspects of residential homes in Ames, Iowa. 🏠📊

## 🚀 Getting Started

## 📖 Competition Description

Real estate pricing is influenced by a wide range of factors beyond simple metrics like the number of bedrooms or lot size. This competition provides a dataset containing **81** **features**, capturing nearly every aspect of a home's characteristics. Participants are required to build predictive models to estimate home prices. 💰📉

### 🛠 Skills Practiced

- 🔍 **Feature Engineering**: Identifying and constructing new variables to improve model performance.
- 📈 **Advanced Regression Techniques**: Implementing methods such as **random forest**, **gradient boosting**, and **stacking models** to achieve high accuracy.

## 📂 Dataset and File Descriptions

The dataset consists of the following files:

- 📄 `train.csv`: The training dataset containing house features and sale prices.
- 📄 `test.csv`: The test dataset containing house features (without sale prices).
- 📄 `sample_submission.csv`: A sample submission file demonstrating the required format.
- 📄 `data_description.txt`: A file explaining the meaning of each column in the dataset.

### 🎯 Target Variable

- **SalePrice**: The final sale price of a home (measured in USD). 💵

### 🔑 Key Features

- 📏 **LotFrontage**: Linear feet of street connected to property
- 🏆 **OverallQual**: Overall material and finish quality
- 🏗 **YearBuilt**: Year the house was built
- 🏠 **TotalBsmtSF**: Total square feet of basement area
- 📐 **GrLivArea**: Above-grade (ground) living area in square feet
- 🚗 **GarageCars**: Number of cars that fit in the garage

## 🏋️‍♂️ Model Training Approach

### 🧹 Data Preprocessing

1. 🕵️‍♂️ **Handling Missing Values**: Filling missing values using mean, median, mode, and custom logic (e.g., grouping by neighborhood).
2. ✂ **Outlier Removal**: Using IQR (Interquartile Range) to filter extreme values.
3. 🛠 **Feature Engineering**:
   - Created new features such as `HouseAge`, `TotalBathrooms`, and `TotalSF`.
   - Applied **K-Means Clustering** on the `Neighborhood` feature.
4. 🎭 **Encoding Categorical Variables**:
   - **Label Encoding** for ordinal categorical features.
   - **One-Hot Encoding** for nominal categorical features.
5. 📏 **Feature Scaling**:
   - Used **StandardScaler** for numerical features.
   - Applied **log transformation** to the `SalePrice` variable to reduce skewness.

### 🏆 Model Selection and Hyperparameter Tuning

We experimented with multiple models and fine-tuned their hyperparameters using **GridSearchCV** and **K-Fold Cross-Validation (k=5)**. The following models were used:

#### 🤖 **Trained Models**

1. 📊 **Linear Regression**
2. 🌲 **Random Forest Regressor**
3. 🚀 **XGBoost Regressor**
4. 🐱 **CatBoost Regressor** *(Best Performing Model)*
5. 🌿 **LightGBM Regressor**
6. 🔗 **Stacked Model (Ensemble of Best Models)**

#### 🔗 **Stacking Model Implementation**

To further improve performance, we used a **stacking regressor**, which combines predictions from multiple models to create a meta-model:

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

stacked_model = StackingRegressor(
    estimators=[
        ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.05)),
        ('catboost', CatBoostRegressor(iterations=500, learning_rate=0.05, verbose=0)),
        ('lgbm', LGBMRegressor(n_estimators=500, learning_rate=0.05))
    ],
    final_estimator=Ridge(alpha=1.0)
)
```

### 📊 Performance Evaluation

Models were evaluated using **Root Mean Squared Error (RMSE)**:

- ✅ RMSE measures the error between the predicted and actual sale prices, with lower values indicating better performance.
- ✅ The metric is applied to the **log-transformed SalePrice** to ensure errors in high-value and low-value houses are treated equally.

### 🏆 Best Model Results

- 🥇 **Stacking Regressor achieved the best RMSE on the test set**, leading to our final submission.
- 🥈 The **CatBoost model** also provided competitive results and was used as a backup submission.

## 📑 Submission Format

The submission file (`submission.csv`) must follow the format below:

```
Id,SalePrice
1461,169000.1
1462,187724.1233
1463,175221.0
```

## 🔑 Key Takeaways

- 🎯 **Feature Engineering** is crucial for achieving high performance.
- 🤖 **Ensemble Models** (Stacking, CatBoost, LightGBM) significantly improve accuracy.
- ⚙ **Hyperparameter Tuning** plays a major role in model optimization.
- 🏠 **Handling Missing Data Correctly** prevents biases in the model.

## 🙌 Acknowledgments

This project is based on the **Ames Housing Dataset**, compiled by **Dean De Cock** for data science education. Special thanks to the Kaggle community for sharing valuable insights and methodologies. 💡

## 🔮 Future Improvements

- 🤖 Experimenting with **Neural Networks (MLPs)** for regression.
- 🏆 **Automated Feature Selection** using Lasso regression.
- 🎯 Implementing a **Bayesian Optimization-based Hyperparameter Tuning** instead of GridSearchCV.

🚀 *Happy Coding!*
