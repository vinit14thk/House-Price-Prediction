House Price Prediction Using Machine Learning

About the Project

This project predicts house prices using machine learning techniques in Python. It explores housing data, preprocesses categorical features, trains regression models, and evaluates their performance.

🚀 Live Demo

Try the deployed application here: "House Price Prediction App" (https://house-price-prediction-v.streamlit.app/)

Enter property details to get an estimated house price using the trained Linear Regression model.

Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook

Dataset

The dataset contains 545 housing records with information about price, area, bedrooms, bathrooms, stories, parking, and other property features.

Machine Learning Models

1. Linear Regression
2. Random Forest Regressor

Model Evaluation

Metric| Linear Regression| Random Forest
MAE| 970,043.40| 1,021,546.04
RMSE| 1,324,506.96| 1,400,565.97
R² Score| 0.6529| 0.6119

Linear Regression performed better than Random Forest on the test split used in this project.

Visualizations

- Actual vs. predicted house prices
- Distribution of prediction errors
- Model comparison chart
- Regression coefficient analysis

How to Run

1. Download or clone this repository.
2. Install the required libraries:
   "pip install pandas numpy matplotlib seaborn scikit-learn jupyter"
3. Open "house_price_prediction.ipynb" in Jupyter Notebook.
4. Ensure "housing.csv" is in the same folder.
5. Run the notebook cells in order.

Limitations

The model's predictions are estimates based on the available dataset. Its performance may vary for properties outside the dataset's range.

Author

Created as a machine learning portfolio project.
