# 🚗 Car Price Prediction using Linear Regression

This project predicts the selling price of a used car based on its brand, fuel type, transmission, kilometers driven, and other features. It's built using a Linear Regression model and trained on a dataset downloaded from Google Drive.

---

## 🔍 What the project does

- Downloads a real car dataset using `gdown`
- Preprocesses the data:
  - Extracts brand from car name
  - Calculates age of the car
  - One-hot encodes categorical features
- Trains a Linear Regression model to predict selling price
- Lets the user enter their own car details to get a price estimate 💰

---

## 🧪 Technologies used

- Python
- pandas, numpy
- scikit-learn
- gdown (to download dataset)
- Linear Regression

---

## 🛠️ How to Run

1. Make sure Python is installed
2. Install required packages:
   ```bash
   pip install -r requirements.txt
python src/price_predictor.py
