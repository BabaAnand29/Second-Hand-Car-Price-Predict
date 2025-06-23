# Install required packages
# !pip install -q gdown scikit-learn pandas

# Import libraries
import gdown
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Step 1: Download dataset from Google Drive
file_url = 'https://drive.google.com/uc?id=11_r1Ozwm4F4KCTVuhlyzlBRqytA0bOz-'
output_file = 'car_data.csv'
gdown.download(file_url, output_file, quiet=False)

# Step 2: Load the dataset
df = pd.read_csv(output_file)

# Step 3: Feature engineering
df['brand'] = df['name'].apply(lambda x: x.split()[0])
df['car_age'] = datetime.datetime.now().year - df['year']
df.drop(['name', 'year'], axis=1, inplace=True)

# Step 4: Define features and target
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

# Step 5: Define preprocessing
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
numerical_cols = ['km_driven', 'car_age']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Step 6: Create and train the pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Step 7: Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("✅ Model trained successfully!")
print("📊 RMSE:", round(rmse, 2))
print("📈 R² Score:", round(r2_score(y_test, y_pred), 2))

# Step 9: Make prediction with custom input
print("\n🚗 Enter your own car details for price prediction:\n")
fuel = input("Fuel type (Petrol/Diesel/CNG/LPG/Electric): ")
seller_type = input("Seller type (Individual/Dealer/Trustmark Dealer): ")
transmission = input("Transmission (Manual/Automatic): ")
owner = input("Owner type (First Owner/Second Owner/...): ")
brand = input("Brand (e.g. Maruti, Hyundai, Toyota): ")
km_driven = int(input("Kilometers driven: "))
car_age = int(input("Age of the car (in years): "))

# Create DataFrame for prediction
input_df = pd.DataFrame([{
    'fuel': fuel,
    'seller_type': seller_type,
    'transmission': transmission,
    'owner': owner,
    'brand': brand,
    'km_driven': km_driven,
    'car_age': car_age
}])

# Predict and print result
predicted_price = model.predict(input_df)[0]
print(f"\n💰 Estimated Selling Price: ₹{round(predicted_price):,}")
