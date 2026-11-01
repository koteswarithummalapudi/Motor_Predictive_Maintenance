
import joblib

# Load the trained machine learning model
model = joblib.load("motor_model.joblib")

print("=== MOTOR HEALTH PREDICTION SYSTEM ===")

vibration = float(input("Enter vibration value: "))
temperature = float(input("Enter temperature in Celsius: "))

# Use the same input features used during training
input_data = [[vibration, temperature]]

prediction = model.predict(input_data)

print("\nPredicted Motor Health:", prediction[0])
