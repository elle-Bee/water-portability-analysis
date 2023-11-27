import joblib
import numpy as np


def get_user_input():
    ph = float(input("Enter the value for pH (0-14): "))
    Hardness = float(input("Enter the value for hardness of water (in mg/L): "))
    Solids = float(input("Enter the total dissolved solids (in ppm): "))
    Chloramines = float(input("Enter the amount of chloramines (in ppm): "))
    Sulfate = float(input("Enter the amount of sulfates dissolved (in mg/L): "))
    Conductivity = float(input("Enter the electrical conductivity of water (in μS/cm): "))
    Organic_carbon = float(input("Enter the amount of organic carbon (in ppm): "))
    Trihalomethanes = float(input("Enter the amount of trihalomethanes in (μg/L): "))
    Turbidity = float(input("Enter the measure of light emitting property of water (in NTU): "))

    return np.array([ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes,
                     Turbidity]).reshape(1, -1)


def main():
    # Load the pre-trained model
    model_filename = "knn_model.joblib"
    model = joblib.load(model_filename)

    user_input = get_user_input()

    # Make a prediction using the loaded model
    prediction = model.predict(user_input)

    if prediction == 1:
        print("The water should be portable")
    else:
        print("The water is predicted to be non portable")

if __name__ == "__main__":
    main()