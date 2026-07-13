# Loads the trained model and computes evaluation metrics such as Accuracy, Precision, Recall, F1-score (classification) or RMSE/MAE/MSE (forecasting).
# RMSE, MAE, MSE

"""
====================================================
Module : evaluate.py
Project: Airline Passenger Forecasting
Purpose: Evaluate Model Performance
====================================================
"""

import os
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.predict import Predictor


class Evaluator:
    """
    Evaluate forecasting model.
    """

    def __init__(self):

        self.predictor = Predictor()


    def evaluate(self):

        # --------------------------------
        # Get Actual and Predicted Values
        # --------------------------------

        actual, predicted = self.predictor.predict()


        # --------------------------------
        # Calculate Metrics
        # --------------------------------

        mae = mean_absolute_error(
            actual,
            predicted
        )

        mse = mean_squared_error(
            actual,
            predicted
        )

        rmse = np.sqrt(mse)


        print("\nModel Evaluation\n")

        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")


        # --------------------------------
        # Save Metrics
        # --------------------------------

        os.makedirs(
            "outputs",
            exist_ok=True
        )


        with open(
            "outputs/metrics.txt",
            "w"
        ) as file:

            file.write("Airline Passenger Forecasting\n")
            file.write("============================\n\n")

            file.write(
                f"MAE  : {mae:.4f}\n"
            )

            file.write(
                f"MSE  : {mse:.4f}\n"
            )

            file.write(
                f"RMSE : {rmse:.4f}\n"
            )


        print("\nMetrics saved successfully.")
        print("Location : outputs/metrics.txt")


        return mae, mse, rmse



if __name__ == "__main__":


    evaluator = Evaluator()


    mae, mse, rmse = evaluator.evaluate()