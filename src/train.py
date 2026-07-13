"""
====================================================
Module : train.py
Project: Airline Passenger Forecasting
Purpose: Train the Deep Learning Model
====================================================
"""

import os
import joblib

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.sequence_generator import SequenceGenerator
from src.train_test_split import TimeSeriesSplit
from src.model import ModelBuilder


class Trainer:
    """
    Train the RNN/LSTM/GRU model.
    """

    def __init__(
        self,
        data_path="data/airline-passengers.csv",
        sequence_length=12,
        model_type="lstm",
        train_size=0.8,
        epochs=50,
        batch_size=16,
    ):

        self.data_path = data_path
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.train_size = train_size
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):

        print("=" * 60)
        print("AIRLINE PASSENGER FORECASTING")
        print("=" * 60)

        # -----------------------------
        # Load Dataset
        # -----------------------------
        loader = DataLoader(self.data_path)
        df = loader.load_data()

        # -----------------------------
        # Preprocess Dataset
        # -----------------------------
        preprocessor = Preprocessor()
        scaled_df = preprocessor.scale_data(df)

        # -----------------------------
        # Generate Sequences
        # -----------------------------
        generator = SequenceGenerator(
            sequence_length=self.sequence_length
        )

        X, y = generator.create_sequences(scaled_df)

        # -----------------------------
        # Train-Test Split
        # -----------------------------
        splitter = TimeSeriesSplit(train_size=self.train_size)

        X_train, X_test, y_train, y_test = splitter.split(X, y)

        # -----------------------------
        # Build Model
        # -----------------------------
        builder = ModelBuilder(
            model_type=self.model_type,
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )

        model = builder.build_model()

        # -----------------------------
        # Train Model
        # -----------------------------
        print("\nTraining Started...\n")

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
        )

        print("\nTraining Completed Successfully.")

        # -----------------------------
        # Evaluate Model
        # -----------------------------
        loss, mae = model.evaluate(
            X_test,
            y_test,
            verbose=0,
        )

        print("\nTest Loss :", round(loss, 4))
        print("Test MAE  :", round(mae, 4))

        # -----------------------------
        # Save Model
        # -----------------------------
        os.makedirs("saved_models", exist_ok=True)

        model.save("saved_models/lstm_model.keras")

        # Save scaler
        joblib.dump(
            preprocessor.scaler,
            "saved_models/scaler.pkl",
        )

        print("\nModel Saved Successfully.")
        print("saved_models/lstm_model.keras")
        print("saved_models/scaler.pkl")

        return model, history


if __name__ == "__main__":

    trainer = Trainer(
        data_path="data/airline-passengers.csv",
        sequence_length=12,
        model_type="lstm",
        train_size=0.8,
        epochs=50,
        batch_size=16,
    )

    trainer.train()