# Placement Predictor

A simple Streamlit app that predicts whether a student is likely to be placed based on their CGPA and IQ score.

## What this project does

This project uses a trained machine learning model to make a basic placement prediction from two input features:

- CGPA
- IQ score

## Features

- Clean web interface built with Streamlit
- Quick prediction based on user input
- Pretrained model and scaler included

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Experimental note

This app is a proof-of-concept trained on a trial dataset using Logistic Regression.

During testing, we discovered a strong feature bias in the dataset:

- CGPA is the dominant factor, with a weight of about 3.27.
- A CGPA of 6.1 or higher almost entirely determines a positive placement outcome.
- IQ has an optimized weight near zero (about -0.05), so it currently has very little influence on the prediction.

Conclusion: the model behaves almost like a single-feature classifier because of the dataset distribution.

## Run the app

Start the app locally with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Project files

- app.py: Streamlit app interface and prediction logic
- placement.csv: dataset used for training
- placement_predictor_ml_model.pkl: trained model
- placement_predictor_ml_scaler.pkl: feature scaler
- requirements.txt: Python dependencies
