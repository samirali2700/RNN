import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from pickle import load


def predict(model_name):
    model = load_model(f"./Models/{model_name}")
    df = pd.read_csv('./staticFiles/uploads/PREDICT.csv')
    df.dropna(inplace=True)
    cols = list(df)[1:6]
    scaler = load(open('./utils/scaler.pkl', 'rb'))
    predict_data = df[cols].astype(float)

    predict_data_scaled = scaler.transform(predict_data)

    predictX = [predict_data_scaled[-300:, 0: predict_data_scaled.shape[1]]]

    predictX = np.array(predictX)

    result = model.predict(predictX)

    return result[0, 0]
