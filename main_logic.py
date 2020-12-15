import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

scalling_cols = [
    "Age",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]


def clean_data(data):
    # Encode Gender
    encoder = pickle.load(open("encoder.pkl", "rb"))
    fitted_values = encoder.transform(data[["Gender"]])
    enc_df = pd.DataFrame(fitted_values.toarray())
    enc_df.rename(columns={0: "Sex"}, inplace=True)
    data = data.join(enc_df["Sex"])
    data.drop(["Gender"], axis=1, inplace=True)

    # Impute missing values with mean
    mean_values = pickle.load(open("mean_values.pkl", "rb"))
    data.fillna(mean_values, inplace=True)

    # Convert Sex field to int
    data.Sex = data.Sex.astype(int)

    # Scale scalable fields with MinMaxScaler
    min_max = pickle.load(open("scaler.pkl", "rb"))
    data[scalling_cols] = min_max.transform(data[scalling_cols])

    return data


def service(input_json):
    model_rfc = pickle.load(open("model_rfc.pkl", "rb"))
    sample_df = pd.DataFrame(input_json, index=[0])
    sample_df = clean_data(sample_df)
    prediction_prob = model_rfc.predict_proba(sample_df)
    return prediction_prob[0][1]
