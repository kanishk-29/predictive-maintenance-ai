import numpy as np, pandas as pd, joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

bundle = joblib.load("model.joblib")
model = bundle["model"]
threshold = 0.10
cols = bundle["features"]

app = FastAPI(title="Predictive Maintenance API") 
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class Reading(BaseModel):
    type: int = Field(ge=0, le=2)
    air_temp: float = Field(ge=290, le=310)
    process_temp: float = Field(ge=300, le=320)
    rpm: float = Field(ge=1000, le=3000)
    torque: float = Field(ge=1, le=100)
    tool_wear: float = Field(ge=0, le=300)

@app.post("/predict")
def predict(r: Reading):
    row = pd.DataFrame([{
        "Type": r.type,
        "Air temperature [K]": r.air_temp,
        "Process temperature [K]": r.process_temp,
        "Rotational speed [rpm]": r.rpm,
        "Torque [Nm]": r.torque,
        "Tool wear [min]": r.tool_wear,
    }])
    # SAME feature engineering as in training
    row["power"] = row["Torque [Nm]"] * row["Rotational speed [rpm]"] * 2 * np.pi / 60
    row["temp_diff"] = row["Process temperature [K]"] - row["Air temperature [K]"]
    row["wear_torque"] = row["Tool wear [min]"] * row["Torque [Nm]"]

    prob = float(model.predict_proba(row[cols])[0, 1])
    return {"failure_probability": round(prob, 4),
            "alert": prob >= threshold}