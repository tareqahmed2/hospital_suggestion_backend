from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv() 

MONGO_URI = os.getenv("MONGO_URI")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

client = MongoClient(MONGO_URI)
db = client['hospital_db']
collection = db['hospitals']

app = FastAPI()
origins = ["http://localhost:3000", "http://127.0.0.1:3000","https://hospital-frontend-orpin.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class RecommendationRequest(BaseModel):
    location: Optional[str] = None
    hospital_name: Optional[str] = None
    test_name: Optional[str] = None
    max_price: Optional[int] = None

def train_model():
    data = []
    for hosp in collection.find():
        for test in hosp['tests']:
            data.append({
                "location": hosp['location'],
                "test_name": test['test_name'],
                "hospital_name": hosp['hospital_name'],
                "price_bdt": test['price_bdt']
            })
    df = pd.DataFrame(data)
    if df.empty:
        return
    le_loc = LabelEncoder()
    le_test = LabelEncoder()
    le_hosp = LabelEncoder()
    df['loc_enc'] = le_loc.fit_transform(df['location'])
    df['test_enc'] = le_test.fit_transform(df['test_name'])
    df['hosp_enc'] = le_hosp.fit_transform(df['hospital_name'])
    X = df[['loc_enc', 'test_enc', 'price_bdt']]
    y = df['hosp_enc']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'hospital_model.pkl')
    joblib.dump(le_loc, 'le_loc.pkl')
    joblib.dump(le_test, 'le_test.pkl')
    joblib.dump(le_hosp, 'le_hosp.pkl')

train_model()
model = joblib.load('hospital_model.pkl')
le_loc = joblib.load('le_loc.pkl')
le_test = joblib.load('le_test.pkl')
le_hosp = joblib.load('le_hosp.pkl')

def serpapi_search(query: str):
    try:
        search = GoogleSearch({
            "q": query,
            "location": "Bangladesh",
            "hl": "en",
            "api_key": SERPAPI_KEY
        })
        results = search.get_dict()
        hospitals = []
        local_res = results.get("local_results")
        organic_res = results.get("organic_results")

        if isinstance(local_res, list):
            for r in local_res:
                hospitals.append({
                    "hospital_name": r.get("title"),
                    "location": r.get("address"),
                    "test_name": "Unknown",
                    "price_bdt": 0,
                    "type": r.get("type") or "Hospital",
                    "image": r.get("thumbnail") or None
                })
        elif isinstance(organic_res, list):
            for r in organic_res:
                hospitals.append({
                    "hospital_name": r.get("title"),
                    "location": r.get("link"),
                    "test_name": "Unknown",
                    "price_bdt": 0,
                    "type": "Hospital",
                    "image": None
                })
        return hospitals
    except Exception as e:
        print("SerpApi error:", e)
        return []

@app.get("/")
def root():
    return {"message": "Hospital Recommendation API running"}

@app.get("/hospitals")
def get_hospitals():
    hospitals = []
    for hosp in collection.find():
        for test in hosp['tests']:
            hospitals.append({
                "hospital_name": hosp['hospital_name'],
                "location": hosp['location'],
                "test_name": test['test_name'],
                "price_bdt": test['price_bdt'],
                "type": test.get("type") or "Unknown"
            })
    return hospitals

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    data = []

    for hosp in collection.find():
        if req.hospital_name and req.hospital_name.lower() not in hosp['hospital_name'].lower():
            continue
        if req.location and req.location.lower() not in hosp['location'].lower():
            continue
        for test in hosp['tests']:
            if req.test_name and req.test_name.lower() not in test['test_name'].lower():
                continue
            if req.max_price and test['price_bdt'] > req.max_price:
                continue
            data.append({
                "hospital_name": hosp['hospital_name'],
                "location": hosp['location'],
                "test_name": test['test_name'],
                "price_bdt": test['price_bdt'],
                "type": test.get("type") or "Unknown",
                "image": None
            })

    if not data:
        query = req.hospital_name or f"hospitals in {req.location}"
        fallback = serpapi_search(query)
        return {"recommended_hospital": fallback[0] if fallback else None, "all_fallback": fallback}

    df = pd.DataFrame(data)
    try:
        X_pred = pd.DataFrame({
            'loc_enc': le_loc.transform(df['location']),
            'test_enc': le_test.transform(df['test_name']),
            'price_bdt': df['price_bdt']
        })
        pred_idx = model.predict(X_pred)
        df['pred_hosp_enc'] = pred_idx
        df['pred_hospital'] = le_hosp.inverse_transform(pred_idx)
    except Exception as e:
        print("Prediction error:", e)
        df['pred_hospital'] = df['hospital_name']

    return {"recommended_hospital": df.iloc[0].to_dict(), "all_hospitals": df.to_dict(orient="records")}
