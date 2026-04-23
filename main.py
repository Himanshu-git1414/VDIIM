import pandas as pd
from fastapi import FastAPI 
from VDIIM import load_and_process_data 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI() 

data_cache = None

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["https://vdiim.vercel.app"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

def get_data():
    global data_cache
    if data_cache is None:
        try:
            data_cache = load_and_process_data()
        except Exception as e:
            print("Error loading data:", e)
            raise e
    return data_cache

@app.get("/") 
def home(): 
    return {"message": "Nifty Volatility API is running"} 

@app.get("/test") 
def test(): 
    return {"status": "working"} 

@app.get("/volatility") 
def get_volatility(limit: int = 100): 
    df, daily_df, hypothesis_result, regression_result = get_data()
    data = df[["open", "high", "low", "close", "rolling_vol_15", 
               "vol_spike", "range", "ema_8", "ema_30"]].reset_index().tail(limit)
    return data.to_dict(orient="records")

@app.get("/daily-summary")
def daily_summary(limit: int = 50):
    try:
        df, daily_df, hypothesis_result, regression_result = get_data()
        data = daily_df.reset_index().tail(limit)
        return data.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/high-vol-days")
def high_vol_days(limit: int = 20):
    df, daily_df, hypothesis_result, regression_result = get_data()
    data = daily_df[daily_df["regime"] == "High Vol"].reset_index().tail(limit)
    return data.to_dict(orient = "records")

@app.get("/intraday-by-date")
def intraday_by_date(date: str):
    try:
        df, daily_df, hypothesis_result, regression_result = get_data()
        filtered = df[df.index.date == pd.to_datetime(date).date()]
        data = filtered[["open", "high", "low", "close", "rolling_vol_15", 
                         "vol_spike", "range", "ema_8", "ema_30"]].reset_index()
        return data.to_dict(orient = "records")
    except Exception as e:
        return {"error" : str(e)}
    
@app.get("/hypothesis-test")
def get_hypothesis():
    df, daily_df, hypothesis_result, regression_result = get_data()
    return hypothesis_result

@app.get("/range-distribution")
def range_distribution():
    df, daily_df, hypothesis_result, regression_result = get_data()
    df_temp = df.copy()
    df_temp["date"] = df_temp.index.date
    daily_temp = daily_df.copy()
    daily_temp["date"] = daily_temp.index.date

    merged = df_temp.merge(daily_temp[["date", "regime"]], on="date")

    high_vol = merged[merged["regime"] == "High Vol"]["range"]
    normal_vol = merged[merged["regime"] == "Normal"]["range"]

    return {
        "high_vol": high_vol.tolist(),
        "normal_vol": normal_vol.tolist()
    }

@app.get("/predict-next-open")
def predict_next_open():
    df, daily_df, hypothesis_result, regression_result = get_data()
    return regression_result