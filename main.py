from fastapi import FastAPI
from utils import util_dua, get_commodities
from text_process import text_summary, text_category
from typing import List
import json

app = FastAPI()

@app.get("/")
def read_root():
    return 'Cendekiaone Machine Learning API'

# Text Summary models name is text_summary.h5
@app.get("/summary")
def predict_summary():
    data = json.loads(request.data)
    text = data["text"]
    return text_summary(text=text)

# Text Category models name is text_category.h5
@app.get("/category")
def predict_category():
    data = json.loads(request.data)
    text = data["text"]
    return text_category(text=text)


