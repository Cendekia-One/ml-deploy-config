from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

app = FastAPI()

# Load the model during startup
model = tf.keras.models.load_model('models/SavedModel')

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    subcategory: list
    # top3_probabilities: list
    # over_0_5_strings: list
    # over_0_5_probabilities: list

@app.get("/")
def read_root():
    return 'Cendekiaone Machine Learning API'

@app.post("/predict")
async def predict_category(request: Request, text_request: TextRequest):
    text = text_request.text
    predictions = model.predict([text])

    subdisciplines_array = ['law', 'philosophy', 'religious studies', 'anthropology', 'archaeology and history', 'economics', 'earth science', 'psychology', 'sociology', 'biology', 'chemistry', 'astronomy', 'physics', 'computer science', 'software development', 'mathematics', 'agriculture', 'architecture', 'business and entrepreneurship', 'education', 'engineering and technology', 'environmental studies and forestry', 'journalism, media studies and communication', 'medicine and health', 'military sciences', 'public policy and administration', 'social work', 'transportation', 'climate change', 'finance', 'inspiration', 'marketing', 'gender studies', 'art', 'data science', 'artificial intelligence', 'spirituality', 'parenting', 'creativity', 'travel', 'social media', 'leadership', 'music', 'cryptocurrency', 'design', 'relationships', 'personal development', 'professional development', 'political science', 'linguistics, languages and literature', 'human physical performance and recreation']

    top_indices = np.argsort(predictions[0])[::-1]
    top3_probabilities = np.array(predictions)[0][top_indices]

    # over_0_5_indices = np.where(predictions > 0.5)[1]
    # over_0_5_probabilities = np.array(predictions)[0][over_0_5_indices]

    # over_0_5_strings = np.array(subdisciplines_array)[over_0_5_indices]
    subcategory = np.array(subdisciplines_array)[top_indices]

    response_data = {
        "subcategory": subcategory.tolist(),
        # "top_probabilities": top3_probabilities.tolist(),
        # "over_0_5_strings": over_0_5_strings.tolist(),
        # "over_0_5_probabilities": over_0_5_probabilities.tolist()
    }

    return TextResponse(**response_data)
