# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Dashboard.Model import HomeCreditRisk, HCRModel

# 2. Create app and model objects
appf = FastAPI()
model = HCRModel()

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the prediction with the probability
@appf.get('/predict')
def predict_risks(id_client:str):
    prediction, probability = model.predict_risks(id_client)
    return {
        'prediction': prediction,
        'probability': probability,
    }
# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(appf, host='127.0.0.1', port=8000)