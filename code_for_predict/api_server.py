from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
from switch_predict import ToeholdPredictorSimplified

app = FastAPI(title="Toehold Switch Prediction API")
predictor = None

class RNAInput(BaseModel):
    sequence: str

@app.on_event("startup")
async def load_model_on_startup():
    global predictor
    print("Initializing model to GPU memory...")

    USE_RNA_ERNIE = True
    VOCAB = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/CatIIIIIIII-RNAErnie-faa2b2d/data/vocab/vocab_1MER.txt"
    ERNIE_PATH = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/CatIIIIIIII-RNAErnie-faa2b2d/output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final"
    ON_MODEL = '/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/Prediction_Translation_Strength/model/adjusted_on_pearson_mse_structure_9_pcc=0.8279.pth'
    OFF_MODEL = '/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/Prediction_Translation_Strength/model/off_pearson_mse_ernie_structure_12_pcc=0.7123.pth'

    predictor = ToeholdPredictorSimplified(
        on_model_path=ON_MODEL,
        off_model_path=OFF_MODEL,
        use_rna_ernie=USE_RNA_ERNIE,
        vocab_path=VOCAB,
        ernie_model_path=ERNIE_PATH,
        auto_select_gpu=True
    )
    print("Model loaded successfully, ready for prediction")

@app.post("/predict")
async def predict(data: RNAInput):
    seq = data.sequence.strip()
    
    if len(seq) != 115:
        raise HTTPException(status_code=400, detail=f"Sequence length must be 115bp, got {len(seq)}")
    
    try:
        res = predictor.predict_single(seq)
        return {
            "on": res['ON'],
            "off": res['OFF'],
            "ratio": res['ON_OFF_Ratio'],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
