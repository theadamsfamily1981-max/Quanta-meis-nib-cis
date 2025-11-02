from fastapi import FastAPI

app = FastAPI(title="QUANTA Selective Inference")

@app.get('/health')
def health():
    return {"status": "ok"}

# Placeholder: selective predict endpoint
@app.post('/v1/predict')
def predict():
    return {"message": "stub"}
