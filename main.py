from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = FastAPI(title="RESCUE RP AI - Microservicio IA", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carga del modelo PLN (BETO) en español
MODELO_BERT_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
print(f"[PLN] Cargando modelo BERT (BETO): {MODELO_BERT_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_BERT_NAME)
model_bert = AutoModelForSequenceClassification.from_pretrained(MODELO_BERT_NAME, num_labels=3)
model_bert.to(device)
model_bert.eval()

# Inicialización y entrenamiento del modelo estructurado (Random Forest)
DATASET_PATH = "misiones_rescue_rp.csv"

def inicializar_y_entrenar_rf():
    if not os.path.exists(DATASET_PATH):
        data = {
            'distancia_km': [45.0, 120.5, 210.0, 85.0, 150.0, 300.0],
            'tipo_aeronave': ['Bell 412', 'UH-60 Black Hawk', 'Mi-17', 'Bell 412', 'UH-60 Black Hawk', 'Mi-17'],
            'tipo_evacuacion': ['MEDEVAC', 'CASEVAC', 'CASEVAC', 'MEDEVAC', 'CASEVAC', 'CASEVAC'],
            'condicion_clima': ['Despejado', 'Lluvia Moderada', 'Niebla/Baja Visibilidad', 'Despejado', 'Vientos Fuertes', 'Niebla/Baja Visibilidad'],
            'tiempo_mision_min': [35.0, 78.5, 140.0, 55.0, 95.0, 180.0]
        }
        pd.DataFrame(data).to_csv(DATASET_PATH, index=False)
    
    df = pd.read_csv(DATASET_PATH)
    X = df[['distancia_km', 'tipo_aeronave', 'tipo_evacuacion', 'condicion_clima']]
    y = df['tiempo_mision_min']
    
    global preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['distancia_km']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['tipo_aeronave', 'tipo_evacuacion', 'condicion_clima'])
        ]
    )
    X_prep = preprocessor.fit_transform(X)
    
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_prep, y)
    return model_rf

model_rf = inicializar_y_entrenar_rf()

class MisionRequest(BaseModel):
    distancia_km: float
    tipo_aeronave: str
    tipo_evacuacion: str
    condicion_clima: str
    descripcion_medica: str

@app.post("/api/analisis-multimodal")
def analizar_mision(datos: MisionRequest):
    try:
        # Flujo Estructurado: Random Forest
        df_input = pd.DataFrame([{
            'distancia_km': datos.distancia_km,
            'tipo_aeronave': datos.tipo_aeronave,
            'tipo_evacuacion': datos.tipo_evacuacion,
            'condicion_clima': datos.condicion_clima
        }])
        X_prep = preprocessor.transform(df_input)
        tiempo_estimado = float(model_rf.predict(X_prep)[0])

        # Flujo No Estructurado: BETO / BERT
        inputs = tokenizer(
            datos.descripcion_medica, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            outputs = model_bert(**inputs)
            clase_predicha = torch.argmax(outputs.logits, dim=1).item()
        
        diccionario_triaje = {
            0: "Leve (Estable / Rutina)",
            1: "Moderado (Atención Prioritaria)",
            2: "Crítico (Código Rojo / Emergencia Vital)"
        }

        return {
            "status": "success",
            "tiempo_estimado_min": round(tiempo_estimado, 2),
            "nivel_triaje": diccionario_triaje.get(clase_predicha, "Desconocido"),
            "codigo_gravedad": clase_predicha
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
