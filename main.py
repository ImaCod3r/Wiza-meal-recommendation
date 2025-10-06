from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

modelo = joblib.load("modelo_pratos.pkl")
encoder_dia_semana = joblib.load("encoder_dia_semana.pkl")
encoder_complexidade = joblib.load("encoder_complexidade.pkl")
encoder_proteina = joblib.load("encoder_proteina.pkl")
encoder_naturalidade = joblib.load("encoder_naturalidade.pkl")
encoder_target = joblib.load("encoder_target.pkl")

app = FastAPI()

class PratoInput(BaseModel):
    DIA_SEMANA: str
    COMPLEXIDADE_MAX: str
    E_FIM_SEMANA: int
    FUNGE_SIM_NAO: int
    PROTEINA_ONTEM: str
    NATURALIDADE: str

@app.get("/")
def home():
    return {"message": "Bem vindo ao Wiza meal recommendation"}

@app.post("/recomendar")
def recomendar(dados: PratoInput):
    
    df = pd.DataFrame([dados.dict()])

    df["DIA_SEMANA_ENCODER"] = encoder_dia_semana.transform(df[["DIA_SEMANA"]])
    df["COMPLEXIDADE_MAX_ENCODER"] = encoder_complexidade.transform(df[["COMPLEXIDADE_MAX"]])

    proteina = encoder_proteina.transform(df[["PROTEINA_ONTEM"]])
    proteina_df = pd.DataFrame(proteina, columns=encoder_proteina.get_feature_names_out(["PROTEINA_ONTEM"]))

    naturalidade = encoder_naturalidade.transform(df[["NATURALIDADE"]])
    naturalidade_df = pd.DataFrame(naturalidade, columns=encoder_naturalidade.get_feature_names_out(["NATURALIDADE"]))

    # Concatenar
    df_final = pd.concat([df, proteina_df, naturalidade_df], axis=1)

    # Selecionar colunas usadas no treino
    X_input = df_final[[
        "E_FIM_SEMANA",
        "FUNGE_SIM_NAO",
        "DIA_SEMANA_ENCODER",
        "COMPLEXIDADE_MAX_ENCODER",
        "PROTEINA_ONTEM_Carne Seca",
        "PROTEINA_ONTEM_Carne Vaca",
        "PROTEINA_ONTEM_Feijão",
        "PROTEINA_ONTEM_Frango",
        "PROTEINA_ONTEM_Miúdos",
        "PROTEINA_ONTEM_Muamba",
        "PROTEINA_ONTEM_Peixe",
        "NATURALIDADE_Benguela",
        "NATURALIDADE_Huíla",
        "NATURALIDADE_Luanda",
        "NATURALIDADE_Malanje",
        "NATURALIDADE_Zaire"
    ]]

    # Fazer previsão
    y_pred = modelo.predict(X_input)
    prato = encoder_target.inverse_transform(y_pred)[0]

    return {"Prato recomendado": prato}