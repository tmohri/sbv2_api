import os
import numpy as np
from pathlib import Path
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.japanese.user_dict import update_dict
import torch
from pydantic import BaseModel

from fastapi import FastAPI, Depends, Header
from typing import List, Dict, Any
from fastapi.security.api_key import APIKeyHeader
import uvicorn
import json
import time
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
update_user_dict = True
default_dict_path = "dict_data/default.csv"
compiled_dict_path = "dict_data/user.dic"
bert_models_model = "ku-nlp/deberta-v2-large-japanese-char-wwm"
bert_models_tokenizer = "ku-nlp/deberta-v2-large-japanese-char-wwm"


class SBV2:
    def __init__(self, model_path):
        logger.remove()

        if update_user_dict:
            print("loading user dict")
            update_dict(default_dict_path = Path(default_dict_path), compiled_dict_path = Path(compiled_dict_path))
        

        if device == "auto":
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.DEVICE = device

        bert_models.load_model(Languages.JP, bert_models_model)
        bert_models.load_tokenizer(Languages.JP, bert_models_tokenizer)

        style_file = glob.glob(f'{model_path}/*.npy',recursive=True)[0]
        config_file = glob.glob(f'{model_path}/*.json',recursive=True)[0]
        model_file = glob.glob(f'{model_path}/*.safetensors',recursive=True)[0]

        print(style_file)
        print(config_file)
        print(model_file)

        
        self.model_TTS = TTSModel(
            model_path=model_file,
            config_path=config_file,
            style_vec_path=style_file,
            device=self.DEVICE
        )

    def call_TTS(self,message):
        sr, audio = self.model_TTS.infer(text=message)

        return sr, audio
    
    def text2speech(self,message):
        sr, audio = self.model_TTS.infer(text=message)
        sd.play(audio, sr)
        sd.wait()

app = FastAPI()

class SBV2_inputs(BaseModel):
    text: str

class SBV2_init(BaseModel):
    modelname: str

# ユーザごとのインスタンスを管理する辞書
user_instances: Dict[str, Dict] = {}

class Dependencies:
    def __init__(self,api_key, model):
        model_path = f"model_assets/{model}"
        self.sbv2 = SBV2(model_path = model_path)

    def get_sbv2(self):   
        return self.sbv2


def get_user_dependencies(api_key: str,model = None):
    #過去にAPIkeyが登録されていない場合は新規登録
    if api_key not in user_instances:
        if model is None:
            raise Exception("model is required for the first time initialization")
        user_instances[api_key] = Dependencies(api_key, model)
        
    #登録されている場合はそのまま返す
    return user_instances[api_key]

API_KEY_NAME = "api_key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
def get_api_key(api_key: str = Depends(api_key_header)):
    return api_key

print("server started")

@app.post("/initialize/")
async def initialize(
    inputs: SBV2_init,
    api_key: str =  Depends(get_api_key)
    ):
    dependencies = get_user_dependencies(api_key, inputs.modelname)
    #初回の実行は`torch.nn.utils.weight_norm`のFutureWarningのせいか、処理時間が長いので、初期化のタイミングで初回の実行を終わらせておく
    _, _ = dependencies.get_sbv2().call_TTS("初期化")
    return {"message": "Initialized"}

@app.post("/process/")
async def process_data(
    inputs: SBV2_inputs,
    api_key: str = Depends(get_api_key),    
):
    dependencies = get_user_dependencies(api_key)
    start_tts = time.time()
    sr, audio = dependencies.get_sbv2().call_TTS(inputs.text)
    print(f"Time taken for TTS: {time.time() - start_tts}")
    return {"audio": audio.tolist(), "sr": sr}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
