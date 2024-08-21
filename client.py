import sys, os
import torch
import requests
import numpy as np
import sounddevice as sd

def init_abv2_api(api_key = "sbv2_amitaro", model_name = "amitaro"):
    init_url = "http://127.0.0.1:8001/initialize/"

    # サーバー側のインスタンスを初期化
    headers = {"api_key": api_key}

    init_inputs = {
        "modelname": model_name,
    }

    init_response = requests.post(init_url, json=init_inputs, headers=headers)
    if init_response.status_code == 200:
        print("Initialization successful.")
    else:
        print("Initialization failed.")
        exit(1)

def call_TTS_API(text,api_key = "sbv2_amitaro"):
    url = "http://127.0.0.1:8001/process/"
    headers = {"api_key": api_key}

    inputs = {
        "text": text,
    }

    response = requests.post(url, json=inputs, headers=headers)
    #response = requests.post(url, headers=headers)
    #print(response)
    # JSONデータとしてレスポンスを解析
    data = response.json()  # ここを修正

    audio = data['audio']
    audio = np.array(audio, dtype=np.float32)
    audio = audio / 32768.0
    sr = data['sr']

    return audio, sr

if __name__ == "__main__":
    init_abv2_api(api_key = "sbv2_amitaro", model_name = "amitaro")
    init_abv2_api(api_key = "sbv2_jvnv-F1-jp", model_name = "jvnv-F1-jp")

    audio, sr = call_TTS_API("こんにちは。",api_key = "sbv2_amitaro")
    sd.play(audio, sr)
    sd.wait()

    audio, sr = call_TTS_API("こんにちは。",api_key = "sbv2_jvnv-F1-jp")
    sd.play(audio, sr)
    sd.wait()
