import os
import pandas as pd

from inference.flair_model import TextClassifierModel
from utils.model_downloading import download_model


model_name = os.environ['MODEL_NAME']
download_model(model_name)
# Ан-комент для Windows
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
model = TextClassifierModel.load("model.pt")

if __name__ == "__main__":
    print("---"*10)
    try:
        data = pd.read_csv("data/input.tsv", sep='\t', header=None, names=["index", "date", "summ", "texts"])
    except Exception as e:
        print(f"Похоже, проблема с загрукой файла...\n\n\n{str(e)}")
        print("---" * 10)
        raise
    
    texts = data["texts"]
    classes = model.predict(texts, 8)
    data["classes"] = classes
    data[["index", "classes"]].to_csv('data/output.tsv', sep='\t', index=False, header=False)
    print("Классификация закончена! Проверьте папку 'data', там лежит файл 'output.tsv' с результатами!")
    print("---"*10)
