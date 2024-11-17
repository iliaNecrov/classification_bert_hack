import os
import boto3


MODELS_INFO = {"tiny": {"pt": "tiny_bert_onnx/model_onnx.pt",
                        "embeddings": "tiny_bert_onnx/embeddings-rubert-base.onnx"},
               "base": {"pt": "USER_base_onnx/model-onnx-user-base.pt",
                        "embeddings": "USER_base_onnx/embeddings-user-base.onnx"}}
BUCKET_NAME = "bucket-bert-hack"
ENDPOINT = "https://storage.yandexcloud.net"


def create_session() -> boto3.session:
    """
    Создать сессию для работы с хранилищем S3.
    Использует учетные данные AWS из переменных окружения.

    :return: клиент S3 с указанной конечной точкой.
    """
    session = boto3.Session(
        aws_access_key_id=(os.environ['AWS_SECRET_KEY_ID']),  # ("YCAJEDYH8sOEKETe5gXbog3r7"),
        aws_secret_access_key=(os.environ['AWS_SECRET_ACCESS_KEY']),  # ("YCP86xjxXtmSoU5NjJslZIc_JuVjglPdOlQXf3h0"),
        region_name="ru-central1",
    )

    return session.client("s3", endpoint_url=ENDPOINT)


def download_model(model_name: str) -> None:
    """
    Скачать выбранную модель и её эмбеддинги из хранилища.
    Файлы сохраняются в текущей директории.

    :param model_name: имя модели для скачивания (например, 'tiny' или 'base').
    """
    s3 = create_session()

    # Скачать основную модель (слой)
    print("Скачивание модели...")
    s3.download_file(BUCKET_NAME, MODELS_INFO[model_name]["pt"], "model.pt")

    # Скачать эмбеддинги
    print("Скачивание эмбедингов... (Может занять время)")
    emb_path = os.path.basename(MODELS_INFO[model_name]["embeddings"])
    s3.download_file(BUCKET_NAME, MODELS_INFO[model_name]["embeddings"], emb_path)

    print("Скачивание прошло успешно!")
    s3.close()