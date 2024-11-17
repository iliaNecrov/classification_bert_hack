from flair.data import Sentence
from flair.models import TextClassifier
from typing import List, Optional

from utils.model_abstract import TextClassifierAbstract

DEFAULT_MESSAGE = "Перевод"

class TextClassifierModel(TextClassifierAbstract):
    """
    Используя фреймворк Flair загрузим 
    """
    
    def __init__(self, model: TextClassifier) -> None:
        self.model = model

    @staticmethod
    def __preprocess(texts: List[Optional[str]]) -> List[str]:
        texts_ = []
        for text in texts:
            if not isinstance(text, str) or not text:
                texts_.append(DEFAULT_MESSAGE)
                print("Пустая строка заменена на значение по умолчанию!")
            else:
                texts_.append(text)
        return texts_
                
    @classmethod
    def load(cls, model_path: str) -> TextClassifierAbstract:
        """
        Загрузка BERT-модели для классификации
        """
        print("Идет загрузка модели...")
        model = TextClassifier.load(model_path)
        print("Модель успешно загружена!")    
        return cls(model=model)
    
    def predict(self, texts: List[Optional[str]], batch_size: int = 8) -> List[str]:
        texts = self.__preprocess(texts)
        print("Препроцесс прошел успешно!")
        sentences: List[Sentence] = [Sentence(text) for text in texts]
        self.model.predict(sentences,
                            mini_batch_size=batch_size,
                              verbose=True)
        tags = [sentence.tag for sentence in sentences]
        return tags