from abc import ABC, abstractmethod
from typing import List, Optional


class TextClassifierAbstract(ABC):

    @classmethod
    def load(cls, model_path: str):
        """Загрузить модель, используя путь до нее"""
        pass

    def predict(self, texts: List[Optional[str]], batch_size: int) -> List[str]:
        """Предсказать классы для списка сообщений"""
        pass
