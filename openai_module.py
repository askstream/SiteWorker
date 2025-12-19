"""Модуль для работы с OpenAI API."""

import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Загружаем переменные окружения
load_dotenv()


class OpenAIClient:
    """Клиент для работы с OpenAI API."""

    def __init__(self, model: str | None = None) -> None:
        """
        Инициализация клиента OpenAI.

        Args:
            model: Название модели OpenAI. Если не указано, читается из .env
                  (OPENAI_MODEL) или используется gpt-4o по умолчанию
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY не найден в переменных окружения. "
                "Создайте файл .env и добавьте OPENAI_API_KEY=your_key"
            )
        self.client = OpenAI(api_key=api_key)
        # Если модель не передана явно, читаем из .env или используем значение по умолчанию
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    def generate_questions(self, text: str, num_questions: int = 5) -> List[str]:
        """
        Генерирует вопросы на основе переданного текста.

        Args:
            text: Текст для анализа
            num_questions: Количество вопросов для генерации

        Returns:
            Список сгенерированных вопросов

        Raises:
            Exception: При ошибках API OpenAI
        """
        prompt = (
            f"Ты пользователь. Какие вопросы у тебя возникли после прочтения "
            f"следующего текста? Сформулируй {num_questions} логичных вопросов, "
            f"которые могли бы задать пользователи.\n\n"
            f"Текст:\n{text}\n\n"
            f"Верни только список вопросов, каждый вопрос с новой строки, "
            f"без нумерации и дополнительных пояснений."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты помощник, который анализирует контент "
                        "и формулирует вопросы от лица пользователя.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )

            questions_text = response.choices[0].message.content.strip()
            questions = [
                q.strip()
                for q in questions_text.split("\n")
                if q.strip() and not q.strip().startswith(("#", "-", "*", "1.", "2."))
            ]

            # Если вопросы начинаются с маркеров, убираем их
            cleaned_questions = []
            for q in questions:
                # Убираем различные маркеры списка
                q = q.lstrip("- *•1234567890. ")
                if q:
                    cleaned_questions.append(q)

            # Если получили меньше вопросов, чем нужно, возвращаем что есть
            # Если больше - берем первые num_questions
            return cleaned_questions[:num_questions] if cleaned_questions else [
                "Не удалось сгенерировать вопросы"
            ]

        except Exception as e:
            raise Exception(f"Ошибка при генерации вопросов через OpenAI: {str(e)}")

