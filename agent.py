"""Агент для парсинга страниц и генерации пользовательских вопросов."""

import logging
import os
from typing import List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from openai_module import OpenAIClient

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PageParser:
    """Класс для парсинга веб-страниц."""

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException,)),
    )
    def fetch_html(url: str) -> str:
        """
        Загружает HTML-контент по URL.

        Args:
            url: URL страницы для загрузки

        Returns:
            HTML-контент страницы

        Raises:
            requests.RequestException: При ошибках загрузки
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Ошибка при загрузке страницы {url}: {str(e)}")
            raise

    @staticmethod
    def extract_text(html: str) -> str:
        """
        Извлекает текстовый контент из HTML.

        Args:
            html: HTML-контент

        Returns:
            Извлеченный текст
        """
        soup = BeautifulSoup(html, "html.parser")

        # Удаляем скрипты и стили
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()

        # Извлекаем текст
        text = soup.get_text(separator=" ", strip=True)

        # Очищаем от лишних пробелов и переносов
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text


class QuestionGeneratorAgent:
    """Агент для генерации пользовательских вопросов на основе контента страницы."""

    def __init__(self, model: str | None = None) -> None:
        """
        Инициализация агента.

        Args:
            model: Модель OpenAI для использования. Если не указано, читается из .env
                  (OPENAI_MODEL) или используется gpt-4o по умолчанию
        """
        self.openai_client = OpenAIClient(model=model)
        self.parser = PageParser()

    def run(self, url: str, num_questions: int = 5) -> List[str]:
        """
        Основной метод агента: парсит страницу и генерирует вопросы.

        Args:
            url: URL страницы для анализа
            num_questions: Количество вопросов для генерации (по умолчанию 5)

        Returns:
            Список сгенерированных вопросов

        Raises:
            Exception: При ошибках парсинга или генерации
        """
        try:
            logger.info(f"Начинаю обработку URL: {url}")

            # Шаг 1: Загружаем HTML
            logger.info("Загрузка HTML...")
            html = self.parser.fetch_html(url)

            # Шаг 2: Извлекаем текст
            logger.info("Извлечение текста из HTML...")
            text = self.parser.extract_text(html)

            if not text or len(text.strip()) < 50:
                raise ValueError(
                    "Не удалось извлечь достаточное количество текста со страницы"
                )

            logger.info(f"Извлечено {len(text)} символов текста")

            # Шаг 3: Генерируем вопросы через OpenAI
            logger.info("Генерация вопросов через OpenAI...")
            questions = self.openai_client.generate_questions(text, num_questions)

            logger.info(f"Сгенерировано {len(questions)} вопросов")
            return questions

        except Exception as e:
            logger.error(f"Ошибка при выполнении агента: {str(e)}")
            raise


def main() -> None:
    """Основная функция для запуска агента из командной строки."""
    import sys

    if len(sys.argv) < 2:
        print("Использование: python agent.py <URL>")
        print("Пример: python agent.py https://example.com")
        sys.exit(1)

    url = sys.argv[1]

    try:
        agent = QuestionGeneratorAgent()
        questions = agent.run(url)

        print("\n" + "=" * 60)
        print("Сгенерированные вопросы:")
        print("=" * 60)
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")
        print("=" * 60)

    except Exception as e:
        print(f"Ошибка: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

