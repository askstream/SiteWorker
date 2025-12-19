"""Агент для парсинга страниц и выполнения различных задач."""

import logging
import os
from typing import Dict, List

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


class ContentClassifierAgent:
    """Агент для классификации типа контента сайта."""

    def __init__(self, model: str | None = None) -> None:
        """
        Инициализация агента.

        Args:
            model: Модель OpenAI для использования. Если не указано, читается из .env
                  (OPENAI_MODEL) или используется gpt-4o по умолчанию
        """
        self.openai_client = OpenAIClient(model=model)
        self.parser = PageParser()

    def run(self, url: str) -> Dict[str, str]:
        """
        Основной метод агента: парсит страницу и классифицирует тип контента.

        Args:
            url: URL страницы для анализа

        Returns:
            Словарь с ключами 'type' (тип сайта) и 'explanation' (краткое объяснение)

        Raises:
            Exception: При ошибках парсинга или классификации
        """
        try:
            logger.info(f"Начинаю классификацию URL: {url}")

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

            # Шаг 3: Классифицируем контент через OpenAI
            logger.info("Классификация контента через OpenAI...")
            classification = self.openai_client.classify_content(text)

            logger.info(f"Тип сайта определен: {classification['type']}")
            return classification

        except Exception as e:
            logger.error(f"Ошибка при выполнении классификации: {str(e)}")
            raise


class UXReviewerAgent:
    """Агент для анализа UX сайта и генерации рекомендаций."""

    def __init__(self, model: str | None = None) -> None:
        """
        Инициализация агента.

        Args:
            model: Модель OpenAI для использования. Если не указано, читается из .env
                  (OPENAI_MODEL) или используется gpt-4o по умолчанию
        """
        self.openai_client = OpenAIClient(model=model)
        self.parser = PageParser()

    def run(self, url: str, num_recommendations: int = 5) -> Dict[str, List[str]]:
        """
        Основной метод агента: парсит страницу и генерирует UX-отчёт.

        Args:
            url: URL страницы для анализа
            num_recommendations: Количество рекомендаций (по умолчанию 5)

        Returns:
            Словарь с ключами:
            - 'strengths' (достоинства)
            - 'weaknesses' (слабые места)
            - 'recommendations' (рекомендации по улучшению)

        Raises:
            Exception: При ошибках парсинга или генерации
        """
        try:
            logger.info(f"Начинаю UX-анализ URL: {url}")

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

            # Шаг 3: Генерируем UX-отчёт через OpenAI
            logger.info("Генерация UX-отчёта через OpenAI...")
            ux_report = self.openai_client.generate_ux_report(text, num_recommendations)

            logger.info(
                f"UX-отчёт сгенерирован: {len(ux_report['recommendations'])} рекомендаций"
            )
            return ux_report

        except Exception as e:
            logger.error(f"Ошибка при выполнении UX-анализа: {str(e)}")
            raise


class SiteAgent:
    """Общий агент для выполнения нескольких задач на странице."""

    def __init__(self, model: str | None = None) -> None:
        """
        Инициализация агента.

        Args:
            model: Модель OpenAI для использования. Если не указано, читается из .env
                  (OPENAI_MODEL) или используется gpt-4o по умолчанию
        """
        self.question_generator = QuestionGeneratorAgent(model=model)
        self.content_classifier = ContentClassifierAgent(model=model)
        self.ux_reviewer = UXReviewerAgent(model=model)
        self.parser = PageParser()

    def run_all(self, url: str, num_questions: int = 5) -> Dict:
        """
        Выполняет все доступные задачи на странице.

        Args:
            url: URL страницы для анализа
            num_questions: Количество вопросов для генерации (по умолчанию 5)

        Returns:
            Словарь с результатами всех задач:
            {
                'questions': List[str],
                'content_type': Dict[str, str],
                'ux_report': Dict[str, List[str]]
            }
        """
        try:
            logger.info(f"Выполняю все задачи для URL: {url}")

            # Загружаем HTML один раз
            logger.info("Загрузка HTML...")
            html = self.parser.fetch_html(url)
            text = self.parser.extract_text(html)

            if not text or len(text.strip()) < 50:
                raise ValueError(
                    "Не удалось извлечь достаточное количество текста со страницы"
                )

            logger.info(f"Извлечено {len(text)} символов текста")

            # Выполняем задачи последовательно
            logger.info("Генерация вопросов...")
            questions = self.question_generator.openai_client.generate_questions(
                text, num_questions
            )

            logger.info("Классификация контента...")
            content_type = self.content_classifier.openai_client.classify_content(text)

            logger.info("Генерация UX-отчёта...")
            ux_report = self.ux_reviewer.openai_client.generate_ux_report(text)

            return {
                "questions": questions,
                "content_type": content_type,
                "ux_report": ux_report,
            }

        except Exception as e:
            logger.error(f"Ошибка при выполнении задач: {str(e)}")
            raise


def main() -> None:
    """Основная функция для запуска агента из командной строки."""
    import sys

    if len(sys.argv) < 2:
        print("Использование: python agent.py <URL> [задача]")
        print("\nДоступные задачи:")
        print("  questions  - Генерация пользовательских вопросов (по умолчанию)")
        print("  classify   - Классификация типа контента")
        print("  ux         - UX-анализ и рекомендации по улучшению")
        print("  all        - Выполнить все задачи")
        print("\nПримеры:")
        print("  python agent.py https://example.com")
        print("  python agent.py https://example.com questions")
        print("  python agent.py https://example.com classify")
        print("  python agent.py https://example.com ux")
        print("  python agent.py https://example.com all")
        sys.exit(1)

    url = sys.argv[1]
    task = sys.argv[2] if len(sys.argv) > 2 else "questions"

    try:
        if task == "questions":
            agent = QuestionGeneratorAgent()
            questions = agent.run(url)

            print("\n" + "=" * 60)
            print("Сгенерированные вопросы:")
            print("=" * 60)
            for i, question in enumerate(questions, 1):
                print(f"{i}. {question}")
            print("=" * 60)

        elif task == "classify":
            agent = ContentClassifierAgent()
            result = agent.run(url)

            print("\n" + "=" * 60)
            print("Классификация типа контента:")
            print("=" * 60)
            print(f"Тип: {result['type']}")
            print(f"\nОбъяснение: {result['explanation']}")
            print("=" * 60)

        elif task == "ux":
            agent = UXReviewerAgent()
            result = agent.run(url)

            print("\n" + "=" * 60)
            print("UX-ОТЧЁТ")
            print("=" * 60)

            print("\n✅ Достоинства:")
            for i, strength in enumerate(result["strengths"], 1):
                print(f"  {i}. {strength}")

            print("\n⚠️  Слабые места:")
            for i, weakness in enumerate(result["weaknesses"], 1):
                print(f"  {i}. {weakness}")

            print("\n💡 Рекомендации по улучшению UX:")
            for i, recommendation in enumerate(result["recommendations"], 1):
                print(f"  {i}. {recommendation}")

            print("\n" + "=" * 60)

        elif task == "all":
            agent = SiteAgent()
            results = agent.run_all(url)

            print("\n" + "=" * 60)
            print("РЕЗУЛЬТАТЫ АНАЛИЗА СТРАНИЦЫ")
            print("=" * 60)

            print("\n📋 Классификация типа контента:")
            print(f"Тип: {results['content_type']['type']}")
            print(f"Объяснение: {results['content_type']['explanation']}")

            print("\n❓ Сгенерированные вопросы:")
            for i, question in enumerate(results["questions"], 1):
                print(f"{i}. {question}")

            print("\n🎨 UX-ОТЧЁТ:")
            print("\n  ✅ Достоинства:")
            for i, strength in enumerate(results["ux_report"]["strengths"], 1):
                print(f"    {i}. {strength}")

            print("\n  ⚠️  Слабые места:")
            for i, weakness in enumerate(results["ux_report"]["weaknesses"], 1):
                print(f"    {i}. {weakness}")

            print("\n  💡 Рекомендации по улучшению UX:")
            for i, recommendation in enumerate(results["ux_report"]["recommendations"], 1):
                print(f"    {i}. {recommendation}")

            print("\n" + "=" * 60)

        else:
            print(f"Неизвестная задача: {task}", file=sys.stderr)
            print("Доступные задачи: questions, classify, ux, all", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Ошибка: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

