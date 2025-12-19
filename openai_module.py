"""Модуль для работы с OpenAI API."""

import os
from typing import Dict, List

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    def classify_content(self, text: str) -> Dict[str, str]:
        """
        Классифицирует тип сайта на основе переданного текста.

        Args:
            text: Текст для анализа

        Returns:
            Словарь с ключами 'type' (тип сайта) и 'explanation' (краткое объяснение)

        Raises:
            Exception: При ошибках API OpenAI
        """
        prompt = (
            f"Классифицируй сайт на основе следующего текста. "
            f"Определи тип сайта (например: лендинг, блог, маркетплейс, "
            f"корпоративный сайт, новостной портал, форум, социальная сеть и т.д.).\n\n"
            f"Текст:\n{text}\n\n"
            f"Верни ответ в формате:\n"
            f"Тип: [тип сайта]\n"
            f"Объяснение: [краткое объяснение, почему именно этот тип]"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты эксперт по классификации веб-сайтов. "
                        "Анализируй контент и определяй тип сайта на основе "
                        "структуры, содержания и назначения.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )

            result_text = response.choices[0].message.content.strip()

            # Парсим ответ
            content_type = ""
            explanation = ""

            lines = result_text.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("Тип:") or line.startswith("Тип :"):
                    content_type = line.split(":", 1)[1].strip()
                elif line.startswith("Объяснение:") or line.startswith("Объяснение :"):
                    explanation = line.split(":", 1)[1].strip()

            # Если не удалось распарсить, пытаемся извлечь из текста
            if not content_type or not explanation:
                # Пробуем найти тип в первой строке
                if not content_type:
                    first_line = lines[0] if lines else ""
                    if ":" in first_line:
                        content_type = first_line.split(":", 1)[1].strip()
                    else:
                        content_type = first_line.strip()

                # Остальное - объяснение
                if not explanation and len(lines) > 1:
                    explanation = " ".join(lines[1:]).strip()

            # Если все еще не нашли, используем весь текст как тип
            if not content_type:
                content_type = result_text[:100] if result_text else "Неизвестный тип"
            if not explanation:
                explanation = "Не удалось определить объяснение"

            return {
                "type": content_type,
                "explanation": explanation,
            }

        except Exception as e:
            raise Exception(f"Ошибка при классификации контента через OpenAI: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    def generate_ux_report(self, text: str, num_recommendations: int = 5) -> Dict[str, List[str]]:
        """
        Генерирует UX-отчёт с рекомендациями по улучшению на основе текста сайта.

        Args:
            text: Текст для анализа
            num_recommendations: Количество рекомендаций (по умолчанию 5)

        Returns:
            Словарь с ключами:
            - 'strengths' (достоинства)
            - 'weaknesses' (слабые места)
            - 'recommendations' (рекомендации по улучшению)

        Raises:
            Exception: При ошибках API OpenAI
        """
        prompt = (
            f"Ты UX-эксперт. Проанализируй следующий текст сайта и создай UX-отчёт.\n\n"
            f"Текст:\n{text}\n\n"
            f"Верни отчёт в следующем формате:\n"
            f"Достоинства:\n"
            f"- [достоинство 1]\n"
            f"- [достоинство 2]\n"
            f"- [достоинство 3]\n\n"
            f"Слабые места:\n"
            f"- [слабое место 1]\n"
            f"- [слабое место 2]\n"
            f"- [слабое место 3]\n\n"
            f"Рекомендации по улучшению UX:\n"
            f"1. [рекомендация 1]\n"
            f"2. [рекомендация 2]\n"
            f"3. [рекомендация 3]\n"
            f"4. [рекомендация 4]\n"
            f"5. [рекомендация 5]\n\n"
            f"Сформулируй {num_recommendations} конкретных и практичных рекомендаций."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты опытный UX-эксперт с глубоким пониманием "
                        "пользовательского опыта, юзабилити и дизайна интерфейсов. "
                        "Ты анализируешь сайты и даёшь конструктивные рекомендации "
                        "по улучшению пользовательского опыта.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1500,
            )

            result_text = response.choices[0].message.content.strip()

            # Парсим ответ
            strengths = []
            weaknesses = []
            recommendations = []

            current_section = None
            lines = result_text.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Определяем текущую секцию
                if "достоинства" in line.lower() or "достоинство" in line.lower():
                    current_section = "strengths"
                    continue
                elif "слабые места" in line.lower() or "слабое место" in line.lower():
                    current_section = "weaknesses"
                    continue
                elif "рекомендации" in line.lower() or "рекомендация" in line.lower():
                    current_section = "recommendations"
                    continue

                # Добавляем элементы в соответствующие списки
                if current_section == "strengths":
                    if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                        item = line.lstrip("- •*").strip()
                        if item:
                            strengths.append(item)
                    elif line and not line.startswith("Достоинства"):
                        strengths.append(line)

                elif current_section == "weaknesses":
                    if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                        item = line.lstrip("- •*").strip()
                        if item:
                            weaknesses.append(item)
                    elif line and not line.startswith("Слабые"):
                        weaknesses.append(line)

                elif current_section == "recommendations":
                    # Рекомендации могут быть пронумерованы
                    if any(line.startswith(f"{i}.") for i in range(1, 10)):
                        item = line.split(".", 1)[1].strip() if "." in line else line
                        if item:
                            recommendations.append(item)
                    elif line.startswith("-") or line.startswith("•") or line.startswith("*"):
                        item = line.lstrip("- •*").strip()
                        if item:
                            recommendations.append(item)
                    elif line and not line.startswith("Рекомендации"):
                        recommendations.append(line)

            # Если не удалось распарсить структурированно, пытаемся извлечь из текста
            if not strengths and not weaknesses and not recommendations:
                # Пробуем найти рекомендации по номерам
                for line in lines:
                    line = line.strip()
                    if any(line.startswith(f"{i}.") for i in range(1, 10)):
                        item = line.split(".", 1)[1].strip() if "." in line else line
                        if item and len(item) > 10:  # Фильтруем слишком короткие
                            recommendations.append(item)

            # Ограничиваем количество рекомендаций
            recommendations = recommendations[:num_recommendations]

            # Если ничего не нашли, возвращаем сообщение об ошибке
            if not recommendations:
                recommendations = ["Не удалось сгенерировать рекомендации"]

            return {
                "strengths": strengths[:5] if strengths else ["Не указаны"],
                "weaknesses": weaknesses[:5] if weaknesses else ["Не указаны"],
                "recommendations": recommendations,
            }

        except Exception as e:
            raise Exception(f"Ошибка при генерации UX-отчёта через OpenAI: {str(e)}")

