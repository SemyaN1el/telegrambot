import os
import asyncio
import logging
import numpy as np
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()


API_TOKEN = os.getenv('API_TOKEN')
ADMIN_CHAT_ID = os.getenv('ADMIN_CHAT_ID')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    MODEL_PATH = "my_roberta_model"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to("cpu")

    if hasattr(model.config, 'id2label'):
        sorted_labels = sorted(model.config.id2label.items(), key=lambda x: int(x[0]))
        ENGLISH_LABELS = [label for _, label in sorted_labels]

        LABEL_MAPPING = {
            "toxic": "Токсичный",
            "severe_toxic": "Сильно токсичный",
            "obscene": "Непристойный",
            "threat": "Угроза",
            "insult": "Оскорбление",
            "identity_hate": "Ненависть к личности"
        }

        RUSSIAN_LABELS = [LABEL_MAPPING.get(label, label) for label in ENGLISH_LABELS]

        logger.info(f"Реальные метки из config: {ENGLISH_LABELS}")
        logger.info(f"Русские метки для вывода: {RUSSIAN_LABELS}")
    else:
        ENGLISH_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        RUSSIAN_LABELS = [
            "Токсичный",
            "Сильно токсичный",
            "Непристойный",
            "Угроза",
            "Оскорбление",
            "Ненависть к личности"
        ]
        logger.warning("Метки не найдены в config, используем стандартный порядок")

    logger.info("Модель загружена")

except Exception as e:
    logger.critical(f"Ошибка загрузки модели: {e}")
    model = None
    tokenizer = None
    RUSSIAN_LABELS = ["Класс 1", "Класс 2", "Класс 3", "Класс 4", "Класс 5", "Класс 6"]

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

TOXICITY_THRESHOLD = 0.5


def analyze_text(text: str) -> str:
    if model is None or tokenizer is None:
        return "Модель недоступна"

    try:
        text = text.lower().strip()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length"
        ).to("cpu")

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

        toxic_indices = np.where(probs > TOXICITY_THRESHOLD)[0]

        if len(toxic_indices) == 0:
            result = "Класс: Корректный\n\nВероятности:\n"
            for label, prob in zip(RUSSIAN_LABELS, probs):
                result += f"{label}: {prob:.3f}\n"
            return result.strip()

        prediction = int(np.argmax(probs))
        result = f"Класс: {RUSSIAN_LABELS[prediction]}\n\nВероятности:\n"

        for label, prob in zip(RUSSIAN_LABELS, probs):
            result += f"{label}: {prob:.3f}\n"

        return result.strip()

    except Exception as e:
        logger.error(f"Ошибка анализа: {e}")
        return f"Ошибка: {str(e)}"


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Отправьте текст для анализа.")


@dp.message()
async def handle_message(message: types.Message):
    if message.text and not message.text.startswith('/'):
        try:
            await bot.forward_message(ADMIN_CHAT_ID, message.chat.id, message.message_id)
        except Exception as e:
            logger.error(f"Ошибка пересылки: {e}")

        result = analyze_text(message.text)
        await message.answer(result)


async def main():
    logger.info(f"Бот запущен с порогом токсичности: {TOXICITY_THRESHOLD}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())