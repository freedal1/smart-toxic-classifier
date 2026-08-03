import sys
import os

# Добавляем путь к проекту, чтобы импортировать app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import predict

if __name__ == "__main__":
    # Если передан аргумент из командной строки
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = predict(text)
        print(f"📝 Текст: {result['text']}")
        print(f"⚠️  Токсичный: {'ДА' if result['toxic'] else 'НЕТ'}")
        print(f"📊 Уверенность: {result['confidence']:.3f}")
    else:
        # Интерактивный режим
        print("🔍 Умный модератор комментариев")
        print("Введите текст для проверки (или 'exit' для выхода):")
        while True:
            text = input("\n👉 Ваш текст: ")
            if text.lower() == 'exit':
                break
            result = predict(text)
            status = "🚨 ТОКСИЧНЫЙ" if result['toxic'] else "✅ БЕЗОПАСНЫЙ"
            print(f"{status} | Уверенность: {result['confidence']:.3f}")