# 🚀 Smart Toxic Classifier

## 📌 Описание проекта

**Smart Toxic Classifier** — система автоматической классификации русскоязычных комментариев на токсичные и нетоксичные.

Цель проекта — разработать полноценное NLP-приложение, которое способно:

- анализировать текстовые сообщения;
- определять вероятность токсичности;
- возвращать результат классификации;
- предоставлять доступ к модели через REST API.

Модель получает текст сообщения и выполняет бинарную классификацию:

- `0` — Non-toxic
- `1` — Toxic

Проект выполнен как NLP-задача с использованием трансформерной модели **RuBERT**, а также включает backend-инфраструктуру на FastAPI.

---

# 🧠 Используемые технологии

## Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- RuBERT
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## Backend

- FastAPI
- Pydantic
- Uvicorn

## Research

- Jupyter Notebook


---

# 📊 Данные

В проекте используется датасет русскоязычных комментариев с разметкой токсичности.

Количество данных:

```
14412 комментариев
```

Распределение классов:

| Класс | Количество |
|------|------------|
| Non-toxic | 9586 |
| Toxic | 4826 |


Перед обучением был проведён анализ данных:

- проверка пропущенных значений;
- анализ баланса классов;
- анализ длины комментариев;
- анализ количества токенов.


Распределение классов:

![Class Distribution](images/class_distribution.png)


Распределение количества токенов:

![Token Distribution](images/token_distribution.png)


После анализа был выбран параметр:

```python
max_length = 128
```

Это позволяет сохранить большую часть информации из комментариев и уменьшить время обучения.

---

# 🤖 Используемые модели

В проекте сравниваются два подхода.


# 1. TF-IDF + Logistic Regression

Базовая модель машинного обучения.

Использует статистическое представление текста через TF-IDF признаки.


Преимущества:

- быстрое обучение;
- простая интерпретация;
- низкое потребление ресурсов.


Недостатки:

- отсутствие понимания контекста;
- слова рассматриваются независимо друг от друга.


---

# 2. RuBERT

Основная модель проекта.


Используется предобученная трансформерная модель:

```
cointegrated/rubert-tiny2
```


Модель была дообучена на датасете токсичных комментариев.


Преимущества:

- учитывает контекст слов;
- анализирует смысл предложения;
- лучше работает со сложными языковыми конструкциями.


После обучения модель сохранена:

```
models/

├── config.json
├── model.safetensors
├── tokenizer_config.json
└── tokenizer.json
```

---

# ⚙️ Обучение модели

Процесс обучения:

1. Загрузка и анализ датасета.
2. Очистка данных.
3. Разделение выборки.
4. Токенизация через RuBERT tokenizer.
5. Подготовка Dataset.
6. Fine-tuning модели.
7. Оценка качества.


Разделение данных:

```
Train: 11529 объектов

Test: 2883 объекта
```


Используемые метрики:

- Accuracy
- Precision
- Recall
- F1-score


Основная метрика:

**F1-score**

Так как она учитывает баланс между Precision и Recall.

---

# 📈 Результаты моделей


| Model | Accuracy | Precision | Recall | F1-score |
|-|-|-|-|-|
| TF-IDF + Logistic Regression | 0.876 | 0.807 | 0.813 | 0.810 |
| RuBERT | 0.915 | 0.865 | 0.885 | 0.875 |


---

# 📌 Вывод

RuBERT показывает значительно лучшее качество классификации.

По сравнению с TF-IDF:

- Accuracy:
```
0.876 → 0.915
```

- Precision:
```
0.807 → 0.865
```

- Recall:
```
0.813 → 0.885
```

- F1-score:
```
0.810 → 0.875
```


Улучшение достигается благодаря способности трансформерной модели учитывать контекст предложения.

---

# 📊 Анализ качества модели


Для оценки модели были построены:


- Classification Report;
- Confusion Matrix;
- Precision-Recall Curve.


## Classification Report


Файл:

```
images/classification_report.csv
```


Результаты:

| Class | Precision | Recall | F1-score |
|-|-|-|-|
| Non-toxic | 0.941 | 0.931 | 0.936 |
| Toxic | 0.865 | 0.885 | 0.875 |


## Confusion Matrix


![Confusion Matrix](images/confusion_matrix.png)


## Precision-Recall Curve


![Precision Recall Curve](images/precision_recall_curve.png)


---

# 🔍 Пример работы модели


Вход:

```
ты полный идиот
```


Ответ:

```
Toxic: True

Confidence: 0.998
```


Другие примеры:


```
привет
→ Non-toxic
→ confidence: 0.011


спасибо большое
→ Non-toxic
→ confidence: 0.001


ненавижу тебя
→ Toxic
→ confidence: 0.993
```

---

# 📂 Структура проекта


```
smart-toxic-classifier/

│
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI приложение
│   ├── model.py         # загрузка модели и predict()
│   └── schemas.py       # Pydantic схемы
│
├── data/
│   └── labeled.csv
│
├── images/
│   ├── class_distribution.png
│   ├── token_distribution.png
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   ├── classification_report.csv
│   ├── rubert_metrics.csv
│   └── model_comparison.csv
│
├── models/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── tokenizer.json
│
├── notebooks/
│   └── smart-toxic-classifier.ipynb
│
├── scripts/
│   └── inference.py     # CLI предсказания
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# ⚙️ Установка


## 1. Клонирование проекта


```bash
git clone https://github.com/freedal1/smart-toxic-classifier.git

cd smart-toxic-classifier
```


## 2. Создание окружения


```bash
python -m venv .venv
```


Активация:


MacOS/Linux:

```bash
source .venv/bin/activate
```


Windows:

```bash
.venv\Scripts\activate
```


## 3. Установка зависимостей


```bash
pip install -r requirements.txt
```


---

# 🚀 Запуск проекта


## Jupyter Notebook


```bash
jupyter notebook notebooks/smart-toxic-classifier.ipynb
```


---

# FastAPI


Запуск:


```bash
uvicorn app.main:app --reload
```


После запуска:


```
http://127.0.0.1:8000
```


Swagger документация:


```
http://127.0.0.1:8000/docs
```


---

# 🔥 REST API


## Endpoint


```
POST /predict
```


Пример запроса:


```json
{
"text": "ты полный идиот"
}
```


Ответ:


```json
{
"text": "ты полный идиот",
"toxic": true,
"confidence": 0.998
}
```


---

# 💻 CLI предсказания


Одно сообщение:


```bash
python scripts/inference.py "ты полный идиот"
```


Результат:


```
📝 Текст: ты полный идиот

⚠️ Токсичный: ДА

📊 Уверенность: 0.998
```


Интерактивный режим:


```bash
python scripts/inference.py
```


---

# 🚧 Дальнейшее развитие


Импорт в другом скрипте:


```bash
from app.model import predict

result = predict("Привет, как дела?")
print(result)
```

Выполнено:

- ✅ обучение RuBERT;
- ✅ сравнение моделей;
- ✅ создание FastAPI сервиса;
- ✅ REST API;
- ✅ вынос логики модели в отдельный модуль;
- ✅ CLI inference.


Планы:

- ⏳ Docker контейнеризация;
- ⏳ создание веб-интерфейса;
- ⏳ оптимизация скорости инференса;
- ⏳ публикация модели на Hugging Face Hub.


---

# 👨‍💻 Автор

Igor Matevosov