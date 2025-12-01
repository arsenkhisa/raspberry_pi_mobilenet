# MobileNetV2

Этот проект реализует систему классификации изображений на Raspberry Pi с использованием нейросети **MobileNetV2**, экспортированной в формат **ONNX**, а также механизма DNN, встроенного в OpenCV. Решение работает полностью оффлайн и не требует TensorFlow или tflite-runtime, что делает его совместимым с современными версиями Python и Raspberry Pi OS.

## Возможности

- Загрузка предобученной сети **MobileNetV2**
- Работа с форматом **ONNX**
- Классификация изображений и вывод **Top-5** предсказаний
- Сохранение изображения с подписью результата
- Работа на Raspberry Pi 3 Model A++ и выше

## Используемые технологии

- Python 3.11 / 3.13
- OpenCV (модуль dnn)
- NumPy
- MobileNetV2 (ONNX Model Zoo)
- Raspberry Pi OS (2025)

## Структура проекта

mobilenet_opencv/
│
├── mobilenetv2.onnx          # модель MobileNetV2
├── imagenet_classes.txt       # 1000 классов ImageNet
├── mobilenet_test.py          # основной код классификации
├── dog.jpg                    # тестовое изображение
├── result.jpg                 # результат с подписью
└── venv/                      # виртуальное окружение

## Установка и запуск

### 1. Клонирование проекта
git clone https://github.com/arsenkhisa/raspberry_pi_mobilenet.git
cd mobilenet_opencv

### 2. Создание виртуального окружения
python -m venv venv
source venv/bin/activate

### 3. Установка зависимостей
pip install python3-numpy python3-opencv

### 4. Запуск классификации
python3 mobilenet_test.py

## Пример вывода

Top-5 predictions:
172: whippet                        0.4710
173: Ibizan hound                   0.1476
171: Italian greyhound              0.0988
158: toy terrier                    0.0782
166: Walker hound                   0.0238