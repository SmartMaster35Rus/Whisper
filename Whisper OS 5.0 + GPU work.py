import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
import time
import os
from tqdm import tqdm

# Проверка доступности CUDA
if torch.cuda.is_available():
    print("Скрипт запущен на CUDA.")
else:
    print("CUDA не доступна.")

# Функция для обработки пакета файлов
def process_files(files, output_dir):
    for filepath in files:
        try:
            filename = os.path.basename(filepath)
            # Путь к конвертированному файлу .wav
            wav_path = os.path.join(output_dir, filename[:-4] + '.wav')
            # Путь к текстовому файлу с результатом
            txt_path = os.path.join(output_dir, filename[:-4] + '.txt')

            with torch.no_grad():
                # Конвертируем .ogg в .wav
                audio = AudioSegment.from_file(filepath)
                audio.export(wav_path, format="wav")

                # Транскрибируем аудиофайл
                result = pipe(wav_path)

                # Записываем результат в текстовый файл
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(filename[:-4] + ': \n"' + result["text"] + '"\n')

        except Exception as e:
            # Обрабатываем исключения
            print(f"Ошибка при обработке файла {filename}: {str(e)}")

# Выбор папки с файлами для обработки
input_directory = input("Введите путь к папке с файлами: ")

# Создание папки для обработанных файлов
output_directory = os.path.join(input_directory, 'processed')
os.makedirs(output_directory, exist_ok=True)

# Рекурсивный поиск всех файлов .ogg в папке и подпапках
ogg_files = []
for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith(".ogg"):
            full_path = os.path.join(root, file)
            ogg_files.append(full_path)

# Инициализация модели Whisper на GPU
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    device="cuda:0",
)

# Итерация по пакетам файлов для обработки
batch_size = 10
num_batches = (len(ogg_files) + batch_size - 1) // batch_size

start_time = time.time()  # Засекаем время выполнения скрипта

for i in tqdm(range(num_batches), desc="Обработка пакетов", unit="пакет"):
    batch_files = ogg_files[i * batch_size: (i + 1) * batch_size]
    process_files(batch_files, output_directory)

end_time = time.time()  # Засекаем время окончания выполнения скрипта

total_time = end_time - start_time  # Вычисляем общее время работы скрипта
num_files = len(ogg_files)  # Количество обработанных файлов

print(f"Обработка завершена. Время работы: {total_time:.2f} секунд. Обработано файлов: {num_files}")
