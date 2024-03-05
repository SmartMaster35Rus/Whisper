import os
from pydub import AudioSegment
from tqdm import tqdm
import time
import shutil
import whisper
import tensorflow as tf

# Настройка TensorFlow для использования GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Путь к папке с файлами
directory = 'D:\\chankiogg\\folder_1_500_files'

# Путь к папке для обработанных файлов
processed_dir = 'D:\\chankiogg\\folder_1_500_files\\success'
os.makedirs(processed_dir, exist_ok=True)

# Получаем список всех файлов .ogg в папке
ogg_files = [f for f in os.listdir(directory) if f.endswith(".ogg")]

# Настройки конфигурации для модели whisper
whisper_config = {
    "model": "large-v3",    #выбираем модель : large-v2, large-v2, Base 
    "temperature": 0,   #не трогать
    "patience": 10, #не трогать
    "suppress_tokens": -1,  #не трогать
    "temperature_increment_on_fallback": 0.2,   #не трогать
    "compression_ratio_threshold": 2.4, #не трогать
    "logprob_threshold": -1,    #не трогать
    "no_speech_threshold": 0.6,     #не трогать
    "fp16": False,  #отключение FP16 потому что работает на FP32
    "device": "cuda"  # Указываем использование GPU
}

# Запоминаем текущее время
start_time = time.time()

# Загружаем базовую модель
model = whisper.load_model("large-v3")

# Итерируемся по файлам пакетами
batch_size = 10
num_batches = (len(ogg_files) + batch_size - 1) // batch_size

for i in tqdm(range(num_batches), desc="Обработка пакетов", unit="пакет"):
    batch_files = ogg_files[i * batch_size: (i + 1) * batch_size]

    for filename in tqdm(batch_files, desc="Обработка файлов", unit="файл", leave=False):
        try:
            # Путь к исходному файлу .ogg
            ogg_path = os.path.join(directory, filename)
            # Путь к конвертированному файлу .wav
            wav_path = os.path.join(directory, filename[:-4] + '.wav')
            # Путь к текстовому файлу с результатом
            txt_path = os.path.join(directory, filename[:-4] + '.txt')

            with tf.device('/GPU:0'):
                # Конвертируем .ogg в .wav
                audio = AudioSegment.from_ogg(ogg_path)
                audio.export(wav_path, format="wav")

                # Транскрибируем аудиофайл
                result = model.transcribe(wav_path)

                # Записываем результат в текстовый файл
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(filename[:-4] + ': \n"' + result["text"] + '"\n')

        except Exception as e:
            # Обрабатываем исключения
            print(f"Ошибка при обработке файла {filename}: {str(e)}")

        finally:
            # Перемещаем обработанные файлы в папку "Processed"
            shutil.move(ogg_path, os.path.join(processed_dir, filename))
            shutil.move(wav_path, os.path.join(processed_dir, filename[:-4] + '.wav'))
            shutil.move(txt_path, os.path.join(processed_dir, filename[:-4] + '.txt'))

# Вычисляем время выполнения скрипта
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Скрипт выполнен за {elapsed_time} секунд")
