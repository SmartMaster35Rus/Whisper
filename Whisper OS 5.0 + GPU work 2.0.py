import torch
import os
import time
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
from tqdm import tqdm
import logging
from moviepy.editor import VideoFileClip

# Настройка логирования
def setup_logging(output_dir):
    log_filename = os.path.join(output_dir, 'processing_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

# Проверка доступности CUDA
if torch.cuda.is_available():
    logging.info("Скрипт запущен на CUDA.")
else:
    logging.info("CUDA не доступна.")

# Функция для извлечения аудио из видео
def extract_audio_from_video(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path, codec='pcm_s16le')  # сохранение в формате WAV
    video.close()

# Функция для обработки аудиофайлов
def process_audio(filepath, output_dir):
    try:
        audio = AudioSegment.from_file(filepath)
        audio.export("temp.wav", format="wav")
        result = pipe("temp.wav")
        text = result["text"]
        filename = os.path.basename(filepath)
        txt_path = os.path.join(output_dir, filename[:-4] + '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info(f"Файл {filename} успешно обработан.")
    except Exception as e:
        logging.error(f"Ошибка при обработке файла {filename}: {str(e)}")

# Функция для обработки видеофайлов
def process_video(filepath, output_dir):
    try:
        audio_filepath = os.path.splitext(filepath)[0] + '.wav'
        extract_audio_from_video(filepath, audio_filepath)
        process_audio(audio_filepath, output_dir)
    except Exception as e:
        logging.error(f"Ошибка при обработке видео файла {os.path.basename(filepath)}: {str(e)}")

# Инициализация модели Whisper на GPU
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device="cuda:0"
)

# Получение пути к папке с файлами
input_directory = input("Введите путь к папке с файлами: ")
output_directory = os.path.join(input_directory, 'processed')
os.makedirs(output_directory, exist_ok=True)

# Настройка логирования в папку output_directory
setup_logging(output_directory)

# Поиск всех файлов .ogg и .mp4
files_to_process = []
for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith(".ogg") or file.endswith(".mp4"):
            full_path = os.path.join(root, file)
            files_to_process.append(full_path)

# Обработка файлов
start_time = time.time()

for filepath in tqdm(files_to_process, desc="Обработка файлов", unit="файл"):
    if filepath.endswith(".ogg"):
        process_audio(filepath, output_directory)
    elif filepath.endswith(".mp4"):
        process_video(filepath, output_directory)

end_time = time.time()
total_time = end_time - start_time
num_files = len(files_to_process)

logging.info(f"Обработка завершена. Время работы: {total_time:.2f} секунд. Обработано файлов: {num_files}")
