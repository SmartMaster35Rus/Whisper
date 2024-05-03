import os
import logging
import subprocess
import time
import tracemalloc
from pydub import AudioSegment
import streamlit as st
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

def setup_logging(output_dir):
    log_filename = os.path.join(output_dir, 'process_log.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return log_filename

def create_output_folder(base_dir, folder_name):
    output_dir = os.path.join(base_dir, folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def extract_audio_from_video(video_path, output_audio_path):
    command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {output_audio_path}"
    subprocess.call(command, shell=True)

def process_audio(filepath, output_dir):
    filename = os.path.basename(filepath)
    try:
        audio = AudioSegment.from_file(filepath)
        temp_audio_path = os.path.join(output_dir, f"temp_{filename}.wav")
        audio.export(temp_audio_path, format="wav")
        result = pipe(temp_audio_path)
        text = result["text"]
        txt_path = os.path.join(output_dir, filename[:-4] + '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        os.remove(temp_audio_path)
        logging.info(f"Файл {filename} успешно обработан.")
    except Exception as e:
        logging.error(f"Ошибка при обработке файла {filename}: {str(e)}")

def process_video(filepath, output_dir):
    try:
        audio_filepath = os.path.splitext(filepath)[0] + '.wav'
        extract_audio_from_video(filepath, audio_filepath)
        process_audio(audio_filepath, output_dir)
    except Exception as e:
        logging.error(f"Ошибка при обработке видео файла {os.path.basename(filepath)}: {str(e)}")

def kill_process():
    subprocess.call(["taskkill", "/F", "/T", "/PID", str(os.getppid())])

def get_files_from_directory(directory_path, file_types):
    files = []
    for root, dirs, files_in_dir in os.walk(directory_path):
        for file in files_in_dir:
            if any(file.endswith(f".{ext}") for ext in file_types):
                files.append(os.path.join(root, file))
    return files

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

st.title('Whisper Ai Web GUI 4.2')

option = st.selectbox('Выберите способ указания файлов:', ('Из директории', 'Из файла со списком'))
file_types = st.multiselect('Выберите типы файлов для обработки:', ['mp4', 'ogg'], default=['mp4', 'ogg'])
directory_path = st.text_input("Введите путь к директории:")
folder_name = st.text_input("Введите имя папки для сохранения обработанных файлов:")
process_button = st.button('Обработать файлы')

if process_button and directory_path and folder_name:
    output_dir = create_output_folder(directory_path, folder_name)
    log_filename = setup_logging(output_dir)
    files_to_process = get_files_from_directory(directory_path, file_types)
    total_size = sum(os.path.getsize(f) for f in files_to_process)
    total_size_mb = total_size / (1024 * 1024)
    processed_size_mb = 0
    start_time = time.time()
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, filepath in enumerate(files_to_process):
        file_size = os.path.getsize(filepath)
        file_size_mb = file_size / (1024 * 1024)
        processed_size_mb += file_size_mb
        if filepath.endswith(".ogg"):
            process_audio(filepath, output_dir)
        elif filepath.endswith(".mp4"):
            process_video(filepath, output_dir)
        progress_bar.progress((i + 1) / len(files_to_process))
        status_text.text(f"Обработка файла {i+1}/{len(files_to_process)}. Обработано {processed_size_mb:.2f} MB из {total_size_mb:.2f} MB")
    end_time = time.time()
    total_time = end_time - start_time
    num_files = len(files_to_process)
    logging.info(f"Обработка завершена. Время работы: {total_time:.2f} секунд. Обработано файлов: {num_files}. Обработано {processed_size_mb:.2f} MB из {total_size_mb:.2f} MB")
    st.write(f"Обработка завершена. Время работы: {total_time:.2f} секунд. Обработано файлов: {num_files}. Обработано {processed_size_mb:.2f} MB из {total_size_mb:.2f} MB")

# Добавление кнопки "Завершить работу"
if st.button('Завершить работу'):
    kill_process()

# Добавление кнопки "Посмотреть лог"
if st.button('Посмотреть лог'):
    with open(log_filename, 'r') as log_file:
        st.text_area("Логи:", log_file.read(), height=300)

# Добавление информации о использовании памяти
def display_memory_usage():
    tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    st.write(f"Текущее использование памяти: {current / 10**6:.2f} MB")
    st.write(f"Пиковое использование памяти: {peak / 10**6:.2f} MB")

display_memory_usage()