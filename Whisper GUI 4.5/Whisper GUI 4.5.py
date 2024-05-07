import os
import logging
import subprocess
import time
import tracemalloc
import streamlit as st

from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def create_output_folder(base_path, folder_name):
    """Создать папку для вывода обработанных файлов."""
    output_path = os.path.join(base_path, folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

def setup_logging(output_dir):
    """Настройка логирования в файл."""
    log_filename = os.path.join(output_dir, 'processing.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    return log_filename

def get_files_from_directory(directory_path, file_types):
    """Получить список файлов определенных типов из директории."""
    files = []
    for root, dirs, files_in_dir in os.walk(directory_path):
        for file in files_in_dir:
            if any(file.endswith(f".{ext}") for ext in file_types):
                files.append(os.path.join(root, file))
    return files

def get_processed_files(output_dir):
    """Получить список уже обработанных файлов в указанной папке."""
    processed_files = set()
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.txt'):
                processed_file_name = os.path.splitext(os.path.basename(file))[0]
                processed_files.add(processed_file_name)
    return processed_files

def filter_unprocessed_files(files_to_process, processed_files):
    """Исключить уже обработанные файлы из списка файлов для обработки."""
    unprocessed_files = []
    for file in files_to_process:
        base_name = os.path.splitext(os.path.basename(file))[0]
        if base_name not in processed_files:
            unprocessed_files.append(file)
    return unprocessed_files

def extract_audio_from_video(video_path, output_audio_path):
    """Извлечь аудио из видео файла."""
    command = f"ffmpeg -y -i \"{video_path}\" -b:a 192k -vn \"{output_audio_path}\""
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

def kill_process():
    """Завершение процесса."""
    subprocess.call(["taskkill", "/F", "/T", "/PID", str(os.getppid())])

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

st.title('Whisper Ai Web GUI 4.5.1b')

option = st.selectbox('Выберите способ указания файлов:', ('Из директории', 'Из файла со списком'))
file_types = st.multiselect('Выберите типы файлов для обработки:', ['mp4', 'ogg', 'mp3'], default=['mp4', 'ogg'])
directory_path = st.text_input("Введите путь к директории:")
folder_name = st.text_input("Введите имя папки для сохранения обработанных файлов:")
process_button = st.button('Обработать файлы')

if process_button and directory_path and folder_name:
    output_dir = create_output_folder(directory_path, folder_name)
    log_filename = setup_logging(output_dir)
    
    files_to_process = get_files_from_directory(directory_path, file_types)
    processed_files = get_processed_files(output_dir)
    files_to_process = filter_unprocessed_files(files_to_process, processed_files)

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
        if filepath.endswith(".mp4"):
            audio_filepath = os.path.splitext(filepath)[0] + '.mp3'
            extract_audio_from_video(filepath, audio_filepath)
            process_audio(audio_filepath, output_dir)
        elif filepath.endswith(".mp3") or filepath.endswith(".ogg"):
            process_audio(filepath, output_dir)
        progress_bar.progress((i + 1) / len(files_to_process))
        status_text.text(f"Обработка файла {i+1}/{len(files_to_process)}. Обработано {processed_size_mb:.2f} MB из {total_size_mb:.2f} MB")
    end_time = time.time()
    total_time = end_time - start_time
    num_files = len(files_to_process)
    logging.info(f"Обработка завершена. Время работы: {total_time:.2f} секунд. Обработано файлов: {num_files}. Обработано {processed_size_mb:.2f} MB из {total_size_mb:.2f} MB")
    st.write(f"Обработка завершена. Время работы: {total_time:.2f} секунд. Обработано файлов: {num_files}. Обработано {processed_size_mb:.2f} MB из {total_size_mb:.2f} MB")

if st.button('Завершить работу'):
    kill_process()

if st.button('Посмотреть лог'):
    with open(log_filename, 'r') as log_file:
        st.text_area("Логи:", log_file.read(), height=300)

def display_memory_usage():
    tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    st.write(f"Текущее использование памяти: {current / 10**6:.2f} MB")
    st.write(f"Пиковое использование памяти: {peak / 10**6:.2f} MB")

display_memory_usage()
