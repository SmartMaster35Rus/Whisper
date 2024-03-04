import os
from pydub import AudioSegment
import subprocess
from tqdm import tqdm
import time
import shutil
from concurrent.futures import ThreadPoolExecutor

# Путь к директории с файлами
directory = 'D:\\chankiogg\\folder_1_500_files'

# Путь к директории для обработанных файлов
processed_dir = 'D:\\chankiogg\\folder_1_500_files\\Success'
os.makedirs(processed_dir, exist_ok=True)

# Получаем список всех .ogg файлов в директории
ogg_files = [f for f in os.listdir(directory) if f.endswith(".ogg")]

# Запоминаем текущее время
start_time = time.time()

# Функция для обработки файла
def process_file(filename):
    # Путь к исходному файлу
    ogg_path = os.path.join(directory, filename)
    # Путь к конвертированному файлу
    wav_path = os.path.join(directory, filename[:-4] + '.wav')
    # Путь к выходному текстовому файлу
    txt_path = os.path.join(directory, filename[:-4] + '.txt')
    
    # Конвертируем ogg в wav
    audio = AudioSegment.from_ogg(ogg_path)
    audio.export(wav_path, format="wav")
    
    # Выполняем распознавание речи с помощью whisper
    result = subprocess.run(['whisper', '--model', 'large-v3', wav_path], stdout=subprocess.PIPE)
    
    # Записываем результат в текстовый файл
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(filename[:-4] + ': \n"' + result.stdout.decode('utf-8') + '"\n')
    
    # Перемещаем обработанные файлы в папку "Выполненные"
    shutil.move(ogg_path, os.path.join(processed_dir, filename))
    shutil.move(wav_path, os.path.join(processed_dir, filename[:-4] + '.wav'))
    shutil.move(txt_path, os.path.join(processed_dir, filename[:-4] + '.txt'))

# Создаем пул потоков с ограничением на количество одновременно выполняющихся потоков
max_workers = 1  # Максимальное количество одновременно выполняющихся потоков (1 для последовательной обработки)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Перебираем все файлы в директории и отправляем задачу на обработку в пул потоков
    futures = [executor.submit(process_file, filename) for filename in ogg_files]
    # Дожидаемся завершения всех задач
    for future in tqdm(futures, desc="Processing files", unit="file"):
        future.result()

# Вычисляем время, затраченное на выполнение скрипта
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Скрипт выполнился за {elapsed_time} секунд")

# Путь к директории, в которой находятся файлы
source_directory = os.path.dirname(os.path.realpath(__file__))  # это путь к папке скрипта

# Получаем список всех файлов в исходной директории
files = os.listdir(source_directory)

# Определяем список расширений файлов, которые нужно переместить
extensions = ['.json', '.srt', '.tsv', '.txt']

# Перебираем все файлы
for file in files:
    # Проверяем, имеет ли файл одно из нужных расширений
    if any(file.endswith(extension) for extension in extensions):
        # Перемещаем файл
        shutil.move(os.path.join(source_directory, file), os.path.join(processed_dir, file))
