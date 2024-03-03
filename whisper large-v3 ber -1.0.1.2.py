import os
from pydub import AudioSegment
import subprocess
from tqdm import tqdm
import time
import shutil

# Путь к директории с файлами
directory = 'D:\\chankiogg\\folder_1_500_files'

# Путь к директории для обработанных файлов
processed_dir = os.path.join(directory, 'Success')
os.makedirs(processed_dir, exist_ok=True)

# Получаем список всех .ogg файлов в директории
ogg_files = [f for f in os.listdir(directory) if f.endswith(".ogg")]

# Запоминаем текущее время
start_time = time.time()

# Инициализируем tqdm для отображения прогресса
pbar = tqdm(ogg_files, desc="Processing files", unit="file")

# Перебираем все файлы в директории
for filename in pbar:
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
    result = subprocess.run(['whisper', '--model', 'large-v3', 'fp16=false', wav_path], stdout=subprocess.PIPE)
    
    # Записываем результат в текстовый файл
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(filename[:-4] + ': \n"' + result.stdout.decode('utf-8') + '"\n')
    
    # Перемещаем обработанный файл в папку"Выполненные"
    shutil.move(ogg_path, os.path.join(processed_dir, filename))
    shutil.move(wav_path, os.path.join(processed_dir, filename[:-4] + '.wav'))
    shutil.move(txt_path, os.path.join(processed_dir, filename[:-4] + '.txt'))

# Вычисляем время, затраченное на выполнение скрипта
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Скрипт выполнился за {elapsed_time} секунд")

# Путь к директории, в которой находятся файлы
source_directory = os.path.dirname(os.path.realpath(__file__))  # это путь к папке скрипта

# Определяем список расширений файлов, которые нужно переместить
extensions = ['.json', '.srt', '.tsv', '.txt'] 

# Перебираем все файлы
for file in files:
    # Проверяем, имеет ли файл одно из нужных расширений
    if any(file.endswith(extension) for extension in extensions):
        # Перемещаем файл
        shutil.move(os.path.join(source_directory, file), os.path.join(processed_dir, file))
