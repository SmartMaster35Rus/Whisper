# Whisper Ai | Large-v3 #

- [Whisper audio transcribe ](https://github.com/SmartMaster35Rus/Whisper/blob/main/Whisper%20OS%205.0%20%2B%20GPU%20work.py)

- [Whisper 2.0 audio + video transcribe ](https://github.com/SmartMaster35Rus/Whisper/blob/main/Whisper%20OS%205.0%20%2B%20GPU%20work%202.0.py) 


### Характеристики ПК
Данный скрипт был разработан и протестирован на следующей конфигурации ПК:

|  Конфигурация  |  Детали спецификаций  |
|----------------|----------------------|
|  Процессор     |  Intel® Core™ i9-14900KF @3.20Ghz  |
|  Память        |  128 ГБ DDR4 4200 МГц (32+32+32+32)  |
|  Диск          |  M.2 PCIe SSD Samsung SSD 980 PRO 1000 ГБ  |
|  Диск          |  M.2 PCIe SSD XPG GAMMIX S11 Pro 1000 ГБ |
|  Дискретная графика  |  NVIDIA GeForce RTX 4090 24 ГБ  |
|  Модель Whisper  |  Whisper Large-v3  |
|  CudaToolkit   |  ver.12.3  |
|  OS   |  Windows 11 Pro |

**Примечание:** Указанные характеристики ПК являются примером и предоставлены для информационных целей. Реальные характеристики вашего ПК могут отличаться.

### Описание

  *Данный скрипт представляет собой инструмент для обработки аудиофайлов формата .ogg и получения транскрипции речи с использованием модели Whisper. Он автоматически конвертирует файлы .ogg в .wav, выполняет распознавание речи и сохраняет результаты в текстовые файлы.*

### 1. Условия выполнения

  Для успешного выполнения скрипта необходимо удовлетворять следующим условиям:
  
    - Установленный Python 3.x.
    - Установлены необходимые библиотеки: torch, transformers, pydub, datasets, tqdm, moviepy
    - Доступ к модели Whisper (Whisper Large-v3) и процессору AutoProcessor.
    - Наличие аудиофайлов формата .ogg, которые требуется обработать.
    - Miniconda

### 2. Установки

  Для работы скрипта необходимо установить следующие компоненты и зависимости:
  
  - Python 3.x: Скачайте и установите Python 3.x с официального веб-сайта Python (https://www.python.org).
  
  - [Miniconda](https://docs.anaconda.com/free/miniconda/index.html)
  
  - Библиотеки: Установите необходимые библиотеки, запустив следующую команду в командной строке/терминале:

**После успешной установки Python и MiniConda**

- Находим в пуск и запускаем **Anaconda Powershell Prompt (miniconda3)**
  
**Для изоляции проекта рекомендуется создать новое виртуальное окружение. Выполните следующие команды в **Anaconda Powershell Prompt (miniconda3)** :**

```shell
conda create -n whisper python=3.9 ##очень важно именно версия 3.9
conda activate whisper

pip install torch transformers pydub datasets tqdm moviepy
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu121
python -m pip install "tensorflow<2.11" ##Верссии tensorflow выше 2.10 не работают на Windows поэтому ставим версию ниже 2.11
pip install jupyter notebook
```


### 3. Проверка настройки TensorFlow GPU

Для использования GPU в TensorFlow, убедитесь, что настройка выполнена корректно. Выполните следующую команду:

```python
  python -c "import tensorflow as whisper; print(whisper.config.list_physical_devices('GPU'))"
```  
  если все настроено правильно вы получаете ответ :
  
```
  [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 4. Запуск Jupyter Notebook

Запустите Jupyter Notebook, чтобы проверить доступность GPU. Введите следующую команду в терминале:
  ```python
  conda activate whisper ##активируем наше виртуальное окружение 
  pip install ipykernel
  python -m ipykernel install --user --name=whisper ##Добавляем наше окружение в Jupyter Notebook
  jupyter notebook ##Откроется браузер сос тратовой страницей Jupyter Notebook 
  ```

Выберите виртуальное окружение whisper (сверху справа) в интерфейсе Jupyter Notebook и перейдите в него. 
Вводим команду:

```python
import tensorflow as whisper
gpus = whisper.config.list_physical_devices('GPU')
print ( gpus )

```
Если все настроено правильно и GPU работает вы получаете ответ : 

```python
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
- Модель Whisper: Для использования модели Whisper Large-v3, необходимо иметь доступ к данной модели. Вы можете получить ее из репозитория моделей Hugging Face или другого источника.

- CudaToolkit: Если вы планируете использовать GPU для обработки аудиофайлов на модели Whisper, убедитесь, что у вас установлена версия CudaToolkit 12.3 или совместимая с вашей версией GPU.

### 5. Использование

1. Убедитесь, что все необходимые зависимости установлены и доступна модель Whisper Large-v3.
2. Поместите аудиофайлы формата .ogg, которые требуется обработать, в папку, указанную в переменной `input_directory` в скрипте.
3. Запустите скрипт и следуйте инструкциям, чтобы указать путь к папке с файлами.
4. Результаты обработки будут сохранены в папке `success`, созданной внутри папки с файлами исходных аудиофайлов. Каждый обработанный файл будет иметь соответствующий текстовый файл с результатом транскрипции.

### 6. Пример

Пример использования скрипта:

$ python Whisper OS 5.0 + GPU work.py
```

  1. Введите путь к папке с аудиофайлами: `/путь/к/папке/с/аудиофайлами` > Формат аудио *.ogg
  
  2. Подождите, пока скрипт обработает все аудиофайлы. `Процесс обработки будет отображаться в прогресс баре и отображаеть кол-во обработанных и оставшихся файлов`
  
  3. Результаты будут сохранены в папке `/путь/к/папке/с/аудиофайлами/success`.

```

**Важно!**
Убедитесь, что вы имеете все необходимые права доступа и разрешения для работы с аудиофайлами и сохранения результатов транскрипции.

**Автор**
Автор скрипта: SmartMaster35Rus

**Вклад**
Если вы хотите внести свой вклад в развитие этого скрипта, пожалуйста, создайте pull request или свяжитесь с автором для обсуждения возможных улучшений.

**Обратная связь**
Если у вас есть вопросы, предложения или обратная связь по скрипту, пожалуйста, свяжитесь с автором по следующему адресу электронной почты: SmartMaster35Rus@yandex.ru.

## Лицензия
Этот скрипт распространяется под лицензией [MIT License](https://github.com/SmartMaster35Rus/Whisper/blob/main/LICENSE.md).
