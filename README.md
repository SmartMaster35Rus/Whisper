# Whisper Ai | Large-v3 #

## Описание скрипта для GitHub - README.md

### Описание
Данный скрипт представляет собой инструмент для обработки аудиофайлов формата .ogg и получения транскрипции речи с использованием модели Whisper. Он автоматически конвертирует файлы .ogg в .wav, выполняет распознавание речи и сохраняет результаты в текстовые файлы.

### Условия выполнения
Для успешного выполнения скрипта необходимо удовлетворять следующим условиям:
- Установленный Python 3.x.
- Установлены необходимые библиотеки: torch, transformers, pydub, datasets, tqdm.
- Доступ к модели Whisper (Whisper Large-v3) и процессору AutoProcessor.
- Наличие аудиофайлов формата .ogg, которые требуется обработать.

### Установки
Для работы скрипта необходимо установить следующие компоненты и зависимости:

- Python 3.x: Скачайте и установите Python 3.x с официального веб-сайта Python (https://www.python.org).

- Библиотеки: Установите необходимые библиотеки, запустив следующую команду в командной строке/терминале:
  ```
  pip install torch transformers pydub datasets tqdm
  ```

- Модель Whisper: Для использования модели Whisper Large-v3, необходимо иметь доступ к данной модели. Вы можете получить ее из репозитория моделей Hugging Face или другого источника.

- CudaToolkit: Если вы планируете использовать GPU для обработки аудиофайлов на модели Whisper, убедитесь, что у вас установлена версия CudaToolkit 12.3 или совместимая с вашей версией GPU.

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

**Примечание:** Указанные характеристики ПК являются примером и предоставлены для информационных целей. Реальные характеристики вашего ПК могут отличаться.

### Использование
1. Убедитесь, что все необходимые зависимости установлены и доступна модель Whisper Large-v3.
2. Поместите аудиофайлы формата .ogg, которые требуется обработать, в папку, указанную в переменной `input_directory` в скрипте.
3. Запустите скрипт и следуйте инструкциям, чтобы указать путь к папке с файлами.
4. Результаты обработки будут сохранены в папке `success`, созданной внутри папки с файлами исходных аудиофайлов. Каждый обработанный файл будет иметь соответствующий текстовый файл с результатом транскрипции.

### Пример
Пример использования скрипта:

1.```bash
$ python script.py
```

2. Введите путь к папке с аудиофайлами: `/путь/к/папке/с/аудиофайлами`

3. Подождите, пока скрипт обработает все аудиофайлы.

4. Результаты будут сохранены в папке `/путь/к/папке/с/аудиофайлами/success`.

### Важно!
Убедитесь, что вы имеете все необходимые права доступа и разрешения для работы с аудиофайлами и сохранения результатов транскрипции.

### Автор
Автор скрипта: SmartMaster35Rus

### Вклад
Если вы хотите внести свой вклад в развитие этого скрипта, пожалуйста, создайте pull request или свяжитесь с автором для обсуждения возможных улучшений.

### Обратная связь
Если у вас есть вопросы, предложения или обратная связь по скрипту, пожалуйста, свяжитесь с автором по следующему адресу электронной почты: SmartMaster35Rus@yandex.ru.
```

# Лицензия #
Этот скрипт распространяется под лицензией [MIT License](https://github.com/yasaxil/Whisper/blob/main/LICENSE.md).
