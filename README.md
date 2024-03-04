# Whisper Large-v3

This repository is dedicated to Whisper Large-v3, a powerful tool for high-quality conversion of audio messages to text. It provides unique settings and configurations to ensure accurate and precise conversion of *.ogg or *.mp3 files to *.text format.

## Features

- Advanced audio processing algorithms for superior quality transcription
- Customizable settings to fine-tune the conversion process
- Support for various audio formats, including *.ogg and *.mp3
- Easy integration with existing systems and applications

## Installation

To use Whisper Large-v3, follow these steps:

"pip install -U openai-whisper"

"pip install git+https://github.com/openai/whisper.git "

"pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git"

"pip install setuptools-rust"

1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Configure the settings according to your specific requirements.
4. Run the application and start converting audio messages to text.

## Usage

Here's how you can use Whisper Large-v3 in your projects:

```python
import whisper

# Получаем список всех файлов .ogg в папке
ogg_files = [f for f in os.listdir(directory) if f.endswith(".ogg")]

# Загружаем базовую модель
model = whisper.load_model("large-v3")

# Конвертируем .ogg в .wav
audio = AudioSegment.from_ogg(ogg_path)
audio.export(wav_path, format="wav")

# Транскрибируем аудиофайл
result = model.transcribe(wav_path)

# Записываем результат в текстовый файл
with open(txt_path, 'w', encoding='utf-8') as f:
f.write(filename[:-4] + ': \n"' + result["text"] + '"\n')
```

Make sure to refer to the [documentation](https://github.com/yasaxil) for detailed instructions and additional examples.

## My optimal setting whisper config

```python
# Настройки конфигурации для модели whisper
whisper_config = {
    "model": "large-v3",
    "temperature": 0,
    "patience": 10,
    "suppress_tokens": -1,
    "temperature_increment_on_fallback": 0.2,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1,
    "no_speech_threshold": 0.6,
    "fp16": False
}
```
## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). For more information, please see the [LICENSE](LICENSE) file.

## Author

This repository is maintained by [yasaxil](https://github.com/yasaxil). If you have any questions or suggestions, feel free to reach out.
