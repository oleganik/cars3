

from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import Image 

INPUT_SHAPE = (96, 54, 3)

model = load_model('cars3.h5')  # Инициализация модели


def process(image_file):
    
    # Открытие обрабатываемого файла
    image = Image.open(BytesIO(image_file)) 

    # Изменение размера изображения в соответствии со входом сети
    resized_image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0])) 

    # Подгонка формы тензора для подачи в модель
    array = np.array(resized_image)[np.newaxis, ...] 

    # Запуск предсказания
    prediction_array = model.predict(array)[0] 

    # Возврат предсказания сети в виде текстовой переменной, хранящей список
    return str(list(prediction_array))
