import os
import numpy as np
import librosa
import pickle
from flask import Flask, request, jsonify, render_template
from tensorflow.keras import layers, models
import soundfile as sf
import io

# Initialize the Flask application
app = Flask(__name__)

# --- Model and Preprocessing Configuration ---
SAMPLING_RATE = 20000
INPUT_LENGTH = 30225
N_CLASSES = 10
CLASS_NAMES = ['chainsaw', 'clock_tick', 'cracking_fire', 'crying_baby', 'dog', 'helicaptor', 'rain', 'roster', 'sea_waves', 'sneezing']

# --- Keras Model Definition ---
def create_acdnet(i_len, n_cls, x=8, SFEB_PS=49, TFEB_PS=[2,2,2,2,2,2]):
    """Defines the ACDNet model architecture."""
    model = models.Sequential()
    model.add(layers.Input(shape=(1, i_len, 1)))

    # SFEB
    model.add(layers.Conv2D(x, (1, 9), strides=(1, 2), padding="valid", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())

    model.add(layers.Conv2D(x * (2**3), (1, 5), strides=(1, 2), padding="valid", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())

    model.add(layers.MaxPooling2D(pool_size=(1, SFEB_PS), strides=(1, SFEB_PS)))
    model.add(layers.Permute((3, 2, 1)))

    # TFEB blocks
    model.add(layers.Conv2D(x * (2**2), (3, 3), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(TFEB_PS[0], TFEB_PS[0]), strides=(TFEB_PS[0], TFEB_PS[0])))

    model.add(layers.Conv2D(x * (2**3), (3, 3), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.Conv2D(x * (2**3), (3, 3), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(TFEB_PS[1], TFEB_PS[1]), strides=(TFEB_PS[1], TFEB_PS[1])))

    model.add(layers.Conv2D(x * (2**4), (3, 3), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.Conv2D(x * (2**4), (3, 3), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(TFEB_PS[2], TFEB_PS[2]), strides=(TFEB_PS[2], TFEB_PS[2])))

    model.add(layers.Conv2D(x * (2**5), (3, 3), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.Conv2D(x * (2**5), (3, 3), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(TFEB_PS[3], TFEB_PS[3]), strides=(TFEB_PS[3], TFEB_PS[3])))

    model.add(layers.Conv2D(x * (2**6), (3, 3), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.Conv2D(x * (2**6), (3, 3), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(TFEB_PS[4], TFEB_PS[4]), strides=(TFEB_PS[4], TFEB_PS[4])))

    # Output block
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(n_cls, (1, 1), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.AveragePooling2D(pool_size=(2, 4), strides=(2,4)))
    model.add(layers.Flatten())
    model.add(layers.Dense(n_cls, activation='softmax'))

    return model

# --- Load Model and Scaler ---
model = create_acdnet(i_len=INPUT_LENGTH, n_cls=N_CLASSES)
model.load_weights('ACDNet_best_custom.keras')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- Audio Preprocessing Function ---
def preprocess_audio(audio_data):
    """Preprocesses audio data to be model-ready."""
    # Padding or truncating
    if len(audio_data) < INPUT_LENGTH:
        audio_data = np.pad(audio_data, (0, INPUT_LENGTH - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:INPUT_LENGTH]

    # Apply Hamming window
    hamming_window = np.hamming(INPUT_LENGTH)
    audio_data = audio_data * hamming_window

    # Scaling
    audio_data = scaler.transform(audio_data.reshape(1, -1))

    # Reshape for the model
    audio_data = audio_data.reshape(1, 1, INPUT_LENGTH, 1)
    return audio_data

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles audio prediction."""
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file found'})

    file = request.files['audio_data']
    audio_data, _ = sf.read(io.BytesIO(file.read()))

    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = librosa.to_mono(audio_data.T)

    processed_audio = preprocess_audio(audio_data)
    prediction = model.predict(processed_audio)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
