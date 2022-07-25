import time, os
import sys
import io
import streamlit as st
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sounddevice as sd
from scipy.io.wavfile import write
import warnings
import spotify
from pydub import AudioSegment
from PIL import Image

AUDIO_FILE_PATH = 'tempDir/recording.wav'

st.set_option('deprecation.showPyplotGlobalUse', False)

if 'pred' not in st.session_state:
    st.session_state.pred = ''
if 'filename' not in st.session_state:
    st.session_state.filename = ''

def init_model():
    model = load_model('models/LSTM.hdf5')
    return model

def display(mel):
    plt.figure(figsize=(10,5))
    librosa.display.specshow(mel, y_axis='mel', x_axis='time')
    plt.title('Mel-Spectrogram of the Recording')
    plt.tight_layout()
    st.pyplot(clear_figure=False)
    
    
def record():
    duration = 5
    sampling_rate = 44100

    recording = sd.rec(int(duration*sampling_rate), 
                       samplerate=sampling_rate, 
                       channels=2)

    sd.wait()

    write(AUDIO_FILE_PATH,sampling_rate,recording)

    
def MelSpectrogram(signal, sr):
    if signal is None:
        if os.path.exists(AUDIO_FILE_PATH):
            signal, sr = librosa.load(AUDIO_FILE_PATH)
        else:
            signal, sr = librosa.load(st.session_state.filename)
        
    #obtain mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(signal, sr=sr, n_fft=2048, hop_length=512, n_mels=40)
    
    #change to dB scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    
    return log_mel_spectrogram


def Classify(model):
    batch = []
    label = ['bus','metro','park','street_traffic']
    
    if os.path.exists(AUDIO_FILE_PATH):
        signal, sr = librosa.load(AUDIO_FILE_PATH)
    else:
        signal, sr = librosa.load(st.session_state.filename)
    
    step = sr*1 # for 1 second duration clip

    for i in range(0, signal.shape[0], step):
        sample = signal[i:i+step]
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mel = MelSpectrogram(sample,sr)
        
        mel = mel.T
        mel = np.expand_dims(mel, axis=2)

        batch.append(mel)

    X_batch = np.array(batch, dtype=np.float32)

    y_pred = model.predict(X_batch)
    y_mean = np.mean(y_pred, axis=0)
    y_pred = np.argmax(y_mean)
    
    return label[y_pred]
    
    
def main():
    
    title='Spotify Song Recommender based on Surrounding Soundscape'
    st.title(title)
    st.text('A song recommender integrated with Spotify API based on the surrounding acoustic scene')
    st.text('')
    st.text('')
    
    with st.sidebar:
        image = Image.open('for_demo/SoundScapify.png')
        st.image(image)
    
    st.subheader('Record/Upload the Audio to be Classified')
    st.caption('Record an audio')
    
    if st.button('Record'):
        with st.spinner('Recording for 5 seconds...'):
            record()
        st.success('Recording completed')
    
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav','mp3'])
    if uploaded_file is not None:
        
        if uploaded_file.name.endswith('wav'):
            audio = AudioSegment.from_wav(uploaded_file)
            file_type = 'wav'
        elif uploaded_file.name.endswith('mp3'):
            audio = AudioSegment.from_mp3(uploaded_file)
            file_type = 'mp3'
        
        st.session_state.filename = 'tempDir/' + uploaded_file.name
        audio.export(st.session_state.filename, format=file_type)
        st.success('File Uploaded')
    
    st.text('')
    st.subheader('Play Audio and Mel-Spectrogram')
    st.caption('Play the Audio Recorded/Uploaded')
    
    if st.button('Play'):        
        if os.path.exists(AUDIO_FILE_PATH):
            #enter the filename with filepath
            audio_file = open(AUDIO_FILE_PATH,'rb')
            
            #reading the file
            audio_bytes = audio_file.read() 

            #displaying the audio
            st.audio(audio_bytes, format='audio/wav')
        
        elif os.path.exists(st.session_state.filename):
            #enter the filename with filepath
            audio_file = open(st.session_state.filename,'rb')
            
            #reading the file
            audio_bytes = audio_file.read() 

            #displaying the audio
            st.audio(audio_bytes, format='audio/wav')
        
        else:
            st.write('No file found. Please record/upload sound first')


    st.caption('Display the Mel-Spectrogram of the Audio Uploaded/Recorded')
    if st.button('Display'):
        if (os.path.exists(AUDIO_FILE_PATH) | os.path.exists(st.session_state.filename)):
            with st.spinner('Loading Mel-Spectrogram'):
                time.sleep(1)
                mel = MelSpectrogram(None, None)
                display(mel)
        else:
            st.write('No file found. Please record/upload sound first')
            
    st.text('')
    st.subheader('Classifier and Song Recommender')
    st.caption('Classify the Audio File Uploaded/Recorded')
    if st.button('Classify'):
        if (os.path.exists(AUDIO_FILE_PATH) | os.path.exists(st.session_state.filename)):
            with st.spinner('Loading...'):
                model = init_model()
                st.session_state.pred = Classify(model)
            st.success('Done')
            st.write('You are at {}'.format(st.session_state.pred))
            
        else:
            st.write('No file found. Please record/upload sound first')
    
    st.caption('Play the recommended songs based on the audio file in Spotify. The total recommended songs = 10 tracks ')
    if st.button('Spotify Play'):
        if (os.path.exists(AUDIO_FILE_PATH) | os.path.exists(st.session_state.filename)):
            st.write('Opening Spotify to play the recommended songs for {} soundscape...'.format(st.session_state.pred))
            time.sleep(0.5)
            
            uris = spotify.GetRecommendation(st.session_state.pred)
            spotify.StartPlayback(uris)
            
        else:
            st.write('No file found. Please record/upload sound first')
    

if __name__ == '__main__':
    main()