from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

class Whisper:

    def record() -> None:
        """Saves to whisper/output.wav"""

        # Define parameters
        duration = 5  # Duration in seconds
        sample_rate = 44100  # Sample rate in Hz

        # Capture audio
        print("Recording...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
        sd.wait()  # Wait until recording is finished
        print("Recording complete.")

        # Save the captured audio to a WAV file
        wav.write('whisper/output.wav', sample_rate, np.int16(audio_data * 32767))  # Convert to int16
        print("Audio saved to 'whisper/output.wav'.")

    def predict() -> str:

        audio_file = "whisper/output.wav"
        speech_array, sampling_rate = torchaudio.load(audio_file)

        if sampling_rate != 16000:
          resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
          speech_array = resampler(speech_array)

        inputs = processor(speech_array.reshape((-1,)), sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription