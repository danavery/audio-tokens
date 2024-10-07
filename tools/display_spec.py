import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Load the spectrogram data from the .npy file
spectrogram = np.load("test.npy")

# Display the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(
    spectrogram, sr=22050, hop_length=64, x_axis="time", y_axis="mel"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram")
plt.show()
