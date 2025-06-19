import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
import sounddevice as sd
import threading
import time  # ⬅️ For timing analysis

plt.style.use('dark_background')


class AudioAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Analyzer using FFT")
        self.master.geometry("400x360")
        self.master.configure(bg="#1e1e1e")

        self.audio_data = None
        self.fs = 44100
        self.analysis_time = 0.0  # ⬅️ Store analysis duration

        self.setup_widgets()

    def setup_widgets(self):
        btn_upload = tk.Button(self.master, text="Upload Audio", command=self.upload_audio,
                               font=('Arial', 12), bg="#333", fg="cyan", width=20)
        btn_upload.pack(pady=10)

        btn_record = tk.Button(self.master, text="Record Audio", command=self.open_record_dialog,
                               font=('Arial', 12), bg="#333", fg="magenta", width=20)
        btn_record.pack(pady=10)

        self.btn_analyze = tk.Button(self.master, text="Show Analysis", command=self.show_analysis,
                                     font=('Arial', 12), bg="#333", fg="lime", width=20, state=tk.DISABLED)
        self.btn_analyze.pack(pady=10)

        self.btn_data = tk.Button(self.master, text="Show Data Report", command=self.show_data_report,
                                  font=('Arial', 12), bg="#333", fg="orange", width=20, state=tk.DISABLED)
        self.btn_data.pack(pady=10)

    def upload_audio(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            self.fs, data = wavfile.read(path)
            if data.ndim > 1:
                data = data[:, 0]  # Take first channel if stereo
            self.audio_data = data.astype(np.float32)
            self.btn_analyze.config(state=tk.NORMAL)
            self.btn_data.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Audio loaded successfully!")

    def open_record_dialog(self):
        self.rec_win = Toplevel(self.master)
        self.rec_win.title("Record Audio")
        self.rec_win.geometry("300x150")
        self.rec_win.configure(bg="#111")

        self.recording = False
        self.audio_buffer = []

        btn_start = tk.Button(self.rec_win, text="Start Recording", command=self.start_recording,
                              bg="#444", fg="white", font=('Arial', 11))
        btn_start.pack(pady=10)

        btn_stop = tk.Button(self.rec_win, text="Stop Recording", command=self.stop_recording,
                             bg="#555", fg="white", font=('Arial', 11))
        btn_stop.pack(pady=10)

    def start_recording(self):
        self.audio_buffer = []
        self.recording = True
        threading.Thread(target=self.record_thread).start()

    def stop_recording(self):
        self.recording = False
        sd.wait()
        self.audio_data = np.concatenate(self.audio_buffer).astype(np.float32)
        self.btn_analyze.config(state=tk.NORMAL)
        self.btn_data.config(state=tk.NORMAL)
        self.rec_win.destroy()
        messagebox.showinfo("Done", "Recording stopped and loaded.")

    def record_thread(self):
        def callback(indata, frames, time_, status):
            if self.recording:
                self.audio_buffer.append(indata.copy())

        with sd.InputStream(samplerate=self.fs, channels=1, callback=callback):
            while self.recording:
                sd.sleep(100)

    def show_analysis(self):
        if self.audio_data is None:
            messagebox.showerror("Error", "No audio data found.")
            return

        start_time = time.time()  # ⏱️ Start timer

        audio = self.audio_data.flatten()
        n = len(audio)
        duration = n / self.fs
        time_axis = np.linspace(0, duration, n)

        frequencies = np.fft.fftfreq(n, d=1 / self.fs)
        magnitude = np.abs(fft(audio)) / n

        self.analysis_time = time.time() - start_time  # ⏱️ Stop timer

        # Plotting
        fig, axs = plt.subplots(3, 1, figsize=(14, 10), facecolor='#0f0f0f')
        fig.suptitle("Audio Signal Analysis", fontsize=18, fontweight='bold', color='cyan')

        axs[0].plot(time_axis, audio, color='#00ffff')
        axs[0].fill_between(time_axis, audio, color='#00ffff', alpha=0.3)
        axs[0].set_title("Waveform (Time Domain)", fontsize=14, color='white')
        axs[0].set_xlabel("Time (s)", color='white')
        axs[0].set_ylabel("Amplitude", color='white')
        axs[0].tick_params(colors='white')
        axs[0].grid(True, color='#444444')

        axs[1].plot(frequencies[:n // 2], magnitude[:n // 2], color='#ff00ff')
        axs[1].fill_between(frequencies[:n // 2], magnitude[:n // 2], color='#ff00ff', alpha=0.3)
        axs[1].set_title("Frequency Spectrum", fontsize=14, color='white')
        axs[1].set_xlabel("Frequency (Hz)", color='white')
        axs[1].set_ylabel("Magnitude", color='white')
        axs[1].set_xlim(0, 6000)
        axs[1].tick_params(colors='white')
        axs[1].grid(True, color='#444444')

        Pxx, freqs, bins, im = axs[2].specgram(audio, NFFT=1024, Fs=self.fs,
                                               noverlap=512, cmap='inferno', scale='dB', mode='psd')
        axs[2].set_title("Spectrogram (Time vs Frequency)", fontsize=14, color='white')
        axs[2].set_xlabel("Time (s)", color='white')
        axs[2].set_ylabel("Frequency (Hz)", color='white')
        axs[2].set_ylim(1, 6000)
        axs[2].tick_params(colors='white')

        # Highlight key frequencies
        for f in [70, 352, 1500, 5500]:
            axs[2].axhline(y=f, color='white', linestyle='--', linewidth=0.6)
            label = f"{f / 1000:.1f} KHz" if f >= 1000 else f"{f} Hz"
            axs[2].text(bins[-1] + 0.05, f, label, color='white', va='center', fontsize=9)

        cbar = fig.colorbar(im, ax=axs[2], label='Intensity (dB)', orientation='vertical', pad=0.01)
        cbar.ax.tick_params(colors='white', labelsize=9)

        # Show analysis time on graph
        fig.text(0.5, 0.01, f"Analysis Time: {self.analysis_time:.3f} seconds",
                 ha='center', fontsize=10, color='yellow')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def show_data_report(self):
        if self.audio_data is None:
            messagebox.showerror("Error", "No audio data to report.")
            return

        audio = self.audio_data.flatten()
        n = len(audio)
        duration = n / self.fs
        mean_amp = np.mean(audio)
        max_amp = np.max(audio)
        min_amp = np.min(audio)

        frequencies = np.fft.fftfreq(n, d=1 / self.fs)
        magnitude = np.abs(fft(audio)) / n
        peak_freq = abs(frequencies[np.argmax(magnitude[:n // 2])])

        # Report Window
        report_win = Toplevel(self.master)
        report_win.title("Audio Analysis Report")
        report_win.geometry("360x270")
        report_win.configure(bg="#1c1c1c")

        report = f"""
Audio Report Summary:
---------------------------
Sample Rate        : {self.fs} Hz
Duration           : {duration:.2f} s
Total Samples      : {n}
Peak Frequency     : {peak_freq:.2f} Hz
Mean Amplitude     : {mean_amp:.4f}
Max Amplitude      : {max_amp:.4f}
Min Amplitude      : {min_amp:.4f}
Analysis Time      : {self.analysis_time:.3f} sec
        """

        lbl = tk.Label(report_win, text=report, justify='left', font=('Courier', 10),
                       bg="#1c1c1c", fg="white", padx=10, pady=10)
        lbl.pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzer(root)
    root.mainloop()
