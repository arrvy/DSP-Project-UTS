"""
MODUL DASAR PEMROSESAN SINYAL DIGITAL (VERSI MODULAR SEDERHANA)

Tujuan:
- Tetap mudah dibaca untuk pemula.
- Bisa dipanggil dari file lain (tidak auto-jalan saat di-import).
- Tetap menghasilkan analisa dan gambar seperti skrip awal.

Contoh pakai dari file lain:
    import src.references as ref
    hasil = ref.run_full_analysis(show_plots=False)

Contoh jalankan langsung:
    PYTHONPATH=. python src/references.py
"""

import os
import importlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import windows

__all__ = [
    "get_default_config",
    "generate_synthetic_signal",
    "load_wav_signal",
    "load_audio_signal",
    "compute_basic_operations",
    "analyze_signal",
    "run_full_analysis",
    "run_full_analysis_from_wav",
    "run_full_analysis_from_audio",
]

#! UBAH PARAMETER DI SINI
def get_default_config():
    """Konfigurasi default agar gampang diubah dari luar modul."""
    return {
        "fs": 1000,
        "duration": 1.0,
        "f_bass": 50,
        "f_treble": 300,
        "A_bass": 1.5,
        "A_treble": 0.8,
        "noise_std": 0.2,
        #"clip_level": 1.2, DEFAULT
        "clip_level": .5,
        "ma_window": 20,
        "threshold": 0.1,
        "view_samples": 1000,
        "time_view_strategy": "auto",
        "view_start_sample": 0,
    }


def apply_plot_style():
    """Styling global plot supaya konsisten."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.dpi": 100,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.4,
        }
    )


def build_time_axis(fs, duration):
    """Membuat indeks n dan axis waktu t dari fs dan durasi."""
    N = int(fs * duration)
    n = np.arange(N)
    t = n / fs
    return n, t


def generate_synthetic_signal(
    t,
    f_bass=50,
    f_treble=300,
    A_bass=1.5,
    A_treble=0.8,
    noise_std=0.2,
):
    """Membuat sinyal komposit: bass + treble + noise Gaussian."""
    x_clean = A_bass * np.sin(2 * np.pi * f_bass * t) + A_treble * np.sin(
        2 * np.pi * f_treble * t
    )
    x = x_clean + noise_std * np.random.randn(len(t))
    return x_clean, x


def load_wav_signal(file_path):
    """Load WAV lalu normalisasi ke rentang [-1, 1]."""
    fs_wav, data = wavfile.read(file_path)
    data = data.astype(float)

    # Kalau stereo, ambil rata-rata dua channel agar jadi mono.
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    max_abs = np.max(np.abs(data))
    if max_abs > 0:
        data = data / max_abs

    t = np.arange(len(data)) / fs_wav
    return fs_wav, t, data


def load_audio_signal(file_path, target_fs=None):
    """
    Load audio umum (wav/mp3/aac/m4a/format lain yang didukung backend).

    - Kalau WAV dan target_fs=None: pakai loader WAV cepat.
    - Selain itu: pakai librosa agar format terkompresi bisa dibaca.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".wav" and target_fs is None:
        return load_wav_signal(file_path)

    try:
        librosa = importlib.import_module("librosa")
    except ImportError as exc:
        raise ImportError(
            "Untuk file non-WAV (mis. mp3/aac), install dulu: pip install librosa audioread"
        ) from exc

    data, fs_audio = librosa.load(file_path, sr=target_fs, mono=True)
    data = data.astype(float)

    max_abs = np.max(np.abs(data))
    if max_abs > 0:
        data = data / max_abs

    t = np.arange(len(data)) / fs_audio
    return fs_audio, t, data


def get_amplitude(X_fft, pos_idx, N):
    """Ambil amplitudo sisi frekuensi positif dan normalisasi."""
    return np.abs(X_fft[pos_idx]) * 2 / N


def normalize_peak(x):
    """Normalisasi peak ke kisaran sekitar [-1, 1]."""
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return x.copy()
    return x / max_abs


def normalize_zscore(x):
    """Normalisasi z-score: (x - mean) / std."""
    std = np.std(x)
    if std == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def choose_time_view_window(x, view_samples, strategy="auto", start_sample=0):
    """
    Tentukan window sampel yang dipakai untuk plot domain waktu.

    strategy:
    - "start": ambil dari awal (atau dari start_sample)
    - "peak": ambil area sekitar puncak |x|
    - "auto": jika awal sinyal terlalu hening, pindah ke area puncak
    """
    N = len(x)
    if N == 0:
        return 0, 0

    show_n = max(1, min(view_samples, N))

    if strategy == "start":
        start = max(0, min(int(start_sample), N - show_n))
        return start, start + show_n

    if strategy == "peak":
        center = int(np.argmax(np.abs(x)))
        start = max(0, min(center - (show_n // 2), N - show_n))
        return start, start + show_n

    # strategy == "auto"
    first = x[:show_n]
    overall_std = float(np.std(x))
    first_std = float(np.std(first))
    overall_peak = float(np.max(np.abs(x)))
    first_peak = float(np.max(np.abs(first)))

    # Jika bagian awal sangat kecil dibanding keseluruhan, ambil area puncak.
    if overall_std > 0 and overall_peak > 0:
        low_variance = first_std < (0.15 * overall_std)
        low_peak = first_peak < (0.20 * overall_peak)
        if low_variance and low_peak:
            center = int(np.argmax(np.abs(x)))
            start = max(0, min(center - (show_n // 2), N - show_n))
            return start, start + show_n

    return 0, show_n


def build_operation_signals(N, kernel_len=41):
    """
    Buat sinyal uji dasar (impulse/step/ramp/ones/zeros) dan kernel operasi.

    Definisi yang dipakai (diskrit):
    - impulse : delta[n]
    - step    : u[n]
    - ramp    : n * u[n]

    Kernel dibuat pendek agar konvolusi/korelasi tetap efisien untuk sinyal
    panjang (mis. audio sampling tinggi).
    """
    if kernel_len % 2 == 0:
        kernel_len += 1

    n = np.arange(N) - (N // 2)

    pointwise = {
        "impulse": np.zeros(N),
        "step": (n >= 0).astype(float),
        "ramp": (n * (n >= 0)).astype(float),
        "ones": np.ones(N),
        "zeros": np.zeros(N),
    }
    pointwise["impulse"][N // 2] = 1.0

    half_k = kernel_len // 2
    n_k = np.arange(-half_k, half_k + 1)

    kernels = {
        "impulse": (n_k == 0).astype(float),
        "step": (n_k >= 0).astype(float),
        "ramp": (n_k * (n_k >= 0)).astype(float),
        "ones": np.ones(kernel_len),
        "zeros": np.zeros(kernel_len),
    }

    return {
        "n": n,
        "n_kernel": n_k,
        "pointwise": pointwise,
        "kernels": kernels,
    }


def compute_basic_operations(x):
    """
    Operasi dasar sinyal digital:
    - penjumlahan
    - perkalian
    - konvolusi
    - korelasi
    - normalisasi
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    signals = build_operation_signals(N)

    point = signals["pointwise"]
    kernels = signals["kernels"]

    # Penjumlahan: x + nilai tertentu (konstanta).
    addition_constants = {
        "minus_1": -1.0,
        "minus_05": -0.5,
        "plus_05": 0.5,
        "plus_1": 1.0,
    }
    addition = {k: x + c for k, c in addition_constants.items()}

    # Perkalian pointwise pakai sinyal standar.
    use_keys = ["impulse", "ramp", "step", "ones"]
    multiplication = {k: x * point[k] for k in use_keys}

    # Konvolusi dan korelasi pakai kernel impulse/step/ramp/ones.
    convolution = {k: np.convolve(x, kernels[k], mode="same") for k in use_keys}
    correlation = {k: np.correlate(x, kernels[k], mode="same") for k in use_keys}
    correlation["self"] = np.correlate(x, x, mode="same")

    normalization = {
        "peak": normalize_peak(x),
        "zscore": normalize_zscore(x),
    }

    return {
        "signals": signals,
        "addition_constants": addition_constants,
        "addition": addition,
        "multiplication": multiplication,
        "convolution": convolution,
        "correlation": correlation,
        "normalization": normalization,
    }


def analyze_signal(x, fs, clip_level=1.2, ma_window=20):
    """Operasi dasar sinyal: clipping, power, RMS, MA, FFT, Re/Im, phase."""
    N = len(x)

    # Operasi domain waktu
    x_clipped = np.clip(x, -clip_level, clip_level)
    x_power = x**2
    rms_val = np.sqrt(np.mean(x_power))
    kernel = np.ones(ma_window) / ma_window
    x_ma = np.convolve(x, kernel, mode="same")

    # FFT
    X = np.fft.fft(x)
    X_c = np.fft.fft(x_clipped)
    X_ma = np.fft.fft(x_ma)
    freqs = np.fft.fftfreq(N, d=1 / fs)

    pos_idx = freqs >= 0
    f_pos = freqs[pos_idx]

    amp = get_amplitude(X, pos_idx, N)
    amp_c = get_amplitude(X_c, pos_idx, N)
    amp_ma = get_amplitude(X_ma, pos_idx, N)

    re_X = np.real(X[pos_idx])
    im_X = np.imag(X[pos_idx])
    mag_X = np.sqrt(re_X**2 + im_X**2) * 2 / N
    phase_X = np.arctan2(im_X, re_X)

    peak_idx = int(np.argmax(amp))
    peak_freq = f_pos[peak_idx]

    return {
        "N": N,
        "x_clipped": x_clipped,
        "x_power": x_power,
        "rms_val": rms_val,
        "kernel": kernel,
        "x_ma": x_ma,
        "X": X,
        "X_c": X_c,
        "X_ma": X_ma,
        "freqs": freqs,
        "pos_idx": pos_idx,
        "f_pos": f_pos,
        "amp": amp,
        "amp_c": amp_c,
        "amp_ma": amp_ma,
        "re_X": re_X,
        "im_X": im_X,
        "mag_X": mag_X,
        "phase_X": phase_X,
        "peak_idx": peak_idx,
        "peak_freq": peak_freq,
    }


def save_or_show(fig, filename, output_dir=".", save_plots=True, show_plots=True):
    """Helper kecil untuk simpan/show plot."""
    fig.tight_layout()
    if save_plots:
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=150)
        print(f"Tersimpan: {save_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_time_domain(
    t,
    x,
    analysis,
    clip_level,
    ma_window,
    view_samples=500,
    time_view_strategy="auto",
    view_start_sample=0,
    output_dir=".",
    save_plots=True,
    show_plots=True,
):
    """Plot (5a): domain waktu."""
    x_clipped = analysis["x_clipped"]
    x_power = analysis["x_power"]
    x_ma = analysis["x_ma"]
    rms_val = analysis["rms_val"]

    start, end = choose_time_view_window(
        x,
        view_samples=view_samples,
        strategy=time_view_strategy,
        start_sample=view_start_sample,
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    fig.suptitle("Analisa Kawasan Waktu", fontsize=14, fontweight="bold")

    axes[0].plot(t[start:end], x[start:end], color="steelblue")
    axes[0].set_title("(i) Sinyal x[n] - Domain Waktu")
    axes[0].set_ylabel("Amplitudo")

    axes[1].plot(t[start:end], x[start:end], label="Original", alpha=0.6)
    axes[1].plot(
        t[start:end],
        x_clipped[start:end],
        label=f"Clipped +/-{clip_level}",
        alpha=0.9,
        color="tomato",
    )
    axes[1].axhline(clip_level, color="red", linestyle="--", linewidth=0.8, label="Batas clip")
    axes[1].axhline(-clip_level, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_title("(ii) Clipping - Pemotongan Sinyal")
    axes[1].set_ylabel("Amplitudo")
    axes[1].legend(loc="upper right")

    axes[2].fill_between(
        t[start:end],
        0,
        x_power[start:end],
        color="orange",
        alpha=0.6,
        label="Daya sesaat",
    )
    axes[2].axhline(
        rms_val**2,
        color="red",
        linestyle="--",
        label=f"Daya rata-rata (RMS^2={rms_val**2:.3f})",
    )
    axes[2].set_title(f"(iii) Sinyal Daya  |  (iv) RMS = {rms_val:.4f}")
    axes[2].set_ylabel("Daya (amplitudo^2)")
    axes[2].legend(loc="upper right")

    axes[3].plot(t[start:end], x[start:end], label="Original", alpha=0.4, color="steelblue")
    axes[3].plot(
        t[start:end],
        x_ma[start:end],
        label=f"Moving Avg (M={ma_window})",
        color="darkblue",
        linewidth=2,
    )
    axes[3].set_title(f"(v) Konvolusi Moving Average (M={ma_window}) - Efek Low-pass")
    axes[3].set_ylabel("Amplitudo")
    axes[3].set_xlabel("Waktu (s)")
    axes[3].legend(loc="upper right")

    save_or_show(fig, "5a_time_domain.png", output_dir, save_plots, show_plots)


def _plot_four_operation_panels(
    axis_x,
    data_dict,
    keys,
    pretty_name,
    title,
    filename,
    view_samples=500,
    output_dir=".",
    save_plots=True,
    show_plots=True,
):
    """Helper plot 4 subplot vertikal dengan area fokus sekitar n=0."""
    total_n = len(axis_x)
    show_n = min(view_samples, total_n)
    start = max(0, (total_n // 2) - (show_n // 2))
    end = start + show_n
    x_view = axis_x[start:end]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, key in zip(axes, keys):
        ax.plot(x_view, data_dict[key][start:end])
        ax.set_title(f"Operand: {pretty_name[key]}")
        ax.set_ylabel("Amplitudo")

    axes[-1].set_xlabel("n (sample)")
    save_or_show(fig, filename, output_dir, save_plots, show_plots)


def plot_addition_operations(axis_n, basic_ops, view_samples=500, output_dir=".", save_plots=True, show_plots=True):
    """Plot hasil penjumlahan x dengan konstanta tertentu."""
    _plot_four_operation_panels(
        axis_n,
        basic_ops["addition"],
        keys=["minus_1", "minus_05", "plus_05", "plus_1"],
        pretty_name={
            "minus_1": "x + (-1.0)",
            "minus_05": "x + (-0.5)",
            "plus_05": "x + 0.5",
            "plus_1": "x + 1.0",
        },
        title="Operasi Penjumlahan Sinyal",
        filename="5a_ops_addition.png",
        view_samples=view_samples,
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )


def plot_multiplication_operations(axis_n, basic_ops, view_samples=500, output_dir=".", save_plots=True, show_plots=True):
    """Plot hasil perkalian x dengan impulse/ramp/step/ones."""
    _plot_four_operation_panels(
        axis_n,
        basic_ops["multiplication"],
        keys=["impulse", "ramp", "step", "ones"],
        pretty_name={
            "impulse": "Impulse",
            "ramp": "Ramp = n*u[n]",
            "step": "Step = u[n]",
            "ones": "Konstan 1",
        },
        title="Operasi Perkalian Sinyal",
        filename="5a_ops_multiplication.png",
        view_samples=view_samples,
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )


def plot_convolution_operations(axis_n, basic_ops, view_samples=500, output_dir=".", save_plots=True, show_plots=True):
    """Plot hasil konvolusi x dengan kernel impulse/ramp/step/ones."""
    _plot_four_operation_panels(
        axis_n,
        basic_ops["convolution"],
        keys=["impulse", "ramp", "step", "ones"],
        pretty_name={
            "impulse": "Kernel Impulse",
            "ramp": "Kernel Ramp = n*u[n]",
            "step": "Kernel Step = u[n]",
            "ones": "Kernel Konstan 1",
        },
        title="Operasi Konvolusi Sinyal",
        filename="5a_ops_convolution.png",
        view_samples=view_samples,
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )


def plot_correlation_and_normalization(axis_n, basic_ops, view_samples=500, output_dir=".", save_plots=True, show_plots=True):
    """Plot representatif untuk korelasi dan normalisasi."""
    total_n = len(axis_n)
    show_n = min(view_samples, total_n)
    start = max(0, (total_n // 2) - (show_n // 2))
    end = start + show_n
    n_view = axis_n[start:end]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Operasi Korelasi dan Normalisasi", fontsize=13, fontweight="bold")

    axes[0, 0].plot(n_view, basic_ops["correlation"]["impulse"][start:end], color="steelblue")
    axes[0, 0].set_title("Korelasi dengan Kernel Impulse")
    axes[0, 0].set_ylabel("Nilai")

    axes[0, 1].plot(n_view, basic_ops["correlation"]["self"][start:end], color="tomato")
    axes[0, 1].set_title("Auto-korelasi (x dikorelasikan dengan x)")

    axes[1, 0].plot(n_view, basic_ops["normalization"]["peak"][start:end], color="purple")
    axes[1, 0].set_title("Normalisasi Peak")
    axes[1, 0].set_xlabel("n (sample)")
    axes[1, 0].set_ylabel("Amplitudo")

    axes[1, 1].plot(n_view, basic_ops["normalization"]["zscore"][start:end], color="green")
    axes[1, 1].set_title("Normalisasi Z-Score")
    axes[1, 1].set_xlabel("n (sample)")

    save_or_show(fig, "5a_ops_corr_norm.png", output_dir, save_plots, show_plots)


def plot_amp_energy(analysis, f_targets, output_dir=".", save_plots=True, show_plots=True):
    """Plot (5b-i): spektrum amplitudo dan energi."""
    f_pos = analysis["f_pos"]
    amp = analysis["amp"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle("(5b-i) Spektrum Amplitudo & Energi", fontsize=13, fontweight="bold")

    axes[0].plot(f_pos, amp, color="steelblue")
    axes[0].set_title("Spektrum Amplitudo |X[k]| (dinormalisasi)")
    axes[0].set_ylabel("|X[k]|")

    for f_target in f_targets:
        idx = int(np.argmin(np.abs(f_pos - f_target)))
        axes[0].annotate(
            f"{f_pos[idx]:.0f} Hz\nA={amp[idx]:.2f}",
            xy=(f_pos[idx], amp[idx]),
            xytext=(f_pos[idx] + 15, amp[idx] * 0.9),
            arrowprops={"arrowstyle": "->", "color": "red"},
            fontsize=9,
            color="red",
        )

    axes[1].plot(f_pos, amp**2, color="green")
    axes[1].set_title("Spektrum Energi |X[k]|^2")
    axes[1].set_ylabel("|X[k]|^2")
    axes[1].set_xlabel("Frekuensi (Hz)")

    for ax in axes:
        ax.set_xlim(0, 500)

    save_or_show(fig, "5b_i_amp_energy.png", output_dir, save_plots, show_plots)


def plot_real_imag_phase(analysis, output_dir=".", save_plots=True, show_plots=True):
    """Plot (5b-ii): komponen riil, imajiner, dan phase."""
    f_pos = analysis["f_pos"]
    re_X = analysis["re_X"]
    im_X = analysis["im_X"]
    phase_X = analysis["phase_X"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("(5b-ii) Komponen Riil & Imajiner Spektrum DFT", fontsize=13, fontweight="bold")

    axes[0].plot(f_pos, re_X, color="steelblue", label="Re{X[k]}")
    axes[0].set_title(
        "Re{X[k]} - Kontribusi Cosine\n"
        "Bernilai mendekati 0 karena sinyal dibangun dari sin(), bukan cos()"
    )
    axes[0].set_ylabel("Nilai Riil")
    axes[0].axhline(0, color="black", linewidth=0.5)

    axes[1].plot(f_pos, im_X, color="tomato", label="Im{X[k]}")
    axes[1].set_title(
        "Im{X[k]} - Kontribusi Sine\n"
        "Peak negatif di 50 Hz dan 300 Hz karena konvensi DFT"
    )
    axes[1].set_ylabel("Nilai Imajiner")
    axes[1].axhline(0, color="black", linewidth=0.5)

    axes[2].plot(f_pos, np.degrees(phase_X), color="purple", linewidth=0.8)
    axes[2].set_title(
        "Phase phi[k] = arctan(Im/Re) dalam derajat\n"
        "Frekuensi dominan mendekati -90 derajat karena komponen berupa sine"
    )
    axes[2].set_ylabel("Phase (derajat)")
    axes[2].set_xlabel("Frekuensi (Hz)")
    axes[2].axhline(-90, color="red", linestyle="--", linewidth=0.7, label="-90 derajat (pure sine)")
    axes[2].axhline(0, color="green", linestyle="--", linewidth=0.7, label="0 derajat (pure cosine)")
    axes[2].legend()
    axes[2].set_ylim(-200, 200)

    for ax in axes:
        ax.set_xlim(0, 500)

    save_or_show(fig, "5b_ii_real_imag.png", output_dir, save_plots, show_plots)


def plot_windowing(x, analysis, output_dir=".", save_plots=True, show_plots=True):
    """Plot (5b-iii): perbandingan window rectangular, hanning, hamming, blackman."""
    N = analysis["N"]
    pos_idx = analysis["pos_idx"]
    f_pos = analysis["f_pos"]

    wind_dict = {
        "Rectangular": np.ones(N),
        "Hanning": windows.hann(N),
        "Hamming": windows.hamming(N),
        "Blackman": windows.blackman(N),
    }

    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    fig.suptitle("(5b-iii) Spektrum dengan Window Berbeda", fontsize=13, fontweight="bold")

    for ax, (name, w) in zip(axes, wind_dict.items()):
        Xw = np.fft.fft(x * w)
        ampw = np.abs(Xw[pos_idx]) * 2 / N
        ax.plot(f_pos, ampw)
        ax.set_title(f"Window: {name}")
        ax.set_ylabel("|X[k]|")
        ax.set_xlim(0, 500)

    axes[-1].set_xlabel("Frekuensi (Hz)")
    save_or_show(fig, "5b_iii_windowing.png", output_dir, save_plots, show_plots)


def identify_frequency_components(analysis, threshold=0.1, bass_limit=200):
    """Cari komponen frekuensi yang amplitudonya lewat threshold."""
    f_pos = analysis["f_pos"]
    amp = analysis["amp"]
    found = []

    for f_val, a_val in zip(f_pos, amp):
        if a_val > threshold:
            label = "BASS (< 200 Hz)" if f_val < bass_limit else "TREBLE (>= 200 Hz)"
            found.append((f_val, a_val, label))

    return found


def plot_clipping_impact(analysis, clip_level, output_dir=".", save_plots=True, show_plots=True):
    """Plot (5b-v): dampak clipping di domain frekuensi."""
    f_pos = analysis["f_pos"]
    amp = analysis["amp"]
    amp_c = analysis["amp_c"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle("(5b-v) Dampak Clipping pada Spektrum Frekuensi", fontsize=13, fontweight="bold")

    axes[0].plot(f_pos, amp, color="steelblue", label="Original")
    axes[0].set_title("Spektrum Original - hanya ada 50 Hz dan 300 Hz")

    axes[1].plot(f_pos, amp_c, color="tomato", label="Clipped")
    axes[1].set_title(f"Spektrum Clipped (+/-{clip_level}) - muncul harmonik baru karena distorsi")

    for ax in axes:
        ax.set_ylabel("|X[k]|")
        ax.set_xlim(0, 500)
        ax.legend()
    axes[-1].set_xlabel("Frekuensi (Hz)")

    save_or_show(fig, "5b_v_clipping_freq.png", output_dir, save_plots, show_plots)


def build_comparison_spectrum(t, cfg):
    """Buat sinyal pembanding dan spektrumnya untuk cosine similarity."""
    x2 = cfg["A_bass"] * np.sin(2 * np.pi * cfg["f_bass"] * t)
    x2 = x2 + 0.7 * np.sin(2 * np.pi * cfg["f_treble"] * t)

    X2 = np.fft.fft(x2)
    return x2, X2


def plot_similarity(analysis, amp2, similarity, output_dir=".", save_plots=True, show_plots=True):
    """Plot (5b-vi): kemiripan spektrum antara dua sinyal."""
    f_pos = analysis["f_pos"]
    amp = analysis["amp"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(f_pos, amp, label="Sinyal 1 (dengan noise)", alpha=0.8)
    ax.plot(f_pos, amp2, label="Sinyal 2 (clean, amplitudo berbeda)", alpha=0.8, linestyle="--")
    ax.set_title(f"(5b-vi) Kemiripan Spektrum - Cosine Similarity = {similarity:.4f}")
    ax.set_xlabel("Frekuensi (Hz)")
    ax.set_ylabel("|X[k]|")
    ax.set_xlim(0, 500)
    ax.legend()

    save_or_show(fig, "5b_vi_similarity.png", output_dir, save_plots, show_plots)


def plot_ma_spectrum(analysis, ma_window, output_dir=".", save_plots=True, show_plots=True):
    """Plot (5b-vii): spektrum sebelum dan sesudah moving average."""
    f_pos = analysis["f_pos"]
    amp = analysis["amp"]
    amp_ma = analysis["amp_ma"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle("(5b-vii) Spektrum Setelah Moving Average", fontsize=13, fontweight="bold")

    axes[0].plot(f_pos, amp, color="steelblue", label="Original")
    axes[0].set_title("Spektrum Original")

    axes[1].plot(f_pos, amp_ma, color="purple", label=f"Setelah MA (M={ma_window})")
    axes[1].set_title(f"Setelah Moving Average M={ma_window} - komponen 300 Hz melemah")

    for ax in axes:
        ax.set_ylabel("|X[k]|")
        ax.set_xlim(0, 500)
        ax.legend()
    axes[-1].set_xlabel("Frekuensi (Hz)")

    save_or_show(fig, "5b_vii_ma_spectrum.png", output_dir, save_plots, show_plots)


def run_full_analysis(
    config=None,
    signal=None,
    fs=None,
    show_plots=True,
    save_plots=True,
    output_dir=".",
    verbose=True,
):
    """
    Pipeline utama untuk dipanggil dari file lain.

    Parameters:
    - config: dict untuk override config default.
    - signal: array sinyal eksternal. Kalau None, otomatis generate sinyal sintetik.
    - fs: sampling rate untuk signal eksternal. Jika None, pakai config['fs'].
    - show_plots: tampilkan plot ke layar.
    - save_plots: simpan plot ke file png.
    - output_dir: folder output gambar.
    - verbose: print ringkasan ke console.
    """
    cfg = get_default_config()
    if config is not None:
        cfg.update(config)

    apply_plot_style()

    if fs is None:
        fs = cfg["fs"]
    else:
        cfg["fs"] = fs

    if fs <= 0:
        raise ValueError("fs harus lebih besar dari 0.")

    if signal is None:
        _, t = build_time_axis(fs, cfg["duration"])
        x_clean, x = generate_synthetic_signal(
            t,
            f_bass=cfg["f_bass"],
            f_treble=cfg["f_treble"],
            A_bass=cfg["A_bass"],
            A_treble=cfg["A_treble"],
            noise_std=cfg["noise_std"],
        )
    else:
        x_clean = None
        x = np.asarray(signal, dtype=float)
        t = np.arange(len(x)) / fs

    analysis = analyze_signal(
        x,
        fs,
        clip_level=cfg["clip_level"],
        ma_window=cfg["ma_window"],
    )
    basic_ops = compute_basic_operations(x)

    if verbose:
        print(f"RMS sinyal    : {analysis['rms_val']:.4f}")
        if x_clean is not None:
            rms_theory = np.sqrt(cfg["A_bass"] ** 2 / 2 + cfg["A_treble"] ** 2 / 2)
            print(f"RMS teoritis  : {rms_theory:.4f}  (tanpa noise)")

        diff_mag = np.max(np.abs(analysis["amp"] - analysis["mag_X"]))
        print("\nCek konsistensi magnitude:")
        print(f"max |amp - mag_X| = {diff_mag:.2e}  (harus mendekati 0)")

        peak_idx = analysis["peak_idx"]
        peak_freq = analysis["peak_freq"]
        print(f"\nFrekuensi dominan: {peak_freq:.1f} Hz  (amplitudo = {analysis['amp'][peak_idx]:.4f})")

    plot_time_domain(
        t,
        x,
        analysis,
        clip_level=cfg["clip_level"],
        ma_window=cfg["ma_window"],
        view_samples=max(cfg["view_samples"], int(0.05 * fs)),
        time_view_strategy=cfg["time_view_strategy"],
        view_start_sample=cfg["view_start_sample"],
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    axis_n = basic_ops["signals"]["n"]

    plot_addition_operations(
        axis_n,
        basic_ops,
        view_samples=cfg["view_samples"],
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    plot_multiplication_operations(
        axis_n,
        basic_ops,
        view_samples=cfg["view_samples"],
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    plot_convolution_operations(
        axis_n,
        basic_ops,
        view_samples=cfg["view_samples"],
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    plot_correlation_and_normalization(
        axis_n,
        basic_ops,
        view_samples=cfg["view_samples"],
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    if verbose:
        add_zero_err = np.max(np.abs((x + 0.0) - x))
        mul_one_err = np.max(np.abs((x * basic_ops["signals"]["pointwise"]["ones"]) - x))
        print("\n--- Operasi Dasar Sinyal Digital ---")
        print("Penjumlahan  : x + konstanta tertentu")
        print("Perkalian    : x * impulse/ramp/step/1")
        print("Konvolusi    : x dikonvolusi kernel impulse/ramp/step/1")
        print("Korelasi     : x dikorelasikan kernel impulse/ramp/step/1")
        print("Normalisasi  : peak dan z-score")
        print(f"Cek identitas x + 0: max error = {add_zero_err:.2e}")
        print(f"Cek identitas x * 1: max error = {mul_one_err:.2e}")

    plot_amp_energy(
        analysis,
        f_targets=[cfg["f_bass"], cfg["f_treble"]],
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    plot_real_imag_phase(
        analysis,
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    if verbose:
        print("\n--- Nilai Re, Im, Magnitude, Phase di frekuensi dominan ---")
        for f_target in [cfg["f_bass"], cfg["f_treble"]]:
            idx = int(np.argmin(np.abs(analysis["f_pos"] - f_target)))
            print(
                f"f = {analysis['f_pos'][idx]:.0f} Hz | Re = {analysis['re_X'][idx]:8.2f} | "
                f"Im = {analysis['im_X'][idx]:8.2f} | |X| = {analysis['mag_X'][idx]:.4f} | "
                f"phi = {np.degrees(analysis['phase_X'][idx]):.1f} derajat"
            )

    plot_windowing(
        x,
        analysis,
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    comps = identify_frequency_components(analysis, threshold=cfg["threshold"], bass_limit=200)
    if verbose:
        print("\n--- (5b-iv) Identifikasi Komponen Frekuensi ---")
        for f_val, a_val, label in comps:
            print(f"  f = {f_val:6.1f} Hz | Amp = {a_val:.4f} | {label}")

    plot_clipping_impact(
        analysis,
        clip_level=cfg["clip_level"],
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    x2, X2 = build_comparison_spectrum(t, cfg)
    amp2 = get_amplitude(X2, analysis["pos_idx"], analysis["N"])
    similarity = np.dot(analysis["amp"], amp2) / (
        np.linalg.norm(analysis["amp"]) * np.linalg.norm(amp2)
    )

    if verbose:
        print("\n--- (5b-vi) Kemiripan Spektrum ---")
        print(f"Cosine similarity: {similarity:.4f}  (1.0 = identik)")

    plot_similarity(
        analysis,
        amp2,
        similarity,
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    plot_ma_spectrum(
        analysis,
        ma_window=cfg["ma_window"],
        output_dir=output_dir,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    if verbose:
        print("\n" + "=" * 50)
        print("RINGKASAN OUTPUT FILE:")
        print("  5a_time_domain.png      - Plot kawasan waktu (5a-i s/d v)")
        print("  5a_ops_addition.png     - Operasi penjumlahan (4 subplot)")
        print("  5a_ops_multiplication.png - Operasi perkalian (4 subplot)")
        print("  5a_ops_convolution.png  - Operasi konvolusi (4 subplot)")
        print("  5a_ops_corr_norm.png    - Korelasi & normalisasi")
        print("  5b_i_amp_energy.png     - Spektrum amplitudo & energi")
        print("  5b_ii_real_imag.png     - Komponen Re, Im, Phase")
        print("  5b_iii_windowing.png    - Perbandingan 4 window")
        print("  5b_v_clipping_freq.png  - Dampak clipping di frekuensi")
        print("  5b_vi_similarity.png    - Kemiripan spektrum")
        print("  5b_vii_ma_spectrum.png  - Efek moving average di frekuensi")
        print("=" * 50)

    return {
        "config": cfg,
        "time": t,
        "signal": x,
        "signal_clean": x_clean,
        "basic_operations": basic_ops,
        "analysis": analysis,
        "components": comps,
        "similarity": similarity,
        "signal_compare": x2,
        "amp_compare": amp2,
    }


def run_full_analysis_from_wav(
    file_path,
    config=None,
    show_plots=True,
    save_plots=True,
    output_dir=".",
    verbose=True,
):
    """Shortcut analisa full langsung dari file WAV dengan fs asli rekaman."""
    fs_wav, _, x_wav = load_wav_signal(file_path)
    return run_full_analysis(
        config=config,
        signal=x_wav,
        fs=fs_wav,
        show_plots=show_plots,
        save_plots=save_plots,
        output_dir=output_dir,
        verbose=verbose,
    )


def run_full_analysis_from_audio(
    file_path,
    config=None,
    target_fs=None,
    show_plots=True,
    save_plots=True,
    output_dir=".",
    verbose=True,
):
    """
    Shortcut analisa full langsung dari file audio (wav/mp3/aac/dll).

    target_fs:
    - None  -> pakai fs asli file
    - angka -> resample ke fs tersebut (via librosa)
    """
    fs_audio, _, x_audio = load_audio_signal(file_path, target_fs=target_fs)
    return run_full_analysis(
        config=config,
        signal=x_audio,
        fs=fs_audio,
        show_plots=show_plots,
        save_plots=save_plots,
        output_dir=output_dir,
        verbose=verbose,
    )


if __name__ == "__main__":
    run_full_analysis()
