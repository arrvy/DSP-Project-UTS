"""
=============================================================
DASAR MATEMATIS PEMROSESAN SINYAL DIGITAL
Untuk keperluan: Analisa Kawasan Waktu & Frekuensi
Tools: NumPy, SciPy, Matplotlib
=============================================================

STRUKTUR FILE INI:
  BAGIAN 0 — Import & Konfigurasi Global
  BAGIAN 1 — Generate / Load Sinyal
  BAGIAN 2 — Operasi Dasar Matematika Sinyal
  BAGIAN 3 — Analisa Kawasan Waktu (5a)
  BAGIAN 4 — Analisa Kawasan Frekuensi (5b)
  BAGIAN 5 — Helper Functions
"""

# ==============================================================
# BAGIAN 0 — IMPORT & KONFIGURASI GLOBAL
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.io import wavfile

# Styling global — supaya semua plot konsisten
plt.rcParams.update({
    'font.size'       : 11,
    'axes.titlesize'  : 12,
    'axes.labelsize'  : 11,
    'figure.dpi'      : 100,
    'lines.linewidth' : 1.5,
    'axes.grid'       : True,
    'grid.alpha'      : 0.4,
})


# ==============================================================
# BAGIAN 1 — GENERATE / LOAD SINYAL
# ==============================================================

# --- Parameter sampling ---
fs = 1000        # Frekuensi sampling (Hz) — minimal 2x frekuensi tertinggi (Nyquist)
T  = 1.0         # Durasi sinyal (detik)
N  = int(fs * T) # Jumlah sampel total
n  = np.arange(N)          # Array indeks diskrit [0, 1, 2, ..., N-1]
t  = n / fs                # Array waktu kontinu (detik)

# --- Opsi A: Sinyal sintetik komposit ---
# Komposisi: bass (50 Hz) + treble (300 Hz) + noise
# Ini ideal untuk pembelajaran karena kita TAHU isi frekuensinya
f_bass   = 50     # Hz
f_treble = 300    # Hz
A_bass   = 1.5    # Amplitudo komponen bass
A_treble = 0.8    # Amplitudo komponen treble

x_clean = (A_bass   * np.sin(2 * np.pi * f_bass   * t) +
           A_treble * np.sin(2 * np.pi * f_treble * t))
x = x_clean + 0.2 * np.random.randn(N)   # tambah noise Gaussian


# --- Opsi B: Load dari file WAV (uncomment jika pakai audio nyata) ---
# fs_wav, data = wavfile.read("garpu_tala_440hz.wav")
# x = data.astype(float) / np.max(np.abs(data))   # normalisasi ke [-1, 1]
# N = len(x)
# t = np.arange(N) / fs_wav


# ==============================================================
# BAGIAN 2 — OPERASI DASAR MATEMATIKA SINYAL
# ==============================================================
# Semua fungsi di sini adalah FONDASI dari analisa 5a dan 5b.
# Pahami ini dulu sebelum lihat plot-nya.

# --- 2.1 Clipping ---
# Clipping = memotong sinyal yang melebihi batas threshold.
# Secara matematis: x_clip[n] = clip(x[n], -A, +A)
# Efek di frekuensi: memunculkan harmonik baru (distorsi nonlinier)
clip_level = 1.2
x_clipped  = np.clip(x, -clip_level, clip_level)

# --- 2.2 Sinyal Daya ---
# Daya sesaat: p[n] = x[n]^2
# Ini adalah representasi energi sinyal di setiap titik waktu
x_power = x ** 2

# --- 2.3 RMS (Root Mean Square) ---
# RMS = ukuran "besar" sinyal secara efektif
# Formula: RMS = sqrt( (1/N) * sum(x[n]^2) )
# Untuk sinyal sinusoidal A*sin: RMS = A / sqrt(2)
rms_val = np.sqrt(np.mean(x_power))
print(f"RMS sinyal    : {rms_val:.4f}")
print(f"RMS teoritis  : {np.sqrt(A_bass**2/2 + A_treble**2/2):.4f}  (tanpa noise)")

# --- 2.4 Moving Average (Konvolusi) ---
# Moving average = low-pass filter sederhana
# Formula: y[n] = (1/M) * sum(x[n-k]) untuk k = 0..M-1
# Di frekuensi: ini menekan komponen frekuensi tinggi
M      = 20
kernel = np.ones(M) / M               # kernel konvolusi (bobot rata-rata)
x_ma   = np.convolve(x, kernel, mode='same')
# mode='same' → output panjang sama dengan input (ada edge effect di ujung)

# --- 2.5 FFT & Frekuensi Axis ---
# FFT menghasilkan array kompleks X[k] = Re{X[k]} + j*Im{X[k]}
# k = indeks frekuensi, bukan frekuensi dalam Hz
# Konversi: f_Hz = k * fs / N
X    = np.fft.fft(x)          # FFT sinyal asli
X_c  = np.fft.fft(x_clipped)  # FFT sinyal clipped
X_ma = np.fft.fft(x_ma)       # FFT sinyal moving average
freqs = np.fft.fftfreq(N, d=1/fs)   # array frekuensi dalam Hz

# Ambil sisi frekuensi positif saja (0 Hz sampai fs/2)
# Sisi negatif adalah mirror (conjugate symmetry) untuk sinyal real
pos_idx = freqs >= 0
f_pos   = freqs[pos_idx]

# Normalisasi amplitudo: kalikan 2 (kompensasi sisi negatif), bagi N (normalisasi)
def get_amplitude(X_fft, pos_idx, N):
    return np.abs(X_fft[pos_idx]) * 2 / N

amp    = get_amplitude(X,    pos_idx, N)
amp_c  = get_amplitude(X_c,  pos_idx, N)
amp_ma = get_amplitude(X_ma, pos_idx, N)

# --- 2.6 Komponen Riil dan Imajiner ---
# X[k] = Re{X[k]} + j * Im{X[k]}
# Re{X[k]} : kontribusi basis cosine di frekuensi k
# Im{X[k]} : kontribusi basis sine di frekuensi k (negatif untuk sin biasa)
#
# Untuk sinyal x[n] = A*sin(2pi*f0*n/fs):
#   Re{X[f0]} ≈ 0         (tidak ada komponen cosine)
#   Im{X[f0]} ≈ -A*N/2    (ada komponen sine)
#
# Untuk sinyal x[n] = A*cos(2pi*f0*n/fs):
#   Re{X[f0]} ≈ A*N/2     (ada komponen cosine)
#   Im{X[f0]} ≈ 0         (tidak ada komponen sine)
re_X = np.real(X[pos_idx])
im_X = np.imag(X[pos_idx])

# --- 2.7 Magnitude & Phase dari Re/Im ---
# Magnitude = sqrt(Re^2 + Im^2) → ini spektrum amplitudo
# Phase     = arctan(Im / Re)   → ini sudut fase sinyal di tiap frekuensi
mag_X   = np.sqrt(re_X**2 + im_X**2) * 2 / N
phase_X = np.arctan2(im_X, re_X)     # dalam radian, gunakan arctan2 (handle kuadran)

# Cek konsistensi: mag_X harus sama dengan amp
print(f"\nCek konsistensi magnitude:")
print(f"max |amp - mag_X| = {np.max(np.abs(amp - mag_X)):.2e}  (harus ≈ 0)")

# --- 2.8 Frekuensi Dominan ---
peak_idx  = np.argmax(amp)
peak_freq = f_pos[peak_idx]
print(f"\nFrekuensi dominan: {peak_freq:.1f} Hz  (amplitudo = {amp[peak_idx]:.4f})")


# ==============================================================
# BAGIAN 3 — PLOT KAWASAN WAKTU (5a)
# ==============================================================

fig, axes = plt.subplots(4, 1, figsize=(12, 14))
fig.suptitle("Analisa Kawasan Waktu", fontsize=14, fontweight='bold')

# (i) Sinyal asli
axes[0].plot(t[:500], x[:500], color='steelblue')
axes[0].set_title("(i) Sinyal x[n] — Domain Waktu")
axes[0].set_ylabel("Amplitudo")

# (ii) Clipping — overlay original vs clipped
axes[1].plot(t[:500], x[:500],         label="Original",              alpha=0.6)
axes[1].plot(t[:500], x_clipped[:500], label=f"Clipped ±{clip_level}", alpha=0.9, color='tomato')
axes[1].axhline( clip_level, color='red', linestyle='--', linewidth=0.8, label='Batas clip')
axes[1].axhline(-clip_level, color='red', linestyle='--', linewidth=0.8)
axes[1].set_title("(ii) Clipping — Pemotongan Sinyal")
axes[1].set_ylabel("Amplitudo")
axes[1].legend(loc='upper right')

# (iii) Sinyal daya + (iv) RMS ditampilkan sebagai garis referensi
axes[2].fill_between(t[:500], 0, x_power[:500], color='orange', alpha=0.6, label='Daya sesaat')
axes[2].axhline(rms_val**2, color='red', linestyle='--', label=f'Daya rata-rata (RMS²={rms_val**2:.3f})')
axes[2].set_title(f"(iii) Sinyal Daya  |  (iv) RMS = {rms_val:.4f}")
axes[2].set_ylabel("Daya (amplitudo²)")
axes[2].legend(loc='upper right')

# (v) Moving Average
axes[3].plot(t[:500], x[:500],    label="Original",           alpha=0.4, color='steelblue')
axes[3].plot(t[:500], x_ma[:500], label=f"Moving Avg (M={M})", color='darkblue', linewidth=2)
axes[3].set_title(f"(v) Konvolusi Moving Average (M={M}) — Efek Low-pass")
axes[3].set_ylabel("Amplitudo")
axes[3].set_xlabel("Waktu (s)")
axes[3].legend(loc='upper right')

plt.tight_layout()
plt.savefig("5a_time_domain.png", dpi=150)
plt.show()
print("Tersimpan: 5a_time_domain.png")


# ==============================================================
# BAGIAN 4 — PLOT KAWASAN FREKUENSI (5b)
# ==============================================================

# --- (i) Spektrum Amplitudo & Energi ---
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
fig.suptitle("(5b-i) Spektrum Amplitudo & Energi", fontsize=13, fontweight='bold')

axes[0].plot(f_pos, amp, color='steelblue')
axes[0].set_title("Spektrum Amplitudo |X[k]| (dinormalisasi)")
axes[0].set_ylabel("|X[k]|")
# Annotasi peak
for f_target in [f_bass, f_treble]:
    idx = np.argmin(np.abs(f_pos - f_target))
    axes[0].annotate(f'{f_pos[idx]:.0f} Hz\nA={amp[idx]:.2f}',
                     xy=(f_pos[idx], amp[idx]),
                     xytext=(f_pos[idx]+15, amp[idx]*0.9),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=9, color='red')

axes[1].plot(f_pos, amp**2, color='green')
axes[1].set_title("Spektrum Energi |X[k]|²")
axes[1].set_ylabel("|X[k]|²")
axes[1].set_xlabel("Frekuensi (Hz)")

for ax in axes:
    ax.set_xlim(0, 500)

plt.tight_layout()
plt.savefig("5b_i_amp_energy.png", dpi=150)
plt.show()


# --- (ii) Komponen Riil & Imajiner ← FOKUS UTAMA ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle("(5b-ii) Komponen Riil & Imajiner Spektrum DFT", fontsize=13, fontweight='bold')

# Re{X[k]}
axes[0].plot(f_pos, re_X, color='steelblue', label='Re{X[k]}')
axes[0].set_title("Re{X[k]} — Kontribusi Cosine\n"
                  "Bernilai ≈0 di sini karena sinyal kita dibangun dari sin(), bukan cos()")
axes[0].set_ylabel("Nilai Riil")
axes[0].axhline(0, color='black', linewidth=0.5)

# Im{X[k]}
axes[1].plot(f_pos, im_X, color='tomato', label='Im{X[k]}')
axes[1].set_title("Im{X[k]} — Kontribusi Sine\n"
                  "Peak negatif di 50Hz dan 300Hz karena DFT konvensi: sin → Im negatif")
axes[1].set_ylabel("Nilai Imajiner")
axes[1].axhline(0, color='black', linewidth=0.5)

# Phase angle
axes[2].plot(f_pos, np.degrees(phase_X), color='purple', linewidth=0.8)
axes[2].set_title("Phase φ[k] = arctan(Im/Re) dalam derajat\n"
                  "Frekuensi dominan mendekati -90° karena komponen adalah sine")
axes[2].set_ylabel("Phase (°)")
axes[2].set_xlabel("Frekuensi (Hz)")
axes[2].axhline(-90, color='red', linestyle='--', linewidth=0.7, label='-90° (pure sine)')
axes[2].axhline(  0, color='green', linestyle='--', linewidth=0.7, label='0° (pure cosine)')
axes[2].legend()
axes[2].set_ylim(-200, 200)

for ax in axes:
    ax.set_xlim(0, 500)

plt.tight_layout()
plt.savefig("5b_ii_real_imag.png", dpi=150)
plt.show()

# Print nilai numerik di frekuensi dominan untuk laporan
print("\n--- Nilai Re, Im, Magnitude, Phase di frekuensi dominan ---")
for f_target in [f_bass, f_treble]:
    idx = np.argmin(np.abs(f_pos - f_target))
    print(f"f = {f_pos[idx]:.0f} Hz | Re = {re_X[idx]:8.2f} | Im = {im_X[idx]:8.2f} | "
          f"|X| = {mag_X[idx]:.4f} | φ = {np.degrees(phase_X[idx]):.1f}°")


# --- (iii) Windowing ---
wind_dict = {
    'Rectangular': np.ones(N),
    'Hanning'    : windows.hann(N),
    'Hamming'    : windows.hamming(N),
    'Blackman'   : windows.blackman(N),
}
# Windowing diterapkan ke sinyal SEBELUM FFT.
# Tujuan: mengurangi spectral leakage (kebocoran energi ke frekuensi tetangga).
# Rectangular = tidak ada window (FFT biasa), leakage paling besar.
# Blackman = leakage paling kecil, tapi resolusi frekuensi juga berkurang.

fig, axes = plt.subplots(4, 1, figsize=(12, 14))
fig.suptitle("(5b-iii) Spektrum dengan Window Berbeda", fontsize=13, fontweight='bold')

for ax, (name, w) in zip(axes, wind_dict.items()):
    Xw   = np.fft.fft(x * w)
    ampw = np.abs(Xw[pos_idx]) * 2 / N
    ax.plot(f_pos, ampw)
    ax.set_title(f"Window: {name}")
    ax.set_ylabel("|X[k]|")
    ax.set_xlim(0, 500)

axes[-1].set_xlabel("Frekuensi (Hz)")
plt.tight_layout()
plt.savefig("5b_iii_windowing.png", dpi=150)
plt.show()


# --- (iv) Identifikasi Komponen: Bass vs Treble ---
threshold = 0.1   # batas minimum amplitudo untuk dianggap komponen signifikan
print("\n--- (5b-iv) Identifikasi Komponen Frekuensi ---")
for i, (f_val, a_val) in enumerate(zip(f_pos, amp)):
    if a_val > threshold:
        label = "BASS (< 200 Hz)" if f_val < 200 else "TREBLE (≥ 200 Hz)"
        print(f"  f = {f_val:6.1f} Hz | Amp = {a_val:.4f} | {label}")


# --- (v) Dampak Clipping pada Frekuensi ---
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
fig.suptitle("(5b-v) Dampak Clipping pada Spektrum Frekuensi", fontsize=13, fontweight='bold')

axes[0].plot(f_pos, amp,   color='steelblue', label='Original')
axes[0].set_title("Spektrum Original — hanya ada 50 Hz dan 300 Hz")

axes[1].plot(f_pos, amp_c, color='tomato', label='Clipped')
axes[1].set_title(f"Spektrum Clipped (±{clip_level}) — muncul harmonik baru akibat distorsi nonlinier")

for ax in axes:
    ax.set_ylabel("|X[k]|")
    ax.set_xlim(0, 500)
    ax.legend()
axes[-1].set_xlabel("Frekuensi (Hz)")

plt.tight_layout()
plt.savefig("5b_v_clipping_freq.png", dpi=150)
plt.show()


# --- (vi) Kemiripan Spektrum (Cosine Similarity) ---
# Cosine similarity mengukur sudut antara dua vektor spektrum.
# Nilai mendekati 1.0 = sangat mirip, mendekati 0 = tidak mirip.
# Formula: similarity = (A · B) / (||A|| * ||B||)
x2   = A_bass * np.sin(2*np.pi*f_bass*t) + 0.7 * np.sin(2*np.pi*f_treble*t)
X2   = np.fft.fft(x2)
amp2 = np.abs(X2[pos_idx]) * 2 / N

similarity = np.dot(amp, amp2) / (np.linalg.norm(amp) * np.linalg.norm(amp2))
print(f"\n--- (5b-vi) Kemiripan Spektrum ---")
print(f"Cosine similarity: {similarity:.4f}  (1.0 = identik)")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(f_pos, amp,  label="Sinyal 1 (dengan noise)", alpha=0.8)
ax.plot(f_pos, amp2, label="Sinyal 2 (clean, amplitudo berbeda)", alpha=0.8, linestyle='--')
ax.set_title(f"(5b-vi) Kemiripan Spektrum — Cosine Similarity = {similarity:.4f}")
ax.set_xlabel("Frekuensi (Hz)"); ax.set_ylabel("|X[k]|")
ax.set_xlim(0, 500); ax.legend()
plt.tight_layout()
plt.savefig("5b_vi_similarity.png", dpi=150)
plt.show()


# --- (vii) Spektrum Setelah Moving Average ---
# Moving average = konvolusi dengan kernel rata-rata = low-pass filter.
# Efek di frekuensi: sinc response — frekuensi tinggi dilemahkan.
# Semakin besar M, semakin agresif pelemahan frekuensi tinggi.

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
fig.suptitle("(5b-vii) Spektrum Setelah Moving Average", fontsize=13, fontweight='bold')

axes[0].plot(f_pos, amp,    color='steelblue', label='Original')
axes[0].set_title("Spektrum Original")
axes[1].plot(f_pos, amp_ma, color='purple',    label=f'Setelah MA (M={M})')
axes[1].set_title(f"Setelah Moving Average M={M} — komponen 300 Hz melemah, 50 Hz bertahan")

for ax in axes:
    ax.set_ylabel("|X[k]|"); ax.set_xlim(0, 500); ax.legend()
axes[-1].set_xlabel("Frekuensi (Hz)")
plt.tight_layout()
plt.savefig("5b_vii_ma_spectrum.png", dpi=150)
plt.show()


# ==============================================================
# BAGIAN 5 — RINGKASAN OUTPUT
# ==============================================================
print("\n" + "="*50)
print("RINGKASAN OUTPUT FILE:")
print("  5a_time_domain.png      — Plot kawasan waktu (5a-i s/d v)")
print("  5b_i_amp_energy.png     — Spektrum amplitudo & energi")
print("  5b_ii_real_imag.png     — Komponen Re, Im, Phase ← FOKUS")
print("  5b_iii_windowing.png    — Perbandingan 4 window")
print("  5b_v_clipping_freq.png  — Dampak clipping di frekuensi")
print("  5b_vi_similarity.png    — Kemiripan spektrum")
print("  5b_vii_ma_spectrum.png  — Efek moving average di frekuensi")
print("="*50)