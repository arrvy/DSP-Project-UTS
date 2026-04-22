# DSP-Project-UTS

NAMA :
1. Aulia Firmansyah (H1A024081)
2. Nashr Ardy Wahyono (H1A024087)
3. Hariri Febrianto Arkansyah (H1A024137)

Project dasar Digital Signal Processing (DSP) untuk UTS.

Fokus project:
- Operasi matematis dasar sinyal digital.
- Analisa domain waktu dan domain frekuensi.
- Arsitektur modular: logic utama ada di modul, script main hanya sebagai router input dan config.


## 1) Struktur Project

- `main.py`
	- Entry point sederhana.
	- Pilih sumber input (`synthetic` atau `audio`), set path audio, set override config.
	- Tidak berisi implementasi plot utama.

- `src/references.py`
	- Modul utama DSP.
	- Berisi generate/load sinyal, operasi dasar, analisa waktu, analisa frekuensi, dan semua plot.

- `assets/`
	- Tempat file audio input (mis. WAV/MP3/AAC).

- `report.ipynb`
	- Notebook untuk eksperimen/presentasi.


## 2) Fitur Utama

### A. Sumber Input Modular
- `synthetic`: sinyal sintetik (bass + treble + noise).
- `audio`: file audio nyata (WAV/MP3/AAC, dll) dengan sampling rate otomatis dari file.

### B. Operasi Matematis Dasar
Implementasi operasi di modul mencakup:
- Penjumlahan: `x + konstanta`.
- Perkalian: `x * operand` (impulse, step, ramp, ones).
- Konvolusi: `x * h` dengan kernel impulse/step/ramp/ones.
- Korelasi: korelasi terhadap kernel dan auto-korelasi (`x` dengan `x`).
- Normalisasi: peak normalization dan z-score.

Definisi sinyal dasar (diskrit):
- Impulse: delta[n]
- Step: u[n]
- Ramp: n*u[n]

### C. Analisa Domain Waktu
- Plot sinyal utama di domain waktu.
- Clipping sinyal.
- Daya sinyal (instantaneous power).
- RMS sinyal.
- Moving average (konvolusi sederhana).

### D. Analisa Domain Frekuensi
- Spektrum amplitudo dan energi.
- Komponen real, imajiner, dan phase.
- Perbandingan window (Rectangular, Hanning, Hamming, Blackman).
- Identifikasi komponen frekuensi (bass vs treble).
- Dampak clipping di domain frekuensi.
- Kemiripan spektrum (cosine similarity).
- Spektrum setelah moving average.


## 3) Parameter Konfigurasi Default

Konfigurasi default ada di fungsi `get_default_config()` pada `src/references.py`:

- `fs`: frekuensi sampling default (dipakai mode synthetic, atau fallback).
- `duration`: durasi sinyal synthetic (detik).
- `f_bass`, `f_treble`: frekuensi komponen synthetic.
- `A_bass`, `A_treble`: amplitudo komponen synthetic.
- `noise_std`: level noise Gaussian.
- `clip_level`: batas clipping (+/-).
- `ma_window`: panjang moving average (sample).
- `threshold`: ambang deteksi komponen frekuensi.
- `view_samples`: jumlah sample yang ditampilkan di plot waktu.


## 4) Dependencies

Install minimal:

- numpy
- scipy
- matplotlib

Install tambahan untuk input MP3/AAC:

- librosa
- audioread

Contoh install:

		pip install numpy scipy matplotlib librosa audioread

Catatan:
- Untuk sebagian file AAC/MP3, backend decode bisa butuh ffmpeg di sistem.


## 5) Cara Menjalankan

### A. Dari main.py

1. Buka `main.py`.
2. Atur parameter di bagian bawah file:
	 - `INPUT_SOURCE = "synthetic"` atau `"audio"`
	 - `AUDIO_PATH = ...` jika mode audio
	 - `CONFIG_OVERRIDES = {...}` jika ingin override parameter
3. Jalankan:

		python main.py

### B. Contoh Penggunaan dari Notebook / Script Lain

Panggil lewat main:

		import main
		result = main.main(
				input_source="audio",
				audio_path="assets/contoh.mp3",
				show_plots=True,
				save_plots=False,
				verbose=True,
		)

Panggil langsung modul:

		import src.references as ref
		result = ref.run_full_analysis(show_plots=True, save_plots=False)

Atau langsung dari audio:

		import src.references as ref
		result = ref.run_full_analysis_from_audio(
				file_path="assets/contoh.aac",
				show_plots=True,
				save_plots=False,
		)


## 6) Output Hasil

Secara default project bisa:
- Menampilkan plot interaktif (`show_plots=True`).
- Menyimpan file gambar (`save_plots=True`).

Nama output plot utama (jika save aktif) mencakup:
- `5a_time_domain.png`
- `5a_ops_addition.png`
- `5a_ops_multiplication.png`
- `5a_ops_convolution.png`
- `5a_ops_corr_norm.png`
- `5b_i_amp_energy.png`
- `5b_ii_real_imag.png`
- `5b_iii_windowing.png`
- `5b_v_clipping_freq.png`
- `5b_vi_similarity.png`
- `5b_vii_ma_spectrum.png`


## 7) Ringkasan Arsitektur

- `main.py`: input source + config router.
- `src/references.py`: seluruh implementasi DSP dan visualisasi.

Tujuan desain ini: mudah dipakai ulang, gampang dipahami, dan tetap fleksibel untuk synthetic maupun audio real.
