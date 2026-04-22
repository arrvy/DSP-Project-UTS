"""
MAIN SCRIPT DSP UTS

File ini sengaja dibuat simple:
- urus input source (synthetic / wav)
- urus config
- panggil semua analisa dari src.references

Cara jalan:
    PYTHONPATH=. python main.py
"""

import os

import src.references as ref


def run_analysis_by_source(
    input_source,
    audio_path,
    config,
    show_plots,
    save_plots,
    output_dir,
    verbose,
):
    """Router analisa berdasarkan sumber input."""
    source = input_source.lower().strip()

    if source == "synthetic":
        return ref.run_full_analysis(
            config=config,
            show_plots=show_plots,
            save_plots=save_plots,
            output_dir=output_dir,
            verbose=verbose,
        )

    if source in ("audio", "wav", "mp3", "aac"):
        if not audio_path:
            raise ValueError("Mode audio butuh parameter audio_path.")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File audio tidak ditemukan: {audio_path}")

        return ref.run_full_analysis_from_audio(
            file_path=audio_path,
            config=config,
            show_plots=show_plots,
            save_plots=save_plots,
            output_dir=output_dir,
            verbose=verbose,
        )

    raise ValueError("input_source harus 'synthetic' atau 'audio'.")


def main(
    input_source="synthetic",
    audio_path=None,
    config_overrides=None,
    show_plots=True,
    save_plots=False,
    output_dir=".",
    verbose=True,
):
    """
    Entry point modular.

    input_source:
    - "synthetic" -> generate sinyal sintetik
    - "audio"     -> load WAV/MP3/AAC dan otomatis pakai fs file
    """
    config = ref.get_default_config()
    if config_overrides:
        config.update(config_overrides)

    print(f"Menjalankan analisa utama dari src.references... mode={input_source}")

    result = run_analysis_by_source(
        input_source=input_source,
        audio_path=audio_path,
        config=config,
        show_plots=show_plots,
        save_plots=save_plots,
        output_dir=output_dir,
        verbose=verbose,
    )
    print(f"Sampling rate yang dipakai: {result['config']['fs']} Hz")
    print("\nSelesai. main.py hanya sebagai input/config router.")
    return result


if __name__ == "__main__":
    # Ganti ke "audio" jika ingin analisa dari file WAV/MP3/AAC., "synthetic" jika mau menggunakan sound/data bawaan
    INPUT_SOURCE = "synthetic"

    # Isi path jika INPUT_SOURCE = "audio".
    # Contoh: os.path.join("references", "garpu_tala_440hz.wav")
    # Contoh: os.path.join("references", "contoh.mp3")
    # Contoh: os.path.join("references", "contoh.aac")
    AUDIO_PATH = os.path.join("assets", "67 sound.mp3")

    # Override config jika perlu (contoh: {"duration": 2.0, "ma_window": 50})
    CONFIG_OVERRIDES = None

        

    main(
        
        input_source=INPUT_SOURCE,
        audio_path=AUDIO_PATH,
        config_overrides=CONFIG_OVERRIDES,
        show_plots=True,
        save_plots=True,
        output_dir="./assets/image",
        verbose=True,
    )
