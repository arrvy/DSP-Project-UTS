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
    wav_path,
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

    if source == "wav":
        if not wav_path:
            raise ValueError("Mode wav butuh parameter wav_path.")
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"File WAV tidak ditemukan: {wav_path}")

        return ref.run_full_analysis_from_wav(
            file_path=wav_path,
            config=config,
            show_plots=show_plots,
            save_plots=save_plots,
            output_dir=output_dir,
            verbose=verbose,
        )

    raise ValueError("input_source harus 'synthetic' atau 'wav'.")


def main(
    input_source="synthetic",
    wav_path=None,
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
    - "wav"       -> load WAV dan otomatis pakai fs file
    """
    config = ref.get_default_config()
    if config_overrides:
        config.update(config_overrides)

    print(f"Menjalankan analisa utama dari src.references... mode={input_source}")

    result = run_analysis_by_source(
        input_source=input_source,
        wav_path=wav_path,
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
    # Ganti ke "wav" jika ingin analisa dari file WAV.
    INPUT_SOURCE = "synthetic"

    # Isi path jika INPUT_SOURCE = "wav".
    WAV_PATH = os.path.join("references", "garpu_tala_440hz.wav")

    # Override config jika perlu (contoh: {"duration": 2.0, "ma_window": 50})
    CONFIG_OVERRIDES = None

    main(
        input_source=INPUT_SOURCE,
        wav_path=WAV_PATH,
        config_overrides=CONFIG_OVERRIDES,
        show_plots=True,
        save_plots=False,
        output_dir=".",
        verbose=True,
    )
