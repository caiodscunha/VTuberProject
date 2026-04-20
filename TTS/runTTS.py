from TTSscript import TTSGenerator

if __name__ == "__main__":
    # Example usage
    tts = TTSGenerator()
    tts.generate_audio(
        text="TEXTO QUE SERÁ FALADO",
        ref_text="FALA DO AUDIO REFERENCIA",
        ref_audio="refAudio.wav",
        output_path="TempAudios\\out.wav",
        denoise = True,
        class_temperature=1,
        speed=1.4
    )