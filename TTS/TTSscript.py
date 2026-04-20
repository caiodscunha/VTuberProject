# !pip install omnivoice
from omnivoice import OmniVoice
import soundfile as sf
import torch
import os

class TTSGenerator:
    def __init__(self, 
                 model_path: str = None, 
                 device: str = "cuda:0", 
                 dtype=torch.float16):
        """
        Initializes the TTS Generator and loads the OmniVoice model.
        """
        if model_path is None:
            # Pega o diretório atual onde este script está salvo e navega até a pasta Model/OmniVoice
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "Model", "OmniVoice")
            
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.model = None
        self.__load_model()

    def __load_model(self):
        """
        Loads the OmniVoice model into memory.
        """
        print(f"Loading model from {self.model_path}...")
        self.model = OmniVoice.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=self.dtype
        )
        print("Model loaded successfully.")

    def generate_audio(
        self, 
        text: str, 
        ref_text: str = None, 
        ref_audio: str = None, 
        output_path: str = "TempAudios\\out.wav", 
        sample_rate: int = 24000, 
        language: str = "pt",
        instruct: str = "",
        denoise: bool = True,
        class_temperature: int = 0,
        speed: float = 1.0
        ):
        """
        Generates audio from text, with optional zero-shot voice cloning if ref_text and ref_audio are provided, and saves it to a WAV file.
        
        :param text: The text you want the model to speak.
        :param ref_text: (Optional) Transcription of the reference audio for voice cloning.
        :param ref_audio: (Optional) Path to the reference audio sample file.
        :param output_path: Path where the output audio will be saved.
        :param sample_rate: Audio sampling rate (default 24kHz).
        :param language: Language of the text (default "pt").
        :param instruct: Whether to use instruction-based generation (default True).
        :param denoise: Whether to denoise the audio (default True).
        :param class_temperature: Temperature for class-based generation (default 1.0).
        :param speed: Speed of the generated audio (default 1.0).
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot generate audio.")
            
        print("Generating audio...")
        
        if ref_audio is not None and ref_text is not None:
            # Garante que ref_audio tenha o caminho completo baseado no local do script
            if not os.path.isabs(ref_audio):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                ref_audio = os.path.join(script_dir, ref_audio)

            audio = self.model.generate(
                text=text,
                ref_text=ref_text,
                ref_audio=ref_audio,
                language=language,
                instruct= instruct,
                denoise = denoise,
                class_temperature = class_temperature,
                speed = speed
            )
        else:
            audio = self.model.generate(
                text=text,
                language=language,
                instruct= instruct,
                denoise = denoise,
                class_temperature = class_temperature,
                speed = speed
            ) 
        
        # Se 'output_path' for um caminho relativo, junte com a pasta em que este script está
        if not os.path.isabs(output_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, output_path)

        # Garante que a pasta de destino exista antes de tentar salvar o arquivo
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        sf.write(output_path, audio[0], sample_rate)
        print(f"Audio successfully saved to {output_path}")
        
        return output_path
