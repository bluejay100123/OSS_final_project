import torch
import queue
import speech_recognition as sr
import numpy as np
import os
import time
import random
import tempfile
import platform
from collections import deque

from mic.utils import get_logger

import openai

class WhisperMic:
    def __init__(self,model="base",device=("cuda" if torch.cuda.is_available() else "cpu"),english=False,verbose=False,energy=300,pause=2,dynamic_energy=False,save_file=False, model_root="~/.cache/whisper",mic_index=None,implementation="faster_whisper",hallucinate_threshold=220):

        self.logger = get_logger("whisper_mic", "info")
        self.energy = energy
        self.hallucinate_threshold = hallucinate_threshold
        self.pause = pause
        self.dynamic_energy = dynamic_energy
        self.save_file = save_file
        self.verbose = verbose
        self.english = english

        self.platform = platform.system()

        if self.platform == "darwin":
            if device == "mps":
                self.logger.warning("Using MPS for Mac, this does not work but may in the future")
                device = "mps"
                device = torch.device(device)

        if (model != "large" and model != "large-v2" and model != "large-v3" and model != "tiny.en" and model != "base.en" and model != "small.en" and model != "medium.en") and self.english:
            model = model + ".en"

        model_root = os.path.expanduser(model_root)

        self.faster = False
        if (implementation == "faster_whisper"):
            try:
                from faster_whisper import WhisperModel
                self.audio_model = WhisperModel(model, download_root=model_root, device="cuda", compute_type="float16")            
                self.faster = True    # Only set the flag if we succesfully imported the library and opened the model.
            except ImportError:
                self.logger.error("faster_whisper not installed, falling back to whisper")
                import whisper
                self.audio_model = whisper.load_model(model, download_root=model_root).to(device)

        else:
            import whisper
            self.audio_model = whisper.load_model(model, download_root=model_root).to(device)
        
        self.temp_dir = tempfile.mkdtemp() if save_file else None

        self.audio_queue = queue.Queue()
        self.result_queue: "queue.Queue[str]" = queue.Queue()
        
        self.break_threads = False
        self.mic_active = False

        self.banned_results = [""," ","\n",None]

        if save_file:
            self.file = open("transcribed_text.txt", "w+", encoding="utf-8")

        self.__setup_mic(mic_index)

        os.environ["OPENAI_API_KEY"] = ""

        self.client = openai.OpenAI()


    def __setup_mic(self, mic_index):
        if mic_index is None:
            self.logger.info("No mic index provided, using default")
        self.source = sr.Microphone(sample_rate=16000, device_index=mic_index)

        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy
        self.recorder.pause_threshold = self.pause
        self.recorder.dynamic_energy_threshold = self.dynamic_energy

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        self.logger.info("Mic setup complete")

    # Whisper takes a Tensor while faster_whisper only wants an NDArray
    def __preprocess(self, data):
        is_audio_loud_enough = self.is_audio_loud_enough(data)
        if self.faster:
            return np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0,is_audio_loud_enough
        else:
            return torch.from_numpy(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0),is_audio_loud_enough
        
    def is_audio_loud_enough(self, frame):
        audio_frame = np.frombuffer(frame, dtype=np.int16)
        amplitude = np.mean(np.abs(audio_frame))
        return amplitude > self.hallucinate_threshold

    
    def __get_all_audio(self, min_time: float = -1.):
        audio = bytes()
        got_audio = False
        time_start = time.time()
        while not got_audio or time.time() - time_start < min_time:
            while not self.audio_queue.empty():
                audio += self.audio_queue.get()
                got_audio = True

        data = sr.AudioData(audio,16000,2)
        data = data.get_raw_data()
        return data
    

    # Handles the task of getting the audio input via microphone. This method has been used for listen() method
    def __listen_handler(self, timeout, phrase_time_limit):
        try:
            with self.source as microphone:
                audio = self.recorder.listen(source=microphone, timeout=timeout, phrase_time_limit=phrase_time_limit)
            self.__record_load(0, audio)
            audio_data = self.__get_all_audio()
            self.__transcribe(data=audio_data)
        except sr.WaitTimeoutError:
            self.result_queue.put_nowait("Timeout: No speech detected within the specified time.")
        except sr.UnknownValueError:
            self.result_queue.put_nowait("Speech recognition could not understand audio.")


    # This method is similar to the __listen_handler() method but it has the added ability for recording the audio for a specified duration of time
    def __record_handler(self, duration, offset):
        with self.source as microphone:
            audio = self.recorder.record(source=microphone, duration=duration, offset=offset)
        
        self.__record_load(0, audio)
        audio_data = self.__get_all_audio()
        self.__transcribe(data=audio_data)


    # This method takes the recorded audio data, converts it into raw format and stores it in a queue. 
    def __record_load(self,_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.audio_queue.put_nowait(data)

    def __transcribe_forever(self) -> None:
        while True:
            if self.break_threads:
                break
            self.__transcribe()

    def __transcribe(self,data=None, realtime: bool = False):
        if data is None:
            audio_data = self.__get_all_audio()
        else:
            audio_data = data
        audio_data,is_audio_loud_enough = self.__preprocess(audio_data)

        if is_audio_loud_enough:
            predicted_text = ''
            # faster_whisper returns an iterable object rather than a string
            if self.faster:
                self.verbose = False
                segments, info = self.audio_model.transcribe(audio_data, language="ko")
                for segment in segments:
                    predicted_text += segment.text
                return predicted_text
        return None

    def stt_to_gpt(self, stt="안녕하세요!", num=5, responses=deque()):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 라이브 스트림 방송을 보는 한국인 시청자야. 너는 스트리머에게 댓글 형식으로 너의 의견이나 반응을 표현하게 되어있어. 댓글은 한글로 작성하며 의성어도 사용가능해. 스트리머가 말하는 내용의 핵심을 짚고 이와 관련한 내용으로 댓글은 반말로 작성해야해. 기분에 따라 이모티콘이나 의성어를 활용할 줄 알아야하고 장난기 많은 고등학생처럼 반말해야해."},
                {"role": "user", "content": stt}
            ],
            n=num,
            temperature=1,
            max_tokens=30
        )
        for i in range(num):
            responses.append(response.choices[i].message.content)

    # Complete this function
    def listen(self, timeout=None, phrase_time_limit=None, responses=deque(), lock=None):
        if self.mic_active:
            self.logger.info("Microphone is already in use.")
            return
        
        self.mic_active = True
        try:
            with self.source as microphone:
                audio = self.recorder.listen(source=microphone, timeout=timeout, phrase_time_limit=phrase_time_limit)
                self.__record_load(0, audio)
                audio_data = self.__get_all_audio()
                transcription = self.__transcribe(data=audio_data)
                print("listen function")
                if transcription is not None:
                    if lock:
                        with lock:
                            print(f'stt_to_gpt: {transcription}')
                            self.stt_to_gpt(stt=transcription, num=random.randint(5,9), responses=responses)
                    else:
                        print("stt_to_gpt no_lock")
                        self.stt_to_gpt(stt=transcription, responses=responses)
        finally:
            self.mic_active = False