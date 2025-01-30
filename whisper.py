import time
import keyboard
import os
import pyaudio
import wave
import whisper

class SpeechToTextManager:
    model = None

    def __init__(self):
        try:
            self.model = whisper.load_model("base")  # Load the Whisper model
        except Exception as e:
            exit(f"Failed to load Whisper model: {e}")

    def record_audio(self, filename, duration=10):
        # Records audio from the default microphone
        chunk = 1024  # Record in chunks of 1024 samples
        format = pyaudio.paInt16  # 16 bits per sample
        channels = 1
        rate = 44100  # 44.1 kHz sampling rate

        p = pyaudio.PyAudio()

        print("Recording...")

        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

        frames = []

        for _ in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)

        print("Finished recording.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

    def speechtotext_from_mic(self):
        temp_file = "temp_audio.wav"
        self.record_audio(temp_file)

        print("Transcribing audio...")
        result = self.model.transcribe(temp_file)
        os.remove(temp_file)  # Clean up temporary file
        text_result = result["text"]

        print(f"We got the following text: {text_result}")
        return text_result

    def speechtotext_from_file(self, filename):
        print("Listening to the file...")
        result = self.model.transcribe(filename)
        text_result = result["text"]

        print(f"Recognized: {text_result}")
        return text_result

    def speechtotext_from_file_continuous(self, filename):
        print("Processing the audio file continuously...")
        result = self.model.transcribe(filename)
        final_result = result["text"].strip()

        print(f"\n\nHere's the result we got from continuous file read:\n\n{final_result}\n\n")
        return final_result

    def speechtotext_from_mic_continuous(self, stop_key='p'):
        temp_file = "temp_audio.wav"
        all_results = []
        done = False

        def stop_recording():
            nonlocal done
            done = True

        print("Continuous Speech Recognition is now running, press '{}' to stop.".format(stop_key))
        keyboard.add_hotkey(stop_key, stop_recording)

        while not done:
            self.record_audio(temp_file, duration=5)  # Record 5 seconds chunks
            print("Transcribing audio chunk...")
            result = self.model.transcribe(temp_file)
            os.remove(temp_file)  # Clean up temporary file
            all_results.append(result["text"])

        final_result = " ".join(all_results).strip()
        print(f"\n\nHere's the result we got:\n\n{final_result}\n\n")
        return final_result


# Tests
if __name__ == '__main__':
    TEST_FILE = "D:\\Video Editing\\Misc - Ai teaches me to pass History Exam\\Audio\\Misc - Ai teaches me to pass History Exam - VO 1.wav"

    speechtotext_manager = SpeechToTextManager()

    while True:
        # Uncomment the test you'd like to run
        # speechtotext_manager.speechtotext_from_mic()
        # speechtotext_manager.speechtotext_from_file(TEST_FILE)
        # speechtotext_manager.speechtotext_from_file_continuous(TEST_FILE)
        result = speechtotext_manager.speechtotext_from_mic_continuous()
        print(f"\n\nHERE IS THE RESULT:\n{result}")
        time.sleep(60)