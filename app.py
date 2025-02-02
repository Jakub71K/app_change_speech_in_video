import os
import tempfile
import streamlit as st
import subprocess
import re
import uuid
import openai
import cv2
import numpy as np
import imageio_ffmpeg as ffmpeg

from io import BytesIO
from pydub import AudioSegment
from dotenv import dotenv_values
from openai import OpenAI
from PIL import ImageFont, ImageDraw, Image
from pydub.effects import speedup


AUDIO_TRANSCRIBE_MODEL = "whisper-1"
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

env = dotenv_values(".env")

# Funkcja do uzyskania klienta OpenAI
def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

if "changes_saved" not in st.session_state:
    st.session_state.changes_saved = False  # Domyślnie zmiany nie są zapisane

USD_TO_PLN = 4.04  # Kurs wymiany USD na PLN

# Koszty usług OpenAI w USD (przykładowe wartości, mogą się zmieniać)
COSTS = {
    "whisper": 0.006,  # USD za minutę audio
    "gpt-3.5-turbo": 0.002,  # USD za 1K tokenów
    "gpt-4": 0.03,  # USD za 1K tokenów
    "tts-1": 0.015  # USD za 1K znaków
}

def add_cost(service, amount, tokens=0):
    """Dodaje koszt usługi i liczy użyte tokeny"""
    if "cost_usd" not in st.session_state:
        st.session_state.cost_usd = 0.0
    if "cost_pln" not in st.session_state:
        st.session_state.cost_pln = 0.0

    cost = COSTS.get(service, 0) * amount
    st.session_state.cost_usd += cost
    st.session_state.cost_pln += cost * USD_TO_PLN


#  Funkcja do zwracania długości wideo w sekundach
def get_video_duration(video_path):
    ffprobe_command = [
        'ffprobe', '-i', video_path, '-show_entries', 'format=duration',
        '-v', 'quiet', '-of', 'csv=p=0'
    ]
    result = subprocess.run(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        duration = float(result.stdout.strip())
        return duration
    except ValueError:
        raise RuntimeError("Nie udało się odczytać długości wideo.")

# Funkcja do wyodrębnienia audio z wideo za pomocą ffmpeg
def extract_audio(video_path):
    audio_path = os.path.join("/tmp", f"{uuid.uuid4()}.mp3")  
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y']

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return audio_path  # Zwrot poprawnej ścieżki, jeśli ffmpeg się powiodło
    except subprocess.CalledProcessError as e:
        st.error(f" Błąd FFmpeg podczas ekstrakcji audio: {e.stderr}")
        return None  # Zwracamy `None`, aby program nie kontynuował z błędnym plikiem

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02}:{seconds:02}"

# Funkcja do transkrypcji audio za pomocą OpenAI
def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"

    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
    )

    if hasattr(transcript, "segments") and isinstance(transcript.segments, list):
        transcription_with_timestamps = []
        for segment in transcript.segments:
            transcription_with_timestamps.append({
                "word": re.sub(r"[^\w\s]", "", segment.text),  # ZAMIENIAMY segment["text"] na segment.text
                "start": getattr(segment, "start", 0),  # Poprawna metoda dostępu do atrybutu
                "end": getattr(segment, "end", 0)
            })
        return transcription_with_timestamps
    else:
        raise ValueError(f"Nie udało się znaleźć segmentów w odpowiedzi: {transcript}")



# Funkcja do dzielenia tekstu/audio na segmenty
def split_text_into_segments(transcription, max_duration=5, max_chars_per_second=14):
    segments = []
    current_segment = {"word": "", "start": None, "end": None}
    dynamic_max_chars = int(max_duration * max_chars_per_second)

    for entry in transcription:
        text = entry["word"]
        start = entry.get("start", 0)
        end = entry.get("end", 0)
        duration = end - start

        if len(text) > dynamic_max_chars:
            words = re.findall(r'\S+', text)
            temp_segment = {"word": "", "start": start, "end": start}
            for word in words:
                if len(temp_segment["word"]) + len(word) + 1 > dynamic_max_chars:
                    temp_segment["end"] = temp_segment["start"] + (len(temp_segment["word"]) / max_chars_per_second)
                    segments.append(temp_segment)
                    temp_segment = {"word": word, "start": temp_segment["end"], "end": None}
                else:
                    temp_segment["word"] += (" " + word if temp_segment["word"] else word)

            if temp_segment["word"]:
                temp_segment["end"] = temp_segment["start"] + (len(temp_segment["word"]) / max_chars_per_second)
                segments.append(temp_segment)
            continue

        if current_segment["start"] is None:
            current_segment["start"] = start

        current_end = current_segment["end"] if current_segment["end"] is not None else start

        if (len(current_segment["word"]) + len(text) <= dynamic_max_chars and
                (current_end - current_segment["start"] <= max_duration)):
            current_segment["word"] += (" " + text if current_segment["word"] else text)
            current_segment["end"] = end
        else:
            segments.append(current_segment)
            current_segment = {"word": text, "start": start, "end": end}

    if current_segment["word"]:
        segments.append(current_segment)

    return segments

# Funkcja do tłumaczenia treści mowy zawartej w pliku audio
def translate_audio(audio_bytes):
    client = openai.OpenAI(api_key=st.session_state["openai_api_key"])
    try:
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "audio.mp3"  # Wymagana nazwa pliku dla multipart/form-data
        response = client.audio.translations.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json"  # Pobieranie szczegółowych segmentów
        )

        if response and hasattr(response, "segments") and isinstance(response.segments, list):
            # Tworzenie segmentów na podstawie odpowiedzi
            segments = [
                {
                    "word": re.sub(r"[^\w\s]", "", segment.text),  # Poprawnie używamy segment.text
                    "start": getattr(segment, "start", 0),  # Użycie getattr() zamiast segment.get()
                    "end": getattr(segment, "end", 0)
                }
                for segment in response.segments
            ]

            # Uzupełnianie brakujących segmentów ciszą
            segments = fill_missing_segments(segments)
            return segments
        else:
            raise RuntimeError("Odpowiedź tłumaczenia jest pusta lub nie zawiera segmentów.")
    except Exception as e:
        raise RuntimeError(f"Błąd podczas tłumaczenia audio: {e}")


# Funkcja do generowania nowego audio na podstawie edytowanego tekstu
def generate_audio_from_text(transcription, voice):
    if not transcription or not all("word" in entry and "start" in entry and "end" in entry for entry in transcription):
        raise ValueError("Nieprawidłowe dane transkrypcji. Sprawdź segmenty.")

    audio_segments = []
    mp3_audio_path = os.path.join("/tmp", f"generated_audio_{uuid.uuid4()}.mp3")

    num_chars = sum(len(entry["word"]) for entry in transcription)  # Liczenie znaków w transkrypcji

    for i, entry in enumerate(transcription):
        text = entry["word"].strip()
        start = entry.get("start", 0)
        end = entry.get("end", 0)

        # Czas trwania segmentu w milisekundach
        segment_duration_ms = int((end - start) * 1000)

        try:
            if text:
                # Generowanie audio z tekstu
                openai_client = get_openai_client()
                response = openai_client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text
                )

                # Zapis na plik tymczasowy
                segment_audio_path = os.path.join("/tmp", f"segment_{uuid.uuid4()}.mp3")
                with open(segment_audio_path, "wb") as f:
                    f.write(response.content)

                # Załaduj segment audio
                segment_audio = AudioSegment.from_file(segment_audio_path, format="mp3")

                # Dostosowanie długości segmentu do docelowej
                audio_duration_ms = len(segment_audio)

                if audio_duration_ms < segment_duration_ms:
                    # Jeśli audio jest za krótkie, dodaj ciszę
                    silence_gap = segment_duration_ms - audio_duration_ms
                    segment_audio += AudioSegment.silent(duration=silence_gap)

                elif audio_duration_ms > segment_duration_ms:
                    speed_factor = audio_duration_ms / segment_duration_ms
                    if speed_factor > 1.0:  # Sprawdzamy, czy wartość speed_factor jest poprawna
                        segment_audio = speedup(segment_audio, playback_speed=speed_factor)

            else:
                # Jeśli segment nie zawiera tekstu, dodaj ciszę
                segment_audio = AudioSegment.silent(duration=segment_duration_ms)
            audio_segments.append(segment_audio)
        except Exception as e:
            raise RuntimeError(f"Błąd podczas generowania segmentu audio: {e}")

    if audio_segments:
        try:
            # Połączenie wszystkich segmentów audio
            combined_audio = sum(audio_segments)
            combined_audio.export(mp3_audio_path, format="mp3", bitrate="192k", parameters=["-q:a", "0", "-ar", "44100"])
        
            # Oblicz koszt generowania audio (TTS-1: 0.015 USD za 1K znaków)
            add_cost("tts-1", num_chars / 1000)

        finally:
            # Zwolnienie pamięci używanej przez segmenty
            for segment in audio_segments:
                del segment

        if not os.path.exists(mp3_audio_path) or os.path.getsize(mp3_audio_path) == 0:
            raise RuntimeError("Plik MP3 jest nieprawidłowy lub pusty.")

        return mp3_audio_path
    else:
        raise RuntimeError("Nie utworzono żadnych segmentów audio.")

# Funkcja do uzupełniania brakujących segmentów ciszą, jeśli występują przerwy czasowe między sąsiednimi segmentami transkrypcji.
def fill_missing_segments(transcription):
    if not transcription:
        return []  # Zwróć pustą listę, jeśli brak danych

    filled_transcription = []
    for i, segment in enumerate(transcription):
        filled_transcription.append(segment)

        # Sprawdź, czy istnieje luka między obecnym a następnym segmentem
        if i + 1 < len(transcription):
            current_end = segment["end"]
            next_start = transcription[i + 1]["start"]

            # Jeśli luka jest większa niż 0, dodaj pusty segment
            if next_start > current_end:
                filled_transcription.append({
                    "start": current_end,
                    "end": next_start,
                    "word": ""  # Pusty tekst dla luki
                })
    return filled_transcription

# Funkcja do połączenia audio z wideo za pomocą ffmpeg
def combine_video_with_audio(video_path, audio_path, output_path):
    ffmpeg_executable = ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_executable, '-i', video_path, '-i', audio_path,
        '-c:v', 'libx264', '-preset', 'fast',  # Rekodowanie do H.264
        '-c:a', 'aac', '-b:a', '192k', '-map', '0:v', '-map', '1:a', '-shortest',
        '-movflags', 'faststart',  # Optymalizacja do strumieniowania
        '-y', output_path  # Wymuszenie nadpisania pliku
    ]
    process = subprocess.run(command, capture_output=True, text=True)

    if process.returncode != 0:
        error_message = process.stderr if process.stderr else "Nieznany błąd FFmpeg"
        raise RuntimeError(f"Błąd FFmpeg: {error_message}")
    
if "video" in st.session_state and "audio" in st.session_state:
    if st.session_state.video and st.session_state.audio:
        output_video_path = os.path.join(tempfile.gettempdir(), f"output_video_{uuid.uuid4()}.mp4")

        #  Debugowanie: Sprawdź, czy audio istnieje
        st.write(f"🔎 Używane audio przed scaleniem: {st.session_state.audio}")
        
        if not os.path.exists(st.session_state.audio) or os.path.getsize(st.session_state.audio) == 0:
            st.error(" Nie znaleziono wygenerowanego pliku audio! Upewnij się, że zapisano zmiany.")
        else:
            combine_video_with_audio(st.session_state.video, st.session_state.audio, output_video_path)
            st.success("Scalanie zakończone!")

            #  Debugowanie: Sprawdź, czy nowe wideo istnieje
            if os.path.exists(output_video_path):
                st.write(f" Nowe wideo zapisane.")
            else:
                st.error(" Nie udało się zapisać nowego pliku wideo.")

# Funkcja weryfikująca poprawność klucza API OpenAI, sprawdzając możliwość wykonania zapytania testowego.
def verify_openai_api_key(api_key):
    try:
        # Ustaw klucz API
        openai.api_key = api_key
        # Wykonaj testowe zapytanie (np. lista modeli)
        openai.Model.list()
        return True
    except openai.error.AuthenticationError:
        return False
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")
        return False

# Funkcja dodająca tekst do video
def add_text_to_video(video_path, output_path, transcription, font_path="arial.ttf", font_size=24):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Nie można otworzyć pliku wideo.")

    # Pobierz informacje o wideo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Kodek do zapisu

    # Dynamiczne skalowanie wielkości czcionki
    font_scale = max(0.5, min(2, width / 1280))  # Skalowanie w stosunku do rozdzielczości
    adjusted_font_size = int(font_size * font_scale)

    # Załaduj poprawną czcionkę obsługującą polskie znaki
    font_paths = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",  # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux (alternatywa)
        "/Library/Fonts/Arial.ttf",  # MacOS
        "C:/Windows/Fonts/arial.ttf",  # Windows
    ]
    font_path = next((path for path in font_paths if os.path.exists(path)), font_path)

    try:
        font = ImageFont.truetype(font_path, adjusted_font_size)
    except IOError:
        raise RuntimeError(f"Nie można załadować czcionki: {font_path}")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    current_text = ""
    transcription_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Oblicz aktualny czas w sekundach
        current_time = frame_idx / fps

        # Aktualizuj tekst na podstawie czasu
        while (transcription_index < len(transcription) and
               current_time >= transcription[transcription_index]["start"]):
            current_text = transcription[transcription_index]["word"]
            transcription_index += 1

        # Rysuj tekst na ramce za pomocą PIL
        if current_text:
            # Konwertuj ramkę (OpenCV) na format PIL
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            # Oblicz pozycję tekstu na środku ekranu
            text_bbox = draw.textbbox((0, 0), current_text, font=font)  # Pobierz rozmiar tekstu
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (width - text_width) // 2  # Wyśrodkowanie tekstu
            text_y = height - 50  # Pozycja tekstu na dole

            # Rysuj tekst na wideo
            draw.text((text_x, text_y), current_text, font=font, fill=(255, 255, 255))

            # Konwertuj ramkę z powrotem na format OpenCV
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Sprawdź, czy plik wideo został wygenerowany poprawnie
    if os.path.exists(output_path):
        st.write(f"✅ Plik napisów utworzony: {output_path}")
    else:
        st.error(f"🚨 Nie udało się utworzyć pliku wideo z napisami: {output_path}")

# Funkcja tłumacząca tekst za pomocą modelu GPT
def translate_text_to_polish(text, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Można zmienić na "gpt-4" dla lepszych wyników
            messages=[
                {"role": "system", "content": "You are a translation assistant. Translate the text from English to Polish. The length of the translated text should be as close as possible to the length of the original. "},
                {"role": "user", "content": text}
            ]
        )
        # Zwraca przetłumaczony tekst
        translated_text = response.choices[0].message.content

        # Oblicz koszt tłumaczenia (GPT-3.5: 0.002 USD za 1K tokenów)
        num_tokens = len(text) / 4  # Przybliżona liczba tokenów (1 token ≈ 4 znaki)
        add_cost("gpt-3.5-turbo", num_tokens / 1000)

        return translated_text
    except Exception as e:
        raise RuntimeError(f"Błąd podczas tłumaczenia: {e}")

#
# Aplikacja Streamlit
#

def main():
    if "video" not in st.session_state:
        st.session_state.video = None

    if "audio" not in st.session_state:
        st.session_state.audio = None

    if "transcription" not in st.session_state:
        st.session_state.transcription = None

    st.title(":movie_camera: VocalCraft :movie_camera:")
    st.header("Twoje narzędzie do edycji wideo")
    st.text("VocalCraft to aplikacja do edycji wideo, która oferuje transkrypcję mowy z wideo na tekst w języku polskim i angielskim(z automatycznym tłumaczeniem). Pozwala na edycję tekstu z możliwością dostosowania treści nowego wideo. Generowanie nowych ścieżek audio na podstawie edytowanego tekstu. Dodawanie napisów do wideo oraz łączenie ścieżek audio z wideo, które można pobrać na dysk.")

    st.sidebar.title("Koszty OpenAI")
    st.sidebar.markdown(f"**Suma kosztów:**")
    st.sidebar.metric(label="USD", value=f"${st.session_state.get('cost_usd', 0):.4f}")
    st.sidebar.metric(label="PLN", value=f"{st.session_state.get('cost_pln', 0):.4f} zł")



    if not st.session_state.get("openai_api_key"):
        if "OPENAI_API_KEY" in env:
            st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
        else:
            st.info("Dodaj swój klucz API OpenAI, aby móc korzystać z tej aplikacji")
            st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
            # Sprawdź format klucza
            if re.match(r'^sk-[A-Za-z0-9]{48}$', st.session_state["openai_api_key"]):
                if verify_openai_api_key(st.session_state["openai_api_key"]):
                    st.success("Klucz API jest prawidłowy. Możesz korzystać z aplikacji.")
                    st.rerun()  # Przeładuj aplikację
                else:
                    st.error("Nieprawidłowy klucz API. Wprowadź poprawny klucz.")
                    st.stop

    if not st.session_state.get("openai_api_key"):
        st.stop()

    if "video" not in st.session_state:
        st.session_state.video = None
    if "audio" not in st.session_state:
        st.session_state.audio = None
    if "transcription" not in st.session_state:
        st.session_state.transcription = None

    uploaded_file = st.file_uploader("Prześlij plik wideo o maksymalnej długości 3 minut oraz rozmiarze nieprzekraczającym 200MB.", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("Plik jest za duży. Maksymalny rozmiar to 200MB.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

        try:
            # Sprawdzenie długości wideo
            video_duration = get_video_duration(video_path)
            if video_duration > 180:  # 3 minuty
                st.error(f"Przesłane wideo jest za długie ({video_duration:.2f} sekundy). Maksymalna długość to 180 sekund.")
                return
            else:
                st.session_state.video = video_path
                st.success(f"Wideo zostało załadowane. Długość: {video_duration:.2f} sekundy.")
        except Exception as e:
            st.error(f"Błąd podczas przetwarzania wideo: {e}")
            return

        if st.button("Wyodrębnij audio"):
            with st.spinner("Proszę czekać"):
                st.session_state.audio = extract_audio(st.session_state.video)
                st.success("Ścieżka audio została wyodrębniona, następnym krokiem jest transkrypcja na interesujący Cię język.")

    if st.session_state.audio:
        with open(st.session_state.audio, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")
        if st.button("Transkrypcja audio - język polski"):
            with st.spinner("Proszę czekać"):
                with open(st.session_state.audio, "rb") as f:
                    audio_bytes = f.read()
                try:
                    # Transkrypcja audio na tekst
                    transcription_segments = transcribe_audio(audio_bytes)

                    # Tłumaczenie tekstu każdego segmentu na polski
                    translated_segments = []
                    for segment in transcription_segments:
                        try:
                            # Przetłumacz tekst segmentu
                            translated_text = translate_text_to_polish(segment["word"], st.session_state["openai_api_key"])
                            # Dodaj przetłumaczony segment do wynikowej listy
                            translated_segments.append({
                                "word": translated_text,
                                "start": segment["start"],
                                "end": segment["end"]
                            })
                        except Exception as translation_error:
                            st.error(f"Błąd tłumaczenia segmentu: {translation_error}")
                            # Dodaj oryginalny segment w przypadku błędu tłumaczenia
                            translated_segments.append(segment)

                    # Zapisanie przetłumaczonych segmentów w stanie aplikacji
                    st.session_state.transcription = translated_segments

                    st.success("Transkrypcja zakończona! Po zakończeniu edycji tekstu zapisz zmiany.")
                except Exception as e:
                    st.error(f"Błąd transkrypcji: {e}")

        if st.button("Transkrypcja audio - język angielski"):
            with st.spinner("Proszę czekać"):
                with open(st.session_state.audio, "rb") as f:
                    audio_bytes = f.read()
                try:
                    translated_segments = translate_audio(audio_bytes)
                    st.session_state.transcription = translated_segments
                    st.success("Transkrypcja audio zakończona! Po zakończeniu edycji tekstu zapisz zmiany.")
                except Exception as e:
                    st.error(f"Błąd tłumaczenia: {e}")

    # Prędkość mowy klienta TTS (znaki na sekundę)
    TTS_SPEED = 12  # Możesz zmienić tę wartość na podstawie testów

    # Wypełnianie brakujących segmentów
    if st.session_state.transcription:
        st.session_state.transcription = fill_missing_segments(st.session_state.transcription)
    else:
        st.session_state.transcription = []

    # Edycja transkrypcji
    if st.session_state.transcription:
        st.write("Edycja transkrypcji")
        transcription_valid = True

        for i, entry in enumerate(st.session_state.transcription):
            start = entry.get("start", 0)
            end = entry.get("end", 0)
            segment_duration = end - start

            if start is None or not isinstance(start, (int, float)):
                st.error(f"Wpis {i}: 'start' jest nieprawidłowy (wartość: {start}).")
                transcription_valid = False
                continue

            if end is None or not isinstance(end, (int, float)):
                st.error(f"Wpis {i}: 'end' jest nieprawidłowy (wartość: {end}).")
                transcription_valid = False
                continue

            if end < start:
                st.error(f"Wpis {i}: 'end' ({end}) jest mniejszy niż 'start' ({start}).")
                transcription_valid = False
                continue

            # Oryginalny tekst segmentu
            original_text = entry["word"]
            original_max_chars = len(original_text)

            # Maksymalna liczba znaków (oryginalna liczba znaków lub dynamiczny limit)
            dynamic_max_chars = int(segment_duration * TTS_SPEED)
            max_chars = max(original_max_chars, dynamic_max_chars)

            # Formatowanie czasu na mm:ss
            formatted_start = format_time(start)
            formatted_end = format_time(end)

            current_text = entry["word"]
            edited_text = st.text_input(
                f"Od {formatted_start} do {formatted_end}, max {max_chars} znaków):",
                value=current_text,
                key=f"text_input_{i}",
                max_chars=max_chars
            )

            # Sprawdzanie długości wprowadzanego tekstu
            if len(edited_text) > max_chars:
                st.warning(
                    f"Segment {i}: Tekst przekracza maksymalną długość {max_chars} znaków. "
                    "Zostanie obcięty."
                )
                edited_text = edited_text[:max_chars]

            entry["word"] = edited_text

        if transcription_valid:
            voice_option = st.selectbox(
                "Wybierz głos:", 
                ["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"], 
                index=0
            )

            if st.button("Zapisz zmiany"):
                with st.spinner("Proszę czekać"):
                    transcription_path = os.path.join(tempfile.gettempdir(), "transcription.txt")
                    with open(transcription_path, "w") as f:
                        for entry in st.session_state.transcription:
                            f.write(f"{entry['start']:.2f} {entry['end']:.2f} {entry['word']}\n")
                    st.spinner(text="Proszę czekać, analizujemy dane")

                    try:
                        generated_audio_path = generate_audio_from_text(st.session_state.transcription, voice_option)
                        st.session_state.audio = generated_audio_path
                        st.success("Nowe audio zostało wygenerowane! Teraz czas na scalenie nowego audio z oryginalnym wideo.")
                        st.session_state.changes_saved = True

                        if os.path.exists(generated_audio_path):
                            with open(generated_audio_path, "rb") as audio_file:
                                audio_bytes = audio_file.read()
                                st.audio(audio_bytes, format="audio/mp3")
                        else:
                            st.error("Nie można znaleźć wygenerowanego pliku audio")
                    except Exception as e:
                        st.error(f"Błąd podczas generowania nowego audio: {str(e)}")

        if not st.session_state.changes_saved:
            st.info("Aby przejść dalej i móc scalić audio i wideo oraz dodać napisy zapisz zmiany.")
        else:
            if st.session_state.video and st.session_state.audio:
                # Pierwszy przycisk: Scal audio i wideo
                if st.button("Scal audio i wideo"):
                    with st.spinner("Proszę czekać"):
                        output_video_path = os.path.join(tempfile.gettempdir(), f"output_video_{uuid.uuid4()}.mp4")
                        try:
                            combine_video_with_audio(st.session_state.video, st.session_state.audio, output_video_path)

                            # Wyświetl scalone wideo
                            with open(output_video_path, "rb") as video_file:
                                video_bytes = video_file.read()
                                st.video(video_bytes)
                                st.download_button(
                                    "Pobierz wideo",
                                    data=video_bytes,
                                    file_name="Wideo.mp4",
                                    mime="video/mp4"
                                )
                                if os.path.exists(st.session_state.audio):
                                    with open(st.session_state.audio, "rb") as audio_file:
                                        audio_bytes = audio_file.read()
                                        st.download_button(
                                            "Pobierz audio",
                                            data=audio_bytes,
                                            file_name="Audio.mp3",
                                            mime="audio/mp3"
                                        )
                        except Exception as e:
                            st.error(f"Błąd podczas scalania wideo i audio: {e}")

                # Drugi przycisk: Scal audio, wideo i dodaj napisy
                if st.button("Scal audio i wideo oraz dodaj napisy"):
                    with st.spinner("Proszę czekać"):
                        temp_video_path = os.path.join("/tmp", f"temp_video_with_text_{uuid.uuid4()}.mp4")
                        output_video_path = os.path.join(tempfile.gettempdir(), f"output_video_with_text_and_audio_{uuid.uuid4()}.mp4")
                        try:
                            # Dodaj tekst do wideo
                            add_text_to_video(st.session_state.video, temp_video_path, st.session_state.transcription)
                            
                            # Połącz wideo z oryginalnym dźwiękiem
                            combine_video_with_audio(temp_video_path, st.session_state.audio, output_video_path)

                            # Wyświetl wideo z tekstem i dźwiękiem
                            with open(output_video_path, "rb") as video_file:
                                video_bytes = video_file.read()
                                st.video(video_bytes)
                                st.download_button(
                                    "Pobierz wideo z tekstem i dźwiękiem",
                                    data=video_bytes,
                                    file_name="Wideo_z_napisami.mp4",
                                    mime="video/mp4"
                                )
                            # Dodanie przycisku do pobrania samego audio
                            if os.path.exists(st.session_state.audio):
                                with open(st.session_state.audio, "rb") as audio_file:
                                    audio_bytes = audio_file.read()
                                    st.download_button(
                                        "Pobierz audio",
                                        data=audio_bytes,
                                        file_name="Audio.mp3",
                                        mime="audio/mp3"
                                    )

                        except Exception as e:
                            st.error(f"Błąd podczas dodawania tekstu do wideo lub łączenia z dźwiękiem: {e}")
if __name__ == "__main__":
    main()