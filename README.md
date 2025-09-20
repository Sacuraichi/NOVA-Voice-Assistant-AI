import os
import re
import time
import queue
import webbrowser
from datetime import datetime
import subprocess
import requests
from googletrans import Translator  # pip install googletrans==4.0.0-rc1

# Speech recognition and TTS
import speech_recognition as sr
import pyttsx3

# Optional: OpenAI (only if available and API key is set)
USE_OPENAI = False
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")
try:
    from openai import OpenAI  # new SDK style
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        USE_OPENAI = True
except Exception:
    USE_OPENAI = False
    openai_client = None

# Optional: Vosk offline speech recognition (if installed)
VOSK_AVAILABLE = False
try:
    import vosk  # type: ignore
    VOSK_AVAILABLE = True
except Exception:
    VOSK_AVAILABLE = False

WAKE_WORDS = [r"\bhey nova\b", r"\bokay nova\b", r"\bhi nova\b"]
NAME = "Nova"

# ------------------ Utilities ------------------

def say(text: str):
    """Speak the given text with pyttsx3 (non-blocking)."""
    engine.say(text)
    engine.runAndWait()


def clean_text(t: str) -> str:
    """Normalize recognized text for easier intent parsing."""
    return re.sub(r"[^a-z0-9 ?!.,'-]+", " ", t.lower()).strip()


def heard_wake_word(text: str) -> bool:
    for pat in WAKE_WORDS:
        if re.search(pat, text):
            return True
    return False


def extract_after_wake_word(text: str) -> str:
    """Remove the wake word from the utterance so only the command remains."""
    t = text
    for pat in WAKE_WORDS:
        t = re.sub(pat, " ", t)
    return clean_text(t).strip()


def now_time_str():
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def now_date_str():
    return datetime.now().strftime("%A, %B %d, %Y")


def open_quick_site(cmd: str) -> bool:
    sites = {
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
        "facebook": "https://www.facebook.com",
        "gmail": "https://mail.google.com",
        "spotify": "https://open.spotify.com",
        "twitter": "https://x.com",
        "reddit": "https://www.reddit.com",
        "github": "https://github.com",
    }
    for key, url in sites.items():
        if key in cmd:
            webbrowser.open(url)
            say(f"Opening {key}.")
            return True
    return False


def do_web_search(query: str):
    webbrowser.open(f"https://www.google.com/search?q={query}")
    say("Here is what I found on the web.")


def small_talk(cmd: str) -> bool:
    patterns = {
        r"\bwho (are|r) you\b": f"I'm {NAME}, your voice assistant.",
        r"\bwhat('?| i)s your name\b": f"My name is {NAME}.",
        r"\bhow are you\b": "I'm doing great! How can I help?",
        r"\bthank(s| you)\b": "You're welcome!",
        r"\bgood (morning|afternoon|evening|night)\b": "Hello! How can I assist you?",
        r"\btell me a joke\b": "Why did the computer show up at work late? It had a hard drive!",
    }
    for pat, reply in patterns.items():
        if re.search(pat, cmd):
            say(reply)
            return True
    return False


def rule_based_handler(cmd: str) -> bool:
    """Handle common commands without using GPT."""
    if not cmd:
        return True

    if re.search(r"\b(exit|quit|goodbye|stop)\b", cmd):
        say("Goodbye!")
        raise SystemExit(0)

    if "time" in cmd:
        say(f"The time is {now_time_str()}.")
        return True

    if "date" in cmd or "day" in cmd:
        say(f"Today is {now_date_str()}.")
        return True

    if re.search(r"\bopen\b", cmd) and open_quick_site(cmd):
        return True

    m = re.search(r"(search|look up|google)\s+(.*)", cmd)
    if m:
        do_web_search(m.group(2))
        return True

    if small_talk(cmd):
        return True

    return False


def gpt_answer(prompt: str) -> str:
    """Ask OpenAI for a helpful response, fall back to a default string on error."""
    if not USE_OPENAI or not openai_client:
        return ""

    try:
        # Prefer chat.completions if available; otherwise responses API would be similar.
        # Using a conservative max_tokens and temperature for helpfulness.
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are Nova, a concise, friendly voice assistant. Keep answers short and practical for TTS."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return ""


# ------------------ Speech I/O ------------------

def transcribe_with_google(recognizer: sr.Recognizer, audio: sr.AudioData) -> str:
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""


def transcribe_with_vosk(recognizer: sr.Recognizer, audio: sr.AudioData) -> str:
    if not VOSK_AVAILABLE:
        return ""
    try:
        import json
        model_path = os.environ.get("VOSK_MODEL", "")  # set to a directory with a Vosk model
        if not model_path or not os.path.isdir(model_path):
            return ""  # no model available
        m = vosk.Model(model_path)
        rec = vosk.KaldiRecognizer(m, 16000)
        data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        rec.AcceptWaveform(data)
        result = json.loads(rec.Result())
        return result.get("text", "")
    except Exception:
        return ""


def listen_and_transcribe(recognizer: sr.Recognizer, mic: sr.Microphone, timeout=3, phrase_time_limit=6) -> str:
    with mic as source:
        # Dynamic energy threshold helps in noisy rooms
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        print("🎧 Listening...")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            return ""
    # Try offline first if configured; else Google
    text = transcribe_with_vosk(recognizer, audio)
    if not text:
        text = transcribe_with_google(recognizer, audio)
    return clean_text(text)


# ------------------ Main loop ------------------

def main():
    global engine

    # Initialize TTS
    engine = pyttsx3.init()
    engine.setProperty("rate", 185)   # speaking speed
    engine.setProperty("volume", 0.9) # volume

    # Pick a voice if available (try female first)
    try:
        voices = engine.getProperty("voices")
        preferred = None
        for v in voices:
            name = (v.name or "").lower()
            if any(k in name for k in ["zira", "female", "zira", "salli", "hazel"]):
                preferred = v.id
                break
        if preferred:
            engine.setProperty("voice", preferred)
    except Exception:
        pass

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    mic_index_env = os.environ.get("MIC_INDEX")
    mic_index = int(mic_index_env) if mic_index_env and mic_index_env.isdigit() else None

    # Pick a microphone
    try:
        if mic_index is not None:
            mic = sr.Microphone(device_index=mic_index)
        else:
            mic = sr.Microphone()
    except OSError as e:
        print("No microphone found. Plug one in and try again.")
        return

    say(f"Hello! I'm {NAME}. Say 'Hey Nova' to wake me.")
    print("Tip: Say 'hey nova' then your command. Examples:")
    print(" - hey nova, what's the time?")
    print(" - hey nova, open YouTube")
    print(" - hey nova, search Python voice recognition")
    print(" - hey nova, tell me a joke")
    print(" - hey nova, stop")

    while True:
        utterance = listen_and_transcribe(recognizer, mic)
        if not utterance:
            continue

        print("Heard:", utterance)

        # Wake-word mode: only act when user says the wake word
        if not heard_wake_word(utterance):
            continue

        cmd = extract_after_wake_word(utterance)
        if not cmd:
            say("Yes?")
            # listen once more for the actual command
            cmd = listen_and_transcribe(recognizer, mic, timeout=3, phrase_time_limit=6)
            print("Command:", cmd)

        # First try rule-based skills
        handled = rule_based_handler(cmd)

        # If not handled, try GPT (if configured)
        if not handled:
            answer = gpt_answer(cmd)
            if answer:
                say(answer)
            else:
                say("I didn't catch a specific command. I opened a web search for you.")
                do_web_search(cmd)

def rule_based_handler(cmd: str) -> bool:
    if not cmd:
        return True

    # Exit command
    if re.search(r"\b(exit|quit|goodbye|stop)\b", cmd):
        say("Goodbye!")
        raise SystemExit(0)

    # Time & Date
    if "time" in cmd:
        say(f"The time is {now_time_str()}.")
        return True
    if "date" in cmd or "day" in cmd:
        say(f"Today is {now_date_str()}.")
        return True

    # Open common sites
    if re.search(r"\bopen\b", cmd) and open_quick_site(cmd):
        return True

    # App launcher (Windows/macOS/Linux adjust paths
    # Open Notepad
    if "open notepad" in cmd:
        subprocess.Popen(["notepad"])
        say("Opening Notepad.")
        return True
    
    # Open Windows Calculator
    if "open calculator" in cmd:
        subprocess.Popen(["calc.exe"])
        say("Opening Calculator.")
        return True
    
    # Open Windows Settings
    if "open settings" in cmd:
        subprocess.Popen("start ms-settings:", shell=True)
        say("Opening Settings.")
        return True
    
    # Open Paint
    if "open paint" in cmd:
        subprocess.Popen("mspaint.exe")
        say("Opening Paint.")
        return True
    
    # Local music
    if "play music" in cmd:
        music_file = "C:/Users/alter/Music/_Earthquake_ by Flirtations(MP3_128K).mp3"  # change path
        os.startfile(music_file)  # works on Windows
        say("Playing your music.")
        return True
    
        # Small talk / fallback
    if small_talk(cmd):
        return True
    
    # Translation (Tagalog <-> English)
    if "translate" in cmd:
        try:
            from googletrans import Translator
            translator = Translator()

        # Extract the part after 'translate'
            parts = cmd.split("translate", 1)[1].strip()

            if not parts:
                say("Please tell me what to translate.")
                return True

            if "to tagalog" in parts:
                text = parts.replace("to tagalog", "").strip()
                if text:
                    result = translator.translate(text, dest="tl")
                    say(f"In Tagalog: {result.text}")
                else:
                    say("Please give me some text to translate to Tagalog.")
                return True

            elif "to english" in parts:
                text = parts.replace("to english", "").strip()
                if text:
                    result = translator.translate(text, dest="en")
                    say(f"In English: {result.text}")
                else:
                    say("Please give me some text to translate to English.")
                return True

            else:
                say("Please specify 'to Tagalog' or 'to English'.")
                return True

        except Exception as e:
            say("Sorry, I couldn’t translate that.")
            print("Translation error:", e)
            return True
        
    # Weather (requires OPENWEATHER_API_KEY)
    if "weather" in cmd:
        try:
            import requests

            api_key = os.environ.get("OPENWEATHER_API_KEY")
            if not api_key:
               say("Weather service not configured. Please set your OPENWEATHER_API_KEY.")
               return True

        # Detect city name
            if "weather in" in cmd:
                city = cmd.split("weather in", 1)[1].strip().title()
            else:
                city = "Manila"  # default city

            if not city:
                city = "Manila"

            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            r = requests.get(url).json()

            if r.get("cod") == 200:
                temp = r["main"]["temp"]
                desc = r["weather"][0]["description"]
                say(f"The weather in {city} is {desc} with {temp}°C.")
            elif r.get("cod") == "404":
                say(f"Sorry, I couldn’t find weather information for {city}.")
            else:
                say("I couldn’t fetch the weather right now.")

            return True
        
        except Exception as e:
            say("Something went wrong with the weather service.")
            print("Weather error:", e)
            return True
        
    # --- Open VS Code ---
    if "open vs code" in cmd or "open visual studio code" in cmd:
        try:
        # Default install path for VS Code on Windows
            vscode_path = r"C:\Users\alter\AppData\Local\Programs\Microsoft VS Code\Code.exe"

        # Expand %USERNAME% to actual user folder
            vscode_path = os.path.expandvars(vscode_path)

            if os.path.exists(vscode_path):
                subprocess.Popen(vscode_path)
                say("Opening Visual Studio Code.")
            else:
                say("I couldn’t find Visual Studio Code on your system.")
            return True
    
        except Exception as e:
            say("Something went wrong while opening Visual Studio Code.")
            print("VS Code error:", e)
            return True

    # --- Open Microsoft Excel ---
    if "open excel" in cmd or "open ms excel" in cmd:
        try:
            excel_path = r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"
            if os.path.exists(excel_path):
                subprocess.Popen(excel_path)
                say("Opening Microsoft Excel.")
            else:
                say("I couldn’t find Microsoft Excel on your system.")
            return True
        except Exception as e:
            say("Something went wrong while opening Excel.")
            print("Excel error:", e)
            return True
        
        # --- Open Microsoft PowerPoint ---
    if "open powerpoint" in cmd or "open ms powerpoint" in cmd or "open ppt" in cmd:
        try:
            ppt_path = r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE"
            if os.path.exists(ppt_path):
                subprocess.Popen(ppt_path)
                say("Opening Microsoft PowerPoint.")
            else:
                say("I couldn’t find Microsoft PowerPoint on your system.")
            return True
        except Exception as e:
            say("Something went wrong while opening PowerPoint.")
            print("PowerPoint error:", e)
            return True
        
        # --- Open Microsoft Word ---
    if "open word" in cmd or "open ms word" in cmd:
        try:
            word_path = r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"
            if os.path.exists(word_path):
                subprocess.Popen(word_path)
                say("Opening Microsoft Word.")
            else:
                say("I couldn’t find Microsoft Word on your system.")
            return True
        except Exception as e:
            say("Something went wrong while opening Word.")
            print("Word error:", e)
            return True
        
    return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting. Bye!")
