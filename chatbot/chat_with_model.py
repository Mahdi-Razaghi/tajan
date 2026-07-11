"""
Chatbot Interface for Intent Recognition
---------------------------------------
Author and programmer of this project: Mahdi Razaghi
Date: 2026
"""


import re
import sys
import json
import nltk
import random
import pickle
import threading
import numpy as np
import tkinter as tk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from hazm import Lemmatizer, Normalizer
from tensorflow.keras.models import load_model
from tkinter import scrolledtext, font, messagebox

# ------------------------------------------------------------
# 1. WordNet fallback (for offline/low-bandwidth environments)
# ------------------------------------------------------------
# Attempt to download WordNet only once, with a short timeout.
# If download fails, we use a rule-based lemmatizer.
try:
    # Download quietly, raise on error, and set a timeout (10 sec)
    nltk.download('wordnet', quiet=True, raise_on_error=True)
    _has_wordnet = True
except Exception:
    _has_wordnet = False
    print("Warning: WordNet not available. Using simple fallback lemmatizer.")


class SimpleLemmatizer:
    """
    A lightweight, rule-based English lemmatizer.
    Serves as a fallback when WordNet is unavailable.
    Handles common irregular nouns/verbs and basic plural/verb forms.
    """

    def __init__(self):
        self.irregular_nouns = {
            "men": "man", "women": "woman", "children": "child",
            "mice": "mouse", "geese": "goose", "feet": "foot",
            "teeth": "tooth", "people": "person"
        }
        self.irregular_verbs = {
            "went": "go", "gone": "go", "done": "do", "did": "do",
            "seen": "see", "saw": "see", "been": "be", "was": "be",
            "were": "be", "had": "have", "has": "have"
        }

    def lemmatize(self, word):
        word = word.lower()
        # Check irregular forms
        if word in self.irregular_nouns:
            return self.irregular_nouns[word]
        if word in self.irregular_verbs:
            return self.irregular_verbs[word]

        # Plural rules
        if re.search(r'ies$', word):
            return re.sub(r'ies$', 'y', word)
        if re.search(r'ves$', word):
            return re.sub(r'ves$', 'f', word)
        if re.search(r'ses$|xes$|zes$|ches$|shes$', word):
            return word[:-2]
        if word.endswith('s') and not word.endswith('ss'):
            return word[:-1]

        # Verb forms (ing, ed)
        if re.search(r'([a-z])\1ing$', word):
            return word[:-4]
        if word.endswith('ing') and len(word) > 4:
            return word[:-3]
        if re.search(r'([a-z])\1ed$', word):
            return word[:-3]
        if word.endswith('ed') and len(word) > 3:
            return word[:-2]

        return word


# Choose the appropriate English lemmatizer
if _has_wordnet:
    lemmatizer_en = WordNetLemmatizer()
else:
    lemmatizer_en = SimpleLemmatizer()
    print("Using SimpleLemmatizer for English words.")

# ------------------------------------------------------------
# 2. Data loading and preprocessing utilities
# ------------------------------------------------------------
def load_data():
    """Load intents JSON and preprocessed pickle files."""
    try:
        with open("intents.json", "r", encoding="utf-8") as f:
            intents_data = json.load(f)
        intents = intents_data["intents"]
    except Exception as e:
        messagebox.showerror("Error", f"Cannot load intents.json: {e}")
        sys.exit(1)

    try:
        with open("preprocessed_data.pkl", "rb") as f:
            preprocessed = pickle.load(f)
        word_to_token = preprocessed["word_to_token"]
        max_len = preprocessed["max_len"]
        num_to_tag = preprocessed["num_to_tag"]
    except Exception as e:
        messagebox.showerror("Error", f"Cannot load preprocessed_data.pkl: {e}")
        sys.exit(1)

    return intents, word_to_token, max_len, num_to_tag


def custom_word_tokenize(text):
    """Tokenize text preserving words, numbers, and punctuation."""
    pattern = r"""
        \w+(?:'\w+)?   # words with optional apostrophe (e.g., don't)
        | [^\w\s]      # any non-alphanumeric, non-space character
        | \-           # hyphens
    """
    return re.findall(pattern, text, re.VERBOSE)


def detect_language(word: str) -> str:
    """Detect language of a single word: 'en', 'fa', 'mixed', or 'other'."""
    has_english = bool(re.search(r'[a-zA-Z]', word))
    has_persian_arabic = bool(re.search(
        r'[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]', word
    ))
    if has_english and not has_persian_arabic:
        return 'en'
    if has_persian_arabic and not has_english:
        return 'fa'
    if has_english and has_persian_arabic:
        return 'mixed'
    return 'other'


def preprocess_word(word, lemmatizer_fa, normalizer, lemmatizer_en):
    """Normalize, lemmatize, and clean a word based on its language."""
    word = word.strip()
    if not word:
        return None
    lang = detect_language(word)
    if lang == 'en':
        return lemmatizer_en.lemmatize(word.lower())
    if lang == 'fa':
        normalized = normalizer.normalize(word)
        lemmatized = lemmatizer_fa.lemmatize(normalized)
        return lemmatized.replace('#', '')
    return None  # discard mixed or other


def sentence_to_sequence(sentence, word_to_token, max_len,
                         lemmatizer_fa, normalizer, lemmatizer_en):
    """
    Convert a raw sentence into a padded sequence of token IDs.
    Uses the same vocabulary and preprocessing as during training.
    """
    ignore_letters = [
        '!', '?', ',', '.', '\u200c', "؟", "/",
        "#", "$", "^", "&", "*", "@", "%", "~"
    ]
    tokens = custom_word_tokenize(sentence)
    token_ids = []
    for token in tokens:
        if token in ignore_letters:
            continue
        processed = preprocess_word(token, lemmatizer_fa,
                                    normalizer, lemmatizer_en)
        if processed is not None:
            token_id = word_to_token.get(processed, 1)  # 1 = unknown token
            token_ids.append(token_id)

    if not token_ids:
        token_ids = [1]  # fallback to unknown token

    # Pad or truncate to max_len
    if len(token_ids) < max_len:
        token_ids += [0] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]

    return np.array([token_ids], dtype=np.int32)


def get_random_response(tag, intents):
    """Return a random response for the given intent tag."""
    for intent in intents:
        if intent["tag"] == tag:
            responses = intent.get("context", [])
            if responses:
                return random.choice(responses)
            return "هیچ پاسخی برای این مورد تعریف نشده است."
    return "متأسفم، پاسخی برای این مورد ندارم."


# ------------------------------------------------------------
# 3. GUI Application
# ------------------------------------------------------------
class ChatBotGUI:
    """Main GUI class for the chatbot."""

    def __init__(self, root):
        self.root = root
        self.root.title("تجن - دستیار هوشمند 🤖")
        self.root.geometry("600x700")
        self.root.configure(bg='#2C3E50')
        self.root.resizable(True, True)

        # Font for Persian/English text
        self.font_peyk = font.Font(family='Segoe UI', size=11)

        # Load resources
        (self.intents, self.word_to_token,
         self.max_len, self.num_to_tag) = load_data()
        # Load the trained model (disable compilation for inference)
        self.model = load_model("best_model.keras", compile=False)
        # Persian NLP tools
        self.lemmatizer_fa = Lemmatizer()
        self.normalizer = Normalizer()
        self.lemmatizer_en = lemmatizer_en

        # Build UI
        self.create_widgets()

        # Welcome message
        self.display_message("ربات", "سلام! من تجن هستم. چطور می‌توانم کمکتان کنم؟")

    def create_widgets(self):
        """Create and arrange all GUI components."""
        main_frame = tk.Frame(self.root, bg='#2C3E50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Chat display area (scrollable)
        self.chat_area = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, font=self.font_peyk,
            bg='#ECF0F1', fg='#2C3E50', state='disabled',
            height=20, borderwidth=2, relief="groove"
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Input frame
        input_frame = tk.Frame(main_frame, bg='#34495E')
        input_frame.pack(fill=tk.X, pady=5)

        self.entry = tk.Entry(
            input_frame, font=self.font_peyk, bg='white', fg='black',
            insertbackground='black', relief='flat', bd=3
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.entry.bind("<Return>", self.send_message)

        self.send_btn = tk.Button(
            input_frame, text="ارسال", command=self.send_message,
            bg='#E67E22', fg='white', font=('Segoe UI', 10, 'bold'),
            relief='flat', padx=10, cursor='hand2'
        )
        self.send_btn.pack(side=tk.RIGHT)

        # Status bar
        self.status_label = tk.Label(
            main_frame, text=" آماده", bg='#2C3E50', fg='#2ECC71',
            font=('Segoe UI', 9), anchor='w'
        )
        self.status_label.pack(fill=tk.X, pady=(5, 0))

    def display_message(self, sender, message):
        """Insert a message into the chat area with appropriate styling."""
        self.chat_area.config(state='normal')
        if sender == "شما":
            self.chat_area.insert(tk.END, f"🧑 شما: {message}\n", "user")
            self.chat_area.tag_config(
                "user", foreground="#2980B9",
                font=('Segoe UI', 10, 'bold')
            )
        else:
            self.chat_area.insert(tk.END, f"🤖 تجن : {message}\n", "bot")
            self.chat_area.tag_config(
                "bot", foreground="#E67E22",
                font=('Segoe UI', 10, 'bold')
            )
        self.chat_area.see(tk.END)
        self.chat_area.config(state='disabled')

    def send_message(self, event=None):
        """Handle user input and start a background thread for prediction."""
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.display_message("شما", user_input)
        self.entry.delete(0, tk.END)
        self.status_label.config(text="⏳ در حال پردازش...", fg="#F39C12")
        self.root.update_idletasks()

        # Run prediction in a separate thread to keep GUI responsive
        thread = threading.Thread(target=self.get_bot_response,
                                  args=(user_input,))
        thread.daemon = True
        thread.start()

    def get_bot_response(self, user_input):
        """
        Convert user input to sequence, run model prediction,
        and return a response based on confidence threshold.
        """
        try:
            # Preprocess input into a padded sequence
            seq = sentence_to_sequence(
                user_input,
                self.word_to_token,
                self.max_len,
                self.lemmatizer_fa,
                self.normalizer,
                self.lemmatizer_en
            )
            # Predict probabilities
            pred_probs = self.model.predict(seq, verbose=0)[0]
            pred_class = np.argmax(pred_probs)
            confidence = pred_probs[pred_class]

            # Confidence threshold (adjust as needed)
            if confidence < 0.4:
                response = ("متأسفم، مزی زاتوجه نشدم. "
                            "لطفاً سوال خود را واضح‌تر بپرسید.")
                tag = "unknown"
            else:
                tag = self.num_to_tag[pred_class]
                response = get_random_response(tag, self.intents)

            # Update GUI in the main thread
            self.root.after(0, self.display_message, "ربات", response)
            # Optional debug output (can be removed in production)
            print(f"[DEBUG] Tag: {tag} | Confidence: {confidence:.2f}")
            self.root.after(0, self.update_status, " آماده")

        except Exception as e:
            error_msg = f"خطا در پردازش: {str(e)}"
            self.root.after(0, self.display_message, "ربات", error_msg)
            self.root.after(0, self.update_status, "! خطا")

    def update_status(self, text):
        """Update the status label color based on message."""
        self.status_label.config(text=text)
        if text == " آماده":
            self.status_label.config(fg="#2ECC71")
        elif "خطا" in text:
            self.status_label.config(fg="#E74C3C")
        else:
            self.status_label.config(fg="#F39C12")


# ------------------------------------------------------------
# 4. Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    # Set random seed for reproducible response selection (optional)
    random.seed(42)
    root = tk.Tk()
    app = ChatBotGUI(root)
    root.mainloop()