# 🤖 Tajan — Intelligent User Intent Recognition Assistant


## 🧠 Project Overview

**Tajan** is an intelligent intent-recognition system built on deep bidirectional neural networks. Given a sentence or question from the user, it identifies the intended purpose from among **419 distinct topics** and returns a relevant, accurate, and efficient response.

The project was designed and implemented to provide a **fast, accurate, customizable** solution that runs **fully offline** — making it suitable for a wide range of industries such as customer support, education, healthcare, e-commerce, and municipal services.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Smart Bilingual Processing** | Automatic detection of Persian and English with appropriate lemmatizers |
| **High Response Speed** | Responds in under 0.5 seconds — suitable for thousands of concurrent users |
| **Solid Accuracy** | Achieves approximately **67% accuracy** across 419 intents |
| **Fully Offline Execution** | No internet connection required — ideal for underserved regions |
| **Easy Customization** | Build your own assistant by editing a single JSON file |
| **Smart Data Augmentation** | Uses Persian synonyms to increase training data by **1.8x** |
| **Graphical User Interface** | Simple interaction built with Tkinter |
| **High Scalability** | Adaptable to different industries by changing the dataset |
| **Low Cost** | Built entirely on free, open-source libraries |
| **Complete Documentation** | Step-by-step guide for installation, execution, and customization |

---

## 📋 System Requirements

| Requirement | Details |
|-------------|---------|
| **Python** | Version 3.10 or higher (recommended) |
| **pip** | Python package manager |
| **RAM** | Minimum 4 GB (for model training) |
| **OS** | Windows, Linux, or macOS (all supported) |

> **Note:** The project has been tested on Python 3.10

---

## 📁 Project Structure

```
tajan/
│
├── README.md
├── requirements.txt
│
├── Model_learning_code_and_files/
│   ├── tajan.ipynb
│   ├── intents.json
│   └── synonym_dict.json
│
├── chatbot/
│   ├── chat_with_model.py
│   ├── best_model.keras
│   ├── intents.json
│   └── preprocessed_data.pkl
│
└── generated_files/
    ├── training_log.csv
    ├── augmented_intents.json
    ├── output.png
    ├── marge.py
    └── logs/
```

---

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
cd tajan
pip install -r requirements.txt
```

> **Tip:** If you face network restrictions, use:
> ```bash
> pip install -r requirements.txt --index-url https://pypi.org/simple/
> ```

---

### Step 2: Run the Chatbot

```bash
cd chatbot
python chat_with_model.py
```

> **Note:** The initial model load may take up to 30 seconds.

---

### Step 3: Retrain the Model (Optional)

```bash
cd Model_learning_code_and_files
jupyter notebook tajan.ipynb
```

---

## 📂 Folder Descriptions

| Folder | Description |
|--------|-------------|
| `chatbot/` | Code and model for running the assistant |
| `Model_learning_code_and_files/` | Training code, dataset, and synonym dictionary |
| `generated_files/` | Output files from training (optional, for review) |

---

## ⚠️ Important Notes

- The first run may take up to **30 seconds** to load the model.
- Keep all files in their original folders — do not change the directory structure.
- No internet connection is required after installation.

---



## 📄 License

This project is licensed under the **MIT License** – feel free to use, modify, and distribute with proper attribution.

---

## 👨‍💻 Author

**Mahdi Razaghi**  

Email : phytonabcdefg1380@gmail.com

---

> 💡 *"A smarter future is built today."*
