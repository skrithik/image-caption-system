# 🔹 Project Report: AI Image Caption Generator

---

## 🌍 Introduction
This project aims to build a deep learning-based image captioning system that automatically generates descriptive textual captions for images. We leverage a CNN + LSTM architecture to extract image features and generate context-aware sentences using Greedy and Beam Search decoding strategies.

---

## 📚 Dataset Description
- **Dataset:** [Flickr8k Dataset (Kaggle)](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Images:** 8000 images depicting various everyday scenes
- **Captions:** Each image is associated with 5 human-written captions

### Preprocessing
- Removed punctuation and digits
- Converted to lowercase
- Added special tokens: `start`, `end`
- Tokenized using Keras Tokenizer

---

## 🤖 Model Architecture & Methodology

### 1. **Feature Extraction (Encoder)**
- Used **InceptionV3** pre-trained on ImageNet
- Removed the final classification layer to obtain a 2048-dimensional feature vector
- Features extracted for each image and stored in dictionaries (`train_image_features`, etc.)

### 2. **Text Processing**
- Cleaned all captions
- Tokenized using `Tokenizer`
- Padded to `max_caption_length`
- Converted outputs to one-hot vectors for training

### 3. **Decoder (Caption Generator)**
- Input 1: Extracted image features → Dense(256)
- Input 2: Sequence of tokens → Embedding → LSTM(256)
- Combined both inputs → Dense → Softmax

### 4. **Training**
- Used custom data generator to feed sequences
- Optimizer: `Adam(learning_rate=0.01, clipnorm=1.0)`
- Loss: `categorical_crossentropy`
- Callbacks: `EarlyStopping`, `LearningRateScheduler`
- Saved trained model as `caption_model.h5`
- Saved tokenizer as `tokenizer.pkl`

---

## 🔧 Decoding Strategies

### Greedy Search
- Picks the word with maximum probability at each step
- Fast but may miss better captions

### Beam Search
- Keeps top-K probable sequences at each step
- Higher quality captions by exploring multiple paths
- Configurable with `K_beams`

---

## 🌐 Web App (app.py)
Built using **Streamlit** for an interactive image captioning interface.

### Features:
- Upload image
- Generates two captions: Greedy and Beam
- Displays side-by-side results
- Animated UI with CSS (`style.css`)
- Session metrics: caption count, processing time

---

## ⚖️ Technologies Used

| Component           | Library/Tool        |
|--------------------|---------------------|
| Deep Learning       | TensorFlow, Keras   |
| Image Processing    | PIL, NumPy          |
| NLP / BLEU          | NLTK                |
| Web Framework       | Streamlit           |
| Pretrained Network  | InceptionV3         |
| Styling             | Custom CSS          |

---

## 🚫 Limitations
- BLEU scores may be low for unseen images
- Beam search is slower than greedy
- No attention mechanism (yet)

---

## ✨ Future Improvements
- Add attention layer (e.g., Bahdanau or Luong)
- Use larger datasets like MS-COCO
- Enable download/share/export options in app
- Integrate BLEU evaluation on test set

---

## 💡 How to Reproduce Training
Run the notebook:

```bash
backend.ipynb
```
This file walks through:
- Data loading & cleaning
- Feature extraction
- Model training
- Saving weights & tokenizer

Outputs:
- `model/caption_model.h5`
- `model/tokenizer.pkl`

---


