# image-caption-system
# 🖼️ AI Image Caption Generator

This project generates image captions using deep learning (CNN + LSTM) with both **Greedy Search** and **Beam Search** decoding strategies. Built with **TensorFlow/Keras** and a stylish **Streamlit UI** for a smooth user experience.

---

## 📂 Repository Structure

```bash
.
├── app.py                 # Streamlit UI application
├── utils.py               # Feature extraction and caption generation logic
├── model/
│   ├── caption_model.h5   # Trained caption model weights
│   └── tokenizer.pkl      # Tokenizer object used during training
├── backend.ipynb          # Full training process to generate weights (optional)
├── requirements.txt       # All dependencies
└── README.md              # You are here!
```

---

## ✅ How to Run the App

### 1. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

Then open your browser at: `http://localhost:8501`

---

## 🛠️ To Train or Reproduce the Model Weights
If you want to generate `caption_model.h5` and `tokenizer.pkl` yourself:

> Run the Jupyter Notebook:
```
backend.ipynb
```
It walks you through the entire training pipeline using the Flickr8k dataset:
- Data cleaning
- Tokenization
- Feature extraction using InceptionV3
- LSTM training
- Model saving

---

## 🔗 Dataset
The dataset used for training and caption generation is the **Flickr8k** dataset, downloaded from Kaggle:

📎 [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

---

## 🔹 Features
- CNN Encoder + LSTM Decoder architecture
- InceptionV3 for image embeddings
- Custom caption generation using:
  - Greedy Search
  - Beam Search
- Beautiful, modern UI with animated loading and caption effects
- Session statistics and state tracking

---

## 💚 Acknowledgements
- Dataset: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Libraries: TensorFlow, Streamlit, NLTK, PIL

---

