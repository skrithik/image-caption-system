import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Load pre-trained InceptionV3 model
inception_model = InceptionV3(weights='imagenet')
inception_model = tf.keras.Model(inputs=inception_model.input, outputs=inception_model.layers[-2].output)

# Image preprocessing and feature extraction
def preprocess_image(uploaded_file):
    img = load_img(uploaded_file, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = inception_model.predict(img, verbose=0)
    return features.flatten()

# Greedy decoding
def greedy_generator(image_features, tokenizer, caption_model, max_caption_length, cnn_output_dim):
    in_text = 'start '
    for _ in range(max_caption_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_caption_length).reshape((1, max_caption_length))
        preds = caption_model.predict([image_features.reshape(1, cnn_output_dim), seq], verbose=0)
        idx = np.argmax(preds)
        word = tokenizer.index_word.get(idx, '')
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text.replace('start ', '').replace(' end', '').strip().lower()

# Beam search decoding
def beam_search_generator(image_features, tokenizer, caption_model, max_caption_length, cnn_output_dim, K_beams=3, log=False):
    start = [tokenizer.word_index['start']]
    beams = [[start, 0.0]]
    for _ in range(max_caption_length):
        candidates = []
        for seq, score in beams:
            padded = pad_sequences([seq], maxlen=max_caption_length).reshape((1, max_caption_length))
            preds = caption_model.predict([image_features.reshape(1, cnn_output_dim), padded], verbose=0)[0]
            for w in np.argsort(preds)[-K_beams:]:
                new_seq = seq + [w]
                new_score = score + (np.log(preds[w]) if log else preds[w])
                candidates.append([new_seq, new_score])
        beams = sorted(candidates, key=lambda x: x[1])[-K_beams:]

    best_seq = beams[-1][0]
    words = [tokenizer.index_word.get(i, '') for i in best_seq if i in tokenizer.index_word]
    if words and words[0] == 'start':
        words = words[1:]
    if 'end' in words:
        words = words[:words.index('end')]
    return ' '.join(words).strip().lower()
