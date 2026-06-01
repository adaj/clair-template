import os
# Set TF_CPP_MIN_LOG_LEVEL to 3 to avoid the logging of warnings from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set TF_CUDNN_DETERMINISTIC to 1 to enforce determinism for cuDNN
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import numpy as np
import unidecode
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub


# URL to the TFHub model
TFHUB_LANGUAGE_FEATURIZER = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
STOP_WORDS_LANGUAGES = [file.split('.')[0] for file in os.listdir('nlu/stop_words') if file.endswith('.txt')]


def create_encoder(language_featurizer: str = TFHUB_LANGUAGE_FEATURIZER):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    encoder = hub.KerasLayer(language_featurizer)(text_input)
    encoder = tf.keras.Model(text_input, encoder)
    return encoder


def remove_special_characters(string):
    return unidecode.unidecode(string.encode('cp1252', 'ignore').decode('cp1252').strip())


class TopicEmbeddings:

    def __init__(self,
                 encoder,
                 keywords: str,
                 stop_words_file: str = None):
        assert keywords is not None, "Provide some keywords."
        # Join keywords together in one string
        if isinstance(keywords, list):
            keywords = ", ".join(keywords)
        # Load stop words if in STOP_WORDS_LANGUAGES
        self.stop_words = []
        if stop_words_file and stop_words_file.split('.')[0][-2:] in STOP_WORDS_LANGUAGES:
            with open(stop_words_file, 'r', encoding='utf-8') as f:
                self.stop_words = f.read().split('\n')
        # Apply preprocessing and remove stop words if provided
        if self.stop_words:
            keywords = ' '.join([word for word in keywords.split() if word not in self.stop_words])
        keywords = remove_special_characters(keywords).lower()
        self.keywords = keywords
        # Create encoder and compute topic_embeddings of keywords
        self.encoder = encoder
        self.topic_embeddings = self.encoder(tf.constant(self.keywords))

    def get_topic_similarity(self, text: str):
        # Apply preprocessing and remove stop words if provided
        if self.stop_words:
            text = ' '.join([word for word in text.split() if word not in self.stop_words])
        text = remove_special_characters(text).lower()
        # Compute embeddings
        sentence_embeddings = self.encoder(tf.constant(text))
        similarity = np.inner(sentence_embeddings, self.topic_embeddings)[0][0]
        # Cut noise if similarity is too low
        similarity -= 0.05
        # Apply relu to avoid the negative noise
        return np.maximum(similarity, 0)