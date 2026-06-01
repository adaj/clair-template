import os
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass
import yaml
import fire
from pprint import pprint

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, cohen_kappa_score
import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow_text
import tensorflow_hub as hub
import tensorflow_addons as tfa
import wandb
from wandb.keras import WandbCallback


PUNCTUATION_TOKENS = {
    "?": "QUESTION_MARK",
}


@dataclass
class Config:
    """
    Default configuration for training IntentClassifier.
    
    Attributes:
    ------------
    examples_file : str
        Path to the intents YAML file.
    architecture : str
        Version or identifier for the model architecture.
    wandb_project : Optional[str]
        Weights & Biases project name, if used for logging.
    load_model : Optional[str]
        Path to a pre-trained model, if starting from an existing model.
    language_featurizer : Union[str, List[str]]
        URL or identifier for the language feature extractor (e.g., Universal Sentence Encoder).
    sent_hl_units : Union[int, List[int]]
        Number of hidden layer units or list of hidden units for the sentence encoder.
    sent_dropout : Union[float, List[float]]
        Dropout rate or list of dropout rates for the sentence encoder.
    epochs : int
        Number of training epochs.
    callback_patience : int
        Number of epochs with no improvement to trigger early stopping.
    learning_rate : Union[float, List[float]]
        Learning rate or list of learning rates for training optimization.
    """
    dataset_name: str
    codes : List[str] = None
    architecture: str = "v0.1.5"
    wandb_project: Optional[str] = None
    min_words: int = 1
    language_featurizer: Union[str, List[str]] = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
    sent_hl_units: Union[int, List[int]] = 32
    sent_dropout: Union[float, List[float]] = 0.1
    l1_reg: float = 0.01
    l2_reg: float = 0.01
    epochs: int = 500
    callback_patience: int = 20
    learning_rate: Union[float, List[float]] = 5e-3
    validation_split: float = 0.2


def remove_duplicate_words(text):
    words = text.split()
    seen = set()
    result = []

    for word in words:
        if word not in seen:
            seen.add(word)
            result.append(word)

    return ' '.join(result)

class IntentClassifier:
    # TODO: cross_validation

    def __init__(self, config = None, load_model = None, dataset_name = None, examples_file = None, handle_punctuation = False):
        self.handle_punctuation = handle_punctuation
        # Load config
        self.config = None
        if config is None:
            # Load from a model
            if load_model is not None:
                self.model = tf.keras.models.load_model(load_model)
                print(f"Loaded keras model from {load_model}.")
                config_path = os.path.join(load_model, "config.yml")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        self.config = Config(**yaml.safe_load(f))
                else:
                    print(f"WARNING: No config file found at {config_path}.")

            # Use default if config was not provided
            if self.config is None:
                if dataset_name is None or examples_file is None:
                    raise ValueError("Both dataset_name and examples_file must be provided if config is None.")
                self.config = Config(dataset_name)
                # self.config.codes needs to be set (later)
        elif isinstance(config, str):
            with open(config, 'r') as f:
                self.config = Config(**yaml.safe_load(f))
        elif isinstance(config, Config):
            self.config = config
        else:
            raise ValueError("config must be a path to a YAML file, a Config object, or None.")

        # Load intents from the examples file if provided
        self.examples_file = examples_file
        if examples_file is not None:
            pprint(f"Loading intents from {examples_file}...")
            with open(examples_file, 'r') as f:
                self.intents_data = yaml.safe_load(f)

            # Preprocess intents
            input_text = []
            labels = []
            for i in self.intents_data:
                input_text += i['examples']
                labels += [i['intent']]*len(i['examples'])
            input_text = np.array(input_text)
            labels = np.array(labels)

            # Preprocess input_text
            # 1 - Iterate on input_text and replace punctuation with " <punctuation>" (apparently it helps the sentence encoder)
            if self.handle_punctuation:
                for i, text in enumerate(input_text):
                    for p, t in PUNCTUATION_TOKENS.items():
                        input_text[i] = input_text[i].replace(p, f" {t} ").strip()

            # Shuffle data
            indices = np.arange(len(labels))
            np.random.shuffle(indices)
            self.input_text = input_text[indices]
            self.labels = labels[indices]
            self.codes = np.unique(self.labels)
            self.config.codes = self.codes.tolist()
        else:
            # If the model is used only to predict, there is no need to provide the intent examples
            # The codes must be available in the config
            self.codes = self.config.codes
        # Initialize stop_words
        self.stop_words = []

        # Set up one-hot encoder
        if len(self.codes) == 1:
            self.codes = self.codes[0]
        self.onehot_encoder = OneHotEncoder(categories=[self.codes],)\
                                .fit(np.array(self.codes).reshape(-1, 1))
        
    def load_stop_words(self, stop_words_file: str):
        with open(stop_words_file, 'r', encoding='utf-8') as f:
            self.stop_words = f.read().split('\n')
        print(f"Loaded {len(self.stop_words)} stop words from {stop_words_file}.")
        return self
    
    def make_model(self, config: Config):
        # Set the random seed for reproducibility
        seed = 42
        tf.random.set_seed(seed)  # Assuming you have a random_seed in your config

        # Extract config values
        sent_hl_units, sent_dropout = config.sent_hl_units, config.sent_dropout
        l1_reg, l2_reg = config.l1_reg, config.l2_reg 
        output_size = len(self.codes)

        # Build model
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)  # Set seed in initializer
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text_input")
        encoder = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3',
                                trainable=False,
                                name="sent_encoder")(text_input)
        sent_hl = tf.keras.layers.Dense(sent_hl_units,
                                        kernel_initializer=initializer,
                                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                        activation=None,  # No activation here yet
                                        name='sent_hl')(encoder)
        sent_hl_norm = tf.keras.layers.BatchNormalization()(sent_hl)  # Add batch normalization
        sent_hl_activation = tf.keras.layers.Activation('relu')(sent_hl_norm)  # Activation after batch normalization
        sent_hl_dropout = tf.keras.layers.Dropout(sent_dropout, seed=seed)(sent_hl_activation)  # Set seed in dropout
        sent_output = tf.keras.layers.Dense(output_size,
                                            kernel_initializer=initializer,
                                            activation='softmax',
                                            name="sent_output")(sent_hl_dropout)
        model = tf.keras.Model(inputs=text_input, outputs=sent_output)
        return model
    
    def setup_wandb(self, config = None):
        if self.config.wandb_project:
            self.wandb_run = wandb.init(project=self.config.wandb_project,
                                        config=config.__dict__)

    def finish_wandb(self):
        if self.config.wandb_project and self.wandb_run:
            self.wandb_run.finish()

    def train(self, save_model: Optional[str] = None, tf_verbosity: int = 1):
        pprint(self.config.__dict__)
        assert self.examples_file is not None, "examples_file must be provided when the IntentClassifier was created."
        # Extract config values
        learning_rate = self.config.learning_rate
        epochs = self.config.epochs

        # Extract one-hot encoded labels
        labels_ohe = self.onehot_encoder\
                            .transform(self.labels.reshape(-1, 1))\
                            .toarray()

        # New model from scratch
        self.model = self.make_model(self.config)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(\
                learning_rate=learning_rate),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                     tfa.metrics.F1Score(num_classes=len(self.codes),
                                         average='micro')])

        # Set callbacks
        callbacks = []
        if self.config.callback_patience > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                    patience=self.config.callback_patience,
                    restore_best_weights=True)
            )
        if self.config.wandb_project:
            self.setup_wandb(self.config)
            callbacks.append(WandbCallback(monitor="accuracy",
                                           mode='max',
                                           save_model=False,
                                           labels=self.codes))

        # Apply preprocessing and remove stop words
        if self.stop_words:
            new_input_text = []
            for text in self.input_text:
                if self.stop_words:
                    text = ' '.join([word for word in text.lower().split() if word not in self.stop_words])
                text = remove_duplicate_words(text)
                new_input_text.append(text)
            self.input_text = np.array(new_input_text)

        if self.config.min_words:
            new_input_text = []
            for text in self.input_text:
                # Count words in input_text except by ? . , !
                words = [w for w in text.split() if w not in ["?", ".", ",", "!"]]
                if len(words) <= self.config.min_words:
                    new_input_text.append("")
                else:
                    new_input_text.append(text)
            self.input_text = np.array(new_input_text)

        # Train model
        self.model.fit(self.input_text, labels_ohe,
                       validation_split=self.config.validation_split,
                       # batch_size=16,
                       shuffle=True,
                       epochs=epochs,
                       verbose=tf_verbosity,
                       callbacks=callbacks)
        
        # Save model
        if save_model is not None:
            Path(os.path.dirname(save_model))\
                .mkdir(parents=True, exist_ok=True)
            self.model.save(save_model)
            # Save config into a yaml file inside the model directory
            config_path = os.path.join(save_model, "config.yml")
            with open(config_path, 'w') as f:
                f.write(yaml.dump(self.config.__dict__))
            print(f"Model saved to {save_model}.")
        print(save_model)
        return

    def predict_one(self, input_text: str, get_certainty: bool = False):
        # Preprocess input_text
        # 1 - Replace punctuation with " <punctuation>" (apparently it helps the sentence encoder)
        if self.handle_punctuation:
            for p, t in PUNCTUATION_TOKENS.items():
                input_text = input_text.replace(p, f" {t} ").strip()

        # 2 - Remove stopwords
        if self.stop_words:
            input_text = ' '.join([word for word in input_text.lower().split() if word not in self.stop_words])
        # 3 - Clear input if it has min_words or less
        if self.config.min_words:
            if len(input_text.split()) <= self.config.min_words:
                input_text = ""

        # Get prediction
        pred = self.model.predict([input_text])
        intent = self.onehot_encoder.inverse_transform(pred)[0][0]
        if get_certainty:
            if get_certainty=="all":
                return intent, {code: pred[0][i] for i, code in enumerate(self.codes)}
            return intent, max(pred[0])
        return intent

    def cross_validation(self, n_splits: int = 3):
        assert self.examples_file is not None, "examples_file must be provided when the IntentClassifier was created."
        results = []
        kf = KFold(n_splits=n_splits)
        for i, (train_index, test_index) in enumerate(kf.split(self.input_text)):
            print(f"Fold {i+1}/{n_splits}")
            labels_ohe = self.onehot_encoder\
                            .transform(self.labels.reshape(-1, 1))\
                            .toarray()
            self.model.fit(self.input_text[train_index], labels_ohe[train_index],
                           epochs=self.config.epochs, verbose=0)
            preds = self.model.predict(self.input_text[test_index])
            preds = self.onehot_encoder.inverse_transform(preds)
            labels = self.onehot_encoder.inverse_transform(labels_ohe[test_index])
            # print(classification_report(labels, preds))
            res = classification_report(labels, preds, output_dict=True)
            res['kappa'] = cohen_kappa_score(labels, preds)
            results.append(res)
        # Average the accuracy and f1 scores
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['macro avg']['f1-score'] for r in results])
        avg_kappa = np.mean([r['kappa'] for r in results])
        print(f"Average accuracy: {avg_accuracy}")
        print(f"Average f1-score: {avg_f1}")
        print(f"Average kappa: {avg_kappa}")
        return results


if __name__=="__main__":
    fire.Fire(IntentClassifier)
