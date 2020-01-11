import logging
import os
import re
from datetime import datetime
from typing import Tuple

import tensorflow as tf

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_text(
        data_file: str
) -> str:
    """
    Load text file to learn from
    :param data_file: str, file location
    :return: str, cleaned text file as string
    """
    with open(data_file) as f:
        text = f.read()
    text = re.sub('\SPEECH \d+', '', text)
    text = re.sub('\\n\\n+', '\n', text)
    text = text[4:]

    return text


def split_input_target(
        chunk: str
) -> Tuple[str, str]:
    """
    Given a string, split into input text and target text.
    :param chunk: str, text to split
    :return: Tuple[str, str], input and target text
    """
    input_text = chunk[:-1]
    target_text = chunk[1:]

    return input_text, target_text


def build_model(
        vocab_size: int,
        embedding_dim: int,
        rnn_units: int,
        batch_size: int
) -> tf.keras.Model:
    """
    Create the neural network.
    :param vocab_size: int, size of vocabulary
    :param embedding_dim: int, dimension of word embedding
    :param rnn_units: int, size of GRU layer
    :param batch_size: int, batch size
    :return: tf.keras.Model, neural network model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            batch_input_shape=[batch_size, None]
        ),
        tf.keras.layers.GRU(
            rnn_units,
            return_sequences=True,
            stateful=True,
            recurrent_initializer='glorot_uniform'
        ),
        tf.keras.layers.Dense(vocab_size)
    ])

    return model


def loss_func(
        labels: tf.Tensor,
        logits: tf.Tensor
) -> tf.keras.losses.SparseCategoricalCrossentropy:
    """
    Loss function.
    :param labels: tf.Tensor, true output
    :param logits: tf.Tensor, predicted output
    :return: Sparse categorical crossentropy loss function
    """
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def generate_text(
        model: tf.keras.Model,
        start_string: str,
        char2idx: dict,
        idx2char: dict,
        num_generate: int = 1000,
        temperature: float = 0.25
) -> str:
    """
    Generate prediction of text given start_string.
    :param model: tf.keras.Model,
    :param start_string: str, input string to start the prediction
    :param char2idx: dict, character-to-index dictionary
    :param idx2char: dict, index-to-character dictionary
    :param num_generate: int, number of characters to generate
    :param temperature: float, Low/high temperatures results in more predictable/surprising text.
    :return: str, predicted text given start_string
    """
    # Converting our start string to numbers
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


def get_tensorboard_callback(
        log_dir: str = 'logs'
) -> tf.keras.callbacks.TensorBoard:
    """
    get tensorboard callback
    :param log_dir: str, directory for tensorboard logs
    :return: tf.keras.callbacks.TensorBoard, tensorboard callback
    """
    logdir = os.path.join(log_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        logdir,
        histogram_freq=1
    )

    return tensorboard_callback


def get_checkpoint_callback(
        checkpoint_dir: str
) -> tf.keras.callbacks.ModelCheckpoint:
    """
    get checkpoint callback
    :param checkpoint_dir: str, directory for model checkpoints
    :return: tf.keras.callbacks.ModelCheckpoint, model checkpoint callback
    """
    checkpoint_prefix = os.path.join(f'./{checkpoint_dir}', "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )

    return checkpoint_callback
