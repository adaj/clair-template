import sys
sys.path.append('.')
import time
import os
import io
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import numpy as np
import streamlit as st
import yaml
from skfuzzy import control as ctrl
from triggering.fuzzy import membership
import matplotlib.pyplot as plt


TFHUB_LANGUAGE_FEATURIZER="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"

@st.cache(show_spinner=False)
def create_encoder(language_featurizer: str = TFHUB_LANGUAGE_FEATURIZER):
    # Load language featurizer
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    encoder = hub.KerasLayer(language_featurizer)(text_input)
    encoder = tf.keras.Model(text_input, encoder)
    return encoder

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_fuzzy_sim():
    # Load fuzzy TSIM
    with open(os.path.join("triggering", "apt-base", "inputs", "memberships.yml"), 'r') as f:
        antecedents_data = yaml.safe_load(f)
    universe = np.linspace(0, 1, 100+1)
    fuzzy_sim = ctrl.Antecedent(universe, label='Similarity')
    for a in antecedents_data:
        if a['name'] == 'TSIM':
            for term in a['terms']:
                fuzzy_sim[term['name']] = membership(universe, **term)
    return fuzzy_sim

@st.cache(show_spinner=False, allow_output_mutation=True)
def similarity(statement, targets):
    sentence_emb = MUSE(tf.constant(statement))
    target_emb = MUSE(tf.constant(targets))
    similarity = np.inner(sentence_emb, target_emb)[0][0]
    return np.maximum(similarity, 0)


# Streamlit app
st.title("📊 Semantic Similarity Calculator")
st.markdown("Created by [Adelson de Araujo](https://github.com/adaj/)")

fuzzy_sim = load_fuzzy_sim()

# Display the fuzzy TSIM
st.markdown("* Reference results from clustering previous data with K=3")
fig, ax = plt.subplots()
plt.rcParams['font.size'] = 18
fuzzy_sim.view()
plt.tight_layout()
# Adjusting font sizes
# Save the plot to a BytesIO buffer
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
# Display the plot in Streamlit with specified width
st.image(buf, caption="Reference results from clustering previous data with K=3", 
         use_column_width=False, width=300)  # Adjust  as needed
# Close the buffer
buf.close()

# Display the input fields
# User inputs
statement = st.text_input("Statement")
# Dynamic number of target inputs
targets = st.text_input("Target (comma-separated for multiple values)")

# Load the language featurizer
MUSE = create_encoder("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

# Calculate and display the result when the 'Compute' button is pressed
if st.button('Compute'):
    st.write("Calculating...")
    st.write("Statement:", statement)
    st.write("Targets:", targets)
    t0 = time.time()
    result = similarity(statement, targets)
    t1 = time.time()
    st.success(f"Result: {round(result, 4)}")  # Displays in green
    st.write(f"Time taken: {round(t1-t0, 2)} seconds")


if __name__ == "__main__":
    pass  # this is here just to structure the code as a typical Streamlit app


