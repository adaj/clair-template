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
import pandas as pd

from nlu.intents import IntentClassifier


@st.cache(show_spinner=False)
def load_model():
    clf = IntentClassifier(config_file_path=None, 
                           intents_file_path="nlu/intents/statement_intents_examples.yml",
                           load_model="nlu/intents/statement_intents_v1")
    return clf

@st.cache(show_spinner=False, allow_output_mutation=True)
def get_intent(statement, clf):
    intent, probs = clf.predict_one(statement,
                                    get_certainty="all")
    return intent, probs


# Streamlit app
st.title("🎭 Statement Intent Classifier")
st.markdown("Created by [Adelson de Araujo](https://github.com/adaj/)")


# Display the input fields
# User inputs
statement = st.text_input("Statement")

clf = load_model()

# Calculate and display the result when the 'Compute' button is pressed
if st.button('Compute'):
    st.write("Calculating...")
    st.write("Statement:", statement)
    t0 = time.time()
    intent, probs = get_intent(statement, clf)
    t1 = time.time()
    # Convert dictionary to pandas DataFrame
    df = pd.DataFrame(list(probs.items()), columns=['Key', 'Value'])
    st.table(df.set_index('Key'))
    st.write(f"Time taken: {round(t1-t0, 2)} seconds")


if __name__ == "__main__":
    pass  # this is here just to structure the code as a typical Streamlit app


