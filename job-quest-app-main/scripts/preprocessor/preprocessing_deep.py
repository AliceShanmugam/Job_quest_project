import pandas as pd
from transformers import AutoTokenizer
from transformers import TFAutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scripts.model.registry import save_category
import tensorflow as tf
import numpy as np

# Check if GPU is available


# Module to preprocess the data before fitting the model
# Tokenizer (= Vectorizer) then Embeddings
# Finally, split the embedding into X_train, y_train, X_val, y_val with Stratify( )

# Embedding: Tokenize = Vectorize
def X_embed(df):
    """
    Return de X tensor from the df
    """
    print("START X TOKENIZER")
    #Define tokenizer
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny", padding_side = "right")
    print("Saving Tokenizer")
    tokenizer.save_pretrained('./outputs/tokenizers/')
    print("Tokenizer Saved")
    #define X
    X = df["description"]
    X_token = tokenizer(X.to_list(), max_length=512, padding = "max_length", truncation = True, return_tensors="tf")

    #Embed using the right model
    model = TFAutoModel.from_pretrained("prajjwal1/bert-tiny", from_pt = True)
    print("Saving Embedder")
    model.save_pretrained('./outputs/embedders/')
    print("Embedder Saved")

    if tf.config.list_physical_devices('GPU'):
        device_name = 'GPU:0'
    else:
        device_name = 'CPU:0'

    print(f'Launch embedding with {device_name}')
    with tf.device(device_name):
        # OLD CODE
        # X_outputs = model.predict(X_token["input_ids"]).last_hidden_state
        # NEW CODE
        batch_size = 256
        total_samples = X_token["input_ids"].shape[0]
        X_outputs = []

        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_outputs = model.predict(X_token["input_ids"][start_idx:end_idx]).last_hidden_state
            X_outputs.append(batch_outputs)

        X_outputs = np.concatenate(X_outputs, axis=0)

    print('Embedding done')
    print(X_outputs.shape)

    return X_outputs


def y_onehot(df):
    """
    Return de y as one hot encode
    """
    print("START ONE HOT ENCODING")
    y = df[["title"]]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y)
    cat = enc.categories_

    y_output = enc.transform(y).toarray()
    print(y_output.shape)

    save_category(category=cat)
    # return a tuple of a ndarray and a list
    return y_output, cat

# Train split with stratify, use stratify on categories to have a prportional amount of data from each category
def train_val_split(df: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    print("Embedding X")
    X = X_embed(df)
    print("One Hot Encode y")
    y = y_onehot(df)[0]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify= df['title'])
    return X_train, X_val, y_train, y_val

if __name__ == '__main__':
    print('Preprocessing function')
