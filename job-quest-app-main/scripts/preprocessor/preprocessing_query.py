# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# from transformers import AutoTokenizer, TFAutoModel

# def preproc_query(text):
#     """
#     Input a text and preproc only a text, will be used for the query of our API
#     """
#     tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny", padding_side = "right")
#     query_token = tokenizer(text, max_length=512, padding = "max_length", truncation = True, return_tensors="tf")
#     model2 = TFAutoModel.from_pretrained("prajjwal1/bert-tiny", from_pt = True)

#     #query_outputs = model2.predict(text).last_hidden_state
#     query_outputs = model2.predict(query_token["input_ids"]).last_hidden_state

#     return query_outputs

from transformers import AutoTokenizer, TFAutoModel

def load_pretrained_components():
    # Load the tokenizer and model from the saved directory
    tokenizer = AutoTokenizer.from_pretrained('./outputs/tokenizers/')
    model = TFAutoModel.from_pretrained('./outputs/embedders/')
    return tokenizer, model

tokenizer, model = load_pretrained_components()

def preproc_query(text):
    # Tokenize the input text
    query_token = tokenizer(text, max_length=512, padding="max_length", truncation=True, return_tensors="tf")

    # Generate model output
    query_outputs = model(query_token["input_ids"]).last_hidden_state
    return query_outputs
