from transformers import AutoTokenizer, TFAutoModel

# Download and cache the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny", padding_side="right")
embedder = TFAutoModel.from_pretrained("prajjwal1/bert-tiny", from_pt=True)

# Save the tokenizer and model
tokenizer.save_pretrained('./outputs/tokenizers/')
embedder.save_pretrained('./outputs/embedders/')
