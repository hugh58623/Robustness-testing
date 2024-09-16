from transformers import RobertaTokenizer, T5ForConditionalGeneration

# Model name
model_name = "distilbert-base-uncased"

# Download and cache the model

model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-large')

# Download and cache the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-large')

# Save the model and tokenizer to a directory (optional)
model.save_pretrained('./codet5-large')
tokenizer.save_pretrained('./codet5-large')

