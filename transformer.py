
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(123)
print(generator("Hey readers, today is", max_length=20, num_return_sequences=3))
