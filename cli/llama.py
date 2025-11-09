import torch
from transformers import AutoTokenizer, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig

model_path = 'Meta-Llama-3.1-8B-Instruct'


# Load the tokenizer directly from the model path
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model configuration from params.json
config = LlamaConfig.from_json_file(f'{model_path}/params.json')

# load the model with the specific configs.
model = LlamaForCausalLM(config=config)

# Load the weights of the model
state_dict = torch.load(f'{model_path}/consolidated.00.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

model.eval()

# generate tokens and generate output
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(inputs['input_ids'])

# print the output you asked it
output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output)