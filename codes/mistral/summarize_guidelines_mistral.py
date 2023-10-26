import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd 

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",offload_folder="offload",torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
guidelines=pd.read_csv('../ICD-11 Guidelines.csv')

device = "cuda:0"


prefix=""" ### Instruction: Summarize this passage as short as possible

"""
suffix=""" Answer:
"""
inputs = [prefix + g +suffix for g in guidelines['ICD11 Guidelines']]

encodeds = tokenizer(inputs[0], return_tensors="pt", add_special_tokens=True)
model_inputs = encodeds.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0]

        )


