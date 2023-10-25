import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("syzymon/long_llama_3b_v1_1")
model = AutoModelForCausalLM.from_pretrained("syzymon/long_llama_3b_v1_1",
                                             torch_dtype=torch.float32, 
                                             trust_remote_code=True)

prompt = """ 
    Summarize the below text in 3 sentences:
    Q2  Panic Disorder. Essential (Required) Features: Recurrent unexpected panic attacks that are not restricted to particular stimuli or situations. Panic attacks are discrete episodes of intense fear or apprehension also characterized by the rapid and concurrent onset of several characteristic symptoms. These symptoms may include, but are not limited to, the following:
    Palpitations or increased heart rate
    Sweating
    Trembling
    Sensations of shortness of breath
    Feelings of choking
    Chest pain
    Nausea or abdominal distress
    Feelings of dizziness or lightheadedness
    Chills or hot flushes
    Tingling or lack of sensation in extremities (i.e., paresthesias)
    Depersonalization or derealization
    Fear of losing control or going mad
    Fear of imminent death
    
    Panic attacks are followed by persistent concern or worry (e.g., for several weeks) about their recurrence or their perceived negative significance (e.g., that the physiological symptoms may be those of a myocardial infarction), or behaviours intended to avoid their recurrence (e.g., only leaving the home with a trusted companion).
    
    The symptoms are sufficiently severe to result in significant impairment in personal, family, social, educational, occupational, or other important areas of functioning.
    
    Panic attacks can occur in other Anxiety and Fear-Related Disorders as well as other Mental and Behavioural Disorders and therefore the presence of panic attacks is not in itself sufficient to assign a diagnosis of Panic Disorder.
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = model(input_ids=input_ids)

generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=1024*2,
    num_beams=1,
    last_context_length=1792,
    do_sample=True,
    temperature=0.3,
)

print(tokenizer.decode(generation_output[0]))

