from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

access_token = "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"
model = "meta-llama/Llama-2-7b-chat-hf" 

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=access_token, device_map = 'auto')

llama_pipeline = pipeline(
    "text-generation",  # LLM task
    tokenizer=tokenizer,
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def get_llama_response(prompt: str) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=20,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        repetition_penalty=1.1,
        temperature=0.7,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

prompt = """
<s> 
[INST] <<SYS>> Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. <</SYS>>\n\n

[vignette]:
PK is a 36-year-old woman who lives with her husband and two young daughters, ages 7 and 10, and works in a government office. Last week, she went to see a gastroenterologist with complaints of chronic stomach pain and was diagnosed with an ulcer. In an attempt to identify potential causes of the ulcer, the gastroenterologist asked PK about her stress level. PK mentioned that she had been feeling extremely nervous and worried during the past couple of years, even though nothing unusual or bad has happened. The gastroenterologist suggested that PK speak to a mental health professional to help her manage her anxiety. Presenting symptoms: During the initial mental health evaluation, PK admits that she worries almost all the time, and now she is concerned that her worry is damaging her health. PK says that she has always been “a worrier” and that she thinks that this quality has helped her to be attentive to her family and conscientious in her job.  However, she says that during the past couple of years it has gotten “out of control” and she worries to the point that she feels overwhelmed and exhausted.  When asked what she worries about, she says, “It could be anything—whether my kids might get sick, whether I locked the front door, whether I can finish a report for work on time—whatever I am thinking about can be something to worry about.” She provides the example of when her daughters have a minor cold, she worries that it will turn into something serious to the extent that she has to take them to the doctor, even though she knows that they are in good health overall and that all children get colds. As another example, PK says that her husband is very social and often invites his co-workers over for dinner, which PK says “really stresses her out” because she worries that the meal will turn out badly and that this will somehow damage her husband’s career. On most days, she finds it very difficult to sleep, as she lies in bed thinking about all the things she has to do the next day, and worries that she will not be able to get everything done.  She feels very tense, and often has pain in her shoulders and upper back that she recognizes comes from the tension. What is the most likely diagnosis for PK and why? Your options are the following: Generalized Anxiety Disorder,Panic Disorder, Agoraphobia, Specific Phobia, Social Anxiety Disorder, Separation Anxiety Disorder, Selective Mutism, Other Anxiety and Fear-Related Disorder, Unspecified Anxiety and Fear-Related Disorder. 

\n[/INST]\n\n
[RESPONSE]:
"""
get_llama_response(prompt)