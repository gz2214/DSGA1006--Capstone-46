from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "syzymon/long_llama_3b_v1_1"  # Replace with your actual model name

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             torch_dtype=torch.float32, 
                                             trust_remote_code=True)

llama_pipeline = pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model,
    torch_dtype=torch.float16,
)

def get_llama_summary(prompt: str) -> None:
    """
    Generate a summary from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's summary.
    """
    summary_prompt = f"Please summarize the following text in three sentences:\n{prompt}"
    sequences = llama_pipeline(
        summary_prompt,
        do_sample=True,
        top_k=10,
        temperature=0.3,
        max_new_tokens=1024*2,
        last_context_length=1792,
    )
    for seq in sequences:
        print(f"Summary: {seq['generated_text']}")

# Text to summarize
text_to_summarize = """Q2 Panic Disorder. Essential (Required) Features: Recurrent unexpected panic attacks that are not restricted to particular stimuli or situations. [....] Panic attacks can occur in other Anxiety and Fear-Related Disorders as well as other Mental and Behavioural Disorders and therefore the presence of panic attacks is not in itself sufficient to assign a diagnosis of Panic Disorder."""

# Generate summary
get_llama_summary(text_to_summarize)
