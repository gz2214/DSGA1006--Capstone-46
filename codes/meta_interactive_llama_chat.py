from typing import List, Optional
import fire
from llama import Llama, Dialog

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialog = [
        {
            "role": "system",
            "content": """\
                You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
        },
    ]

    while True:
        print("\n")
        user_input = input("You: ")
        
        dialog.append({"role": "user", "content": user_input})
        
        results = generator.chat_completion(
            [dialog],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        assistant_response = results[0]['generation']['content']
        print(f"\nAssistant: {assistant_response}")
        
        dialog.append({"role": "assistant", "content": assistant_response})
        
        continue_chat = input("\nDo you want to continue the conversation? (y/n): ")

        if continue_chat.lower() not in ['yes', 'y']:
            break

if __name__ == "__main__":
    fire.Fire(main)
