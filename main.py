import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import readline
from colorama import init, Fore, Style
init()

def setup_model():
    print(f"{Fore.YELLOW}Loading Deepseek model... This may take a few minutes.{Style.RESET_ALL}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Let's use the instruct model for better responses
    model_name = "deepseek-ai/deepseek-coder-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=256):
    # Format the prompt for better instruction following
    formatted_prompt = f"Human: {prompt}\n\nAssistant: Let me answer that directly:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,  # Enable sampling
        temperature=0.7,
        top_p=0.95,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    response = response.split("Assistant: Let me answer that directly:")[-1].strip()
    return response

def main():
    print(f"{Fore.CYAN}Initializing Deepseek REPL...{Style.RESET_ALL}")
    model, tokenizer = setup_model()
    print(f"\n{Fore.GREEN}Deepseek REPL ready! Type 'exit' to quit.{Style.RESET_ALL}\n")
    
    while True:
        try:
            prompt = input(f"{Fore.BLUE}Deepseek>{Style.RESET_ALL} ")
            if prompt.lower() in ['exit', 'quit']:
                break
            if prompt.strip():
                print(f"{Fore.GREEN}Response:{Style.RESET_ALL}")
                response = generate_response(model, tokenizer, prompt)
                print(f"{Fore.WHITE}{response}{Style.RESET_ALL}\n")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Use 'exit' to quit{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
