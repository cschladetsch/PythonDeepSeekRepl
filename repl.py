import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import readline
from colorama import init, Fore, Style
import os
import signal
init()

def signal_handler(sig, frame):
    print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
    exit(0)

def load_local_model():
    print(f"{Fore.YELLOW}Loading local model...{Style.RESET_ALL}")
    model_path = os.path.join(os.getcwd(), "models", "deepseek_model")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run model_setup.py first.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=9999):
    formatted_prompt = f"System: You are a helpful AI assistant. Provide thoughtful and accurate responses.\nHuman: {prompt}\nAssistant: "
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(formatted_prompt, "").strip()
    response = response.split('System:')[0].strip()
    response = response.split('Human:')[0].strip()
    response = response.split('Assistant:')[0].strip()
    
    return response

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"{Fore.CYAN}Initializing Deepseek REPL...{Style.RESET_ALL}")
    try:
        model, tokenizer = load_local_model()
        print(f"\n{Fore.GREEN}Deepseek REPL ready! Type 'exit' to quit or press Ctrl-C{Style.RESET_ALL}\n")
        
        while True:
            try:
                prompt = input(f"{Fore.BLUE}Deepseek>{Style.RESET_ALL} ")
                if prompt.lower() in ['exit', 'quit']:
                    break
                if prompt.strip():
                    print(f"{Fore.GREEN}Response:{Style.RESET_ALL}")
                    response = generate_response(model, tokenizer, prompt)
                    print(f"{Fore.WHITE}{response}{Style.RESET_ALL}\n")
            except EOFError:
                break
    except FileNotFoundError as e:
        print(f"{Fore.RED}{str(e)}{Style.RESET_ALL}")
        return
    
    print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
