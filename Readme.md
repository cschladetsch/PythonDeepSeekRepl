# Local Deepseek REPL

An interactive Python REPL interface for running Deepseek's language models locally, featuring 4-bit quantization for efficient memory usage.

## Sample

(Image)[resources/screen.png]

## Features
- Interactive command-line interface with colored output
- Optimized model loading with 4-bit quantization
- Support for the Deepseek-coder-7b-instruct model
- Clean exit handling with Ctrl-C
- Local model caching for faster subsequent loads

## Requirements
- Python 3.12
- NVIDIA GPU with CUDA support
- Minimum 8GB VRAM (16GB recommended)
- 50GB free disk space for model storage

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate colorama bitsandbytes
```

## Usage

1. First time setup - download and cache the model:
```bash
python3 model_setup.py
```

2. Start the REPL:
```bash
python3 repl.py
```

Exit using:
- Press Ctrl-C
- Type 'exit' or 'quit'

## Project Structure

- `model_setup.py`: Downloads and caches the model locally
- `repl.py`: Main REPL interface with interactive prompt
- `main.py`: Alternative implementation with direct model loading
- `run.py`: Simple script for testing model responses

## Example Interactions

```
Deepseek> Write a function to calculate fibonacci
Response: Here's a function to calculate Fibonacci numbers:

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

Deepseek> 2 + 2
Response: 4
```

## Memory Usage
The implementation uses 4-bit quantization via the BitsAndBytesConfig to reduce VRAM usage while maintaining model performance. This allows the model to run on GPUs with 8GB VRAM, though 16GB is recommended for optimal performance.

## License
MIT
```
