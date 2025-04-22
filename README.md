# Ovis2 Local Inference Runner

A tool for running the Ovis2-16B-GPTQ-Int4 model locally on consumer hardware.

![Ovis2 Logo](https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/3IK823BZ8w-mz_QfeYkDn.png)

## Overview

This tool enables local inference with the Ovis2-16B-GPTQ-Int4 multimodal model, allowing you to run powerful vision-language capabilities on your own hardware. The tool handles image input and text generation in a simple, streamlined interface.

## Requirements

- NVIDIA GPU with at least 16GB VRAM
- Python 3.10
- Git
- Conda (recommended for environment management)
- Linux operating system (tool has only been tested on Linux)

## Installation

### 1. Create the Conda environment and clone the repository

```bash
conda create -n ovis2_run python=3.10 -y
conda activate ovis2_run
git clone git@github.com:ninjavfx/ovis2_run.git
cd ovis2_run
```

### 2. Install the dependencies

```bash
pip install torch==2.4.0
pip install -r requirements.txt
pip install flash-attn==2.7.0.post2
pip install gptqmodel

Or just run install.sh which does it for you
```

### 3. Download the Model

Create a directory for the model and download it using the Hugging Face CLI:

```bash
mkdir models_16B
huggingface-cli download AIDC-AI/Ovis2-16B-GPTQ-Int4 --local-dir ./models_16B
```


## Usage

Run the inference script with the following parameters:

```bash
python ovis2_run.py --image_path IMAGE_PATH --prompt PROMPT --model_path MODEL_PATH [--max_tokens MAX_TOKENS] [--output_dir OUTPUT_DIR] [--save_raw]
```

### Parameters

- `--image_path`: Path to the input image file (required)
- `--prompt`: Text prompt to guide the model's response (required)
- `--model_path`: Path to the downloaded model directory (required)
- `--max_tokens`: Maximum number of tokens to generate (optional)
- `--output_dir`: Directory to save output files (optional, defaults to image directory)
- `--save_raw`: Flag to save the raw model output (optional)

### Example

```bash
python ovis2_run.py --model_path="./models_16B" --image_path="my_image.jpg" --prompt="Describe the image without using any special characters except for commas and periods"
```

This command will generate a description of the image and save it to `my_image.txt` in the same directory as the source image.

## Output Formatting Note

The prompt example includes "without using any special characters except for commas and periods" to ensure cleaner output formatting. This helps avoid unexpected characters in the generated text.

## License

This project is released under the [MIT License](LICENSE.md).

## Acknowledgments

- This tool builds upon the [AIDC-AI/Ovis](https://github.com/AIDC-AI/Ovis) framework
- Ovis2-16B-GPTQ-Int4 model created by [AIDC-AI](https://github.com/AIDC-AI)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
