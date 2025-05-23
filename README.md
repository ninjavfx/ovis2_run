# Ovis2 Local Inference Runner

A tool for running the Ovis2-16B-GPTQ-Int4 model locally on consumer hardware.

![Ovis2 Logo](https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/3IK823BZ8w-mz_QfeYkDn.png)

## Overview

This tool enables local inference with the Ovis2-16B-GPTQ-Int4 multimodal model, allowing you to run powerful vision-language capabilities on your own hardware. The tool handles image input and text generation in a simple, streamlined interface.

## Requirements

- NVIDIA GPU with at least 16GB VRAM
- CUDA 12.1
- Python 3.10
- Git
- Conda (recommended for environment management)
- Linux operating system (tool has only been tested on Linux)

## Installation

### 1. Create the Conda environment and clone the repository

```bash
conda create -n ovis2_run python=3.10 -y
conda activate ovis2_run
git clone https://github.com/ninjavfx/ovis2_run.git
cd ovis2_run
```

### 2. Install the dependencies

```bash
pip install torch==2.4.0
pip install -r requirements.txt
pip install flash-attn==2.7.0.post2
pip install gptqmodel

```

Or you can just use the included install script

```bash
./install.sh
```

### 3. Download the Model

Create a directory for the model and download it using the Hugging Face CLI.  
(Please note that if you have more than 24GB of VRAM you could try and download the 32B parameters)

```bash
mkdir models_16B
huggingface-cli download AIDC-AI/Ovis2-16B-GPTQ-Int4 --local-dir ./models_16B
```


## Usage

Run the inference script with the following parameters:

```bash
python ovis2_run.py --image_path IMAGE_PATH --prompt PROMPT --model_path MODEL_PATH [--max_tokens MAX_TOKENS] [--output_dir OUTPUT_DIR] 
```

### Parameters

- `--image_path`: Path to the input image file (required)
- `--prompt`: Text prompt to guide the model's response (required)
- `--model_path`: Path to the downloaded model directory (required)
- `--max_tokens`: Maximum number of tokens to generate (optional, defaults to 1024)
- `--output_dir`: Directory to save output files (optional, defaults to image directory)
- `--temperature`: Sampling temperature for generation (optional, defaults to 0.7)
- `--top_p`: Top-p sampling parameter (optional, defaults to 0.9)
- `--max_partition`: Maximum image partition for high-resolution image (optional, defaults to 9)
- `--no_sample`: Use greedy decoding instead of sampling, this ignores temperature and top_p (more consistent, less creative )


### Example

```bash
python ovis2_run.py --model_path="./models_16B" --image_path="my_image.jpg" --prompt="Describe the image without using any special characters except for commas and periods"
```

This command will generate a description of the image and save it to `my_image.txt` in the same directory as the source image.

### Graphical User Interface

For a more user-friendly experience, you can run the included gradio web interface:

```bash
python ovis2_run_gui.py --model_path "./models_16B/"
```

This will launch a web interface that allows you to upload images and interact with the model through your browser.

![Gradio UI](./extras/gradio.png)

## License

This project is released under the [MIT License](LICENSE.md).

## Acknowledgments

- This tool builds upon the [AIDC-AI/Ovis](https://github.com/AIDC-AI/Ovis) framework
- Ovis2-16B-GPTQ-Int4 model created by [AIDC-AI](https://github.com/AIDC-AI)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
