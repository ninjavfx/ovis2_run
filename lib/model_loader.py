# model_loader.py
import torch
from transformers import GenerationConfig
from gptqmodel import GPTQModel  # This should match your existing import structure


def load_model(model_path="AIDC-AI/Ovis2-16B-GPTQ-Int4", device="cuda:0"):
    """
    Load the Ovis2 model with GPTQ quantization.

    Args:
        model_path (str): Path to the model or model name on HuggingFace
        device (str): Device to run the model on (e.g., "cuda:0")

    Returns:
        tuple: (model, text_tokenizer, visual_tokenizer)
    """
    try:
        # Set the device
        torch.cuda.set_device(device)

        # Load the model
        model = GPTQModel.load(model_path, device=device, trust_remote_code=True)

        # Set generation config
        model.model.generation_config = GenerationConfig.from_pretrained(model_path)

        # Get tokenizers
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        return model, text_tokenizer, visual_tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        raise
