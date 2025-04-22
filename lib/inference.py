# inference.py
import torch

def generate_response(
    model,
    text_tokenizer,
    visual_tokenizer,
    query,
    images,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    max_partition=9,
    do_sample=True,
):
    """
    Generate a response from the model based on the input query and images.

    Args:
        model: The loaded Ovis2 model
        text_tokenizer: Text tokenizer for the model
        visual_tokenizer: Visual tokenizer for the model
        query (str): The text query including the <image> tag
        images (list): List of PIL Image objects
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        top_p (float): Top-p sampling parameter
        max_partition (int): Maximum image partition for high-resolution images

    Returns:
        str: The generated response
    """
    try:
        # Preprocess the inputs
        prompt, input_ids, pixel_values = model.preprocess_inputs(
            query, images, max_partition=max_partition
        )

        # Create attention mask
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)

        # Save input length to determine what's newly generated
        input_length = input_ids.shape[0]

        # Prepare inputs for the model
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)

        # Convert pixel_values list to tensor and move to device
        if isinstance(pixel_values, list):
            pixel_values = [
                pv.unsqueeze(0).to(device=model.device) if pv is not None else None
                for pv in pixel_values
            ]
        else:
            pixel_values = pixel_values.unsqueeze(0).to(device=model.device)

        # Generate the response
        with torch.no_grad():
            # Set up generation parameters based on sampling choice
            gen_kwargs = {
                "inputs": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": 1.2,
            }
            
            # Add sampling parameters only if do_sample is True
            if do_sample:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                })
            else:
                gen_kwargs.update({
                    "do_sample": False,
                    "eos_token_id": model.generation_config.eos_token_id,
                    "pad_token_id": text_tokenizer.pad_token_id,
                })
                
            outputs = model.generate(**gen_kwargs)

        # Simply decode the entire output with skip_special_tokens=True
        generated_text = text_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()
        
        return generated_text

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error generating response: {str(e)}\n\nDetails:\n{error_details}"
