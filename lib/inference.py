# inference.py
import torch
import re


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
            outputs = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.2,
            )

        try:
            # Get the full output text
            full_text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Also get just the new tokens
            new_tokens_text = text_tokenizer.decode(
                outputs[0, input_length:], skip_special_tokens=True
            )
            
            # Try to extract assistant's response using patterns similar to CLI version
            clean_text = full_text
            
            # Try to extract assistant's response from chat format
            assistant_pattern = r"<\|im_start\|>assistant(.*?)(?=<\|im_end\|>|$)"
            assistant_parts = re.findall(assistant_pattern, clean_text, re.DOTALL)
            if assistant_parts:
                # Get the last assistant response
                clean_text = assistant_parts[-1].strip()
            
            # If the response still seems too short, use new tokens
            if len(clean_text) < len(new_tokens_text) * 0.8:
                clean_text = new_tokens_text
                
            # Final cleanup for readability
            clean_text = clean_text.strip()
            
            return clean_text
            
        except Exception as e:
            # Fall back to direct token decoding if processing fails
            return text_tokenizer.decode(
                outputs[0, input_length:], skip_special_tokens=True
            ).strip()

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error generating response: {str(e)}\n\nDetails:\n{error_details}"
