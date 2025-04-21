import argparse
import torch
import gc
import os
from PIL import Image
from gptqmodel import GPTQModel
import re


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run Ovis model with an image and prompt"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to use")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the Ovis model"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output text file (defaults to same location as image)",
    )
    parser.add_argument(
        "--save_raw",
        action="store_true",
        help="Also save the raw, unprocessed output for debugging",
    )

    # Parse arguments
    args = parser.parse_args()

    # Extract filename and create output path
    image_path = args.image_path
    image_basename = os.path.basename(image_path)
    image_name = os.path.splitext(image_basename)[0]  # Remove extension
    output_file = f"{image_name}.txt"

    # Determine output directory
    if args.output_dir:
        # Use specified output directory
        output_dir = args.output_dir
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Default to image directory
        output_dir = os.path.dirname(image_path)

    # Construct full output path
    output_path = os.path.join(output_dir, output_file)
    # Path for raw output if requested
    raw_output_path = os.path.join(output_dir, f"{image_name}_raw.txt")

    # Memory management environment variable
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128,expandable_segments:True"
    )

    # Clear CUDA cache and garbage collect
    torch.cuda.empty_cache()
    gc.collect()

    # Load image
    image = Image.open(args.image_path)

    # Set device
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("Starting model load...")

    # Load model with optimized settings
    try:
        model = GPTQModel.load(
            args.model_path,
            device=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        print("Model loaded successfully!")

        # Get tokenizers
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        # Format prompt
        query = f"<image>\n{args.prompt}"

        # Preprocess input
        print("Processing inputs...")
        prompt, input_ids, pixel_values = model.preprocess_inputs(
            query, [image], max_partition=9
        )
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)

        # Save the original input length to determine what's newly generated
        input_length = input_ids.shape[0]

        # Prepare for model
        input_ids = input_ids.unsqueeze(0).to(device=device)
        attention_mask = attention_mask.unsqueeze(0).to(device=device)

        if isinstance(pixel_values, list):
            pixel_values = [pv.unsqueeze(0).to(device=device) for pv in pixel_values]
        else:
            pixel_values = pixel_values.unsqueeze(0).to(device=device)

        # Generate output
        print(f"\nGenerating output for {image_basename}...")
        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
            )

        try:
            # Decode full output
            full_text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Also try getting just the newly generated tokens
            new_tokens = outputs[0, input_length:]
            new_tokens_text = text_tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            # Save raw output if requested
            if args.save_raw:
                with open(raw_output_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                print(f"Raw output saved to: {raw_output_path}")

            # Try multiple extraction approaches to get clean text

            # Start with the full text
            clean_text = full_text

            # Try to extract assistant's response from chat format
            assistant_pattern = r"<\|im_start\|>assistant(.*?)(?=<\|im_end\|>|$)"
            assistant_parts = re.findall(assistant_pattern, clean_text, re.DOTALL)
            if assistant_parts:
                # Get the last assistant response
                clean_text = assistant_parts[-1].strip()

            # Try to remove any metadata part after a colon
            # But only if it doesn't disrupt sentences midway
            if ":" in clean_text and len(clean_text.split(":")[0]) > 100:
                candidate = clean_text.split(":")[0].strip()
                if (
                    candidate.endswith(".")
                    or candidate.endswith("!")
                    or candidate.endswith("?")
                ):
                    clean_text = candidate

            # Extract well-formed complete sentences if available
            sentences = re.findall(r"[A-Z][^.!?]*[.!?]", clean_text)
            if sentences and sum(len(s) for s in sentences) > 100:
                # Only use this approach if we have substantial sentence content
                clean_text = " ".join(sentences)

            # If the output is very short, it might be truncated
            # Try using the new tokens directly
            if len(clean_text) < 100 and len(new_tokens_text) > len(clean_text):
                # Extract sentences from the new tokens text
                new_sentences = re.findall(r"[A-Z][^.!?]*[.!?]", new_tokens_text)
                if new_sentences:
                    clean_text = " ".join(new_sentences)
                else:
                    # Just use the raw new tokens if no sentences found
                    clean_text = new_tokens_text

            # Final cleanup - careful with what we remove to avoid truncating content
            clean_text = re.sub(
                r":[^.!?]*$", "", clean_text
            )  # Remove metadata after the last sentence
            clean_text = re.sub(
                r"%.*$", "", clean_text
            )  # Remove anything after percent sign
            clean_text = re.sub(r"\\n", "\n", clean_text)  # Replace escaped newlines

            # Less aggressive non-text character removal
            clean_text = re.sub(r'[^\w\s.,;:!?\'"\-\(\)\[\]{}/]', " ", clean_text)
            clean_text = re.sub(r"\s+", " ", clean_text)  # Normalize whitespace
            clean_text = clean_text.strip()

            # If the output still seems truncated, try checking for a cutoff point
            # and make a note in the output
            if not (
                clean_text.endswith(".")
                or clean_text.endswith("!")
                or clean_text.endswith("?")
            ):
                clean_text = clean_text + " [output may be truncated]"

            # Print clean output
            print("\n" + "=" * 50)
            print("CLEAN OUTPUT:")
            print("=" * 50)
            print(clean_text)
            print("=" * 50)

            # Save the output file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(clean_text)
            print(f"Output saved to: {output_path}")

        except Exception as e:
            print(f"Error when processing output: {e}")
            import traceback

            traceback.print_exc()

            # Try a simpler approach - just use the new tokens directly
            try:
                simple_text = (
                    new_tokens_text
                    if "new_tokens_text" in locals()
                    else text_tokenizer.decode(
                        outputs[0, input_length:], skip_special_tokens=True
                    )
                )

                print("\nFalling back to direct token decoding:")
                print(
                    simple_text[:100] + "..." if len(simple_text) > 100 else simple_text
                )

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(simple_text)
                print(f"Fallback output saved to: {output_path}")
            except Exception as inner_e:
                print(f"Simple decoding also failed: {inner_e}")
                print("Could not generate any output file.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Final cleanup
        if "model" in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()

