import argparse
import os
import gc
import sys
import torch
from PIL import Image

# Add the parent directory to sys.path to enable importing from lib/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.model_loader import load_model
from lib.inference import generate_response


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
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--max_partition",
        type=int,
        default=9,
        help="Maximum image partition for high-resolution images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling (more consistent but less creative)",
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Set do_sample based on the --no_sample flag
    args.do_sample = not args.no_sample

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

    # Memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128,expandable_segments:True"
    )
    torch.cuda.empty_cache()
    gc.collect()

    # Load image
    try:
        image = Image.open(args.image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Set device
    device = args.device
    try:
        torch.cuda.set_device(device)
    except Exception as e:
        print(f"Error setting device: {e}. Using default device.")

    print("Starting model load...")

    try:
        # Use the model_loader module
        model, text_tokenizer, visual_tokenizer = load_model(
            model_path=args.model_path, device=device
        )

        print("Model loaded successfully!")

        # Format prompt
        query = f"<image>\n{args.prompt}"

        print(f"Processing input for {image_basename}...")

        # Use the inference module
        clean_text = generate_response(
            model=model,
            text_tokenizer=text_tokenizer,
            visual_tokenizer=visual_tokenizer,
            query=query,
            images=[image],
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_partition=args.max_partition,
            do_sample=args.do_sample,
        )

        # Print clean output
        print("\n" + "=" * 50)
        print("OUTPUT:")
        print("=" * 50)
        print(clean_text)
        print("=" * 50)

        # Save the output file - remove newlines for the file output
        file_output = clean_text.replace("\n", " ").replace("\r", "")
        # Also normalize multiple spaces
        while "  " in file_output:
            file_output = file_output.replace("  ", " ")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(file_output)
        print(f"Output saved to: {output_path}")

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
