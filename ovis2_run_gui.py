import os
import argparse
import gradio as gr
from PIL import Image
import sys
import torch
import tempfile

# Import your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.model_loader import load_model
from lib.inference import generate_response

# Default settings
DEFAULT_MODEL_PATH = "AIDC-AI/Ovis2-16B-GPTQ-Int4"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_PARTITION = 9  # For high-resolution image handling
DEFAULT_DO_SAMPLE = False  # Changed from True to False


def parse_args():
    parser = argparse.ArgumentParser(description="Ovis2-16B Gradio Web Interface")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model or model name on HuggingFace",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the Gradio interface on"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public link for sharing"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the model on"
    )
    return parser.parse_args()


class Ovis2WebUI:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, device="cuda:0"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.text_tokenizer = None
        self.visual_tokenizer = None

        # Load model if CUDA is available
        if torch.cuda.is_available():
            print(f"Loading model from {model_path}...")
            self.load_model()
        else:
            print("CUDA not available. Model will be loaded when needed.")

    def load_model(self):
        """Load the Ovis2 model and tokenizers"""
        try:
            # Use your CLI's model loading function
            self.model, self.text_tokenizer, self.visual_tokenizer = load_model(
                model_path=self.model_path, device=self.device
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def process_single_image(
        self, 
        image, 
        prompt, 
        max_new_tokens, 
        temperature, 
        top_p, 
        max_partition,
        do_sample
    ):
        """Process a single image with the given prompt"""
        if self.model is None:
            self.load_model()

        # Prepare the image
        if isinstance(image, str):
            image = Image.open(image)

        # Format the query
        query = f"<image>\n{prompt}"

        # Generate the response using your CLI's inference function
        response = generate_response(
            model=self.model,
            text_tokenizer=self.text_tokenizer,
            visual_tokenizer=self.visual_tokenizer,
            query=query,
            images=[image],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_partition=max_partition,
            do_sample=do_sample,
        )

        return response


def save_text_to_file(text):
    """Save text to a temporary file and return the path for download"""
    if not text or text.strip() == "":
        return None

    # Process the text: replace newlines with spaces but keep carriage returns
    processed_text = text.replace("\n", " ")
    
    # Normalize multiple spaces to single spaces
    while "  " in processed_text:
        processed_text = processed_text.replace("  ", " ")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as f:
        f.write(processed_text)
        return f.name


def create_ui(ui_instance):
    with gr.Blocks(title="Ovis2-16B Web UI") as demo:
        gr.Markdown("# Ovis2-16B Multimodal Interface")
        gr.Markdown("""
        This interface allows you to interact with the Ovis2-16B-GPTQ-Int4 model, 
        a powerful multimodal AI that can process images.
        """)

        # Single Image Interface
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image in detail.",
                    lines=10,  # Changed from 3 to 10
                    max_lines=10,  # Fixed at 10 lines
                )
                with gr.Accordion("Advanced Options", open=False):
                    do_sample = gr.Checkbox(
                        label="Use Sampling (more creative but less consistent)",
                        value=DEFAULT_DO_SAMPLE,  # Now False by default
                    )
                    with gr.Group(visible=DEFAULT_DO_SAMPLE) as sampling_params:
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=DEFAULT_TEMPERATURE,
                            step=0.1,
                            label="Temperature",
                        )
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=DEFAULT_TOP_P,
                            step=0.05,
                            label="Top-P",
                        )
                    
                    max_new_tokens = gr.Slider(
                        minimum=16,
                        maximum=4096,
                        value=DEFAULT_MAX_NEW_TOKENS,
                        step=16,
                        label="Max New Tokens",
                    )
                    max_partition = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=DEFAULT_MAX_PARTITION,
                        step=1,
                        label="Max Partition (for high-res images)",
                    )

                submit_btn = gr.Button("Generate Response", variant="primary")

            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Response", lines=20)

                # Rename the save button
                save_btn = gr.Button("Save Response", variant="secondary")  # Changed from "Save Response As..."
                download_file = gr.File(label="Download", visible=False)

        # Toggle visibility of sampling parameters based on checkbox
        do_sample.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[do_sample],
            outputs=[sampling_params],
        )
        
        # Connect the generate button
        submit_btn.click(
            fn=ui_instance.process_single_image,
            inputs=[
                image_input,
                prompt_input,
                max_new_tokens,
                temperature,
                top_p,
                max_partition,
                do_sample,
            ],
            outputs=output_text,
        )

        # Connect the save button
        save_btn.click(
            fn=save_text_to_file,
            inputs=[output_text],
            outputs=[download_file],
        ).then(
            lambda: gr.update(visible=True),
            None,
            [download_file],
        )

        return demo


def main():
    args = parse_args()

    # Create the UI instance
    ui = Ovis2WebUI(model_path=args.model_path, device=args.device)

    # Create the Gradio interface
    demo = create_ui(ui)

    # Launch the interface
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
