#!/bin/bash

echo "Installing PyTorch..."
pip install torch==2.4.0

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installing flash-attn..."
pip install flash-attn==2.7.0.post2

echo "Installing gptqmodel..."
pip install gptqmodel

echo "Installation complete!"
