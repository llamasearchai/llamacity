#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate dummy images for testing.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dummy images for testing"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/sample/images",
        help="Directory to save images"
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=8,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=224,
        help="Size of images (height and width)"
    )
    
    return parser.parse_args()


def generate_random_image(size, index):
    """
    Generate a random image with a label.
    
    Args:
        size: Size of the image (height and width)
        index: Image index
        
    Returns:
        PIL Image
    """
    # Create a random background color
    r = random.randint(100, 240)
    g = random.randint(100, 240)
    b = random.randint(100, 240)
    
    # Create a random image
    image = Image.new("RGB", (size, size), (r, g, b))
    draw = ImageDraw.Draw(image)
    
    # Add some random shapes
    for _ in range(5):
        shape_type = random.choice(["rectangle", "ellipse"])
        x1 = random.randint(0, size - 1)
        y1 = random.randint(0, size - 1)
        x2 = random.randint(x1, size - 1)
        y2 = random.randint(y1, size - 1)
        
        # Create a random fill color
        fill_r = random.randint(0, 255)
        fill_g = random.randint(0, 255)
        fill_b = random.randint(0, 255)
        fill_color = (fill_r, fill_g, fill_b)
        
        if shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        else:  # ellipse
            draw.ellipse([x1, y1, x2, y2], fill=fill_color)
    
    # Add a label
    try:
        # Try to load a font
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        # If font not found, use default font
        font = ImageFont.load_default()
    
    label = f"Image {index}"
    
    # For newer versions of Pillow
    try:
        text_width, text_height = draw.textsize(label, font=font)
    except AttributeError:
        # For newer Pillow versions
        text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:4]
    
    position = ((size - text_width) // 2, (size - text_height) // 2)
    
    # Add a background for the text
    text_bg_x1 = position[0] - 5
    text_bg_y1 = position[1] - 5
    text_bg_x2 = position[0] + text_width + 5
    text_bg_y2 = position[1] + text_height + 5
    draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=(255, 255, 255))
    
    # Draw the text
    draw.text(position, label, fill=(0, 0, 0), font=font)
    
    return image


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate images
    for i in range(1, args.num_images + 1):
        image = generate_random_image(args.image_size, i)
        image_path = os.path.join(args.output_dir, f"image{i}.jpg")
        image.save(image_path)
        print(f"Generated image: {image_path}")


if __name__ == "__main__":
    main()