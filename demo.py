import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from transformers import pipeline
import os
import time

def grayscale_to_gradient(depth_image: Image):
    """
    Takes a Pillow Image that represents a depth map and returns 
    a Pillow Image colered image.
    """

    # Load the grayscale depth image using Pillow
    depth_image = depth_image.convert('L')

    # Convert the image to a NumPy array
    depth_array = np.array(depth_image)

    # Normalize if necessary (depends on your image's depth range)
    depth_normalized = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))

    # Apply the colormap (choose 'viridis' for a perceptually uniform gradient)
    colored_depth = cm.magma(depth_normalized)

    # Convert to RGB (remove alpha channel)
    colored_depth = (colored_depth[:, :, :3] * 255).astype(np.uint8)

    # Convert back to a Pillow image
    return Image.fromarray(colored_depth)

def singlefile(directory: str,
               filename: str,
               pipe: pipeline):
    
    """
    Loads a single file scpecified by the directory and file and runs the pipleine
    """

    # load image
    files = os.listdir(directory)
    assert len(files) > 0, "cannot have an empty `./samples/` directory!"

    #get first image
    if filename != '':
        first_image_path = directory+'/'+filename
    else:
        first_image_path = os.path.join(directory, files[0])
    image = Image.open(first_image_path)
    image.show()

    # inference
    start_time = time.time()
    depth_v1 = pipe(image)["depth"]
    end_time = time.time()
    print(f"Estimated Depth:\t {end_time-start_time: .2f}s")
    grayscale_to_gradient(depth_v1).show()

def webcam(pipe: pipeline):
    print("Starting Webcam Demo...")
