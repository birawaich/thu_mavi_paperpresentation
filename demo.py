import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

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