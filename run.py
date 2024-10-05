from transformers import pipeline
from PIL import Image
import os

# load pipe
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

# load image
# Specify the directory containing the images
directory = "./samples"

# List all files in the directory
files = os.listdir(directory)

assert len(files) > 0, "cannot have an empty `./samples/` directory!"

#get first image
first_image_path = os.path.join(directory, files[0])
image = Image.open(first_image_path)
image.show()

# inference
depth = pipe(image)["depth"]
depth.show()