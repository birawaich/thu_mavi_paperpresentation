from transformers import pipeline
from PIL import Image
import os
import time
import demo

# load pipe
# pipe_v1 = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
pipe_v2_small = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
# pipe_v2_large = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")

# load image
# Specify the directory containing the images
directory = "./samples"
filename ="tianjin_orig.JPG"

# List all files in the directory
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
# start_time = time.time()
# depth_v1 = pipe_v1(image)["depth"]
# end_time = time.time()
# print(f"Estimated Depth using V1:\t {end_time-start_time: .2f}s")
# demo.grayscale_to_gradient(depth_v1).show()

start_time = time.time()
depth_v2= pipe_v2_small(image)["depth"]
end_time = time.time()
print(f"Estimated Depth using V1:\t {end_time-start_time: .2f}s")
depth_v2.show()
demo.grayscale_to_gradient(depth_v2).show()
