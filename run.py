from transformers import pipeline
from PIL import Image
import demo


### SETTINGS

MODE = 'WEBCAM' #what to do: either FILE or WEBCAM

PIPE = 'v2_small' #pipe to load

directory = "./samples" #folder to get samples from
filename ="tea.JPG" #specific image (if empty: take first file in samples)

### END SETTINGS

# load pipe
print("Loading Pipes...")
if PIPE == 'v1_small':
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
elif PIPE == 'v2_small':
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
elif PIPE == 'v2_large':
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
else:
    raise LookupError("Unkown pipeline specification '"+PIPE+"'. Adjust the settings.")
print(f"\rDone. Using '{PIPE}'.")

    
if MODE == 'FILE':
    demo.singlefile(directory=directory, filename=filename, pipe=pipe)
elif MODE == 'WEBCAM':
    demo.webcam(pipe=pipe)
else:
    raise LookupError("Unkown mode specification '"+MODE+"'. Adjust the settings.")
