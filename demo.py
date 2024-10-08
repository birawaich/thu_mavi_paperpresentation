import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from transformers import pipeline
import os
import time
import cv2
from screeninfo import get_monitors

def grayscale_to_gradient(depth_image: Image) -> Image:
    """
    Takes a Pillow Image that represents a depth map and returns 
    a Pillow Image colered image.
    """

    # Load the grayscale depth image using Pillow
    depth_image = depth_image.convert('L')

    # do conversion with np array
    colored_depth = _grayscale_to_gradient(np.array(depth_image))

    # Convert back to a Pillow image
    return Image.fromarray(colored_depth)

def _grayscale_to_gradient(depth_image_np: np.ndarray) -> np.ndarray:
    """Private function to convert an np.array directly"""
    depth_array = depth_image_np

    # Normalize if necessary (depends on your image's depth range)
    depth_normalized = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))

    # Apply the colormap (choose 'viridis' for a perceptually uniform gradient)
    colored_depth = cm.magma(depth_normalized)

    # Convert to RGB (remove alpha channel)
    colored_depth = (colored_depth[:, :, :3] * 255).astype(np.uint8)

    return colored_depth


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


def _get_webcam_resolution():
        # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        raise SystemError("Error: Could not open webcam.")
    else:
        # Get the default width and height of the webcam frame
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print(f"Webcam resolution: {int(width)}x{int(height)}")

    # Release the webcam
    cap.release()

    return int(width),int(height)

def webcam(pipe: pipeline):
    print("Starting Webcam Demo...")
    cam_width, cam_height = _get_webcam_resolution()

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    # create the window
    # Create a named window
    cv2.namedWindow('Depth Anything Demo (Webcam)', cv2.WND_PROP_FULLSCREEN)
    # Set the window to full screen
    cv2.setWindowProperty('Depth Anything Demo (Webcam)', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Get the screen resolution
    monitor = get_monitors()[0]  # Get primary monitor
    screen_width, screen_height = monitor.width, monitor.height

    # calculate the resolution based on the smaller factor
    factor_width = screen_width/cam_width
    factor_heiht = screen_height/cam_height
    factor = min(factor_heiht,factor_width)

    # calculate scaling resultion
    output_width = int(factor*cam_width)
    output_height = int(factor*cam_height)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            break  # If there's an issue with capturing, exit the loop
        
        # Convert the frame from OpenCV's BGR format to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        # Convert the frame to a PIL image
        pil_image = Image.fromarray(frame_rgb)

        start_time = time.time()
        pil_image_depth = pipe(pil_image)["depth"]
        end_time = time.time()
        print(f"Estimated Depth:\t {end_time-start_time: .2f}s")
        
        # Apply the transformation
        transformed_pil_image = grayscale_to_gradient(pil_image_depth)
        
        # Convert the transformed PIL image back to a NumPy array (for OpenCV display)
        transformed_frame = np.array(transformed_pil_image)
        
        # Convert the RGB frame back to BGR for OpenCV display
        transformed_frame_bgr = cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR)

        # Resize the frame to fit the full screen resolution
        resized_frame = cv2.resize(transformed_frame_bgr, (output_width, output_height))
        
        # Display the transformed frame
        cv2.imshow('Depth Anything Demo (Webcam)', resized_frame)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped Webcam Demo.")
