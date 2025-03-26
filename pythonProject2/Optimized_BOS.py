import cv2 as cv
import numpy as np
import time

def schlieren_cam(channel=0, gain=10, delay=100, update_interval=4, blend_factor=0.5):
    """
    Synthetic Schlieren System using a webcam and OpenCV.

    Parameters:
        channel (int): Camera channel (0 for default camera).
        gain (int): Gain factor to amplify the intensity of the difference images.
        delay (int): Delay in milliseconds between frames.
        update_interval (int): Number of frames after which the reference frame updates.
        blend_factor (float): Factor for blending the current reference frame with the previous one (0 to 1).
    """
    # Open the camera
    webcam = cv.VideoCapture(0, cv.CAP_DSHOW)  # Use 0 for the default camera

    # Check if the camera is opened successfully
    if not webcam.isOpened():
        print(f"Error: Unable to open camera with channel {channel}.")
        return
    else:
        print("Camera initialized successfully.")

    # Warm up the camera
    print("Warming up the camera...")
    time.sleep(0)

    # Set camera properties


    # Initialize variables
    frame_count = 0
    reference_frame_bw = None  # Placeholder for the reference frame
    previous_reference_frame_bw = None  # Placeholder for the previous reference frame

    # Define the display resolution
    display_width = 1920
    display_height = 1080

    # Start processing the video stream
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        # Convert the current frame to grayscale
        frame_bw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Update the reference frame every 'update_interval' frames
        if frame_count % update_interval == 0:
            if previous_reference_frame_bw is None:
                # If this is the first reference frame, just assign it
                reference_frame_bw = frame_bw
            else:
                # Blend the current reference frame with the previous one
                reference_frame_bw = cv.addWeighted(frame_bw, blend_factor, previous_reference_frame_bw, 1 - blend_factor, 0)
            print(f"Reference frame updated at frame {frame_count}")

        # Ensure the reference frame exists before computing the difference
        if reference_frame_bw is not None:
            # Compute the absolute difference
            diff = cv.absdiff(frame_bw, reference_frame_bw)

            # Smooth the difference image and amplify intensity
            diff_smoothed = cv.medianBlur(diff, 5)
            diff_amplified = cv.multiply(diff_smoothed, gain)

            # Apply a color map for visualization
            diff_colored = cv.applyColorMap(diff_amplified, cv.COLORMAP_JET)

            # Resize the processed image to fit the 1920x1080 display
            diff_resized = cv.resize(diff_colored, (display_width, display_height), interpolation=cv.INTER_LINEAR)

            # Display the resized processed image
            cv.imshow("Schlieren Effect", diff_resized)

        # Store the current reference frame for the next iteration
        previous_reference_frame_bw = reference_frame_bw

        # Increment the frame counter
        frame_count += 1

        # Key handling
        key = cv.waitKey(delay)
        if key == 27:  # ESC key to exit
            
            break
        elif key == 13:  # Enter key to save the frame
            filename = f'schlieren_frame_{int(time.time())}.jpg'
            cv.imwrite(filename, diff_resized)
            print(f"Frame saved as {filename}")

    # Release resources
    webcam.release()
    cv.destroyAllWindows()

# Run the Schlieren System
schlieren_cam(channel=0, gain=5, delay=1, update_interval=5, blend_factor=0.5)