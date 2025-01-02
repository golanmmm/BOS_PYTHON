import cv2 as cv
import numpy as np
import time

def schlieren_cam(channel=0, gain=10, delay=100, update_interval=4, alpha=0.5):
    """
    Synthetic Schlieren System using a webcam and OpenCV.

    Parameters:
        channel (int): Camera channel (0 for default camera).
        gain (int): Gain factor to amplify the intensity of the difference images.
        delay (int): Delay in milliseconds between frames.
        update_interval (int): Number of frames after which the reference frame updates.
        alpha (float): Blending factor for smoothing reference frame updates (0 to 1).
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
    time.sleep(1)

    # Set camera properties
    webcam.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
    webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize variables
    frame_count = 0
    reference_frame_bw = None  # Placeholder for the reference frame
    new_reference_frame_bw = None  # Placeholder for the new reference frame

    # Define the ROI dimensions
    lx, ly = 1920, 1080  # Width and height of the region of interest

    def crop_image(image, lx, ly):
        """Crop the image to a centered region of interest (ROI)."""
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        x1, x2 = cx - lx // 2, cx + lx // 2
        y1, y2 = cy - ly // 2, cy + ly // 2
        return image[y1:y2, x1:x2]



    # Start processing the video stream
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        # Convert the current frame to grayscale
        frame_bw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Update the new reference frame every 'update_interval' frames
        if frame_count % update_interval == 0:
            new_reference_frame_bw = frame_bw
            print(f"New reference frame captured at frame {frame_count}")

        # Blend the reference frame with the new one
        if new_reference_frame_bw is not None:
            if reference_frame_bw is None:
                reference_frame_bw = new_reference_frame_bw  # Initialize the first reference frame
            else:
                reference_frame_bw = cv.addWeighted(reference_frame_bw, 1 - alpha, new_reference_frame_bw, alpha, 0)

        # Ensure the reference frame exists before computing the difference
        if reference_frame_bw is not None:
            # Compute the absolute difference
            diff = cv.absdiff(frame_bw, reference_frame_bw)

            # Crop the ROI
            diff_cropped = crop_image(diff, lx, ly)

            # Smooth the difference image and amplify intensity
            diff_smoothed = cv.medianBlur(diff_cropped, 5)
            diff_amplified = cv.multiply(diff_smoothed, gain)

            # Apply a color map for visualization
            diff_colored = cv.applyColorMap(diff_amplified, cv.COLORMAP_JET)

            # Display the processed image
            cv.imshow("Schlieren Effect", diff_colored)

        # Increment the frame counter
        frame_count += 1

        # Key handling
        key = cv.waitKey(delay)
        if key == 27:  # ESC key to exit
            break
        elif key == 13:  # Enter key to save the frame
            filename = f'schlieren_frame_{int(time.time())}.jpg'
            cv.imwrite(filename, diff_colored)
            print(f"Frame saved as {filename}")

    # Release resources
    webcam.release()
    cv.destroyAllWindows()

# Run the Schlieren System
schlieren_cam(channel=0, gain=1, delay=10, update_interval=2, alpha=0.2)
