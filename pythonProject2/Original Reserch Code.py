import cv2 as cv
import numpy as np


def schlieren_cam(channel=0, gain=5, delay=100):
    """
    Runs a synthetic schlieren system using a webcam.

    Parameters:
        channel (int): Camera channel to use.
        gain (int): Gain factor to amplify the intensity of difference images.
        delay (int): Delay in milliseconds between frames.
    """
    # Open the video file or webcam
    webcam = cv.VideoCapture('Hair Dryer  - original video.mp4')
    if not webcam.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Set resolution (optional, but will use the video resolution if not set)
    webcam.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
    webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    # Capture the initial frame as the background
    ret, first_frame = webcam.read()
    if not ret:
        print("Error: Could not capture the initial frame.")
        webcam.release()
        return
    first_frame_bw = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Define the display resolution (1920x1080)
    display_width = 1920
    display_height = 1080

    # Loop for video stream processing
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Could not read a frame from the webcam.")
            break

        # Convert current frame to grayscale
        frame_bw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Compute the difference
        diff = cv.absdiff(frame_bw, first_frame_bw)

        # Apply a blur and amplify intensity
        diff_smoothed = cv.medianBlur(diff, 5)
        diff_amplified = cv.multiply(diff_smoothed, gain)

        # Apply a color map for better visualization
        diff_colored = cv.applyColorMap(diff_amplified, cv.COLORMAP_JET)

        # Resize the frame to fit the screen (1920x1080) while maintaining the aspect ratio
        diff_resized = cv.resize(diff_colored, (display_width, display_height), interpolation=cv.INTER_LINEAR)

        # Display the resized frame (entire frame, no cropping)
        cv.imshow('Schlieren Effect', diff_resized)

        # Capture key events
        key = cv.waitKey(delay)
        if key == 27:  # ESC key to exit
            break
        elif key == 13:  # Enter key to save the frame
            cv.imwrite(f'schlieren_frame_{cv.getTickCount()}.jpg', diff_resized)
            print("Frame saved.")

    # Release resources
    webcam.release()
    cv.destroyAllWindows()


# Run the schlieren system
schlieren_cam(channel=0, gain=5, delay=5)
