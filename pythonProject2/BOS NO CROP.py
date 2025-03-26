import cv2 as cv
import numpy as np
import time


def schlieren_cam(channel=0, gain=10, delay=100, update_interval=4, alpha=0.5, target_width=1920, target_height=1080,
                  start_frame=0):
    """
    Synthetic Schlieren System using a webcam and OpenCV.

    Parameters:
        channel (int): Camera channel (0 for default camera).
        gain (int): Gain factor to amplify the intensity of the difference images.
        delay (int): Delay in milliseconds between frames.
        update_interval (int): Number of frames after which the reference frame updates.
        alpha (float): Blending factor for smoothing reference frame updates (0 to 1).
        target_width (int): Target screen width (default is 1920).
        target_height (int): Target screen height (default is 1080).
        start_frame (int): Frame number to start processing from (default is 0).
    """
    # Open the video file or camera
    webcam = cv.VideoCapture('temp_video.mp4')

    # Check if the camera or video file is opened successfully
    if not webcam.isOpened():
        print(f"Error: Unable to open camera with channel {channel}.")
        return
    else:
        print("Camera initialized successfully.")

    # Set the starting frame position
    if start_frame > 0:
        webcam.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Starting processing from frame {start_frame}.")

    # Set camera properties for the video resolution (optional if using a webcam)
    webcam.set(cv.CAP_PROP_FRAME_WIDTH, target_width)
    webcam.set(cv.CAP_PROP_FRAME_HEIGHT, target_height)

    # Initialize variables
    frame_count = start_frame  # Start counting from the specified frame
    reference_frame_bw = None  # Placeholder for the reference frame
    new_reference_frame_bw = None  # Placeholder for the new reference frame

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

            # Smooth the difference image and amplify intensity
            diff_smoothed = cv.medianBlur(diff, 5)
            diff_amplified = cv.multiply(diff_smoothed, gain)

            # Apply a color map for visualization
            diff_colored = cv.applyColorMap(diff_amplified, cv.COLORMAP_JET)

            # Resize the output frame to fit within the screen resolution while maintaining the aspect ratio
            frame_height, frame_width = diff_colored.shape[:2]
            aspect_ratio = frame_width / frame_height

            # Calculate the new size keeping the aspect ratio
            if frame_width > target_width or frame_height > target_height:
                if aspect_ratio > 1:  # Wider than tall
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:  # Taller than wide or square
                    new_height = target_height
                    new_width = int(target_height * aspect_ratio)

                # Resize to fit the screen
                diff_colored_resized = cv.resize(diff_colored, (1920, 1080))
            else:
                diff_colored_resized = diff_colored  # If the frame is already smaller than the target resolution

            # Display the processed and resized image
            cv.imshow("Schlieren Effect", diff_colored_resized)

        # Increment the frame counter
        frame_count += 1

        # Key handling
        key = cv.waitKey(delay)
        if key == 27:  # ESC key to exit
            break
        elif key == 13:  # Enter key to save the frame
            filename = f'schlieren_frame_{int(time.time())}.jpg'
            cv.imwrite(filename, diff_colored_resized)
            print(f"Frame saved as {filename}")

    # Release resources
    webcam.release()
    cv.destroyAllWindows()


# Run the Schlieren System
schlieren_cam(channel=0, gain=10, delay=10, update_interval=1, alpha=0.05, target_width=1920, target_height=1080,
              start_frame=9000)

