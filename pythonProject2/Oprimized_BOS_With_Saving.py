import cv2 as cv
import numpy as np

def schlieren_cam(channel=0, gain=10, update_interval=4, blend_factor=0.5, output_filename="output.avi"):
    """
    Synthetic Schlieren System using a webcam and OpenCV, saving the processed video with progress updates.

    Parameters:
        channel (int): Camera channel (0 for default camera).
        gain (int): Gain factor to amplify the intensity of the difference images.
        update_interval (int): Number of frames after which the reference frame updates.
        blend_factor (float): Factor for blending the current reference frame with the previous one (0 to 1).
        output_filename (str): Name of the file to save the processed video.
    """
    # Open the camera or video file
    webcam = cv.VideoCapture('Procced BOS/125HZ IPAD.MOV')

    if not webcam.isOpened():
        print(f"Error: Unable to open camera or video with channel {channel}.")
        return
    else:
        print("Camera/video initialized successfully.")

    # Get original video properties
    fps = int(webcam.get(cv.CAP_PROP_FPS))
    frame_width = int(webcam.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(webcam.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(webcam.get(cv.CAP_PROP_FRAME_COUNT))
    fourcc = cv.VideoWriter_fourcc(*'MP4V')  # Codec for AVI format

    # Define video writer
    out = cv.VideoWriter(output_filename, fourcc, fps, (1920, 1080))

    # Frame processing variables
    frame_count = 0
    reference_frame_bw = None
    previous_reference_frame_bw = None
    display_width, display_height = 1920, 1080

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("End of video or unable to capture video.")
            break

        # Convert the current frame to grayscale
        frame_bw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Update the reference frame periodically
        if frame_count % update_interval == 0:
            if previous_reference_frame_bw is None:
                reference_frame_bw = frame_bw
            else:
                reference_frame_bw = cv.addWeighted(frame_bw, blend_factor, previous_reference_frame_bw, 1 - blend_factor, 0)

        if reference_frame_bw is not None:
            # Compute Schlieren effect
            diff = cv.absdiff(frame_bw, reference_frame_bw)
            diff_smoothed = cv.medianBlur(diff, 5)
            diff_amplified = cv.multiply(diff_smoothed, gain)
            diff_colored = cv.applyColorMap(diff_amplified, cv.COLORMAP_JET)
            diff_resized = cv.resize(diff_colored, (display_width, display_height), interpolation=cv.INTER_LINEAR)

            # Write the frame to the output video file
            out.write(diff_resized)

            progress = (frame_count / total_frames) * 100
            print(f"Processing: {progress:.2f}%", end="\r")

        # Update and display progress
        #progress = (frame_count / total_frames) * 100
        #print(f"Processing: {progress:.2f}%", end="\r")

        previous_reference_frame_bw = reference_frame_bw
        frame_count += 1

    # Release resources
    webcam.release()
    out.release()  # Release the video writer
    print(f"\nProcessed video saved as {output_filename}")

# Run the Schlieren System
schlieren_cam(channel=0, gain=7, update_interval=10, blend_factor=1, output_filename="200HZ_processed_video.mp4")
