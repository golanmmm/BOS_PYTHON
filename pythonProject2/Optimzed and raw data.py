import cv2 as cv
import numpy as np
import os

def images_to_video(image_folder, output_video_path, frame_rate):
    """
    Convert a sequence of images into a video.

    Parameters:
        image_folder (str): Path to the folder containing the image sequence.
        output_video_path (str): Path to save the generated video.
        frame_rate (int): Frames per second for the video.
    """
    # Get a sorted list of image files
    images = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.tif'))])
    if not images:
        print("Error: No images found in the specified folder.")
        return None

    # Read the first image to get dimensions
    first_image = cv.imread(images[0])
    height, width, _ = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write each image to the video
    for image_path in images:
        frame = cv.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video_path}")
    return output_video_path


def bos_from_video(input_file, output_file, gain=10, update_interval=4, blend_factor=0.5, display=False):
    """
    Perform Background Oriented Schlieren (BOS) processing on a video.

    Parameters:
        input_file (str): Path to the input video file.
        output_file (str): Path to save the output BOS video.
        gain (int): Gain factor to amplify the intensity of the difference images.
        update_interval (int): Number of frames after which the reference frame updates.
        blend_factor (float): Factor for blending the current reference frame with the previous one (0 to 1).
        display (bool): Whether to display the BOS output during processing.
    """
    # Open the input video file
    video = cv.VideoCapture(input_file)
    if not video.isOpened():
        print(f"Error: Unable to open video file {input_file}.")
        return

    # Get video properties
    frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(video.get(cv.CAP_PROP_FPS))
    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {input_file}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {frame_rate}, Total frames: {frame_count}")

    # Define the codec and create a VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))

    # Initialize variables
    reference_frame_bw = None
    previous_reference_frame_bw = None
    frame_index = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break  # End of video

        # Convert the current frame to grayscale
        frame_bw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Update the reference frame every 'update_interval' frames
        if frame_index % update_interval == 0:
            if previous_reference_frame_bw is None:
                # If this is the first reference frame, just assign it
                reference_frame_bw = frame_bw
            else:
                # Blend the current reference frame with the previous one
                reference_frame_bw = cv.addWeighted(frame_bw, blend_factor, previous_reference_frame_bw, 1 - blend_factor, 0)

        # Ensure the reference frame exists before computing the difference
        if reference_frame_bw is not None:
            # Compute the absolute difference
            diff = cv.absdiff(frame_bw, reference_frame_bw)

            # Smooth the difference image and amplify intensity
            diff_smoothed = cv.medianBlur(diff, 5)
            diff_amplified = cv.multiply(diff_smoothed, gain)

            # Apply a color map for visualization
            diff_colored = cv.applyColorMap(diff_amplified, cv.COLORMAP_JET)

            # Write the processed frame to the output video
            out.write(diff_colored)

            # Optionally display the result
            if display:
                cv.imshow("BOS Effect", diff_colored)
                if cv.waitKey(1) == 27:  # ESC key to exit display
                    break

        # Store the current reference frame for the next iteration
        previous_reference_frame_bw = reference_frame_bw

        # Increment the frame counter
        frame_index += 1

        if frame_index % 100 == 0:
            print(f"Processed {frame_index}/{frame_count} frames...")

    # Release resources
    video.release()
    out.release()
    cv.destroyAllWindows()
    print(f"BOS video saved as {output_file}")


# Example usage
image_folder = "C001H001S0002 50 CM"  # Folder containing image sequence
temp_video_path = "50_CM_Video.mp4"  # Temporary video file
bos_video_path = "50_CM_Video_BOS.mp4"  # Final BOS video file
frame_rate = 30  # Adjust as per your image sequence

# Step 1: Convert images to video
video_path = images_to_video(image_folder, temp_video_path, frame_rate)

# Step 2: Perform BOS processing on the video
if video_path:
    bos_from_video(video_path, bos_video_path, gain=10, update_interval=16607, blend_factor=1, display=True)
