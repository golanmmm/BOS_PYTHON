import cv2 as cv
import numpy as np
import os

def bos_from_images(
    image_folder,
    output_video_path,
    gain=10,
    reference_interval=1,
    blend_factor=0.5,
    initial_reference=False,
    start_frame=0,
    reference_frame=None,
    output_frame_rate=30,  # New parameter to control video speed
    display=False):
    """
    Perform Background Oriented Schlieren (BOS) processing on a sequence of images.

    Parameters:
        image_folder (str): Path to the folder containing the image sequence.
        output_video_path (str): Path to save the output BOS video.
        gain (int): Gain factor to amplify the intensity of the difference images.
        reference_interval (int): Number of frames after which the reference frame updates.
        blend_factor (float): Factor for blending the current reference frame with the previous one (0 to 1).
        initial_reference (bool): If True, only use the initial frame or `reference_frame` as the reference.
        start_frame (int): Index of the frame to start the analysis from.
        reference_frame (int): Specific frame number to use as the reference frame, regardless of `start_frame`.
        output_frame_rate (int): Frames per second for the output video.
        display (bool): Whether to display the BOS output during processing.
    """
    # Get a sorted list of image files
    images = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.tif'))])
    if not images:
        print("Error: No images found in the specified folder.")
        return None

    # Validate start_frame and reference_frame
    if start_frame >= len(images):
        print(f"Error: Start frame {start_frame} exceeds the total number of frames ({len(images)}).")
        return None

    if reference_frame is not None and reference_frame >= len(images):
        print(f"Error: Reference frame {reference_frame} exceeds the total number of frames ({len(images)}).")
        return None

    # Read the first image to get dimensions
    first_image = cv.imread(images[0])
    height, width, _ = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_video_path, fourcc, output_frame_rate, (width, height))

    # Initialize the reference frame
    if reference_frame is not None:
        ref_image_path = images[reference_frame]
        ref_frame = cv.imread(ref_image_path, cv.IMREAD_GRAYSCALE)
        reference_frame_bw = ref_frame
        print(f"Using frame {reference_frame} as the reference frame.")
    else:
        reference_frame_bw = None

    previous_reference_frame_bw = None

    # Process frames starting from the specified start frame
    for frame_index, image_path in enumerate(images[start_frame:], start=start_frame):
        # Read and convert the current image to grayscale
        frame = cv.imread(image_path)
        frame_bw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Update the reference frame based on the interval or use the specific reference frame
        if reference_frame_bw is None or (not initial_reference and frame_index % reference_interval == 0):
            if previous_reference_frame_bw is None:
                # First reference frame
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

        if frame_index % 100 == 0:
            print(f"Processed {frame_index + 1}/{len(images)} images...")

    # Release resources
    out.release()
    cv.destroyAllWindows()
    print(f"BOS video saved as {output_video_path}")


# Example usage
image_folder = "C001H001S0002 50 CM"  # Folder containing image sequence
bos_video_path = "50_CM_Video_BOS.mp4"  # Final BOS video file

# Perform BOS processing directly on images
bos_from_images(
    image_folder=image_folder,
    output_video_path=bos_video_path,
    gain=20,
    reference_interval=1,  # Adjust reference update interval
    blend_factor=0.5,
    initial_reference=True,  # Use only the reference frame if True
    start_frame=1,  # Start analysis from frame 1000
    reference_frame=1,  # Use frame  as the reference frame
    output_frame_rate=100,  # Set video speed (higher value = faster video)
    display=True
)
