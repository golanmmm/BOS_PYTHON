import cv2
import os

# Replace with your folder path
input_folder = 'C:/Users/amitg/Documents/test_images'
# Name of the output video file with full path
output_video = os.path.join(os.getcwd(), "testiel_video.avi")

try:
    # Get list of image files in the folder
    images = sorted([img for img in os.listdir(input_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])

    if not images:
        raise Exception("No image files found in the input folder")

    # Read the first image to get frame dimensions
    first_image_path = os.path.join(input_folder, images[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        raise Exception(f"Could not read image: {first_image_path}")

    height, width, layers = first_image.shape
    frame_size = (width, height)

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Using XVID codec for better compatibility
    fps = 3  # Frames per second
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    if not video_writer.isOpened():
        raise Exception("Failed to create video writer")

    # Iterate through images and add to the video
    for image in images:
        image_path = os.path.join(input_folder, image)
        frame = cv2.imread(image_path)
        if frame is not None:
            video_writer.write(frame)
        else:
            print(f"Warning: Could not read image {image_path}")

    # Release the video writer
    video_writer.release()
    print(f"Video successfully saved as {output_video}")

except Exception as e:
    print(f"Error: {str(e)}")
    if 'video_writer' in locals():
        video_writer.release()