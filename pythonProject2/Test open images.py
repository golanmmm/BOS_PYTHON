import os

def images_to_video(image_folder, output_video_path, frame_rate):
    images = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        print(f"Error: No images found in the specified folder: {image_folder}")
        return None

    print(f"Found {len(images)} images:")
    for img in images[:5]:  # Show the first 5 image paths
        print(img)

    first_image = cv.imread(images[0])
    height, width, _ = first_image.shape

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for image_path in images:
        frame = cv.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video_path}")
    return output_video_path
