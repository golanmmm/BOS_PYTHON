import cv2

def convert_mov_to_mp4(input_path, output_path):
    # open the video file
    cap = cv2.VideoCapture(input_path)

    # get video properties
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # original codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # define the codec and create a video writer object
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
    out = cv2.VideoWriter(output_path, fourcc_mp4, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # release resources
    cap.release()
    out.release()
    print(f"conversion completed: {output_path}")

# example usage
convert_mov_to_mp4("Procced BOS/115hz iPad .mov", "Procced BOS/115hz iPad .mp4")
