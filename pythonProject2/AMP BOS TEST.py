import eulerian_magnification as em
import cv2
import numpy as np


def process_video_with_eulerian_magnification(input_path, output_path, freq_min=50.0 / 60.0, freq_max=1.0,
                                              amplification=50, pyramid_levels=3):
    """
    Apply Eulerian Video Magnification to amplify subtle changes in a video.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the processed video.
        freq_min (float): Minimum frequency for bandpass filter (Hz).
        freq_max (float): Maximum frequency for bandpass filter (Hz).
        amplification (float): Motion amplification factor.
        pyramid_levels (int): Number of pyramid levels for processing.
    """
    # Load video and get frames as a float array
    vid, fps = em.load_video_float(input_path)

    # Apply Eulerian Magnification
    print("Applying Eulerian Magnification...")
    magnified_vid = em.eulerian_magnification(
        vid,
        fps,
        freq_min=freq_min,
        freq_max=freq_max,
        amplification=amplification,
        pyramid_levels=pyramid_levels
    )

    # Save the processed video
    print("Saving amplified video...")
    em.save_video_float(magnified_vid, fps, output_path)
    print(f"Video saved to {output_path}")


# Example usage
input_video = "input_video.mp4"  # Path to the input video
output_video = "output_video.mp4"  # Path to save the output video

process_video_with_eulerian_magnification(
    input_path=input_video,
    output_path=output_video,
    freq_min=50.0 / 60.0,  # Minimum frequency (Hz)
    freq_max=1.0,  # Maximum frequency (Hz)
    amplification=30,  # Amplification factor
    pyramid_levels=3  # Pyramid levels
)


