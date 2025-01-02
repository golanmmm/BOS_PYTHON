import cv2 as cv
import numpy as np
import random
import os


def generate_bos_speckle_pattern(image_size=(3508, 2480), dot_size=7, dot_density=0.12, contrast=(0, 150)):
    """
    Generate a speckle pattern optimized for BOS and suitable for A4 printing.

    Args:
        image_size (tuple): Size of the output image (height, width) in pixels.
        dot_size (int): Diameter of the dots in pixels.
        dot_density (float): Fraction of the total area covered by dots (0 < density < 1).
        contrast (tuple): Min and max gray levels for the dots.

    Returns:
        speckle_pattern (numpy.ndarray): The generated speckle pattern as a grayscale image.
    """
    height, width = image_size
    min_gray, max_gray = contrast

    # Start with a blank white background
    speckle_pattern = np.full((height, width), 255, dtype=np.uint8)

    # Calculate the total number of dots
    dot_area = np.pi * (dot_size / 2) ** 2  # Area of a single dot
    num_dots = int(dot_density * (height * width) / dot_area)

    # Add random dots to the pattern
    for _ in range(num_dots):
        # Random center for the dot
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        # Random gray level for contrast
        gray_value = random.randint(min_gray, max_gray)

        # Draw the dot
        cv.circle(speckle_pattern, (x, y), dot_size // 2, gray_value, -1)

    return speckle_pattern


if __name__ == "__main__":
    # Parameters for the BOS-optimized speckle pattern
    a4_size = (3508, 2480)  # A4 dimensions at 300 DPI (height, width in pixels)
    dot_size = 3  # Diameter of each dot (in pixels)
    dot_density = 0.90  # Fraction of the total area covered by dots
    contrast = (0, 50)  # Gray levels for dots (min, max)

    # Generate the BOS-optimized speckle pattern
    speckle_pattern = generate_bos_speckle_pattern(a4_size, dot_size, dot_density, contrast)

    # Automatically save the speckle pattern in the current directory
    save_path = os.path.join(os.getcwd(), "speckle_pattern_bos_a4.png")
    try:
        cv.imwrite(save_path, speckle_pattern)
        print(f"Speckle pattern saved successfully at: {save_path}")
    except Exception as e:
        print(f"Error saving the file: {e}")

    # Display the result (optional)
    cv.imshow("BOS Speckle Pattern", speckle_pattern)
    cv.waitKey(0)
    cv.destroyAllWindows()
