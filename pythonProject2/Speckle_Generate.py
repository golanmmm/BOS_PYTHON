import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2  # Ensure you have OpenCV installed for circle drawing


def calculate_speckle_size(pixel_size, L, D):
    """
    Calculate the speckle size based on the user input.

    Args:
    pixel_size (float): The camera pixel size in meters.
    L (float): The distance from the camera to the screen in meters.
    D (float): The distance from the screen to the speckles background in meters.

    Returns:
    speckle_size (float): The calculated speckle size in meters.
    """
    speckle_size = (pixel_size * L) / D
    return speckle_size


def generate_speckle_pattern(image_size, dot_diameter, dot_spacing):
    """
    Generate a gridded dot pattern for the speckle background.

    Args:
    image_size (tuple): Size of the image in pixels (width, height).
    dot_diameter (float): Diameter of each dot in pixels.
    dot_spacing (float): Spacing between adjacent dots in pixels.

    Returns:
    speckle_pattern (ndarray): An array representing the speckle pattern.
    """
    width, height = image_size
    speckle_pattern = np.ones((height, width), dtype=np.uint8) * 0  # Start with a white background

    # Generate a grid of dots with the specified spacing and size
    for y in range(0, height, int(dot_spacing)):
        for x in range(0, width, int(dot_spacing)):
            # Draw a circle (dot) at each grid point
            center = (x, y)
            radius = int(dot_diameter / 2)
            color = 255  # White dot
            speckle_pattern = cv2.circle(speckle_pattern, center, radius, color, -1)

    return speckle_pattern


def main():
    print("Enter the camera pixel size in meters (e.g., 1.12e-6): ")
    #pixel_size = float(input())
    pixel_size = 20e-6

    print("Enter the distance from the camera to the screen in meters (e.g., 2.0): ")
    #L = float(input())
    L= 2.0

    print("Enter the distance from the screen to the speckles background in meters (e.g., 1.0): ")
    #D = float(input())
    D= 2.0

    # Calculate the speckle size
    speckle_size = calculate_speckle_size(pixel_size, L, D)
    print(f"Calculated speckle size: {speckle_size} meters")

    # Parameters for dot size and spacing
    dot_diameter = 75e-6  # Dot diameter in meters (125 µm)
    dot_spacing = 100e-6  # Dot spacing in meters (175 µm)

    # Convert from meters to pixels (assuming 300 DPI printing)
    dpi = 300
    dot_diameter_pixels = dot_diameter * dpi * 254  # Convert to pixels
    dot_spacing_pixels = dot_spacing * dpi * 254  # Convert to pixels

    # Define image size based on A4 paper dimensions (210mm x 297mm)
    a4_width = 210  # mm
    a4_height = 297  # mm

    # Convert A4 dimensions to pixels at 300 DPI
    width_pixels = int(a4_width * dpi / 25.4)  # Convert mm to inches, then to pixels
    height_pixels = int(a4_height * dpi / 25.4)

    print(f"Generating speckle pattern for {width_pixels}x{height_pixels} pixels.")

    # Generate the speckle pattern
    speckle_pattern = generate_speckle_pattern((width_pixels, height_pixels), dot_diameter_pixels, dot_spacing_pixels)

    # Convert the pattern into an image
    speckle_image = Image.fromarray(speckle_pattern)

    # Save and show the speckle pattern
    speckle_image.save("speckle_pattern.png")
    speckle_image.show()


if __name__ == "__main__":
    main()
