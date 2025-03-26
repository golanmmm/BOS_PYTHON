import numpy as np
import cv2


def generate_checkerboard_image(square_size, spacing, output_file="checkerboard.png"):
    """
    Generates a checkerboard pattern as an image with maximum rows and columns to fit A4 paper at 600 DPI.

    :param square_size: Size of each square in pixels
    :param spacing: Space between squares in pixels
    :param output_file: Name of the output image file (PNG for high quality printing)
    """
    # A4 dimensions in pixels at 600 DPI
    dpi = 600
    mm_to_inch = 1 / 25.4
    a4_width = int(210 * dpi * mm_to_inch)  # 210 mm to pixels
    a4_height = int(297 * dpi * mm_to_inch)  # 297 mm to pixels

    # Calculate maximum rows and columns that fit within A4 size
    cols = a4_width // (square_size + spacing)
    rows = a4_height // (square_size + spacing)

    # Create white background
    checkerboard = np.full((a4_height, a4_width, 3), 255, dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                color = (0, 0, 0)  # Black square
            else:
                color = (255, 255, 255)  # White square

            top_left_x = j * (square_size + spacing)
            top_left_y = i * (square_size + spacing)
            bottom_right_x = top_left_x + square_size
            bottom_right_y = top_left_y + square_size

            if bottom_right_x <= a4_width and bottom_right_y <= a4_height:
                cv2.rectangle(checkerboard, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, -1)

    cv2.imwrite(output_file, checkerboard, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Save as PNG with max quality

    print(f"Checkerboard saved as '{output_file}' with {rows} rows and {cols} columns")


def main():
    square_size = int(input("Enter square size (px): "))
    spacing = int(input("Enter spacing (px): "))

    generate_checkerboard_image(square_size, spacing, "checkerboard.png")


if __name__ == "__main__":
    main()
