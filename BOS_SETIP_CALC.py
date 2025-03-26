import numpy as np
from scipy.optimize import minimize

def calculate_distances(f, dA, epsilon_y):
    """
    Calculate the optimal distances ZD, ZA, ZB, zi, and image displacement (Delta y).

    Parameters:
        f (float): Focal length of the lens (meters).
        dA (float): Aperture diameter of the lens (meters).
        epsilon_y (float): Deflection angle (radians).

    Returns:
        dict: A dictionary containing the calculated distances and values.
    """
    # Define the overall blur function (dSigma)
    def overall_blur(params):
        ZD, ZA = params  # Variables to optimize
        ZB = ZA + ZD  # Distance from lens to the background
        zi = 1 / (1 / f - 1 / ZB)  # Distance from lens to the image plane

        # Magnification factors
        M = zi / ZB  # Magnification of the background
        M_prime = zi / ZA  # Magnification for density gradient imaging

        # Geometric blur
        di = dA * (1 - (1 / f) * M_prime * (ZA - f))

        # Diffraction-limited blur
        dd = 2.44 * (f / dA) * (M + 1)

        # Overall blur
        dSigma = (di**2 + dd**2) ** 0.5
        return dSigma

    # Define bounds for ZD and ZA
    bounds = [(0.1, 10), (0.1, 10)]  # ZD and ZA must be positive and reasonable

    # Initial guess
    initial_guess = [1.0, 1.0]

    # Minimize the overall blur function
    result = minimize(overall_blur, initial_guess, bounds=bounds)

    # Extract optimized parameters
    ZD_opt, ZA_opt = result.x
    ZB_opt = ZA_opt + ZD_opt
    zi_opt = 1 / (1 / f - 1 / ZB_opt)

    # Calculate displacement (Delta y) for optimized parameters
    Delta_y = f * (ZD_opt / (ZD_opt + ZA_opt - f)) * epsilon_y

    return {
        "ZD_opt": ZD_opt,
        "ZA_opt": ZA_opt,
        "ZB_opt": ZB_opt,
        "zi_opt": zi_opt,
        "Delta_y": Delta_y
    }

# Main function to get inputs and calculate
if __name__ == "__main__":
    print("Enter camera parameters:")
    f = float(input("Focal length of the lens (meters): "))
    dA = float(input("Aperture diameter of the lens (meters): "))
    epsilon_y = float(input("Deflection angle (radians, e.g., 0.0001): "))

    # Perform calculations
    results = calculate_distances(f, dA, epsilon_y)

    # Display results
    print("\nCalculated Distances:")
    print(f"ZD (distance schlieren to background): {results['ZD_opt']:.3f} m")
    print(f"ZA (distance lens to object): {results['ZA_opt']:.3f} m")
    print(f"ZB (distance lens to background): {results['ZB_opt']:.3f} m")
    print(f"zi (distance lens to image plane): {results['zi_opt']:.3f} m")
    print(f"Image displacement (Delta y): {results['Delta_y']:.6f} m")