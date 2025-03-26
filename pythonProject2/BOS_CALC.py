import numpy as np
from scipy.optimize import minimize

# Given parameters for the BOS setup
f = 0.06  # Focal length of the lens in meters (60 mm)
f_number = 32  # Lens f/number
dA = f / f_number  # Aperture diameter (m)
M = 0.6562  # Fixed magnification
epsilon_y = 1e-3  # Deflection angle (radians)

# Optimize ZD, ZA, ZB for maximum sensitivity and focus
def optimize_distances():
    def lens_equation(params):
        ZD, ZA = params  # Distances to optimize
        ZB = ZD + ZA  # Total distance to the background
        zi = M * ZB  # From magnification: M = zi / ZB
        focus_term = abs(1 / f - (1 / zi + 1 / ZB))  # Ensure lens equation is satisfied
        return focus_term

    # Initial guess and bounds for ZD and ZA
    initial_guess = [1.0, 1.0]  # Start with reasonable values for ZD and ZA (meters)
    bounds = [(0.1, 5.0), (0.1, 5.0)]  # ZD and ZA in a practical range

    # Minimize lens equation error to find ZD and ZA
    result = minimize(lens_equation, initial_guess, bounds=bounds)

    # Extract results
    ZD_opt, ZA_opt = result.x
    ZB_opt = ZD_opt + ZA_opt
    zi_opt = M * ZB_opt  # From magnification

    # Calculate image displacement
    Delta_y = f * (ZD_opt / (ZD_opt + ZA_opt - f)) * epsilon_y

    return ZD_opt, ZA_opt, ZB_opt, zi_opt, Delta_y

# Perform calculations
if __name__ == "__main__":
    ZD_opt, ZA_opt, ZB_opt, zi_opt, Delta_y = optimize_distances()

    # Display the results
    print(f"Optimized ZD (distance schlieren to background): {ZD_opt:.3f} m")
    print(f"Optimized ZA (distance lens to object): {ZA_opt:.3f} m")
    print(f"Optimized ZB (distance lens to background): {ZB_opt:.3f} m")
    print(f"Optimized zi (distance lens to image plane): {zi_opt:.3f} m")
    print(f"Image displacement (Delta y): {Delta_y:.6f} m")
