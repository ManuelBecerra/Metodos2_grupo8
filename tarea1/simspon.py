def simpsons_rule(y, x):
    """
    Compute the integral of y(x) using Simpson's rule.

    Parameters:
    y: List of function values at corresponding x points.
    x: List of x-coordinates.

    Returns:
    The integral value using Simpson's rule.
    """
    if len(y) < 3 or len(x) < 3 or len(y) != len(x):
        raise ValueError("x and y must have the same length, and there must be at least 3 points.")

    n = len(x) - 1  # Number of intervals
    if n % 2 == 1:
        raise ValueError("Simpson's rule requires an even number of intervals (odd number of points).")

    h = (x[-1] - x[0]) / n  # Uniform spacing
    integral = y[0] + y[-1] + 4 * sum(y[1:n:2]) + 2 * sum(y[2:n-1:2])
    return integral * h / 3

# Example data
y = [1, 4, 9, 16]
x = [1, 2, 3, 4]

# Integration using the custom function
result = simpsons_rule(y, x)
print("Result:", result)
