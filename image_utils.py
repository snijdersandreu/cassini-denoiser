import numpy as np

def create_display_image(image_data, method='percentile'):
    """
    Scales raw image data (numpy float array) to a uint8 array [0, 255] 
    suitable for display using either percentile or min-max stretching.

    Args:
        image_data (np.ndarray): Input float image data.
        method (str): Scaling method, either 'percentile' (default) or 'minmax'.

    Returns:
        np.ndarray: uint8 image data scaled to [0, 255].
    """
    if not isinstance(image_data, np.ndarray):
        raise TypeError("Input image_data must be a numpy array.")
        
    if image_data.size == 0:
        # Handle empty array case gracefully
        return np.zeros(image_data.shape, dtype=np.uint8)

    img_float = image_data.astype(np.float64) # Use float64 for precision

    if method == 'percentile':
        p1, p99 = np.percentile(img_float, (1, 99))
        min_val, max_val = p1, p99
    elif method == 'minmax':
        min_val = np.min(img_float)
        max_val = np.max(img_float)
    else:
        raise ValueError("Method must be 'percentile' or 'minmax'.")

    # Handle constant image case or very small range
    if max_val - min_val < 1e-10:
        # Return a constant gray image (e.g., middle gray or based on the value)
        # Avoid division by zero
        # Let's map the constant value to 128 (mid-gray)
        # Or maybe return zeros like before? Let's stick to zeros for now.
        return np.zeros_like(img_float, dtype=np.uint8)

    # Perform scaling
    scaled = (img_float - min_val) / (max_val - min_val)
    
    # Clip to [0, 1] and scale to [0, 255]
    scaled_clamped = np.clip(scaled, 0, 1) * 255.0
    
    # Convert to uint8
    display_arr = scaled_clamped.astype(np.uint8)
    
    return display_arr 