import numpy as np

def tv_denoise(img, weight=0.1, max_iter=100, tolerance=1e-4):
    """
    Apply Total Variation (TV) denoising to a 2D image using the ROF model.
    
    This implements the Rudin-Osher-Fatemi (ROF) total variation denoising model
    using an iterative algorithm. TV denoising preserves edges while removing noise.
    
    Parameters:
    -----------
    img : ndarray
        Input image (2D numpy array)
    weight : float
        Regularization parameter. Higher values = more smoothing.
        Typical range: 0.01 to 1.0
    max_iter : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance
        
    Returns:
    --------
    denoised_img : ndarray
        Denoised image
    """
    # Ensure image is float
    img = np.asarray(img, dtype=np.float64)
    
    # Get image dimensions
    rows, cols = img.shape
    
    # Initialize the denoised image
    u = img.copy()
    
    # Dual variables for the TV optimization
    p = np.zeros((rows, cols, 2))
    
    # Time step for the algorithm
    dt = 0.25
    
    for iteration in range(max_iter):
        u_old = u.copy()
        
        # Compute gradient of u
        grad_u = np.zeros((rows, cols, 2))
        
        # Forward differences for gradient computation
        grad_u[:-1, :, 0] = u[1:, :] - u[:-1, :]  # vertical gradient
        grad_u[:, :-1, 1] = u[:, 1:] - u[:, :-1]  # horizontal gradient
        
        # Update dual variable p
        p_new = p + dt * grad_u
        
        # Normalize p to satisfy |p| <= 1
        p_norm = np.sqrt(p_new[:, :, 0]**2 + p_new[:, :, 1]**2)
        p_norm = np.maximum(p_norm, 1.0)
        p[:, :, 0] = p_new[:, :, 0] / p_norm
        p[:, :, 1] = p_new[:, :, 1] / p_norm
        
        # Compute divergence of p
        div_p = np.zeros((rows, cols))
        
        # Backward differences for divergence computation
        div_p[1:, :] += p[1:, :, 0] - p[:-1, :, 0]  # vertical component
        div_p[:, 1:] += p[:, 1:, 1] - p[:, :-1, 1]  # horizontal component
        div_p[0, :] += p[0, :, 0]  # boundary condition
        div_p[:, 0] += p[:, 0, 1]  # boundary condition
        
        # Update u using the ROF model
        u = img + weight * div_p
        
        # Check for convergence
        change = np.mean(np.abs(u - u_old))
        if change < tolerance:
            break
    
    return u 