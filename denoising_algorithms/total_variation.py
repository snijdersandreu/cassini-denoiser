import numpy as np
from scipy.fft import fft2, ifft2

def _grad(u):
    """
    Computes the forward gradient of a 2D image u.
    Uses forward differences with Neumann boundary conditions.
    """
    h, w = u.shape
    g = np.zeros((h, w, 2), dtype=u.dtype)
    
    # x-gradient
    g[:, :-1, 0] = u[:, 1:] - u[:, :-1]
    
    # y-gradient
    g[:-1, :, 1] = u[1:, :] - u[:-1, :]
    
    return g

def _div(d):
    """
    Computes the backward divergence of a 2D vector field d.
    Uses backward differences with Neumann boundary conditions.
    This is the negative adjoint of the _grad function.
    """
    h, w = d.shape[:2]
    div_d = np.zeros((h, w), dtype=d.dtype)
    
    # Divergence of x-component
    dx = d[:, :, 0]
    div_d[:, 1:] += dx[:, :-1]
    div_d[:, 0] += dx[:, 0]
    div_d[:, 1:] -= dx[:, 1:]

    # Divergence of y-component
    dy = d[:, :, 1]
    div_d[1:, :] += dy[:-1, :]
    div_d[0, :] += dy[0, :]
    div_d[1:, :] -= dy[1:, :]
    
    return div_d
    
def _shrink(x, t):
    """
    Vectorial shrinkage (soft-thresholding) for anisotropic TV.
    """
    # For each pixel, shrink the magnitude of the gradient vector
    norm_x = np.sqrt(x[..., 0]**2 + x[..., 1]**2)
    # Avoid division by zero
    norm_x[norm_x == 0] = 1
    
    # Shrinkage factor
    factor = np.maximum(norm_x - t, 0) / norm_x
    
    # Apply shrinkage
    shrunk_x = x * factor[..., np.newaxis]
    
    return shrunk_x

def tv_denoise(img, lambda_param=10.0, max_iter=100, tolerance=1e-4):
    """
    Apply Total Variation (TV) denoising using the Split Bregman method.
    
    This implements the Split Bregman optimization for the Rudin-Osher-Fatemi (ROF)
    model, which minimizes the following cost function:
    
        min_u TV(u) + (lambda_param / 2) * ||u - f||^2
    
    This method is efficient and robust for edge-preserving noise removal.
    
    Parameters:
    -----------
    img : ndarray
        Input image (2D numpy array).
    lambda_param : float
        Regularization parameter (lambda). Controls the trade-off between
        noise removal and fidelity to the original image.
        Higher values = less smoothing (more fidelity to original).
        Lower values = more smoothing.
    max_iter : int
        Maximum number of iterations.
    tolerance : float
        Convergence tolerance. The algorithm stops when the relative change
        in the solution `u` is below this value.
        
    Returns:
    --------
    denoised_img : ndarray
        Denoised image.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("Input image must be a 2D array.")

    h, w = img.shape
    
    # Algorithm parameters from the Split Bregman paper
    # The fidelity term is (lambda_param / 2) * ||u - f||^2.
    mu = 2.0 * lambda_param  # A common choice for the penalty parameter

    # Pre-compute FFT of the identity and gradient operators
    F_f = fft2(img)
    
    # PSF for gradient operators
    psf_x = np.zeros_like(img)
    psf_x[0, 0] = -1
    psf_x[0, 1] = 1
    psf_y = np.zeros_like(img)
    psf_y[0, 0] = -1
    psf_y[1, 0] = 1
    
    F_dx = fft2(psf_x)
    F_dy = fft2(psf_y)
    
    # Denominator for the u-subproblem
    denom = lambda_param + mu * (np.conj(F_dx) * F_dx + np.conj(F_dy) * F_dy)

    # Initialize variables
    u = img.copy()
    d = np.zeros((h, w, 2), dtype=img.dtype)
    b = np.zeros((h, w, 2), dtype=img.dtype)

    for i in range(max_iter):
        u_old = u.copy()
        
        # --- u-subproblem: solve for u ---
        # This is a screened Poisson equation, solved efficiently in Fourier domain.
        # RHS = lambda*f + mu*div(d-b)
        rhs_div = _div(d - b)
        F_rhs = lambda_param * F_f + mu * fft2(rhs_div)
        F_u = F_rhs / denom
        u = np.real(ifft2(F_u))
        
        # --- d-subproblem: solve for d ---
        # This is a vectorial shrinkage operation.
        grad_u = _grad(u)
        d = _shrink(grad_u + b, 1.0 / mu)
        
        # --- b-update: update Bregman variable ---
        b = b + (grad_u - d)
        
        # --- Check for convergence ---
        change = np.linalg.norm(u - u_old, 'fro') / np.linalg.norm(u, 'fro')
        if change < tolerance:
            print(f"TV Bregman converged at iteration {i+1} with change {change:.2e} < tolerance {tolerance:.2e}")
            break
    else:
        print(f"TV Bregman reached max_iter={max_iter} without converging to tolerance={tolerance:.2e}. Final change: {change:.2e}")

    return u 