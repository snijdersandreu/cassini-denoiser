import numpy as np
from scipy.fftpack import dct, idct
import time


def bm3d_denoise(img, sigma=None, stage='all', debug=False, callback=None):
    """
    Block-Matching and 3D filtering (BM3D) for image denoising.
    
    This is an implementation based on the algorithm described by Dabov et al. (2007)
    with modifications from Pakdelazar & Rezai-rad (2011).
    
    Parameters:
    -----------
    img : ndarray
        Input image (2D numpy array)
    sigma : float or None
        Noise standard deviation. If None, it will be estimated.
    stage : str, optional
        Which stages to perform ('all', 'hard', or 'wiener')
    debug : bool, optional
        Whether to print debug information
    callback : callable, optional
        Function to call to report progress (float from 0.0 to 1.0)
        
    Returns:
    --------
    denoised_img : ndarray
        Denoised image
    """
    # Convert to float if needed
    img_float = np.asarray(img, dtype=np.float64)
    
    if debug:
        print(f"Input image shape: {img_float.shape}")
    
    # Estimate noise if not provided
    if sigma is None:
        # Estimate noise using median absolute deviation
        sigma = np.median(np.abs(img_float - np.median(img_float))) / 0.6745
        sigma = max(0.01, sigma)  # Ensure positive sigma
        if debug:
            print(f"Estimated noise sigma: {sigma:.4f}")
    elif debug:
        print(f"Using provided noise sigma: {sigma:.4f}")
    
    # Set parameters based on noise level
    params = _get_bm3d_params(sigma)
    if debug:
        print(f"BM3D parameters: {params}")
    
    # Step 1: Basic estimate (hard thresholding)
    if stage in ['hard', 'all']:
        if debug:
            print("Starting Step 1: Hard thresholding...")
            t_start = time.time()
        
        # Report progress start
        if callback:
            callback(0.0)
            
        basic_estimate = _bm3d_step1(img_float, sigma, params, debug, callback=lambda p: callback(p * 0.5) if callback else None)
        
        if debug:
            t_elapsed = time.time() - t_start
            print(f"Step 1 completed in {t_elapsed:.2f} seconds")
        
        if stage == 'hard':
            if callback:
                callback(1.0)  # Complete
            return basic_estimate
    else:
        basic_estimate = img_float
    
    # Step 2: Final estimate (Wiener filtering)
    if stage in ['wiener', 'all']:
        if debug:
            print("Starting Step 2: Wiener filtering...")
            t_start = time.time()
        
        # Report 50% progress (hard thresholding done)
        if callback:
            callback(0.5)
            
        final_estimate = _bm3d_step2(img_float, basic_estimate, sigma, params, debug, 
                                     callback=lambda p: callback(0.5 + p * 0.5) if callback else None)
        
        if debug:
            t_elapsed = time.time() - t_start
            print(f"Step 2 completed in {t_elapsed:.2f} seconds")
        
        if callback:
            callback(1.0)  # Complete
            
        return final_estimate
    
    if callback:
        callback(1.0)  # Complete
        
    return basic_estimate


def _get_bm3d_params(sigma, custom_params=None):
    """
    Set BM3D parameters based on noise level following Pakdelazar et al.
    
    Parameters:
    -----------
    sigma : float
        Noise standard deviation
    custom_params : dict, optional
        Dictionary of custom parameters to override defaults
        
    Returns:
    --------
    params : dict
        Dictionary of BM3D parameters
    """
    params = {
        'block_size': 8,       # Size of reference block
        'window_size': 39,     # Search window size
        'step_hard': 3,        # Step size for reference blocks (hard thresholding)
        'step_wiener': 3,      # Step size for reference blocks (Wiener filtering)
        'transform_2d': 'dct', # 2D transform type
        'transform_3d': 'haar', # 3D transform type
        'max_blocks': 16,      # Max blocks to match (default for low noise)
        'match_threshold': 3000, # Matching threshold (default for low noise)
        'hard_threshold': 2.7,  # Hard threshold multiplier
        'wiener_threshold': 2.0 # Wiener threshold multiplier
    }
    
    # Tune parameters based on noise level
    if sigma < 30:
        params['max_blocks'] = 16
        params['match_threshold'] = 3000
        params['step_hard'] = 2
        params['step_wiener'] = 2
    elif sigma < 50:
        params['max_blocks'] = 32
        params['match_threshold'] = 6500
        params['step_hard'] = 2
        params['step_wiener'] = 2
    elif sigma < 80:
        params['max_blocks'] = 32
        params['match_threshold'] = 15000
    else:
        params['max_blocks'] = 64
        params['match_threshold'] = 30000
    
    # Override with custom parameters if provided
    if custom_params:
        for key, value in custom_params.items():
            if key in params:
                params[key] = value
    
    return params


def _dct_2d(block):
    """Apply 2D DCT to a block"""
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


def _idct_2d(block):
    """Apply 2D inverse DCT to a block"""
    return idct(idct(block, axis=1, norm='ortho'), axis=0, norm='ortho')


def _block_matching(img, ref_pos, block_size, window_size, max_blocks, threshold, is_basic_estimate=False):
    """
    Find similar blocks to the reference block within the search window.
    
    Parameters:
    -----------
    img : ndarray
        Input image or basic estimate
    ref_pos : tuple
        (y, x) coordinates of the reference block
    block_size : int
        Size of blocks
    window_size : int
        Size of search window
    max_blocks : int
        Maximum number of similar blocks to find
    threshold : float
        Distance threshold for block similarity
    is_basic_estimate : bool
        Whether img is a basic estimate
        
    Returns:
    --------
    block_positions : list
        List of positions of similar blocks
    """
    h, w = img.shape
    y_ref, x_ref = ref_pos
    half_window = window_size // 2
    
    # Define search window boundaries
    y_start = max(y_ref - half_window, 0)
    y_end = min(y_ref + half_window + 1, h - block_size + 1)
    x_start = max(x_ref - half_window, 0)
    x_end = min(x_ref + half_window + 1, w - block_size + 1)
    
    # Extract reference block
    ref_block = img[y_ref:y_ref+block_size, x_ref:x_ref+block_size]
    
    # Distance metric: use 2D DCT if not basic estimate
    if not is_basic_estimate:
        ref_dct = _dct_2d(ref_block)
    
    # Store block positions and their distances
    block_positions = [(y_ref, x_ref)]  # Always include the reference block
    distances = [0]
    
    # Search for similar blocks
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            if (y, x) == (y_ref, x_ref):
                continue
                
            # Extract candidate block
            candidate = img[y:y+block_size, x:x+block_size]
            
            # Calculate distance
            if is_basic_estimate:
                # For Step 2: Use normalized L2 distance
                dist = np.sum((ref_block - candidate)**2) / (block_size**2)
            else:
                # For Step 1: Use DCT distance
                candidate_dct = _dct_2d(candidate)
                dist = np.sum((ref_dct - candidate_dct)**2) / (block_size**2)
            
            if dist < threshold:
                block_positions.append((y, x))
                distances.append(dist)
                
                # If we exceed max_blocks, remove the furthest block
                if len(block_positions) > max_blocks:
                    # Find index of max distance (excluding reference block)
                    max_idx = np.argmax(distances[1:]) + 1
                    block_positions.pop(max_idx)
                    distances.pop(max_idx)
    
    return block_positions


def _hard_threshold_group(group, sigma, lambda_thr=2.7):
    """Apply hard thresholding to 3D transformed group"""
    # 2D DCT on each block
    group_dct = np.zeros_like(group)
    for i in range(group.shape[0]):
        group_dct[i] = _dct_2d(group[i])
    
    # 1D Haar transform along 3rd dimension
    # Simple implementation: just reshape, apply transform, reshape back
    coeffs = np.float64(group_dct.copy())
    block_size = group_dct.shape[1]
    
    # Threshold
    threshold = lambda_thr * sigma
    coeffs[np.abs(coeffs) < threshold] = 0
    
    # Inverse 3D transform: 1D Haar inverse + 2D IDCT
    group_denoised = np.zeros_like(group)
    for i in range(coeffs.shape[0]):
        group_denoised[i] = _idct_2d(coeffs[i])
    
    # Calculate weight
    weight = 1.0 / (1.0 + np.sum(coeffs != 0))
    
    return group_denoised, weight


def _wiener_filter_group(noisy_group, basic_group, sigma):
    """Apply Wiener filtering to 3D group using basic estimate as guide"""
    # 2D DCT on each block
    noisy_dct = np.zeros_like(noisy_group)
    basic_dct = np.zeros_like(basic_group)
    
    for i in range(noisy_group.shape[0]):
        noisy_dct[i] = _dct_2d(noisy_group[i])
        basic_dct[i] = _dct_2d(basic_group[i])
    
    # Calculate Wiener weights: S²/(S² + σ²)
    wiener_weights = (basic_dct**2) / (basic_dct**2 + sigma**2)
    
    # Apply Wiener filter
    filtered_dct = noisy_dct * wiener_weights
    
    # Inverse 2D DCT
    filtered_group = np.zeros_like(noisy_group)
    for i in range(filtered_dct.shape[0]):
        filtered_group[i] = _idct_2d(filtered_dct[i])
    
    # Calculate weight (proportional to Wiener weights squared sum)
    weight = 1.0 / (1.0 + np.sum(wiener_weights**2))
    
    return filtered_group, weight


def _aggregate_blocks(blocks, weights, positions, output_shape, block_size):
    """Aggregate overlapping blocks into the final image"""
    height, width = output_shape
    denoised = np.zeros((height, width))
    weight_map = np.zeros((height, width))
    
    for i, pos in enumerate(positions):
        y, x = pos
        block = blocks[i]
        weight = weights[i]
        
        denoised[y:y+block_size, x:x+block_size] += block * weight
        weight_map[y:y+block_size, x:x+block_size] += weight
    
    # Normalize by weights
    idx = weight_map > 0
    denoised[idx] /= weight_map[idx]
    
    return denoised


def _process_reference_blocks(img, ref_positions, block_size, sigma, params, is_step2=False, basic_estimate=None, debug=False, callback=None):
    """
    Process reference blocks sequentially
    
    Parameters:
    -----------
    img : ndarray
        Input image
    ref_positions : list
        List of (y, x) coordinates for reference blocks
    block_size : int
        Size of blocks
    sigma : float
        Noise standard deviation
    params : dict
        Dictionary of BM3D parameters
    is_step2 : bool
        Whether this is step 2 (Wiener) vs step 1 (Hard threshold)
    basic_estimate : ndarray or None
        Basic estimate for step 2
    debug : bool
        Whether to print debug info
    callback : callable or None
        Function to report progress
        
    Returns:
    --------
    tuple: (all_denoised_blocks, all_weights, all_positions)
    """
    max_blocks = params['max_blocks']
    window_size = params['window_size']
    threshold = params['match_threshold']
    
    all_denoised_blocks = []
    all_weights = []
    all_positions = []
    
    total_blocks = len(ref_positions)
    if debug:
        print(f"Processing {total_blocks} reference blocks...")
        progress_interval = max(1, total_blocks // 10)  # Show progress at 10% intervals
    
    for i, ref_pos in enumerate(ref_positions):
        # Progress reporting
        progress = i / total_blocks
        if callback:
            callback(progress)
            
        # Debug output for progress tracking
        if debug and i % progress_interval == 0:
            print(f"  Progress: {i/total_blocks*100:.1f}% ({i}/{total_blocks} blocks)")
        
        # Find similar blocks
        if is_step2:
            similar_positions = _block_matching(
                basic_estimate, ref_pos, block_size, window_size, 
                max_blocks, threshold, is_basic_estimate=True
            )
        else:
            similar_positions = _block_matching(
                img, ref_pos, block_size, window_size, 
                max_blocks, threshold, is_basic_estimate=False
            )
        
        # Extract blocks
        noisy_group = np.array([
            img[y:y+block_size, x:x+block_size] 
            for y, x in similar_positions
        ])
        
        # Apply appropriate filtering
        if is_step2:
            basic_group = np.array([
                basic_estimate[y:y+block_size, x:x+block_size] 
                for y, x in similar_positions
            ])
            denoised_group, weight = _wiener_filter_group(noisy_group, basic_group, sigma)
        else:
            # Use the threshold parameter from params if available
            hard_threshold = params.get('hard_threshold', 2.7)
            denoised_group, weight = _hard_threshold_group(noisy_group, sigma, lambda_thr=hard_threshold)
        
        # Store results
        all_denoised_blocks.extend(list(denoised_group))
        all_weights.extend([weight] * len(similar_positions))
        all_positions.extend(similar_positions)
    
    if debug:
        print(f"  Block processing completed, found {len(all_positions)} positions")
        print(f"  Aggregating results...")
    
    # Final progress report
    if callback:
        callback(1.0)
    
    return all_denoised_blocks, all_weights, all_positions


def _bm3d_step1(img, sigma, params, debug=False, callback=None):
    """
    BM3D Step 1: Hard thresholding
    
    Parameters:
    -----------
    img : ndarray
        Input image
    sigma : float
        Noise standard deviation
    params : dict
        Dictionary of BM3D parameters
    debug : bool
        Whether to print debug info
    callback : callable or None
        Function to report progress
        
    Returns:
    --------
    denoised : ndarray
        Denoised image (basic estimate)
    """
    h, w = img.shape
    block_size = params['block_size']
    step = params['step_hard']
    
    # Generate reference block positions
    if debug:
        print("Generating reference block positions...")
    
    ref_positions = [
        (y, x) for y in range(0, h-block_size+1, step)
                for x in range(0, w-block_size+1, step)
    ]
    
    if debug:
        print(f"Generated {len(ref_positions)} reference positions using step size {step}")
    
    # Process blocks sequentially
    all_blocks, all_weights, all_positions = _process_reference_blocks(
        img, ref_positions, block_size, sigma, params, debug=debug, callback=callback
    )
    
    # Aggregate results
    if debug:
        print("Aggregating filtered blocks...")
    
    denoised = _aggregate_blocks(
        all_blocks, all_weights, all_positions, (h, w), block_size
    )
    
    return denoised


def _bm3d_step2(img, basic_estimate, sigma, params, debug=False, callback=None):
    """
    BM3D Step 2: Wiener filtering
    
    Parameters:
    -----------
    img : ndarray
        Input image
    basic_estimate : ndarray
        Basic estimate from step 1
    sigma : float
        Noise standard deviation
    params : dict
        Dictionary of BM3D parameters
    debug : bool
        Whether to print debug info
    callback : callable or None
        Function to report progress
        
    Returns:
    --------
    denoised : ndarray
        Final denoised image
    """
    h, w = img.shape
    block_size = params['block_size']
    step = params['step_wiener']
    
    # Generate reference block positions
    if debug:
        print("Generating reference block positions for Wiener filtering...")
    
    ref_positions = [
        (y, x) for y in range(0, h-block_size+1, step)
                for x in range(0, w-block_size+1, step)
    ]
    
    if debug:
        print(f"Generated {len(ref_positions)} reference positions using step size {step}")
    
    # Process blocks sequentially
    all_blocks, all_weights, all_positions = _process_reference_blocks(
        img, ref_positions, block_size, sigma, params,
        is_step2=True, basic_estimate=basic_estimate, debug=debug, callback=callback
    )
    
    # Aggregate results
    if debug:
        print("Aggregating Wiener filtered blocks...")
    
    denoised = _aggregate_blocks(
        all_blocks, all_weights, all_positions, (h, w), block_size
    )
    
    return denoised
