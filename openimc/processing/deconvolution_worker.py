"""
Worker functions for high resolution IMC deconvolution using Richardson-Lucy deconvolution.
"""

import os
from typing import Tuple
import numpy as np
import tifffile

try:
    from skimage.restoration import richardson_lucy
    from skimage.util import img_as_uint
    _HAVE_SCIKIT_IMAGE = True
except ImportError:
    _HAVE_SCIKIT_IMAGE = False


def RLD_HRIMC_circle(
    image_stack: np.ndarray,
    x0: float = 7.0,
    iterations: int = 4,
    output_format: str = "float"
) -> np.ndarray:
    """
    Apply Richardson-Lucy deconvolution to high resolution IMC image stack.
    
    Args:
        image_stack: Image stack with shape (C, H, W) or (H, W, C)
        x0: Parameter for kernel calculation (default: 7.0)
        iterations: Number of Richardson-Lucy iterations (default: 4)
        output_format: Output format, either 'float' or 'uint16' (default: 'float')
    
    Returns:
        Deconvolved image stack with same shape as input
    """
    if not _HAVE_SCIKIT_IMAGE:
        raise RuntimeError("scikit-image is required for deconvolution. Install with: pip install scikit-image")
    
    # Validate input
    if image_stack.size == 0:
        raise ValueError("Input image stack is empty")
    
    # Ensure input is (C, H, W) format
    # Detection heuristic: For IMC data, channels are typically < 100, while H and W are typically > 1000
    # If first dimension is smallest and < 100, it's likely (C, H, W) - keep as is
    # If last dimension is smallest and < 100, it's likely (H, W, C) - transpose
    if image_stack.ndim == 3:
        dim0, dim1, dim2 = image_stack.shape
        
        # Check if last dimension is smallest and looks like channels (< 100)
        # This would indicate (H, W, C) format
        if dim2 < 100 and dim2 < dim0 and dim2 < dim1:
            # Likely (H, W, C) format, convert to (C, H, W)
            print(f"Detected (H, W, C) format, transposing from {image_stack.shape} to (C, H, W)")
            image_stack = np.transpose(image_stack, (2, 0, 1))
            dim0, dim1, dim2 = image_stack.shape  # Update after transpose
        # Otherwise assume (C, H, W) format (most common case, especially when called from deconvolve_acquisition)
        
        # Now we have (C, H, W)
        n_channels, height, width = dim0, dim1, dim2
        
        # Validate dimensions
        if height < 1 or width < 1:
            raise ValueError(f"Invalid image dimensions: {height}x{width}")
        if n_channels < 1:
            raise ValueError(f"Invalid number of channels: {n_channels}")
    elif image_stack.ndim == 2:
        # Single channel, add channel dimension
        height, width = image_stack.shape
        if height < 1 or width < 1:
            raise ValueError(f"Invalid image dimensions: {height}x{width}")
        image_stack = image_stack[np.newaxis, :, :]
        n_channels = 1
    else:
        raise ValueError(f"Unsupported image dimensionality: {image_stack.ndim}D")
    
    # Predefined arrays for kernel calculation
    Passes = np.array([7,6,5,8,7,
                       7,8,7,6,6,
                       7,9,8,7,8,
                       8,7,7,6,6,
                       7,6,6,5,5,
                       6,6,5,5,4,
                       4,6,4,5,3,
                       4,5,6,6,5,
                       4,5,5,4,3,
                       4,4,3,6,5,
                       4,5,5,4,4,
                       3,3,4,4,3,
                       3,2,4,3,2,
                       2,1,1,3,2,
                       2,1,1,3,3,
                       2,2,2,2,1,
                       1,1,1,3,2,
                       1,2,2,1,1,
                       2,1,1,4,3,
                       2,1])
    
    Contributions = np.array([0.02,0.00108,0.00108,0.0034,0.0196,
                           0.0196,0.0034,0.0034,0.0196,0.0196,
                           0.0034,0.00223,0.00223,0.00223,0.0034,
                           0.0034,0.0034,0.0034,0.0034,0.0034,
                           0.0196,0.00106,0.0196,0.00106,0.0196,
                           0.00108,0.00106,0.00106,0.00106,0.00106,
                           0.00108,0.0196,0.00106,0.0196,0.00106,
                           0.0196,0.0196,0.0034,0.0034,0.0196,
                           0.0196,0.0034,0.0034,0.0196,0.0196,
                           0.0034,0.0034,0.0196,0.00223,0.00223,
                           0.00223,0.0034,0.0034,0.0034,0.0034,
                           0.0034,0.0034,0.0196,0.00108,0.0196,
                           0.00106,0.0196,0.00108,0.00106,0.00106,
                           0.00106,0.00106,0.00108,0.0196,0.00106,
                           0.0196,0.00106,0.0196,0.0034,0.0034,
                           0.0196,0.0196,0.0034,0.0034,0.0196,
                           0.0196,0.0034,0.0034,0.00223,0.00223,
                           0.00223,0.0034,0.0034,0.0034,0.0034,
                           0.00108,0.00196,0.00108,0.00219,0.00219,
                           0.00219,0.00219])
    
    Contributions = Contributions / Contributions.sum()
    
    # Process each channel
    processed_layers = []
    
    print(f"Processing {n_channels} channels, input shape: {image_stack.shape}")
    
    for layer_idx in range(n_channels):
        layer_data = image_stack[layer_idx, :, :]
        
        # Validate layer data
        if layer_data.size == 0:
            raise ValueError(f"Layer {layer_idx} is empty")
        
        if layer_idx == 0:
            print(f"Layer {layer_idx} shape: {layer_data.shape}, expected (H={height}, W={width})")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(layer_data)) or np.any(np.isinf(layer_data)):
            print(f"Warning: Layer {layer_idx} contains NaN or Inf values, replacing with 0")
            layer_data = np.nan_to_num(layer_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate kernel for this channel
        I0 = float(np.max(layer_data))
        
        # Check if I0 is valid
        if I0 <= 0 or np.isnan(I0) or np.isinf(I0):
            print(f"Warning: Layer {layer_idx} has invalid I0={I0}, using default kernel")
            # Use a default kernel if I0 is invalid
            kernel = np.array([[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]])
        else:
            Passes_scaled = I0 - I0 / (1 + np.exp(-(Passes - x0)))
            y_array = Passes_scaled * Contributions
            total_sum = np.sum(y_array)
            
            if total_sum <= 0 or np.isnan(total_sum) or np.isinf(total_sum):
                print(f"Warning: Layer {layer_idx} has invalid kernel sum={total_sum}, using default kernel")
                kernel = np.array([[0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0]])
            else:
                result = list((
                    (y_array[3] + y_array[4] + y_array[11] + y_array[14] + y_array[15] + y_array[20]) / total_sum, 
                    (y_array[0] + y_array[5] + y_array[6] + y_array[7] + y_array[8] + y_array[12] + y_array[16] + y_array[17] + y_array[22]) / total_sum, 
                    (y_array[9] + y_array[10] + y_array[13] + y_array[18] + y_array[19] + y_array[24]) / total_sum,
                    (y_array[31] + y_array[36] + y_array[37] + y_array[38] + y_array[39] + y_array[48] + y_array[51] + y_array[52] + y_array[57]) / total_sum, 
                    (y_array[33] + y_array[40] + y_array[41] + y_array[42] + y_array[43] + y_array[49] + y_array[53] + y_array[54] + y_array[59]) / total_sum, 
                    (y_array[35] + y_array[44] + y_array[45] + y_array[46] + y_array[47] + y_array[50] + y_array[55] + y_array[56] + y_array[61]) / total_sum, 
                    (y_array[68] + y_array[73] + y_array[74] + y_array[75] + y_array[83] + y_array[86]) / total_sum, 
                    (y_array[70] + y_array[76] + y_array[77] + y_array[78] + y_array[79] + y_array[84] + y_array[87] + y_array[88] + y_array[91]) / total_sum, 
                    (y_array[72] + y_array[80] + y_array[81] + y_array[82] + y_array[85] + y_array[89]) / total_sum, 
                ))
                
                kernel = np.array(result / np.sum(result))
                kernel = kernel.reshape(3, 3)
        
        # Normalize the data
        layer_min = layer_data.min()
        layer_max = layer_data.max()
        if layer_max > layer_min:
            layer_data = (layer_data - layer_min) / (layer_max - layer_min)
        else:
            layer_data = layer_data.astype(np.float32)
        
        layer_data_denoise = layer_data
        
        # Clip values to remove 0s (handled badly by RLD)
        layer_data_denoise = np.clip(layer_data_denoise, 1e-4, None)
        
        # Richardson-Lucy deconvolution
        deconvolved_image = richardson_lucy(layer_data_denoise, kernel, num_iter=iterations)
        
        # Denormalize
        if layer_max > layer_min:
            deconvolved_image = deconvolved_image * (layer_max - layer_min) + layer_min
        
        # Verify deconvolved image shape
        if deconvolved_image.shape != (height, width):
            print(f"Warning: Layer {layer_idx} deconvolved shape {deconvolved_image.shape} != expected ({height}, {width})")
        
        processed_layers.append(deconvolved_image)
        
        if layer_idx == 0:
            print(f"Layer {layer_idx} after deconvolution: shape={deconvolved_image.shape}")
    
    # Stack all processed layers
    if len(processed_layers) != n_channels:
        raise ValueError(f"Processed layers count mismatch: expected {n_channels}, got {len(processed_layers)}")
    
    # Verify all layers have the same shape
    if processed_layers:
        expected_layer_shape = processed_layers[0].shape
        for i, layer in enumerate(processed_layers):
            if layer.shape != expected_layer_shape:
                raise ValueError(f"Layer {i} has shape {layer.shape}, expected {expected_layer_shape}")
    
    processed_stack = np.stack(processed_layers, axis=0)  # (C, H, W)
    
    # Debug: verify stack shape
    print(f"After stacking: shape={processed_stack.shape}, expected (C={n_channels}, H={height}, W={width})")
    
    # Verify the stack shape is correct
    if processed_stack.shape[0] != n_channels:
        raise ValueError(f"Stack channel dimension mismatch: expected {n_channels}, got {processed_stack.shape[0]}")
    if processed_stack.shape[1] != height or processed_stack.shape[2] != width:
        print(f"Warning: Stack spatial dimensions mismatch: expected (H={height}, W={width}), got (H={processed_stack.shape[1]}, W={processed_stack.shape[2]})")
    
    # Remove a 2 pixel (1um) border to account for border effect
    # Check if image is large enough for cropping
    n_channels_out, height_out, width_out = processed_stack.shape
    if n_channels_out != n_channels:
        raise ValueError(f"Channel count mismatch: expected {n_channels}, got {n_channels_out}")
    
    if height_out > 6 and width_out > 6:
        processed_stack_cropped = processed_stack[:, 3:-3, 3:-3]
        expected_cropped_h = height_out - 6
        expected_cropped_w = width_out - 6
        print(f"After cropping (3px): shape={processed_stack_cropped.shape}, expected (C={n_channels}, H={expected_cropped_h}, W={expected_cropped_w})")
    else:
        # Image too small, crop less or don't crop
        if height_out > 4 and width_out > 4:
            # Crop 1 pixel instead of 3
            processed_stack_cropped = processed_stack[:, 1:-1, 1:-1]
            expected_cropped_h = height_out - 2
            expected_cropped_w = width_out - 2
            print(f"After cropping (1px): shape={processed_stack_cropped.shape}, expected (C={n_channels}, H={expected_cropped_h}, W={expected_cropped_w})")
        else:
            # Don't crop if too small
            processed_stack_cropped = processed_stack
            print(f"No cropping applied: shape={processed_stack_cropped.shape}")
    
    # Final verification before return
    final_c, final_h, final_w = processed_stack_cropped.shape
    print(f"Final output shape: (C={final_c}, H={final_h}, W={final_w})")
    if final_c != n_channels:
        raise ValueError(f"Final channel count mismatch: expected {n_channels}, got {final_c}")
    
    # Convert format if needed
    if output_format == "uint16":
        # Convert to uint16, scaling to full uint16 range
        # Find min and max across all channels
        stack_min = float(np.min(processed_stack_cropped))
        stack_max = float(np.max(processed_stack_cropped))
        
        if stack_max > stack_min:
            # Scale to [0, 65535] range for uint16
            # First normalize to [0, 1]
            normalized = (processed_stack_cropped - stack_min) / (stack_max - stack_min)
            # Then scale to uint16 range
            processed_stack_cropped = (normalized * 65535.0).astype(np.uint16)
        elif stack_max == stack_min and stack_max >= 0:
            # All values are the same and non-negative
            # Scale to uint16 range if value is in [0, 1], otherwise clip
            if stack_max <= 1.0:
                processed_stack_cropped = (processed_stack_cropped * 65535.0).astype(np.uint16)
            else:
                # Clip to uint16 max
                processed_stack_cropped = np.clip(processed_stack_cropped, 0, 65535).astype(np.uint16)
        else:
            # All values are the same and possibly negative
            # Set to zero
            processed_stack_cropped = np.zeros_like(processed_stack_cropped, dtype=np.uint16)
    else:
        # Keep as float32
        processed_stack_cropped = processed_stack_cropped.astype(np.float32)
    
    return processed_stack_cropped


def deconvolve_acquisition(
    mcd_path: str,
    acq_id: str,
    output_dir: str,
    x0: float = 7.0,
    iterations: int = 4,
    output_format: str = "float",
    channel_names: list = None
) -> str:
    """
    Deconvolve a single acquisition from an MCD file and save as OME-TIFF.
    
    Args:
        mcd_path: Path to the MCD file
        acq_id: Acquisition ID
        output_dir: Output directory for OME-TIFF files
        x0: Parameter for kernel calculation
        iterations: Number of Richardson-Lucy iterations
        output_format: Output format, either 'float' or 'uint16'
        channel_names: List of channel names for OME metadata
    
    Returns:
        Path to the saved OME-TIFF file
    """
    from openimc.data.mcd_loader import MCDLoader
    
    # Load the acquisition
    loader = MCDLoader()
    loader.open(mcd_path)
    
    try:
        # Get all channels for this acquisition
        img_stack = loader.get_all_channels(acq_id)  # Returns (H, W, C)
        
        # Verify image stack is valid
        if img_stack.size == 0:
            raise ValueError(f"Image stack is empty for acquisition {acq_id}")
        
        if img_stack.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, C), got {img_stack.ndim}D array with shape {img_stack.shape}")
        
        # Check image dimensions
        height, width, n_channels = img_stack.shape
        if height < 1 or width < 1:
            raise ValueError(f"Invalid image dimensions: {height}x{width}")
        if n_channels < 1:
            raise ValueError(f"No channels found in acquisition {acq_id}")
        
        # Convert to (C, H, W) for processing
        img_stack = np.transpose(img_stack, (2, 0, 1))
        
        # Apply deconvolution
        print(f"Deconvolving acquisition {acq_id}: shape={img_stack.shape}, x0={x0}, iterations={iterations}, format={output_format}")
        deconvolved_stack = RLD_HRIMC_circle(
            img_stack,
            x0=x0,
            iterations=iterations,
            output_format=output_format
        )  # Returns (C, H, W)
        print(f"Deconvolution complete: output shape={deconvolved_stack.shape}, dtype={deconvolved_stack.dtype}")
        
        # Get acquisition info for filename
        acq_info = loader.list_acquisitions()
        acq = next((a for a in acq_info if a.id == acq_id), None)
        if acq:
            acq_name = acq.name
        else:
            acq_name = f"acquisition_{acq_id}"
        
        # Sanitize filename
        safe_name = "".join(c for c in acq_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename
        output_filename = f"{safe_name}.ome.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        # Check if deconvolved stack is valid
        if deconvolved_stack.size == 0:
            raise ValueError(f"Deconvolved stack is empty for acquisition {acq_id}")
        
        # Verify deconvolved stack shape (should be C, H, W)
        print(f"Before saving: shape={deconvolved_stack.shape}, expected (C, H, W)")
        if deconvolved_stack.ndim != 3:
            raise ValueError(f"Expected 3D array (C, H, W), got {deconvolved_stack.ndim}D array with shape {deconvolved_stack.shape}")
        
        # Verify channel count matches
        expected_channels = len(channel_names) if channel_names else img_stack.shape[2]
        if deconvolved_stack.shape[0] != expected_channels:
            raise ValueError(f"Channel count mismatch: expected {expected_channels}, got {deconvolved_stack.shape[0]}")
        
        # Save as OME-TIFF in CHW format (matches GUI export format)
        # tifffile.imwrite with ome=True can handle (C, H, W) format
        # The GUI export uses this format, so we'll match it for consistency
        metadata = {}
        if channel_names:
            metadata['Channel'] = {'Name': channel_names}
        
        # Save as OME-TIFF in CHW format (same as GUI export)
        try:
            tifffile.imwrite(
                output_path,
                deconvolved_stack,  # Already in (C, H, W) format
                photometric='minisblack',
                metadata=metadata,
                ome=True
            )
            
            # Verify the file was written correctly
            if not os.path.exists(output_path):
                raise IOError(f"Output file was not created: {output_path}")
            
            # Check file size
            file_size = os.path.getsize(output_path)
            if file_size == 0:
                raise IOError(f"Output file is empty: {output_path}")
            
            # Try to read it back to verify
            with tifffile.TiffFile(output_path) as tif:
                if not tif.series:
                    raise IOError(f"TIFF file contains no image series: {output_path}")
                read_shape = tif.series[0].shape
                # tifffile may return shape in different order, so we check if dimensions match
                if set(read_shape) != set(deconvolved_stack.shape):
                    print(f"Warning: Written shape {deconvolved_stack.shape} != read shape {read_shape}")
                else:
                    print(f"File verified: written shape {deconvolved_stack.shape}, read shape {read_shape}")
            
        except Exception as e:
            # Clean up partial file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise IOError(f"Failed to write OME-TIFF file {output_path}: {str(e)}") from e
        
        return output_path
        
    finally:
        loader.close()

