import os
import numpy as np
from typing import Tuple, Optional
from ml.input.types import BinaryIQConfig

def parse_iq_file(
    path: str,
    format_type: str,
    binary_config: Optional[BinaryIQConfig] = None
) -> Tuple[np.ndarray, int]:
    """
    Parses IQ data from NPY, NPZ, or raw binary format into a canonical [2, N] float32 array.
    
    Args:
        path (str): Path to the file.
        format_type (str): 'NPY', 'NPZ', or 'BIN'.
        binary_config (BinaryIQConfig, optional): Config for raw binary decoding.
        
    Returns:
        Tuple[np.ndarray, int]: Canonical float32 IQ array of shape [2, N], and the original sample count.
        
    Raises:
        ValueError: If layouts or configs are invalid, or if NaNs/Infs are present.
    """
    if format_type == "NPY":
        try:
            arr = np.load(path)
        except Exception as e:
            raise ValueError(f"Failed to load NPY file: {str(e)}")
            
    elif format_type == "NPZ":
        try:
            with np.load(path) as data:
                keys = list(data.keys())
                if "X" in keys:
                    arr = data["X"]
                elif len(keys) == 1:
                    arr = data[keys[0]]
                elif len(keys) > 1:
                    raise ValueError(
                        f"NPZ archive has multiple arrays {keys} and none is named 'X'. "
                        "Please provide an archive with a single array or containing key 'X'."
                    )
                else:
                    raise ValueError("NPZ file is empty (contains no arrays).")
        except Exception as e:
            raise ValueError(f"Failed to load NPZ file: {str(e)}")
            
    elif format_type == "BIN":
        if binary_config is None:
            raise ValueError(
                "Format config parameter (binary_config) is required to parse raw binary IQ "
                "(.bin, .dat) files. Silent guessing is disabled."
            )
            
        # Map string dtype to numpy dtype
        dt_str = binary_config.dtype.lower()
        if dt_str == "float32":
            dt = np.dtype(np.float32)
        elif dt_str == "float64":
            dt = np.dtype(np.float64)
        elif dt_str == "int16":
            dt = np.dtype(np.int16)
        elif dt_str == "int32":
            dt = np.dtype(np.int32)
        else:
            try:
                dt = np.dtype(dt_str)
            except Exception:
                raise ValueError(f"Unsupported binary dtype configuration: '{binary_config.dtype}'")
                
        # Set endianness byte order
        if binary_config.endianness == "big":
            dt = dt.newbyteorder(">")
        elif binary_config.endianness == "little":
            dt = dt.newbyteorder("<")
            
        try:
            arr = np.fromfile(path, dtype=dt)
        except Exception as e:
            raise ValueError(f"Failed to read raw binary IQ: {str(e)}")
            
        if len(arr) == 0:
            raise ValueError("Binary IQ file contains 0 samples.")
            
        # Parse based on interleaving layout config
        if binary_config.interleaved:
            # Layout: I0, Q0, I1, Q1 ...
            if len(arr) % 2 != 0:
                arr = arr[:-1]  # Truncate odd trailing sample
            arr = arr.reshape(-1, 2).T  # Shape [2, N]
        else:
            # Layout: I0, I1... then Q0, Q1...
            if len(arr) % 2 != 0:
                arr = arr[:-1]
            arr = arr.reshape(2, -1)  # Shape [2, N]
            
    else:
        raise ValueError(f"Unsupported format type: '{format_type}'")
        
    # Convert loaded arrays of different possible shapes to canonical [2, N]
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Loaded signal data is not a numpy array, got {type(arr)}")
        
    # Check shape layouts
    if np.issubdtype(arr.dtype, np.complexfloating):
        # Layout: Complex [N]
        if arr.ndim == 1:
            arr = np.vstack([arr.real, arr.imag])
        elif arr.ndim == 2:
            if arr.shape[0] == 1:
                arr = np.vstack([arr[0].real, arr[0].imag])
            elif arr.shape[1] == 1:
                flat = arr.squeeze()
                arr = np.vstack([flat.real, flat.imag])
            else:
                raise ValueError(f"Unsupported complex array shape: {arr.shape}. Expected 1D complex array.")
        else:
            raise ValueError(f"Unsupported complex array shape: {arr.shape}. Expected 1D complex array.")
    else:
        # Real array layouts
        if arr.ndim == 1:
            raise ValueError("1D real array layout is ambiguous (cannot distinguish I and Q).")
        elif arr.ndim == 2:
            if arr.shape[0] == 2:
                # Shape [2, N] is already canonical
                pass
            elif arr.shape[1] == 2:
                # Shape [N, 2] -> Transpose
                arr = arr.T
            else:
                raise ValueError(
                    f"Unsupported 2D real array layout shape {arr.shape}. "
                    "Expected shape [2, N] or [N, 2]."
                )
        elif arr.ndim == 3:
            if arr.shape[1] == 2:
                # Shape [N, 2, 128] -> Transpose and flatten to continuous [2, N*128]
                arr = arr.transpose(1, 0, 2).reshape(2, -1)
            else:
                raise ValueError(
                    f"Unsupported 3D real array layout shape {arr.shape}. "
                    "Expected shape [N, 2, L]."
                )
        else:
            raise ValueError(f"Unsupported multidimensional real array shape: {arr.shape}")
            
    # Normalize integers to float32 in range [-1.0, 1.0] dynamically
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        denom = max(abs(info.min), info.max)
        iq_data = arr.astype(np.float32) / float(denom)
    else:
        iq_data = arr.astype(np.float32)
        
    # Final NaN/Inf check
    if np.isnan(iq_data).any() or np.isinf(iq_data).any():
        raise ValueError("Parsed IQ data contains NaNs or infinite values.")
        
    n_samples = iq_data.shape[1]
    if n_samples == 0:
        raise ValueError("IQ file contains 0 samples.")
        
    return iq_data, n_samples
