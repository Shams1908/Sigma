import os
from typing import Set

SUPPORTED_EXTENSIONS: Set[str] = {".wav", ".npy", ".npz", ".bin", ".dat"}

def detect_file_format(path: str, max_size_bytes: int) -> str:
    """
    Deterministically detects the file format by checking the file extension
    and inspecting the file container header bytes.
    
    Args:
        path (str): Path to the target file.
        max_size_bytes (int): Configured maximum allowed file size.
        
    Returns:
        str: 'WAV', 'NPY', 'NPZ', or 'BIN'
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is too large, empty, has unsupported extension, or malformed header.
        IOError: If the file cannot be read.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: '{path}'")
    if not os.path.isfile(path):
        raise ValueError(f"Path is not a regular file: '{path}'")
        
    # Check file size
    file_size = os.path.getsize(path)
    if file_size == 0:
        raise ValueError("File is empty (0 bytes).")
    if file_size > max_size_bytes:
        raise ValueError(
            f"File size ({file_size} bytes) exceeds the configured maximum upload limit "
            f"of {max_size_bytes} bytes."
        )
        
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported extensions are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        
    # Read header bytes to verify format signature
    try:
        with open(path, "rb") as f:
            header = f.read(12)
    except Exception as e:
        raise IOError(f"Failed to read file header: {str(e)}")
        
    if ext == ".wav":
        # Check RIFF and WAVE signatures
        if len(header) < 12 or not header.startswith(b"RIFF") or header[8:12] != b"WAVE":
            raise ValueError("Malformed WAV file: Missing 'RIFF' or 'WAVE' header signature.")
        return "WAV"
        
    if ext == ".npy":
        # Check NumPy magic signature
        if len(header) < 6 or not header.startswith(b"\x93NUMPY"):
            raise ValueError("Malformed NPY file: Missing NumPy header signature.")
        return "NPY"
        
    if ext == ".npz":
        # Check ZIP magic signature (NPZ is a zip archive of NPYs)
        if len(header) < 4 or not header.startswith(b"PK\x03\x04"):
            raise ValueError("Malformed NPZ file: Missing ZIP header signature.")
        return "NPZ"
        
    if ext in {".bin", ".dat"}:
        return "BIN"
        
    # Fallback error
    raise ValueError(f"Internal detection error for file: '{path}'")
