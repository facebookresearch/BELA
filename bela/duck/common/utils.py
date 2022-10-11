import torch

def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.

    Args:
        dtype: torch dtype of supertype float

    Returns:
        float: Tiny value

    Raises:
        TypeError: Given non-float or unknown type
    """

    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")

    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))
