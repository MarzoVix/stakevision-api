import numpy as np

if not hasattr(np, 'sctypes'):
    # Recreate np.sctypes for NumPy 2.0 compatibility.
    # imgaug (pulled in by PaddleOCR) references np.sctypes which was removed
    # in NumPy 2.0. Defining it here before imgaug is imported prevents the
    # AttributeError: np.sctypes was removed in the NumPy 2.0 release.
    np.sctypes = {
        'bool': [np.bool_],
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'object': [np.object_],
    }
