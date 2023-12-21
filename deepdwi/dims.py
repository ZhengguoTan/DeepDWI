"""
Generalized data dimensions for multi-purpose MRI image reconstruction.
"""

DIM_X    = -1  # readout
DIM_Y    = -2  # phase-encoding
DIM_Z    = -3  # slice
DIM_COIL = -4  # coil
DIM_ECHO = -5  # For diffusion MRI, it stores the shots per DWI
DIM_TIME = -6  # For diffusion MRI, it stores the diffusion encodings
DIM_REP  = -7  # repetition