from numpy.ctypeslib import as_ctypes_type
import ctypes

# Must match the data types in in common.h
spin_dtype = ctypes.c_int8
double_ptr = ctypes.POINTER(ctypes.c_double)
spin_t_ptr = ctypes.POINTER(spin_dtype)

class flags(ctypes.Structure):
    _fields_ = [
        ("plt_thermodynamic_functions", ctypes.c_bool),
        ("plt_thermodynamic_functions_for_different_h", ctypes.c_bool),
        ("plt_domains", ctypes.c_bool),
        ("plt_conv_inf_iter", ctypes.c_bool),
        ("plt_conv_inf_lattice", ctypes.c_bool),
        ("anim_state_transitions", ctypes.c_bool)
    ]

class parameters(ctypes.Structure):
    _fields_ = [
        ("flag", flags),
        ("s_0", ctypes.c_double),
        ("t_anim", ctypes.c_double),
        ("n_frames", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("n_h_norm", ctypes.c_int32),
        ("n_t_domain", ctypes.c_int32),
        ("n_t", ctypes.c_int32),
        ("n_rows", ctypes.c_int32),
        ("n_cols", ctypes.c_int32),
        ("n_spins", ctypes.c_int32),
        ("n_pass", ctypes.c_int32),
        ("n_iter_0", ctypes.c_int64),
        ("n_n_iter_mult", ctypes.c_int32),
        ("n_lattice_size_mult", ctypes.c_int32)
    ]

class pointers(ctypes.Structure):
    _fields_ = [
        ("domain", spin_t_ptr),
        ("lattice_anim", spin_t_ptr),
        ("h_norm", double_ptr),
        ("t_domain", double_ptr),
        ("t", double_ptr),
        ("s_f", double_ptr),
        ("s_avg", double_ptr),
        ("u_f", double_ptr),
        ("n_iter_mult", double_ptr),
        ("lattice_size_mult", double_ptr),
        ("s_f_i", double_ptr),
        ("s_f_l", double_ptr)
    ]

# Import the C shared library
lib = ctypes.cdll.LoadLibrary("./libising.so")
start_simulation = lib.start_simulation
start_simulation.restype = ctypes.c_int
start_simulation.argtypes = [parameters, pointers]

# Get the pointer to the data of a numpy array
def get_pointer(nparray):
    return nparray.ctypes.data_as(ctypes.POINTER(as_ctypes_type(nparray.dtype)))
