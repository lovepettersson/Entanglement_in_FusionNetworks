import numpy as np
from timeit import default_timer

import ctypes
import os
import platform ## Used in the c++ wrappers testing on which operating system we are working on

cwd = os.getcwd()

def row_echelon_inZ2_modforlossdecoder(m):
    """ Return Reduced Row Echelon Form of matrix A in the field F_2={0,1}.
    Based on the pseudocode from https://en.wikipedia.org/wiki/Row_echelon_form.
    Returns REF_m, the Reduced Row Echelon Form of the input m, and the transformation matrix inverse_mat such that
    inverse_mat @ m = REF_m  (mod 2), which, if m is invertible (i.e. REF_m is Id), represents the left-inverse of m.

    :param m: The matrix to be put in Reduced Row Echelon Form
    :type m: :class:`np.matrix`
    """

    nr, nc = m.shape
    # inverse_mat = np.eye(nr, dtype = np.uint8)

    REF_m = m.copy()

    lead = 0
    # for r in range(nr):
    for r in range(nr-1): # don't do anything to the last row as that's the logical operator
        if nc <= lead:
            # return REF_m, inverse_mat
            return REF_m
        i = r
        while REF_m[i, lead] == 0:
            i = i+1
            if nr == i:
                i = r
                lead = lead + 1
                if nc == lead:
                    # return REF_m, inverse_mat
                    return REF_m
        # if i != r:
        if i != r and i != (nr-1): # don't swap the last row as that's the logical operator
            # swap rows i and r of the matrix (REF_m), and the i and r rows of transf_mat
            # print('Swapping row', r, ' and row ', i)
            REF_m[[i, r]] = REF_m[[r, i]]
            # inverse_mat[[i, r]] = inverse_mat[[r, i]]

        # for i in range(nr):
        for i in range(r, nr):
            if i != r:
            # if i != r and i != nr: # don't add if it is lo
                # subtract row r to row i of the matrix (REF_m), and the i and r rows of transf_mat
                if REF_m[i, lead] != 0:
                    # print('Adding row', r, ' to row ', i, 'lead:', lead)
                    REF_m[i] = (REF_m[i] - REF_m[r]) % 2
                    # print(REF_m)
                    # inverse_mat[i] = (inverse_mat[i] - inverse_mat[r]) % 2
        lead = lead + 1
    # return REF_m, inverse_mat
    return REF_m



#########################################################################################################
###### Functions to make Gaussian elimination, sped-up version targeting loss decoding


def loss_decoding_gausselim(m, num_lost_qbts):
    # start_func_time = default_timer()
    nr, nc = m.shape

    # start_t = default_timer()
    REF_m = m.copy()
    # end_t = default_timer()
    # print('Time spent copying:', end_t - start_t)

    lead = 0

    time_switch = 0
    time_sum = 0
    time_search = 0

    # for r in range(nr):
    for r in range(nr-1): # don't do anything to the last row as that's the logical operator
        if num_lost_qbts <= lead:
            # end_func_time = default_timer()
            # print('Time spent in func:', end_func_time - start_func_time)
            # print('Time spent switchs:', time_switch)
            # print('Time spent summing:', time_sum)
            # print('Time spent search:', time_search)
            return REF_m
        i = r

        # start_t = default_timer()
        while REF_m[i, lead] == 0:

            i = i+1
            if i == (nr-1): # last row is logical operator
                i = r
                lead = lead + 1
                if num_lost_qbts == lead:
                    # end_func_time = default_timer()
                    # print('Time spent in func:', end_func_time - start_func_time)
                    # print('Time spent switchs:', time_switch)
                    # print('Time spent summing:', time_sum)
                    # print('Time spent search:', time_search)
                    return REF_m
        # end_t = default_timer()
        # time_search += end_t - start_t

        # if i != r:
        if i != r and i != (nr-1): # don't swap the last row as that's the logical operator
            # swap rows i and r of the matrix (REF_m), and the i and r rows of transf_mat

            # start_t = default_timer()

            REF_m[[i, r]] = REF_m[[r, i]]

            # end_t = default_timer()
            # time_switch += end_t - start_t


        # for i in range(nr):
        for i in range(r, nr):
            if i != r:
                # subtract row r to row i of the matrix (REF_m), and the i and r rows of transf_mat
                if REF_m[i, lead] != 0:

                    # start_t = default_timer()

                    REF_m[i] = (REF_m[i] - REF_m[r]) % 2

                    # end_t = default_timer()
                    # time_sum += end_t - start_t

        lead = lead + 1
    # end_func_time = default_timer()
    # print('Time spent in func:', end_func_time - start_func_time)
    # print('Time spent switchs:', time_switch)
    # print('Time spent summing:', time_sum)
    # print('Time spent search:', time_search)
    return REF_m



def loss_decoding_gausselim_print(m, num_lost_qbts):
    # start_func_time = default_timer()
    nr, nc = m.shape

    # start_t = default_timer()
    REF_m = m.copy()
    # end_t = default_timer()
    # print('Time spent copying:', end_t - start_t)

    lead = 0

    time_switch = 0
    time_sum = 0
    time_search = 0

    # for r in range(nr):
    for r in range(nr-1): # don't do anything to the last row as that's the logical operator
        print("Starting row " , r)
        print(REF_m)
        if num_lost_qbts <= lead:
            # end_func_time = default_timer()
            # print('Time spent in func:', end_func_time - start_func_time)
            # print('Time spent switchs:', time_switch)
            # print('Time spent summing:', time_sum)
            # print('Time spent search:', time_search)
            return REF_m
        i = r

        # start_t = default_timer()
        while REF_m[i, lead] == 0:

            i = i+1
            if i == (nr-1): # last row is logical operator
                i = r
                lead = lead + 1
                if num_lost_qbts == lead:
                    # end_func_time = default_timer()
                    # print('Time spent in func:', end_func_time - start_func_time)
                    # print('Time spent switchs:', time_switch)
                    # print('Time spent summing:', time_sum)
                    # print('Time spent search:', time_search)
                    return REF_m
        # end_t = default_timer()
        # time_search += end_t - start_t

        print("Found lead in row ", i, "  col ", lead)

        # if i != r:
        if i != r and i != (nr-1): # don't swap the last row as that's the logical operator
            # swap rows i and r of the matrix (REF_m), and the i and r rows of transf_mat

            # start_t = default_timer()

            print("Switching rows", r, "and", i)
            print(REF_m)

            REF_m[[i, r]] = REF_m[[r, i]]

            # end_t = default_timer()
            # time_switch += end_t - start_t


        # for i in range(nr):
        for i in range(r, nr):
            if i != r:
                # subtract row r to row i of the matrix (REF_m), and the i and r rows of transf_mat
                if REF_m[i, lead] != 0:

                    # start_t = default_timer()
                    print("Summing row", i, " to row ", r)
                    REF_m[i] = (REF_m[i] - REF_m[r]) % 2

                    # end_t = default_timer()
                    # time_sum += end_t - start_t

        lead = lead + 1
    # end_func_time = default_timer()
    # print('Time spent in func:', end_func_time - start_func_time)
    # print('Time spent switchs:', time_switch)
    # print('Time spent summing:', time_sum)
    # print('Time spent search:', time_search)
    return REF_m


#########################################################################################################
###### Functions to make Gaussian elimination  sped-up version targeting loss decoding, fast binary operations in Cpp
###### Works well on Linux
# cpp_libname = "libLossDec.so" ### The cpp library needs to be in the same folder as the Python file using it.
# lib_file_path = os.path.join(cwd, cpp_libname)


os_system = platform.system()
if os_system == 'Windows':
    LTcpp_header = ctypes.cdll.LoadLibrary('./libLossDec_win.dll')
    print('Loaded C++ linear algebra functions for Windows OS')
elif os_system == 'Linux':
    LTcpp_header = ctypes.cdll.LoadLibrary('./libLossDec.so')
    print('Loaded C++ linear algebra functions for Linux OS')
else:
    raise ValueError('Os system not supported: only Windows or Linux')
LTcpp_header.LossDecoder_GaussElimin.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.c_int, ctypes.c_int, ctypes.c_int]
LTcpp_header.LossDecoder_GaussElimin_print.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.c_int, ctypes.c_int, ctypes.c_int]
# LTcpp_header.LossDecoder_GaussElimin_trackqbts.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int]

def loss_decoding_gausselim_fast(m, num_lost_qbts, print = False):
    if m.dtype != np.uint8:
        raise ValueError("The c++ function works only for binary matrices with numpy.uint8 datatype entries.")
    nr, nc = m.shape
    # start_t = default_timer()

    REF_m = m.copy()
    if print:
        LTcpp_header.LossDecoder_GaussElimin_print(REF_m.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                             nr, nc, num_lost_qbts)
    else:
        LTcpp_header.LossDecoder_GaussElimin(REF_m.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                             nr, nc, num_lost_qbts)
    return REF_m

# def loss_decoding_gausselim_fast_trackqbts(m, qbt_syndr_mat, num_lost_qbts):
#     if m.dtype != np.uint8:
#         raise ValueError("The c++ function works only for binary matrices with numpy.uint8 datatype entries.")
#     nr, nc = m.shape
#     # start_t = default_timer()
#
#     REF_m = m.copy()
#     REF_qbt_syndr_mat = qbt_syndr_mat.copy()
#     LTcpp_header.LossDecoder_GaussElimin_trackqbts(REF_m.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
#                                                    REF_qbt_syndr_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
#                                                    nr, nc, num_lost_qbts)
#     return REF_m, REF_qbt_syndr_mat



#########################################################################################################
###### Functions to make Gaussian elimination in binary use operations
def binary_matrix_to_intrepr_array(a, num_bits=8):
    if num_bits == 8:
        my_dtype = np.uint8
    elif num_bits == 16:
        my_dtype = np.uint16
    elif num_bits == 32:
        my_dtype = np.uint32
    elif num_bits == 64:
        my_dtype = np.uint64
    else:
        raise ValueError('num_bits can only be 8, 16, 32, 64')

    over_cols = a.shape[1] % num_bits
    if over_cols > 0:
        cols_to_stack = num_bits - over_cols
    else:
        cols_to_stack = 0
    a_shaped = np.hstack((np.zeros((a.shape[0], cols_to_stack), dtype=my_dtype), a))
    a_shaped = a_shaped.reshape((a_shaped.shape[0], int(a_shaped.shape[1] / num_bits), num_bits))
    return a_shaped.dot(1 << np.arange(a_shaped.shape[-1] - 1, -1, -1, dtype=my_dtype)), cols_to_stack


def row_echelon_inZ2_modforlossdecoder_binaryfast(m, num_bits = 8):
    # print(m)
    start_t = default_timer()
    REF_m, stacked_cols = binary_matrix_to_intrepr_array(m, num_bits=num_bits)
    end_t = default_timer()
    print('time for binarizing:', end_t - start_t)

    nr, nc = REF_m.shape
    nc_bits = nc * num_bits

    bit_lead = stacked_cols
    col_lead = 0
    for r in range(nr-1): # don't do anything to the last row as that's the logical operator
        if nc <= col_lead:
            return REF_m
        i = r
        while (REF_m[i, col_lead] >> (num_bits - bit_lead - 1)) == 0:   ### checks if the bit_lead-th bit (from left) of the number in the col_lead-th columns is 0
            i = i+1
            if nr == i:
                i = r

                bit_lead += 1
                if bit_lead == num_bits:
                    bit_lead = 0
                    col_lead += 1

                if nc == col_lead:
                    return REF_m
        # if i != r:
        if i != r and i != (nr-1): # don't swap the last row as that's the logical operator
            # swap rows i and r of the matrix (REF_m), and the i and r rows of transf_mat
            # print('Swapping row', r, ' and row ', i)
            REF_m[[i, r]] = REF_m[[r, i]]

        # for i in range(nr): ### TODO: MAKE THIS START FROM r+1!!!!!
        for i in range(r, nr): ### TODO: MAKE THIS START FROM r+1!!!!!
            if i != r:
            # if i != r and i != nr: # don't add if it is lo
                # subtract row r to row i of the matrix (REF_m), and the i and r rows of transf_mat
                if ((REF_m[i, col_lead] >> (num_bits - bit_lead - 1))%2)!=0:
                    # print('Adding row', r, ' to row ', i, 'lead:', col_lead*num_bits + bit_lead - stacked_cols, np.binary_repr((REF_m[i, col_lead] >> (num_bits - bit_lead - 1)), width = 8))
                    REF_m[i] = np.bitwise_xor(REF_m[i], REF_m[r])
                    # print(np.unpackbits(REF_m.view(np.uint8), axis=1)[:,stacked_cols:])
        bit_lead += 1
        if bit_lead == num_bits:
            bit_lead = 0
            col_lead += 1
    return REF_m


