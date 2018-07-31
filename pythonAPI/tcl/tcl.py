import numpy as np
import ctypes
from ctypes import cdll
import os
import random

TCL_ROOT = ""
try:
    TCL_ROOT = os.environ['TCL_ROOT']
except:
    print("[TCL] ERROR: TCL_ROOT environment variable is not set. Point TCL_ROOT to the folder which includes TCL_ROOT/lib/libtcl.so")
    exit(-1)

# load TCL library
lib = cdll.LoadLibrary(TCL_ROOT+"/lib/libtcl.so")

def randomNumaAwareInit( A ):
    """
    initializes the passed numpy.ndarray (which have to be created with
    numpy.empty) and initializes it with random data in paralle such that the
    pages are equally distributed among the numa nodes
    """
    lib.randomNumaAwareInit( ctypes.c_void_p(A.ctypes.data),
            ctypes.cast(A.ctypes.shape, ctypes.POINTER(ctypes.c_voidp)),
            ctypes.c_int32(A.ndim) )


def tensorMult( alpha, A, indicesA, B, indicesB, beta,  C, indicesC):
    """
        This function computes the tensor contraction of A and B, yielding C.
        The tensor contraction is of the form:
           C[indicesC] = alpha * A[indicesA] * B[indicesB] + beta * C[indicesC]

        where alpha and beta are scalors and A, B, and C correspond to arbitrary
        dimensional arrays (i.e., tensors). The dimensionality of A, B, and C
        depends on their indices (which need to be separated by commas).

        For instance, the tensor contraction C[m1,n1,m2] = 1.3 * A[k1,m1,k2,m2] * B[k1,k2,n1]
        would be represented as: tensorMult(1.3, A, "k1,m1,k2,m2", B,
        "k1,k2,n1", 0.0, C, "m1,n1,m2").
    """

    dataA = ctypes.c_void_p(A.ctypes.data)
    sizeA = ctypes.cast(A.ctypes.shape, ctypes.POINTER(ctypes.c_voidp))
    outerSizeA = sizeA
    dataB = ctypes.c_void_p(B.ctypes.data)
    sizeB = ctypes.cast(B.ctypes.shape, ctypes.POINTER(ctypes.c_voidp))
    outerSizeB = sizeB
    dataC = ctypes.c_void_p(C.ctypes.data)
    sizeC = ctypes.cast(C.ctypes.shape, ctypes.POINTER(ctypes.c_voidp))
    outerSizeC = sizeC
    indicesA = ctypes.c_char_p(indicesA.encode('utf-8'))
    indicesB = ctypes.c_char_p(indicesB.encode('utf-8'))
    indicesC = ctypes.c_char_p(indicesC.encode('utf-8'))
    useRowMajor = 0
    if( A.flags['C_CONTIGUOUS'] ):
        useRowMajor = 1

    if( A.itemsize == 4 ):
        lib.sTensorMult(ctypes.c_float(alpha), dataA, sizeA, outerSizeA, indicesA,
                                               dataB, sizeB, outerSizeB, indicesB,
                        ctypes.c_float(beta) , dataC, sizeC, outerSizeC, indicesC, useRowMajor)
    else:
        lib.dTensorMult(ctypes.c_double(alpha), dataA, sizeA, outerSizeA, indicesA,
                                                dataB, sizeB, outerSizeB, indicesB,
                        ctypes.c_double(beta) , dataC, sizeC, outerSizeC, indicesC, useRowMajor)

def equal(A, B, numSamples=-1):
    """ Ensures that alle elements of A and B are pretty much equal (due to limited machine precision)

    Parameter:
    numSamples: number of random samples to compare (-1: all). This values is used to approximate this function and speed the result up."
    """
    threshold = 1e-4
    A = np.reshape(A, A.size)
    B = np.reshape(B, B.size)
    error = 0
    samples = list(range(A.size))
    if( numSamples != -1 ):
        samples = random.sample(samples, min(A.size,numSamples))

    for i in samples:
      Aabs = abs(A[i]);
      Babs = abs(B[i]);
      absmax = max(Aabs, Babs);
      diff = Aabs - Babs;
      if( diff < 0 ):
          diff *= -1
      if(diff > 0):
         relError = diff / absmax;
         if(relError > 4e-5 and min(Aabs,Babs) > threshold ):
            error += 1
    return error == 0


def einsum(eq, A, B, out=None):
    """Two term tensor contraction using einsum-like api.

    Parameters
    ----------
    eq : str
        The einsum equation.
    A, B : array_like, len(shape) >= 1
        Tensors to "dot".
    """
    # parse the input equation into indices
    if "->" in eq:
        lhs, C_ix = eq.split('->')
        A_ix, B_ix = lhs.split(',')
    else:
        A_ix, B_ix = eq.split(',')
        idx_rm = [i for i in A_ix if i in B_ix]
        C_ix = "".join(i for i in sorted(A_ix + B_ix) if i not in idx_rm)

    # figure out the output dtype, upcast inputs if they differ in dtype
    if A.dtype == B.dtype:
        dtype = A.dtype
    else:
        dtype = np.common_type(A, B)
        A, B = A.astype(dtype), B.astype(dtype)

    # create out if it is not supplied
    if out is None:
        sz_dict = {i: (A.shape[A_ix.find(i)] if i in A_ix else
                       B.shape[B_ix.find(i)]) for i in A_ix + B_ix}
        out = np.empty([sz_dict[i] for i in C_ix], dtype=dtype)

    # add commas between indices for tcl format
    indA, indB, indC = (",".join(x) for x in (A_ix, B_ix, C_ix))

    # perform the contraction!
    tensorMult(1.0, A, indA, B, indB, 0.0, out, indC)
    return out


einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def tensordot(A, B, axes=2):
    """Simple translation of tensordot syntax to index notation.

    Parameters
    ----------
    A, B : array_like, len(shape) >= 1
        Tensors to "dot".
    axes : int or (2,) array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.
    """
    # convert int argument to (list[int], list[int])
    if isinstance(axes, int):
        axes = range(A.ndim - axes, A.ndim), range(axes)

    # convert (int, int) to (list[int], list[int])
    if isinstance(axes[0], int):
        axes = (axes[0],), axes[1]
    if isinstance(axes[1], int):
        axes = axes[0], (axes[1],)

    # initialize empty indices
    A_ix = [None] * A.ndim
    B_ix = [None] * B.ndim
    C_ix = []

    # fill in repeated indices
    available_ix = iter(einsum_symbols)
    for ax1, ax2 in zip(*axes):
        repeat = next(available_ix)
        A_ix[ax1] = repeat
        B_ix[ax2] = repeat

    # fill in the rest, and maintain output order
    for i in range(A.ndim):
        if A_ix[i] is None:
            leave = next(available_ix)
            A_ix[i] = leave
            C_ix.append(leave)
    for i in range(B.ndim):
        if B_ix[i] is None:
            leave = next(available_ix)
            B_ix[i] = leave
            C_ix.append(leave)

    # form full string and contract!
    einsum_str = "{},{}->{}".format(*map("".join, (A_ix, B_ix, C_ix)))
    return einsum(einsum_str, A, B)
