#!/usr/bin/env python
"""TCL - Tensor Contraction Module based on the C++ tensor contraction library (TCL)"""

from .tcl import randomNumaAwareInit, tensorMult, equal, einsum, tensordot

__all__ = ["randomNumaAwareInit", "tensorMult", "equal", "einsum", "tensordot"]
