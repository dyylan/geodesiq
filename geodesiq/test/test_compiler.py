import pytest

import numpy as np
from ..geodesiq.compiler import Compiler


def test_compiler_qasm_init():
    qasm = 'qreg q[2];\n\nh q[0];\ncx q[0],q[1];'
    comp = Compiler(qasm=qasm)
    assert comp.qasm == qasm