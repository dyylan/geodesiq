# import pytest

# import numpy as np
# import jax.numpy as jnp
# from unittest.mock import MagicMock, patch

# from ..src.optimizer import Optimizer
# from ..src.lie import Basis

# @pytest.fixture
# def mock_basis():

        
#     return MockBasis()

# @pytest.fixture
# def target_unitary():
#     return np.array([[1, 0, 0, 0, 0, 0, 0, 0],
#                      [0, 1, 0, 0, 0, 0, 0, 0],
#                      [0, 0, 1, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 1, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 1, 0, 0, 0],
#                      [0, 0, 0, 0, 0, 1, 0, 0],
#                      [0, 0, 0, 0, 0, 0, 0, 1],
#                      [0, 0, 0, 0, 0, 0, 1, 0]])

# @pytest.fixture
# def init_parameters():
#     return np.array([1.96348585e+00, 
#                      -6.60051338e-04, 
#                      -9.95668892e-04,  
#                      1.25309178e-05,
#                      1.56731411e-05, 
#                      -4.64297134e-05,  
#                      4.51157029e-05, 
#                      -4.33468559e-05,
#                      -1.39946480e-04, 
#                      -4.51157029e-05, 
#                      -4.64297134e-05,
#                      3.92685075e-01,
#                      1.17806509e+00, 
#                      -6.96038082e-04, 
#                      -1.06093922e-03, 
#                      -1.36027719e+00,
#                      2.27928897e-05,  
#                      8.64621387e-06,  
#                      9.49983567e-05, 
#                      -6.07563170e-06,
#                      0.00000000e+00,  
#                      0.00000000e+00,  
#                      0.00000000e+00,  
#                      1.22445556e-05,
#                      0.00000000e+00,  
#                      0.00000000e+00,  
#                      0.00000000e+00, 
#                      -1.36029998e+00,
#                      0.00000000e+00,  
#                      0.00000000e+00,  
#                      0.00000000e+00,  
#                      6.80977949e-01,
#                      -9.88089022e-06, 
#                      -9.49983567e-05,  
#                      8.64621387e-06, 
#                      -1.22445556e-05,
#                      0.00000000e+00,  
#                      0.00000000e+00,
#                      0.00000000e+00,
#                      -6.07563170e-06,
#                      0.00000000e+00,
#                      0.00000000e+00,
#                      0.00000000e+00,
#                      6.80987829e-01,
#                      0.00000000e+00, 
#                      0.00000000e+00,
#                      0.00000000e+00,
#                      3.92889813e-01,
#                      -7.85409183e-01,  
#                      3.59867439e-05,
#                      6.52703271e-05, 
#                      -3.14222334e-06,
#                      0.00000000e+00,  
#                      0.00000000e+00,
#                      0.00000000e+00,
#                      9.65996245e-05,
#                      0.00000000e+00,  
#                      0.00000000e+00,
#                      0.00000000e+00,
#                      -3.92498358e-01,
#                      0.00000000e+00,  
#                      0.00000000e+00,
#                      0.00000000e+00])

# @pytest.fixture
# def max_steps():
#     return 10

# @pytest.fixture
# def precision():
#     return 0.999

# @pytest.fixture
# def max_step_size():
#     return 2

# @pytest.fixture
# def optimizer(target_unitary, mock_basis, init_parameters, max_steps, precision, max_step_size):
#     with patch('your_module.commuting_ansatz', return_value=(np.array([True, False, True]), np.identity(3))), \
#          patch('your_module.prepare_random_parameters', return_value=np.array([0.1, 0.2, 0.3])): 
#         return Optimizer(target_unitary, mock_basis, mock_basis, init_parameters, max_steps, precision, max_step_size)

# def test_optimizer_init(optimizer, target_unitary):
#     assert np.array_equal(optimizer.target_unitary, target_unitary)
#     assert optimizer.max_steps == 10
#     assert optimizer.precision == 0.999
#     assert optimizer.max_step_size == 2

# def test_optimizer_optimize(optimizer):
#     with patch.object(optimizer, 'update_step', return_value=(MagicMock(), 0.999, 0.1)):
#         optimizer.is_succesful = optimizer.optimize()
#         assert optimizer.is_succesful

# def test_optimizer_update_step(optimizer):
#     with patch('your_module.Hamiltonian') as mock_hamiltonian, \
#          patch.object(optimizer, '_new_phi_golden_section_search', return_value=(0.9, 0.95, MagicMock(), 0.1)):
#         mock_hamiltonian.return_value.fidelity.return_value = 0.9
#         new_phi_ham, fidelity_new_phi, step_size = optimizer.update_step()
#         assert fidelity_new_phi == 0.95
#         assert step_size == 0.1

# def test_optimizer_static_methods(mock_basis, target_unitary):
#     params = np.array([0.1, 0.2])
#     compute_matrix_fn = Optimizer.get_compute_matrix_fn(np.identity(2), mock_basis.basis)
#     project_omegas_fn = Optimizer.get_project_omegas_fn(mock_basis.basis, mock_basis.dim)
#     Udagger_dU_contraction_fn = Optimizer.get_Udagger_dU_contraction_fn()
#     fidelity_fn = Optimizer.get_fidelity_fn(mock_basis.basis, target_unitary)

#     # Test compute_matrix_fn
#     matrix = compute_matrix_fn(params)
#     assert matrix.shape == (2, 2)

#     # Test project_omegas_fn
#     omegas = project_omegas_fn(jnp.array([[[0.1, 0.2], [0.3, 0.4]]]))
#     assert omegas.shape == (1, 2)

#     # Test Udagger_dU_contraction_fn
#     contraction = Udagger_dU_contraction_fn(jnp.eye(2), jnp.array([[[0.1, 0.2], [0.3, 0.4]]]))
#     assert contraction.shape == (1, 2, 2)

#     # Test fidelity_fn
#     fidelity = fidelity_fn(0.1, params, np.array([0.1, 0.2]))
#     assert np.isclose(fidelity, 1.0, atol=1e-2)