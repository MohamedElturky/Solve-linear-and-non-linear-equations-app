import pytest
import numpy as np
from GaussElemination import GaussElemination
from Gaussjordan import GaussJordan
from LU import LU
from Jacobi import Jacobi
from GaussSeidel import GaussSeidel
from Method import Equations  

# Define general test cases
general_test_cases = [
    # Test case 1: 2x2 matrix
    (np.array([[3, 2], [1, 2]]), np.array([5, 5]), np.linalg.solve(np.array([[3, 2], [1, 2]]), np.array([5, 5]))),
    # Test case 2: 3x3 matrix
    (np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]), np.array([1, 0, 1]), np.linalg.solve(np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]), np.array([1, 0, 1]))),
    # Test case 3: 4x4 matrix # most of them fail here
    (np.array([[4, 1, 2, 3], [3, 4, 1, 2], [2, 3, 4, 1], [1, 2, 3, 4]]), np.array([10, 11, 12, 13]), np.linalg.solve(np.array([[4, 1, 2, 3], [3, 4, 1, 2], [2, 3, 4, 1], [1, 2, 3, 4]]), np.array([10, 11, 12, 13])))
]

# Define diagonal matrix test cases
diagonal_test_cases = [
    # Test case 1: 2x2 diagonal matrix
    (np.array([[3, 0], [0, 2]]), np.array([6, 4]), np.linalg.solve(np.array([[3, 0], [0, 2]]), np.array([6, 4]))),
    # Test case 2: 3x3 diagonal matrix
    (np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]), np.array([1, 4, 9]), np.linalg.solve(np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]), np.array([1, 4, 9]))),
    # Test case 3: 4x4 diagonal matrix
    (np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]), np.array([1, 4, 9, 16]), np.linalg.solve(np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]), np.array([1, 4, 9, 16])))
]

# Define symmetric matrix test cases
symmetric_test_cases = [
    # Test case 1: 2x2 symmetric matrix
    (np.array([[2, 1], [1, 2]]), np.array([3, 3]), np.linalg.solve(np.array([[2, 1], [1, 2]]), np.array([3, 3]))),
    # Test case 2: 3x3 symmetric matrix
    (np.array([[4, 1, 2], [1, 3, 0], [2, 0, 5]]), np.array([7, 4, 6]), np.linalg.solve(np.array([[4, 1, 2], [1, 3, 0], [2, 0, 5]]), np.array([7, 4, 6]))),
    # Test case 3: 4x4 symmetric matrix
    (np.array([[4, 1, 2, 3], [1, 3, 0, 1], [2, 0, 5, 2], [3, 1, 2, 6]]), np.array([10, 5, 8, 12]), np.linalg.solve(np.array([[4, 1, 2, 3], [1, 3, 0, 1], [2, 0, 5, 2], [3, 1, 2, 6]]), np.array([10, 5, 8, 12])))
]

all_test_cases = general_test_cases + diagonal_test_cases + symmetric_test_cases

@pytest.mark.parametrize("matrix, solution, expected_output", all_test_cases)
def test_gauss_elimination(matrix, solution, expected_output):
    result = GaussElemination( np.array(matrix), np.array(solution)).apply()
    assert np.allclose(result, expected_output  ,  atol=0.1), f"Failed for matrix: \n {matrix} \n Expected: {expected_output} \n Got: \t {result}"


# Repeat the same for other methods (GaussJordan, LU, Jacobi, GaussSeidel)
@pytest.mark.parametrize("matrix, solution, expected_output", all_test_cases)
def test_gauss_jordan(matrix, solution, expected_output):
    result =  GaussJordan( np.array(matrix), np.array(solution)).apply()
    assert np.allclose(result, expected_output  ,  atol=0.1), f"Failed for matrix: \n {matrix} \n Expected: {expected_output} \n Got: \t {result}"



@pytest.mark.parametrize("matrix, solution, expected_output", all_test_cases)
def test_gauss_seidel(matrix, solution, expected_output):
    result,iters = GaussSeidel( np.array(matrix), np.array(solution),guess=np.zeros_like(solution)
                         ,iter=None,tol=5e-2).apply()
    assert np.allclose(result, expected_output  ,  atol=0.1), f"Failed for matrix: \n {matrix} \n Expected: {expected_output} \n Got: \t {result}"


@pytest.mark.parametrize("matrix, solution, expected_output", all_test_cases)
def test_jacobi(matrix, solution, expected_output):
    result,iters = Jacobi( np.array(matrix), np.array(solution),guess=np.zeros_like(solution) , iter=100,tol=1e-5).apply()
    assert np.allclose(result, expected_output  ,  atol=0.1), f"Failed for matrix: \n {matrix} \n Expected: {expected_output} \n Got: \t {result}"


@pytest.mark.parametrize("matrix, solution, expected_output", all_test_cases)
def test_jacobi_no_tol(matrix, solution, expected_output):
    result,iters = Jacobi( np.array(matrix), np.array(solution),guess=np.zeros_like(solution) , iter=100,tol=None).apply()
    assert np.allclose(result, expected_output  ,  atol=0.1), f"Failed for matrix: \n {matrix} \n Expected: {expected_output} \n Got: \t {result}"


@pytest.mark.parametrize("matrix, solution, expected_output", all_test_cases)
def test_jacobi_no_iter(matrix, solution, expected_output):
    result,iters = Jacobi( np.array(matrix), np.array(solution),guess=np.zeros_like(solution) , iter=None,tol=1e-5).apply()
    assert np.allclose(result, expected_output  ,  atol=0.1), f"Failed for matrix: \n {matrix} \n Expected: {expected_output} \n Got: \t {result}"


#LU 

@pytest.mark.parametrize("matrix, solution, expected_output", all_test_cases)
def test_lu_doolittle(matrix, solution, expected_output):
    result = LU( np.array(matrix), np.array(solution), method="Doolittle").apply()
    assert np.allclose(result, expected_output  ,  atol=0.1), f"Failed for matrix: \n {matrix} \n Expected: {expected_output} \n Got: \t {result}"

@pytest.mark.parametrize("matrix, solution, expected_output", all_test_cases)
def test_lu_crout(matrix, solution, expected_output):
    result = LU( np.array(matrix), np.array(solution), method="Crout").apply()
    assert np.allclose(result, expected_output  ,  atol=0.1), f"Failed for matrix: \n {matrix} \n Expected: {expected_output} \n Got: \t {result}"


@pytest.mark.parametrize("matrix, solution, expected_output", general_test_cases)
def test_lu_cholesky(matrix, solution, expected_output):
    result = LU( np.array(matrix), np.array(solution), method="Doolittle").apply()
    assert np.allclose(result, expected_output  ,  atol=0.1), f"Failed for matrix: \n {matrix} \n Expected: {expected_output} \n Got: \t {result}"

@pytest.mark.parametrize("matrix, solution, expected_output", symmetric_test_cases)
def test_lu_cholesky_symmetric(matrix, solution, expected_output):
    result = LU( np.array(matrix), np.array(solution), method="Doolittle").apply()
    assert np.allclose(result, expected_output  ,  atol=0.1), f"Failed for matrix: \n {matrix} \n Expected: {expected_output} \n Got: \t {result}"


if __name__ == "__main__":
    pytest.main()