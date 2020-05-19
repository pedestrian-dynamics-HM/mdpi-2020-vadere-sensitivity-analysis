import numpy as np
import numpy.testing as nptest
from numpy.random import RandomState

from uq.utils.model_function import MatrixModel


# constantine-2015, Example 3.4.1, A quadratic model, p. 45ff
def config_matrix_constantine(case, random_state: RandomState = None):
    bool_diagonal_matrix = False
    bool_constantine = False  # use same matrices as constantine
    m = 10

    if case == 1:
        # Case 1: exponential decay with constant rate
        eig_A = 10 ** np.linspace(2, -2, m)

    elif case == 2:
        # Case 2: exponential decay with a larger gap between the first and second eigenvalues
        eig_A = 10 ** np.linspace(-0.5, -5, m)
        eig_A[0] = 10 ** 2
    elif case == 3:
        # Case 3: exponential decay  with a larger gap between the third and fourth eigenvalue
        tmp_eig_1 = 10 ** np.linspace(2, 1, 3)
        tmp_eig_2 = 10 ** np.linspace(-2, -5, m - 3)
        eig_A = np.concatenate((tmp_eig_1, tmp_eig_2))
    else:
        raise Warning("Case %d not defined" % case)

    # A as diagonal matrix (eigenvalues on diagonal)
    if bool_diagonal_matrix:
        A = np.diag(eig_A)
    else:
        if random_state is None:
            tmp = np.random.rand(m, m)
        else:
            tmp = random_state.rand(m, m)
        Q, R = np.linalg.qr(tmp)
        A = np.matmul(Q, np.matmul(np.diag(eig_A), np.transpose(Q)))

    if case == 1 and bool_constantine:
        # Original matrix of Constantine from https://bitbucket.org/paulcon/computing-active-subspaces/downloads/

        A = [[0.520146613339148, -2.272499724678972, -3.980396033528675, -1.859567306638951, -3.518372147969480,
              -0.077047986466331, 0.022455764013845, -0.043189410012116, 3.645893870833929, 1.021319217711912],
             [-2.272499724678972, 10.054398988263546, 17.611607065967725, 8.255754236776935, 15.501001801537459,
              0.343161702269371, -0.136983933739535, 0.215627562325271 - 16.123109244111316, -4.467439414427460],
             [-3.980396033528675, 17.611607065967725, 30.857014026936504, 14.462878513269160, 27.160219329419597,
              0.600097918417477, -0.242682297724097, 0.373630668448233 - 28.253238396558498, -7.829928927874509],
             [-1.859567306638951, 8.255754236776934, 14.462878513269160, 6.807625611788189, 12.703961335456304,
              0.285864095365142, -0.109691713942052, 0.197914043583960 - 13.237477268899053, -3.652580711803129],
             [-3.518372147969480, 15.501001801537459, 27.160219329419601, 12.703961335456304, 23.950224129146143,
              0.524444724089434, -0.199822947521034, 0.304988254569739 - 24.879441761287691, -6.925167726630556],
             [-0.077047986466331, 0.343161702269371, 0.600097918417478, 0.285864095365142, 0.524444724089434,
              0.012857405344440,
              -0.003557360954465, 0.012147129409185, -0.547128402698593, -0.149716996756593],
             [0.022455764013845, -0.136983933739534, -0.242682297724097, -0.109691713942052, -0.199822947521034,
              -0.003557360954465, 0.022406026790908, -0.001677085883368, 0.220451094866262, 0.049564292387867],
             [-0.043189410012116, 0.215627562325271, 0.373630668448233, 0.197914043583960, 0.304988254569739,
              0.012147129409185,
              -0.001677085883368, 0.027366409565612, -0.331783605628052, -0.077874279369725],
             [3.645893870833929 - 16.123109244111316 - 28.253238396558498 - 13.237477268899053 - 24.879441761287691,
              -0.547128402698593, 0.220451094866262, -0.331783605628052, 25.880785580826764, 7.179153118069505],
             [1.021319217711912, -4.467439414427460, -7.829928927874509, -3.652580711803129, -6.925167726630556,
              -0.149716996756593, 0.049564292387867, -0.077874279369725, 7.179153118069505, 2.013418112803172]]

    test_model = MatrixModel(A)
    test_model.set_eigenvalues(eig_A)

    # config_matrix_constantine: Check Eigenvalues of constructed A
    nptest.assert_array_almost_equal(np.sort(eig_A), np.sort(np.linalg.eigvals(A)))

    x_lower = -1.0 * np.ones(shape=m)  # lower bounds for parameters
    x_upper = np.ones(shape=m)  # upper bounds for parameters

    density_type = "uniform"

    test_input = x_lower

    return test_model, x_lower, x_upper, m, density_type, test_input
