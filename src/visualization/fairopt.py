# This is an implementation inspired by the following paper:
# "Kim, Joon Sik, Jiahao Chen, and Ameet Talwalkar. "Fact: A diagnostic for group fairness trade-offs." International Conference on Machine Learning. PMLR, 2020."

from scipy.optimize import linprog
import numpy as np
import ast

def to_dp_weight(weights):
    try:
        ws = ast.literal_eval(weights)
        if len(ws) >= 3:
            if 3 in ws:
                return 3
            elif 2 in ws:
                return 2
            elif 1 in ws:
                return 1
        return 0
    except Exception:
        return 0
    

def dataset_to_cf(A, Y):
    """
    From the sensitive column A and target Y return the proprotions relative to the confusion matrix.
            Parameters:
                    A (array): Sensitive variable
                    Y (array): Target variable
            Returns:
                    n_N0 (float): % group 0
                    n_N1 (float):  % group 1
                    Y1_G0 (float): % group 0 with Y=1
                    Y1_G1 (float): % group 1 with Y=1
                    Y0_G0 (float):  % group 0 with Y=0
                    Y0_G1 (float): % group 1 with Y=0
    """
    # Number of individuals and groups
    N = A.shape[0]
    N_G0 = N-A.sum()
    N_G1 = N-N_G0
    # percentage of groups
    n_N0 = N_G0/N
    n_N1 = N_G1/N
    # percentage of positive/negative per group
    Y1_G0 = Y[A == 0].sum() / N  # Y=1, G=0
    Y1_G1 = Y[A == 1].sum() / N  # Y=1, G=1
    Y0_G0 = (1-Y[A == 0]).sum() / N  # Y=0, G=0
    Y0_G1 = (1-Y[A == 1]).sum() / N  # Y=0, G=1

    return n_N0, n_N1, Y1_G0, Y1_G1, Y0_G0, Y0_G1, N


def montecarlo_curve(n_N0, n_N1, Y1_G0, Y1_G1, Y0_G0, Y0_G1, size=10000):
    """
    Generate the space (accuracy, demographic parity) with random confusion matrix proportion.
            Parameters:
                    n_N0 (float): % group 0
                    n_N1 (float):  % group 1
                    Y1_G0 (float): % group 0 with Y=1
                    Y1_G1 (float): % group 1 with Y=1
                    Y0_G0 (float):  % group 0 with Y=0
                    Y0_G1 (float): % group 1 with Y=0
                    size (int): dimensions of random points
            Returns:
                    acc (float): accuracy
                    dp (float): demographic parity

    """
    tp1 = np.random.uniform(0, Y1_G1, size=size)
    fn1 = Y1_G1 - tp1
    fp1 = np.random.uniform(0, Y0_G1, size=size)
    tn1 = Y0_G1 - fp1
    tp0 = np.random.uniform(0, Y1_G0, size=size)
    fn0 = Y1_G0 - tp0
    fp0 = np.random.uniform(0, Y0_G0, size=size)
    tn0 = Y0_G0 - fp0

    acc = tp1 + tn1 + tp0 + tn0
    dp = (n_N0*tp1 + n_N0*fp1 - n_N1*tp0 - n_N1*fp0)/(n_N0*n_N1)
    return acc, dp


def acc_given_t(N, n_N0, n_N1, Y1_G0, Y1_G1, Y0_G0, Y0_G1, t, fair_cons='DP'):
    """
    Given the confusion matrix proportions and the fairness constraint (fair_cons) value chosen (t) return z, acc and the output of the optimization problem.
            Parameters:
                    n_N0 (float): % group 0
                    n_N1 (float):  % group 1
                    Y1_G0 (float): % group 0 with Y=1
                    Y1_G1 (float): % group 1 with Y=1
                    Y0_G0 (float):  % group 0 with Y=0
                    Y0_G1 (float): % group 1 with Y=0
                    t (float): fair_cons value desidered to reach
                    fair_cons (str): fairness constraint to impose (e.g. DP or DI)
            Returns:
                    z (array): [TP1, FN1, FP1, TN1, TP0, FN0, FP0, TN0]
                    acc (float): accuracy value
                    res (object): the output of the optimization problem
    """

    if fair_cons == 'DP':
        A_fairness = [n_N0, 0., n_N0, 0., -n_N1, 0., -n_N1, 0.]  # constrain DP
        b_fairness = t*(n_N0*n_N1)  # reformulate the DP for the constrain
    elif fair_cons == 'DI':
        A_fairness = [n_N0, 0., n_N0, 0., -t *
                      n_N1, 0., -t*n_N1, 0.]  # constrain DI
        b_fairness = 0.  # this impose the ratio
    else:
        raise Exception('Error, fairness constraint bad defined.')

    A_eq = [A_fairness,
            # constrain percentage of y=0 for group 1. FP1+TN1
            [0, 0, 1, 1., 0, 0., 0, 0.],
            # constrain percentage of y=0 for group 0. FP0+TN0
            [0, 0., 0, 0., 0, 0., 1, 1],
            # [1, 1, 0, 0., 0, 0., 0, 0], # constrain percentage of y=1 for group 1. TP1+FN1
            # [0, 0., 0, 0., 1, 1, 0, 0], # constrain percentage of y=1 for group 0. TP0+FN0
            [1, 1, 1, 1, 0, 0, 0, 0],  # percentage group 1
            [0, 0., 0, 0., 1, 1, 1, 1],  # percentage group 0
            # [1, 1, 1, 1, 1, 1, 1, 1], # the sum must be 1
            ]

    # A_ub =[
    #        [1, 0, 0, 1, 1, 0, 0, 1],
    #      ]
    #
    # b_ub = [1]

    b_eq = [b_fairness,
            Y0_G1,  # % group 1 with Y=0
            Y0_G0,  # % group 0 with Y=0
            # Y1_G1,
            # Y1_G0,
            n_N1,
            n_N0,
            # 1
            ]

    res = linprog(c=[-1, 0, 0, -1, -1, 0, 0, -1],  # maximize accuracy. TP1+TP0+TN1+TN0
                  A_eq=A_eq,
                  # A_ub=A_ub,
                  b_eq=b_eq,
                  # b_ub=b_ub,
                  x0=[Y1_G1, 0, 0, Y0_G1, Y1_G0, 0, 0, Y0_G0],
                  method='revised simplex',
                  # options={'tol':0.01},
                  bounds=(0, 1))

    conf_mat = (res.x*N).reshape(2, 2, 2).transpose(0, 2, 1)
    conf_mat = conf_mat.astype('int')

    return res.x, -res.fun, conf_mat, res


def curve_dp(A, Y, t, fair_cons='DP'):
    """
    Given A and Y output the accuracy with a target Fairness value (t)
            Parameters:
                    A (array): Sensitive variable
                    Y (array): Target variable
                    t (float): Fariness value desidered to reach
                    fair_cons (str): fairness constraint to impose (e.g. DP or DI)
            Returns:
                    acc (float): accuracy value
                    di (float): disperate impact
                    dp (float): demographic parity
    """
    # Get confusion matrix proportions
    n_N0, n_N1, Y1_G0, Y1_G1, Y0_G0, Y0_G1, N = dataset_to_cf(A, Y)
    # Get counfusion matrix values with DP=t
    z, acc, cm, _ = acc_given_t(A.shape[0],
        n_N0, n_N1, Y1_G0, Y1_G1, Y0_G0, Y0_G1, t, fair_cons)

    return acc, cm
