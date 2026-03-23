import math
import time

import numpy as np

def ErlangLoss(Lambda, Mu, N = None): # take it out once figured out what was the problem for the error. This is now a copy of the function in dispatch_main
    # The Erlangloss Model is exactly the same  as MMN0 and this returns the whole probability distribution
    # Solves the Erlang loss system
    if N == 0: # if there is 0 unit. The block probability is 1
        return [1]
    if N is not None: # If there is a size N, we constitute the Lambda and Mu vectors manually
        Lambda = np.ones(N)*Lambda
        Mu = Mu*(np.arange(N)+1)
    else: # if the Lambdas and Mus are already given in vector form then no need to do anything
        N = len(Lambda)
    LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
    Prod = [np.prod(LoM[0:i]) for i in range(1,N+2)]
    P_n = Prod/sum(Prod)
    return P_n

def Get_Effective_Lambda(offered_load, mu, n_units):
    rho_prev = 0.0
    rho = 1.0
    steps = 0
    while np.abs(rho - rho_prev) > 0.001:
        rho = rho_prev
        lam = rho * mu
        block_prob = ErlangLoss(lam, mu, n_units)[-1]
        rho_prev = rho - (rho * (1 - block_prob) - offered_load) / (
            1 - block_prob - (n_units - rho + rho * block_prob) * block_prob
        )
        steps += 1
        if steps > 100:
            break
    return rho * mu


def SumOfProduct(arr, k):
    n = len(arr)
    dp = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    cur_sum = 0
    for i in range(1, n + 1):
        dp[1][i] = arr[i - 1]
        cur_sum += arr[i - 1]
    for i in range(2, k + 1):
        temp_sum = 0
        for j in range(1, n + 1):
            cur_sum -= dp[i - 1][j]
            dp[i][j] = arr[j - 1] * cur_sum
            temp_sum += dp[i][j]
        cur_sum = temp_sum
    return np.array(dp).sum(axis=1)


class Two_State_Hypercube:
    def __init__(self, data_dict=None):
        self.keys = ['N', 'K', 'Lambda', 'Mu', 'frac_j', 't_mat', 'pre_list', 'pol']
        self.rho_approx = None
        self.q_nj = None
        self.P_n = None
        self.G = None
        self.Q = None
        self.r = None
        self.data_dict = dict.fromkeys(self.keys, None)
        if data_dict is not None:
            for key, value in data_dict.items():
                if key in self.keys:
                    self.data_dict[key] = value

    def Update_Parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.keys:
                self.data_dict[key] = value
                if key == 'pre_list':
                    self.G = None

    def Cal_P_n(self, lambda_in=None):
        n_units = self.data_dict['N']
        lam = self.data_dict['Lambda'] if lambda_in is None else lambda_in
        mu = self.data_dict['Mu']
        p_n = ErlangLoss(lam, mu, n_units)
        if lambda_in is not None:
            self.P_n = p_n
        return p_n

    def Cal_Q(self, p_n=None, r=None):
        n_units = self.data_dict['N']
        mu = self.data_dict['Mu']
        _ = mu
        if self.G is None:
            pre_list = self.data_dict['pre_list']
            self.G = [[np.where(pre_list[:, i] == j)[0] for i in range(n_units)] for j in range(n_units)]
        if p_n is None:
            p_n = self.Cal_P_n()
        if r is None:
            r = np.dot(p_n, np.arange(len(p_n))) / n_units
        q = np.zeros(n_units)
        last_index = len(p_n) - 1
        for j in range(n_units):
            q[j] = sum(
                [
                    (math.factorial(k) / math.factorial(k - j))
                    * (math.factorial(last_index - j) / math.factorial(last_index))
                    * (last_index - k)
                    / (last_index - j)
                    * p_n[k]
                    for k in range(j, last_index)
                ]
            ) / (r ** j * (1 - r))
        self.Q = q
        self.r = r
        return q

    def Larson_Approx(self, use_effective_lambda=True, epsilon=0.00001):
        n_units = self.data_dict['N']
        lam = self.data_dict['Lambda']
        mu = self.data_dict['Mu']
        frac_j = self.data_dict['frac_j']
        pre_list = self.data_dict['pre_list']
        alpha = getattr(self, 'alpha', 0)

        self.Cal_Q()
        rho_i = np.full(n_units, 0.5)
        rho_i_next = np.zeros(n_units)

        while True:
            rho_total = rho_i + (1 - rho_i) * alpha
            if use_effective_lambda:
                offered_load = rho_total.sum()
                lambda_eff = Get_Effective_Lambda(offered_load, mu, n_units)
                p_n = ErlangLoss(lambda_eff, mu, n_units)
                self.P_n = p_n
                self.Cal_Q(p_n=p_n, r=offered_load / n_units)

            for i in range(n_units):
                value = 1
                for k in range(n_units):
                    prod_g_j = 0
                    for j in self.G[i][k]:
                        prod_g_j += lam * frac_j[j] * self.Q[k] * np.prod(rho_total[pre_list[j, :k]])
                    value += prod_g_j / mu
                rho_i_next[i] = (1 - ((1 - rho_i) * alpha)[i]) * (1 - 1 / value)

            if abs(rho_i_next - rho_i).max() < epsilon:
                self.rho_approx = rho_i_next
                return rho_i_next

            rho_i = np.array(rho_i_next)
            rho_i_next = np.zeros(n_units)

    def Get_MRT_Approx(self):
        n_units = self.data_dict['N']
        n_atoms = self.data_dict['K']
        pre_list = self.data_dict['pre_list']
        frac_j = self.data_dict['frac_j']
        t_mat = self.data_dict['t_mat']
        rho = getattr(self, 'rho_total_approx', self.rho_approx)

        if self.P_n is None:
            p_n = self.Cal_P_n()
        else:
            p_n = self.P_n
        self.Cal_Q(r=np.sum(rho) / n_units, p_n=p_n)

        q_nj = np.zeros([n_atoms, n_units])
        for j in range(n_atoms):
            pre_j = pre_list[j]
            for n in range(n_units):
                q_nj[j, pre_j[n]] = self.Q[n] * np.prod(rho[pre_j[:n]]) * (1 - rho[pre_j[n]])
            q_nj[j, :] /= q_nj[j, :].sum()
            q_nj[j, :] *= frac_j[j] * (1 - p_n[-1])

        q_nj /= q_nj.sum()
        self.q_nj = q_nj
        mrt_j = np.sum(q_nj * t_mat, axis=1) / np.sum(q_nj, axis=1)
        mrt = np.sum(q_nj * t_mat)
        return mrt, mrt_j


class Three_State_Hypercube:
    def __init__(self, data_dict=None):
        self.keys_1 = ['N', 'N_1', 'N_2', 'K', 'Lambda_1', 'Mu_1', 'frac_j_1', 't_mat_1', 'pre_list_1']
        self.keys_2 = ['N', 'N_1', 'N_2', 'K', 'Lambda_2', 'Mu_2', 'frac_j_2', 't_mat_2', 'pre_list_2']
        self.data_dict_1 = dict.fromkeys(self.keys_1, None)
        self.data_dict_2 = dict.fromkeys(self.keys_2, None)
        if data_dict is not None:
            for key, value in data_dict.items():
                if key in self.keys_1:
                    self.data_dict_1[key] = value
                if key in self.keys_2:
                    self.data_dict_2[key] = value
        self.time_linearalpha = None

    def Update_Parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.keys_1:
                self.data_dict_1[key] = value
            if key in self.keys_2:
                self.data_dict_2[key] = value

    def Update_alpha(self, subsystem):
        n_units = self.data_dict_1['N']
        n_1 = self.data_dict_1['N_1']
        n_2 = self.data_dict_1['N_2']
        n_sub1 = n_units - n_2
        n_sub2 = n_units - n_1

        if subsystem == 1:
            rho = self.sub1.rho_approx
            alpha = self.sub1.alpha
            n_sub_other = n_sub2
            n_me = n_1
            n_other = n_2
        else:
            rho = self.sub2.rho_approx
            alpha = self.sub2.alpha
            n_sub_other = n_sub1
            n_me = n_2
            n_other = n_1

        return np.array(
            [0] * n_other
            + [
                rho[n + n_me - n_other]
                / (rho[n + n_me - n_other] + (1 - rho[n + n_me - n_other]) * (1 - alpha[n + n_me - n_other]))
                for n in range(n_other, n_sub_other)
            ]
        )

    def Creat_Two_Subsystems(self):
        self.sub1 = self.Subsystem(
            dict((key[:-2], value) for (key, value) in self.data_dict_1.items() if len(key) > 3)
        )
        self.sub2 = self.Subsystem(
            dict((key[:-2], value) for (key, value) in self.data_dict_2.items() if len(key) > 3)
        )
        n_sub1 = self.data_dict_1['N'] - self.data_dict_1['N_2']
        n_sub2 = self.data_dict_2['N'] - self.data_dict_2['N_1']
        n_atoms = self.data_dict_1['K']
        self.sub1.Update_Parameters(N=n_sub1, K=n_atoms)
        self.sub2.Update_Parameters(N=n_sub2, K=n_atoms)
        self.sub1.alpha = np.zeros(n_sub1)
        self.sub2.alpha = np.zeros(n_sub2)

    def Linear_Alpha(self, use_effective_lambda=True, epsilon=0.0001):
        start_time = time.time()
        iterations = 0

        while True:
            iterations += 1
            self.sub1.Larson_Approx(use_effective_lambda=use_effective_lambda)
            self.sub2.alpha = self.Update_alpha(subsystem=1)

            self.sub2.Larson_Approx(use_effective_lambda=use_effective_lambda)
            alpha = self.Update_alpha(subsystem=2)

            if max(abs(alpha - self.sub1.alpha)) < epsilon:
                self.sub1.alpha = alpha
                break

            self.sub1.alpha = alpha

        self.time_linearalpha = time.time() - start_time
        self.ite = iterations
        return self.time_linearalpha

    def Get_MRT_Approx_3state(self):
        n_units = self.data_dict_1['N']
        n_1 = self.data_dict_1['N_1']
        n_2 = self.data_dict_1['N_2']
        rho_1 = self.sub1.rho_approx
        rho_2 = self.sub2.rho_approx

        self.sub1.rho_total_approx = rho_1 + np.append(np.zeros(n_1), rho_2[n_2:])
        self.sub2.rho_total_approx = rho_2 + np.append(np.zeros(n_2), rho_1[n_1:])

        mrt_1, mrt_1_j = self.sub1.Get_MRT_Approx()
        mrt_2, mrt_2_j = self.sub2.Get_MRT_Approx()
        return mrt_1, mrt_2, mrt_1_j, mrt_2_j

    class Subsystem(Two_State_Hypercube):
        def __init__(self, data_dict=None):
            super().__init__(data_dict=data_dict)
            self.alpha = None
            self.rho_total_approx = None

        def Cal_P_n(self):
            n_units = self.data_dict['N']
            lam = self.data_dict['Lambda']
            mu = self.data_dict['Mu']

            lambda_bd = np.ones(n_units) * lam
            sumprod_vec = SumOfProduct(self.alpha, n_units)
            for i in range(n_units):
                num_comb = math.comb(n_units, i + 1)
                lambda_bd[n_units - i - 1] = lam / num_comb * (num_comb - sumprod_vec[i + 1])
                if lam - lambda_bd[n_units - i - 1] < 0.001:
                    break
            mu_bd = mu * (np.array(range(n_units)) + 1)
            return ErlangLoss(lambda_bd, mu_bd)
