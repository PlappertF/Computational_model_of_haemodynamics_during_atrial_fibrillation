from numba import njit
import numpy as np
from math import log, tan, atan, cos, exp, sin, pi
import scipy.special as sc
import heapq
import warnings

"""
You may have the idea at some point that there is a problem when we initialize the AV nodal refractory period with zeros
and don't let the model run for a few beats to reach a steady state before we use the resulting RR intervals. I assumed
that if the model runs for long enough, it forgets the initial conditions. I ran an experiment where I generated a very
very long AA series and ran the model. Then I took out the AA intervals of the first 1000ms and ran the model again.
I compared the second half of the RR series of the two runs and assumed that if the first half is long enough, the RR
intervals of the second half are almost the same. I assumed that the difference between the RR series converges to zero 
when the model runs long enough and the history information of the beginning of the model input series is forgotten.
This was not the case, though. It seems like a classical case of the butterfly effect. If the initial conditions are 
different, the model will always produce different results, no matter how much time is passing. Therefore, setting the AV
nodal refractory period to zero is as good as any initial condition. 
"""

def get_rhythm_parameters(avn_p4_params, desired_rr_char, num_his_act=201):
    if avn_p4_params is None:
        avn_par_ranges = np.array([[250., 600.],  # Minimum refractory period of the slow AV nodal pathway (ms)
                                   [0., 600.],    # Range of refractory period of the slow AV nodal pathway (ms)
                                   [50., 300.],   # Time constant of the refractory period of the slow AV nodal pathway
                                                  # (ms)
                                   [250., 600.],  # Minimum refractory period of the fast AV nodal pathway (ms)
                                   [0., 600.],    # Range of refractory period of the fast AV nodal pathway (ms)
                                   [50., 300.],   # Time constant of the refractory period of the fast AV nodal pathway
                                                  # (ms)
                                   [0., 30.],     # Minimum conduction delay of the slow AV nodal pathway (ms)
                                   [0., 75.],     # Range of conduction delay of the slow AV nodal pathway (ms)
                                   [50., 300.],   # Time constant of the conduction delay of the slow AV nodal pathway
                                                  # (ms)
                                   [0., 30.],     # Minimum conduction delay of the fast AV nodal pathway (ms)
                                   [0., 75.],     # Range of conduction delay of the fast AV nodal pathway (ms)
                                   [50., 300.],   # Time constant of the conduction delay of the fast AV nodal pathway
                                                  # (ms)
                                   [100., 250.],  # Mean arrival rate of atrial impulses used in the Pearson Type IV
                                                  # distribution (ms)
                                   [15., 30.],    # Standard deviation of the arrival rate of atrial impulses in the
                                                  # Pearson Type IV distribution (ms)
                                   [0., 0.4],     # Amplitude of the respiratory modulation of the AV node model
                                   [0.1, 0.4]])   # Frequency of the respiratory modulation of the AV node model (Hz)

    while_counter = 0
    while 1:
        # Parameters for the Pearson Type IV distribution
        p4dist = np.zeros(12)
        if avn_p4_params is None:
            # Get AV node model parameters
            avn_parameters = draw_parameter_set(avn_par_ranges)

            # Parameters for the refractory period of the AV node model
            avn_rp = np.zeros(7)
            avn_rp[0:6] = avn_parameters[0:6]
            avn_rp[6] = 250

            # Parameters for the conduction delay of the AV node model
            avn_cd = np.zeros(7)
            avn_cd[0:6] = avn_parameters[6:12]
            avn_cd[6] = 0

            # Parameters for the respiratory modulation of the AV node model
            avn_resp = avn_parameters[14:16]

            # Parameters for the Pearson Type IV distribution
            p4dist[0:2] = avn_parameters[12:14]
            p4dist[2:5] = np.array([1, 6, 50])
        else:
            avn_rp = avn_p4_params[:7]
            avn_cd = avn_p4_params[7:14]
            avn_resp = avn_p4_params[14:16]

            p4dist[0:5] = avn_p4_params[16:]
        p4dist = get_pearson4_parameters(p4dist)

        fixed_avn_pars = np.concatenate((avn_rp, avn_cd, avn_resp, p4dist))
        vat, _, _ = run_avn_model(fixed_avn_pars, np.ones(46)*np.nan, qu=None, num_his_act=num_his_act)
        rr = np.diff(vat)
        if desired_rr_char is None:  # This is the case if we don't ask for a desired_rr_char
            return avn_rp, avn_cd, avn_resp, p4dist, np.array([np.mean(rr), 0, 0])
        else:  # THis is the case if we pass a desired_rr_char
            # I do it as stacked if clauses, because sample entropy is the most expensive to compute and we don't always
            # have to compute sample entropy.
            rr_mean = np.mean(rr)
            if abs(rr_mean-desired_rr_char[0]) <= 10:
                if np.all(desired_rr_char[1:3] > 0):
                    rr_rmssd = rmssd(rr)
                    if abs(rr_rmssd - desired_rr_char[1]) <= 10:
                        rr_sampen = sample_entropy(rr)
                        if abs(rr_sampen - desired_rr_char[2]) <= 0.1:
                            rr_char = np.array([rr_mean, rr_rmssd, rr_sampen])
                            return avn_rp, avn_cd, avn_resp, p4dist, rr_char
                else:  # If only the RR mean but not RR rmssd and RR sample entropy are desired
                    rr_char = np.array([rr_mean, 0, 0])
                    return avn_rp, avn_cd, avn_resp, p4dist, rr_char

            while_counter += 1
            if (while_counter % 10000) == 0:
                if np.all(desired_rr_char[1:3] > 0):
                    warnings.warn("So far " + str(while_counter) + " parameter sets were generated in search of a " +
                                  "parameter set that results in a RR mean of " + str(desired_rr_char[0]) + "ms, a RR " +
                                  "rmssd of " + str(desired_rr_char[1]) + "ms, and a RR sample entropy of " +
                                  str(desired_rr_char[2]) + ". The search continues until a parameter set is found, " +
                                  "but it might never happen.", UserWarning)
                else:
                    if 12 <= desired_rr_char[0] <= 200:
                        warnings.warn("So far " + str(while_counter) + " parameter sets were generated in search of a " +
                                      "parameter set that results in an average heart rate of " + str(desired_rr_char[0]) +
                                      "bpm. The search continues until a parameter set is found, but it might never " +
                                      "happen.", UserWarning)
                    else:  # 250 <= desired_rr_char[0] <= 5000:  check_input_params makes sure it has to be one of these two
                        warnings.warn("So far " + str(while_counter) + " parameter sets were generated in search of a " +
                                      "parameter set that results in an RR mean of " + str(desired_rr_char[0]) + "ms. " +
                                      "The search continues until a parameter set is found, but it might never happen.",
                                      UserWarning)


@njit(cache=False)
def run_avn_model(fixed_avn_pars, avn_states, qu=None, num_his_act=1, n_aa=2):
    """
    The default of n_aa is 2, because the function 'run_avn_model' runs fastest with n_aa set to 2. If n_aa is set to a
    larger value, the function is slower because the priority queue has to handle more impulses at the same time slowing
    the function down. If n_aa is set to 1, the function is slower as well because drawing 2 AA intervals at the same
    time is much faster than drawing 2 AA intervals one after the other.
    :param fixed_avn_pars: The fixed parameters of the AV node model
        fixed_avn_pars[0:7]: (model.activation.avn_rp)
        fixed_avn_pars[7:14]: (model.activation.avn_cd)
        fixed_avn_pars[14:16]: (model.activation.avn_resp)
        fixed_avn_pars[16:28]: (model.activation.p4dist)
    :param avn_states: The states of the AV node model
        avn_states[0:22]: (model.activation.rt)
        avn_states[22:44]: (model.activation.dt)
        avn_states[44]: (model.activation.aat)
        avn_states[45]: (model.activation.node0_n_imp)
    :param qu: The priority queue of impulses that can be passed from a previous run that should be continued.
    :param num_his_act: The number of His-Bundle activations that are generated
    :param n_aa: The number of AA intervals to draw at once
    """
    # priority queue of impulse arrival times and respective nodes arranged in a heap
    # It is very important that each number in the list has the same datatype, here it is float64.
    # Even if the second element in the tuple could be technically an integer value, it has to be a float64, because the
    # first element must be a float64.
    # For numba to compile this code, I always need to declare q as a list with tuples of two float64 values. Then, if I
    # want to use the queue-list from a previous iteration, I can overwrite the list in the if condition.
    q = [(0.0, 0.0)]  # (arrival time of impulse, node in AV node network)
    if ~np.isnan(avn_states[44]):
        if qu is not None:
            q = qu
        rt = avn_states[0:22]
        dt = avn_states[22:44]
        aat = avn_states[44]
        node0_n_imp = avn_states[45]
    else:
        # The five parameters rt, dt, aat, q and node0_n_imp are None if the function 'run_avn_model' is called for the first
        # time to generate an RR series. The five parameters are not None, if we want to continue with the previous states to
        # continue an RR series.
        # current refractory period values of each node. node 0 is included in this array, but doesn't have a refractory
        # period. I gave the array the length 22 to make the code below easier to read.
        rt = np.zeros(22)
        # current conduction delay values of each node. node 0 is included in this array, but doesn't have a conduction
        # delay. I gave the array the length 22 to make the code below easier to read.
        dt = np.zeros(22)
        # initialize the arrival time of the latest atrial impulse.
        # aat = atrial arrival time
        aat = 0  # value in ms
        node0_n_imp = 1  # number of impulses in priority queue at node 0

    # initialize the array of ventricular activation times.
    # vat = ventricular activation time
    vat = np.zeros(num_his_act)
    vat_idx = 0  # index of ventricular activation that will be saved in vat

    # The while loop stops if the array of ventricular activations 'vat' is filled
    while vat_idx < num_his_act:
        # If there is only one AA interval waiting in node 0 to enter the AV node model, then I produce more AA
        # intervals that are added to the priority queue at node 0
        if node0_n_imp == 0:
            s = generate_pearson4(p4dist=fixed_avn_pars[16:28], n_aa=n_aa)
            for t in s:
                aat += t
                heapq.heappush(q, (aat, 0.0))
            node0_n_imp += n_aa

        # get top impulse in priority queue
        current_time, node = heapq.heappop(q)
        # node of top impulse is node 0 -> conduct the impulse to first node of slow and fast pathway
        if abs(node - 0.0) < 0.1:
            # conduct node to first node of slow pathway
            heapq.heappush(q, (current_time, 1.0))
            # conduct node to first node of fast pathway
            heapq.heappush(q, (current_time, 11.0))
            node0_n_imp -= 1
        else:
            # check if arrival time of the current impulse at the current node is equal or larger than the end of the
            # last refractory period. If this is the case, the impulse is conducted and the code in the statement is
            # executed. Otherwise, nothing happens and the current impulse will be blocked.
            if current_time >= rt[int(node)]:
                # Check if node of current impulse belongs to the slow or fast pathway but not the coupling node.
                if node < 20.9:
                    # c_resp is the scaling factor used for the refractory period and conduction delay to account for the
                    # respiratory modulation
                    c_resp = (1+fixed_avn_pars[14]/2 * sin(2*pi*current_time/1000*fixed_avn_pars[15]))
                    # compute refractory period and conduction delay for slow pathway
                    if node < 10.1:
                        dt[int(node)] = c_resp*(fixed_avn_pars[7] +
                                                fixed_avn_pars[8] * exp(-(current_time - rt[int(node)]) /
                                                                        fixed_avn_pars[9]))
                        rt[int(node)] = (current_time +
                                         c_resp * (fixed_avn_pars[0] +
                                                   fixed_avn_pars[1] * (1-exp(-(current_time - rt[int(node)]) /
                                                                              fixed_avn_pars[2]))))

                    # compute refractory period and conduction delay for fast pathway.
                    else:
                        dt[int(node)] = c_resp*(fixed_avn_pars[10] +
                                                fixed_avn_pars[11] * exp(-(current_time - rt[int(node)]) /
                                                                         fixed_avn_pars[12]))
                        rt[int(node)] = (current_time +
                                         c_resp * (fixed_avn_pars[3] +
                                                   fixed_avn_pars[4] * (1-exp(-(current_time - rt[int(node)]) /
                                                                              fixed_avn_pars[5]))))

                    future_time = current_time + dt[int(node)]
                    # first node of slow or fast pathway
                    if abs(node - 1.0) < 0.1 or abs(node - 11.0) < 0.1:
                        # conduct the impulse to the second node of the slow or fast pathway
                        heapq.heappush(q, (future_time, node + 1.0))

                    elif abs(node - 10.0) < 0.1:  # last node of slow pathway
                        # conduct the impulse backwards on the slow pathway
                        heapq.heappush(q, (future_time, 9.0))
                        # conduct the impulse to the last node of the fast pathway
                        heapq.heappush(q, (future_time, 20.0))
                        # conduct the impulse to the coupling node
                        heapq.heappush(q, (future_time, 21.0))

                    elif abs(node - 20.0) < 0.1:  # last node of fast pathway
                        # conduct the impulse backwards on the fast pathway
                        heapq.heappush(q, (future_time, 19.0))
                        # conduct the impulse to the last node of the slow pathway
                        heapq.heappush(q, (future_time, 10.0))
                        # conduct the impulse to the coupling node
                        heapq.heappush(q, (future_time, 21.0))

                    # center nodes of the slow or fast pathway, it can only be node < 20
                    else:
                        # conduct the impulse forward in the slow or fast pathway
                        heapq.heappush(q, (future_time, node + 1.0))
                        # conduct the impulse backwards in the slow or fast pathway
                        heapq.heappush(q, (future_time, node - 1.0))

                # this can only be the coupling node. current node has index 21
                else:
                    # the coupling node has a fixed refractory period of 250 ms
                    rt[int(node)] = current_time + fixed_avn_pars[6]
                    # the coupling node has a fixed conduction delay of 0 ms
                    # I don't need to update the value of dt[21], because it is always 0.
                    # dt[21] = 0

                    # the impulse successfully leaves the AV node network over the coupling node and is added to the
                    # series of ventricular activation times.
                    vat[vat_idx] = current_time
                    vat_idx += 1

    avn_states[0:22] = rt
    avn_states[22:44] = dt
    avn_states[44] = aat
    avn_states[45] = node0_n_imp
    return vat, avn_states, q


@njit(cache=False)
def generate_pearson4(p4dist, n_aa):
    aa = np.zeros(n_aa)
    i = 0
    while i < n_aa:
        U = np.random.uniform(low=0.0, high=4.0)
        if U <= 1:
            X = p4dist[9] - U * p4dist[11]
            if (abs(X) < pi / 2 and
                    log(np.random.uniform(low=0.0, high=1.0)) <=
                    p4dist[6] * log(abs(cos(X))) - p4dist[5] * X - p4dist[10] and
                    X >= atan((((p4dist[4]-p4dist[0])/p4dist[1])-p4dist[8])/p4dist[7])):
                aa[i] = X
                i += 1
        elif U <= 2:
            X = p4dist[9] - (1 - log(U - 1)) * p4dist[11]
            if (abs(X) < pi / 2 and
                    log(np.random.uniform(low=0.0, high=1.0)) - log(U - 1) <=
                    p4dist[6] * log(abs(cos(X))) - p4dist[5]*X - p4dist[10] and
                    X >= atan((((p4dist[4]-p4dist[0])/p4dist[1])-p4dist[8])/p4dist[7])):
                aa[i] = X
                i += 1
        elif U <= 3:
            X = p4dist[9] + (U - 2) * p4dist[11]
            if (abs(X) < pi / 2 and
                    log(np.random.uniform(low=0.0, high=1.0)) <=
                    p4dist[6] * log(abs(cos(X))) - p4dist[5] * X - p4dist[10] and
                    X >= atan((((p4dist[4]-p4dist[0])/p4dist[1])-p4dist[8])/p4dist[7])):
                aa[i] = X
                i += 1
        else:
            X = p4dist[9] + (1 - log(U - 3) * p4dist[11])
            if (abs(X) < pi / 2 and
                    log(np.random.uniform(low=0.0, high=1.0)) - log(U - 3) <=
                    p4dist[6] * log(abs(cos(X))) -p4dist[5]*X - p4dist[10] and
                    X >= atan((((p4dist[4]-p4dist[0])/p4dist[1])-p4dist[8])/p4dist[7])):
                aa[i] = X
                i += 1
    aa = ((p4dist[7] * np.array([tan(tmp_i) for tmp_i in aa]) + p4dist[8]) * p4dist[1] + p4dist[0])
    return aa


@njit(cache=False)
def draw_parameter_set(par_ranges):
    n_pars = len(par_ranges)
    # parameters = np.zeros(n_pars)
    while True:
        parameters = (par_ranges[:, 0] +
                      np.random.uniform(low=0, high=1, size=n_pars) *
                      (par_ranges[:, 1] - par_ranges[:, 0]))
        if check_av_node_parameters_overlapping(parameters):
            continue
        break
    return parameters


@njit(cache=False)
def check_av_node_parameters_overlapping(pars):
    """
    Check if the refractory period curves of slow pathway and fast pathway are overlapping or check if the conduction
    delay curves of slow pathway and fast pathway are overlapping
    :return: True if refractory period or conduction delay curves are overlapping for t >= 0
             False if curves are not overlapping for t >= 0
    """
    # This can be checked for the time >= 0 with the following six cases
    # The refractory period and conduction delay curves are allowed to overlap for time < 0
    # Case 1: Are the refractory period curves overlapping for t=0
    #         Overlapping if R_min^SP > R_min^FP
    if pars[0] > pars[3]:
        return True
    # Case 2: Are the conduction delay curves overlapping for t=0
    #         Overlapping if D_min^SP < D_min^FP
    if pars[6] < pars[9]:
        return True
    # Case 3: Are the refractory period curves overlapping for t -> infinity
    #         Overlapping if R_min^SP+DeltaR^SP > R_min^FP+DeltaR^FP
    if pars[0] + pars[1] > pars[3] + pars[4]:
        return True
    # Case 4: Are the conduction delay curves overlapping for t -> infinity
    #         Overlapping if D_min^SP+DeltaD^SP < D_min^FP+DeltaD^FP
    if pars[6] + pars[7] < pars[9] + pars[10]:
        return True
    # Case 5: Are the refractory period curves overlapping for 0 < t < infinity
    #         t_rp is the time t at which R^SP(t)-R^FP(t) has its global maximum
    # First avoid zero division error
    if pars[2] == pars[5] or pars[5] == 0 or pars[1] == 0:
        return True
    t_rp = (pars[2] * pars[5] *
            log((pars[2] * pars[4]) / (pars[5] * pars[1])) /
            (pars[2] - pars[5]))
    if (t_rp > 0 and pars[0] + pars[1] * (1 - exp(-t_rp / pars[2])) >
            pars[3] + pars[4] * (1 - exp(-t_rp / pars[5]))):
        return True
    # Case 6: Are the conduction delay curves overlapping for 0 < t < infinity
    #         t_cd is the time t at which D^SP(t)-D^FP(t) has its global minimum
    # First avoid zero division error
    if pars[11] == pars[8] or pars[8] == 0 or pars[10] == 0:
        return True
    t_cd = (pars[8] * pars[11] *
            log((pars[11] * pars[7]) / (pars[8] * pars[10])) /
            (pars[11] - pars[8]))
    if (t_cd > 0 and pars[6] + pars[7] * exp(-t_cd / pars[8]) <
            pars[9] + pars[10] * exp(-t_cd / pars[11])):
        return True
    # If the code reaches this point, then the refractory period and conduction delay curves are not overlapping
    # for t >= 0
    return False


def get_pearson4_parameters(p4dist):
    # The kurtosis in p4dist[3] can be incrementally increased by 0.1 until 4 * c0 * c2 - c1 ** 2 > 0
    skew_kurt_not_realistic = True
    while skew_kurt_not_realistic:
        denominator = (10 * p4dist[3] - 12 * p4dist[2] ** 2 - 18)
        c0 = (4 * p4dist[3] - 3 * p4dist[2]) / denominator
        c1 = (p4dist[2] * (p4dist[3] + 3)) / denominator
        c2 = (2 * p4dist[3] - 3 * p4dist[2] ** 2 - 6) / denominator
        if 4 * c0 * c2 - c1 ** 2 > 0:
            skew_kurt_not_realistic = False
        else:
            p4dist[3] += 0.1

    m = 1 / (2 * c2)
    p4dist[5] = (2 * c1 * (1 - m) / np.sqrt(4 * c0 * c2 - c1 ** 2))
    p4dist[6] = 2 * (m - 1)
    p4dist[7] = np.sqrt(p4dist[6] ** 2 * (p4dist[6] - 1) / (p4dist[6] ** 2 + p4dist[5] ** 2))
    p4dist[8] = p4dist[7] * p4dist[5] / p4dist[6]

    p4dist[9] = atan(-p4dist[5] / p4dist[6])
    p4dist[10] = (p4dist[6]*log(p4dist[7] / np.sqrt(p4dist[6] - 1)) - p4dist[5]*p4dist[9])
    p4dist[11] = exp(-p4dist[10] + log(abs(sc.gamma(m)/sc.gamma(complex(m, p4dist[5]/2)))**2) - sc.gammaln(m) +
                     sc.gammaln(m - 0.5) + log(np.sqrt(np.pi) * p4dist[7]))
    return p4dist


@njit(cache=False)
def sample_entropy_sub_function(matrix):
    a = np.zeros(matrix.shape[1]) == 0
    for i in range(matrix.shape[0]):
        a = np.logical_and(a, matrix[i, :])
    return sum(a) - 1


@njit(cache=False)
def sample_entropy(u):
    N = len(u)
    r = 0.2
    m = 2
    R = r * np.std(u)

    xm = np.zeros((2, N - m))
    xm1 = np.zeros((3, N - m))
    Am = np.zeros(N - m)
    Bm = np.zeros(N - m)

    for i in range(m):
        xm[i, :] = u[i : N - m + i]
    for i in range(m+1):
        xm1[i, :] = u[i : N - m + i]

    for i in range(N - m):
        Bisum = sample_entropy_sub_function(np.abs(xm[:, i:i+1] - xm) <= R)
        Aisum = sample_entropy_sub_function(np.abs(xm1[:, i:i+1] - xm1) <= R)

        Am[i] = Aisum / (N - m - 1)
        Bm[i] = Bisum / (N - m - 1)

    A = np.sum(Am) / (N - m)
    B = np.sum(Bm) / (N - m)

    if A > 0 and B > 0:
        samp_en = -np.log(A / B)
    else:
        samp_en = np.inf

    return samp_en


@njit(cache=False)
def rmssd(rr):
    return np.sqrt(np.mean(np.square(np.diff(rr))))
