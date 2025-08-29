import numpy

def calc_ising_energies(problem_mat_ising: numpy.ndarray, states_ising: numpy.ndarray) -> numpy.ndarray:
    """
    ### Calculate the energies of a given Ising model and spins

    ## Parameters:
    - `problem_mat_ising` - numpy.array of size NxN representing the Ising problem with the external filed on the diagonal (N - Number of spins)
    - `states_ising` - numpy.array of the spins of size N * n_states
    
    ## Returns:
    - `energies` - numpy.array of size [other dimnsions]
    """

    n_spins = states_ising.shape[0]
    assert problem_mat_ising.shape[0] == n_spins, "Row count of states_ising must match the number of spins"

    # To handle single states correctly.
    if states_ising.ndim == 1:
        states_ising.shape = (n_spins, 1)

    h = numpy.diag(problem_mat_ising)
    J = problem_mat_ising - numpy.diag(h)
    
    h.shape = (n_spins, 1)

    # Sum over the spin axis, which is always second from last.
    energies = numpy.sum(h * states_ising + states_ising * (J @ states_ising), -2) 

    # Flip the axis back for 3D case.
    if energies.ndim == 2: 
        energies = numpy.transpose(energies)

    return energies

def probmat_qubo_to_ising(problem_mat_qubo: numpy.ndarray) -> numpy.ndarray:
    """
    ### Converts a QUBO problem to the equivalent Ising problem. 

    ## Parameters:
    - `problem_mat_qubo` - A symmetric real-valued NxN matrix representing a QUBO problem
    
    ## Returns:
    - `problem_mat_ising` - A matrix of size NxN representing the ising problem with the linear part on its diagonal
    - `ising_offset` - The energy offset between the problems
    """
    n_spins = problem_mat_qubo.shape[0]

    # Off diagonal elements
    problem_mat_ising = 0.25 * problem_mat_qubo
    
    # Diagonal elements
    col_sum = numpy.sum(problem_mat_qubo, axis=0)
    problem_mat_ising[numpy.diag_indices(n_spins)] = 0.5 * col_sum
    
    # Offset
    diag_sum = numpy.sum(numpy.diag(problem_mat_qubo))
    linear_offset = 0.5 * diag_sum
    quad_offset = 0.25 * problem_mat_qubo.sum() - 0.25 * diag_sum

    ising_offset = linear_offset + quad_offset
    return problem_mat_ising, ising_offset

def probmat_ising_to_qubo(problem_mat_ising: numpy.ndarray) -> numpy.ndarray:
    """
    ### Converts an Ising problem to the equivalent QUBO problem. 

    ## Parameters:
    - `problem_mat_ising` - The symmetric real-valued NxN matrix representing the Ising problem, with the linear part on its diagonal
    
    ## Returns:
    - `problem_mat_qubo` - The NxN QUBO matrix representing the problem
    - `offset` - The energy offset between the problems
    """
    n_spins = problem_mat_ising.shape[0]
    h = numpy.diag(problem_mat_ising)
    J = problem_mat_ising - numpy.diag(h)
    col_sum = numpy.sum(J, axis=0)
    
    # Off-diagonal elements
    problem_mat_qubo = 4 * J
    
    # Diagonal elements
    problem_mat_qubo[numpy.diag_indices(n_spins)] = -4 * col_sum + 2 * h
    
    # Offset
    offset = numpy.sum(J) - numpy.sum(h)
    
    return problem_mat_qubo, offset

def create_random_initial_states(num_lasers: int, num_states: int, radius_scale_factor: float = 1, seed: int = -1) -> numpy.ndarray:
    """
    ### Creates an array of initial states of size num_lasers * num_states

    ## Parameters:
    - `num_lasers`          - Number of lasers
    - `num_states `         - Number of initial states
    - `radius_scale_factor` - The amplitude scale of the states
    - `seed`                - Seed for reproducibility. -1 - randpm seed
    
    ## Returns:
    - `init_states` - Numpy array of size num_lasers * num_states represensting random laser states
    """
    if seed >= 0:
        rng = numpy.random.default_rng(seed=seed)
        phases = rng.random((num_lasers, num_states))
        amplitudes = radius_scale_factor * rng.random((num_lasers, num_states))
    else:
        phases = numpy.random.rand(num_lasers, num_states)
        amplitudes = radius_scale_factor * numpy.random.rand(num_lasers, num_states)
    init_states = amplitudes * numpy.exp(2j * numpy.pi * phases)
    return init_states

def best_energy_search_xy(state: numpy.ndarray, probmat_ising: numpy.ndarray) -> tuple[numpy.ndarray, float]:
    """
    ### Find the best assignment of spin values (+-1) to lasers, in terms of minimal Ising energy.

    ## Parameters:
    - `state`   - numpy.ndarray of size (N, ) representing the lasers' state (N - Number of lasers)
    - `probmat` - numpy.ndarray of size NxN representing the Ising problem with the external field on the diagonal
    
    ## Returns:
    - `best_state_ising` - Ising state with the minimal Ising energy
    - `best_energy` - Energy of the best state
    """
    is_single_state = len(state.shape) == 1
    # if is_single_state:
    #     state = state.reshape(-1, 1)    
        
    n_lasers = state.shape[0]
    n_states = 1 if is_single_state else state.shape[1]
    
    # Without external field
    if n_lasers == probmat_ising.shape[0]:
        state = state / numpy.abs(state)
        
        idx_states = range(n_states)
        
        # Initialize best state (division relative to the x axis)
        theta = numpy.angle(state) % (2 * numpy.pi)
        is_above_x_axis = numpy.logical_and(theta <= numpy.pi, theta > 0)
        best_state_temp = numpy.zeros_like(state, dtype=numpy.int32)
        best_state_temp[is_above_x_axis] = -2
        best_state_temp = best_state_temp + 1
        
        temp = numpy.sign(numpy.imag(state)) * numpy.real(state)
        ind_sort = numpy.argsort(temp, axis=0)
        
        # Off diagonal elements
        J = probmat_ising.copy()
        numpy.fill_diagonal(J, 0)
        
        # External field
        h = numpy.diag(probmat_ising)
        
        energy_diff = numpy.zeros((2 * n_lasers, n_states))
        energy_diff[0, :] = calc_ising_energies(probmat_ising, best_state_temp)

        for i, k in zip(reversed(range(1, 2 * n_lasers)), range(1, 2 * n_lasers)):
            # Flip current spin sign for each state
            rows_to_flip = [ind_sort[i % n_lasers, idx_state] for idx_state in idx_states]
            best_state_temp[rows_to_flip, idx_states] = -1 * best_state_temp[rows_to_flip, idx_states]  
                 
            # Calc energy difference caused by the flip
            energy_diff[k, :] = 4 * (J[rows_to_flip] * best_state_temp.T).sum(axis=1) * best_state_temp[rows_to_flip, idx_states] \
                 + 2 * h[rows_to_flip] * best_state_temp[rows_to_flip, idx_states]

        energy_cumsum = numpy.cumsum(energy_diff, axis=0)
        idx_min_energy = numpy.argmin(energy_cumsum, axis=0)

        # Reflip to get best state
        for s in idx_states:  
            for i in range(1, 2 * n_lasers - idx_min_energy[s]): 
                best_state_temp[ind_sort[i % n_lasers, s], s] = -1 * best_state_temp[ind_sort[i % n_lasers, s], s]

    # With external field
    elif len(state) == probmat_ising.shape[0] + 1:  
        n_lasers = n_lasers - 1
        state = state / numpy.abs(state)
        state = state / state[0]
        
        idx_states = range(n_states)
        
        # Initialize best state (division relative to the x axis)
        theta = numpy.angle(state[1:]) % (2 * numpy.pi)
        is_above_x_axis = numpy.logical_and(theta <= numpy.pi, theta > 0)
        best_state_temp = numpy.zeros_like(state[1:], dtype=numpy.int32)
        best_state_temp[is_above_x_axis] = -2
        best_state_temp = best_state_temp + 1

        # to order them all along the projected imaginary axis
      
        temp = numpy.sign(numpy.imag(state[1:])) * numpy.real(state[1:])
        ind_sort = numpy.argsort(temp, axis=0)
        
        # Off diagonal elements
        J = probmat_ising.copy()
        numpy.fill_diagonal(J, 0)
        
        # External field
        h = numpy.diag(probmat_ising)
        
        energy_diff = numpy.zeros((2 * n_lasers, n_states))
        energy_diff[0, :] = calc_ising_energies(probmat_ising, best_state_temp)

        for i, k in zip(reversed(range(1, 2 * n_lasers)), range(1, 2 * n_lasers)):
            # Flip current spin sign for each state
            rows_to_flip = [ind_sort[i % n_lasers, idx_state] for idx_state in idx_states]
            best_state_temp[rows_to_flip, idx_states] = -1 * best_state_temp[rows_to_flip, idx_states]  
                 
            # Calc energy difference caused by the flip
            energy_diff[k, :] = 4 * (J[rows_to_flip] * best_state_temp.T).sum(axis=1) * best_state_temp[rows_to_flip, idx_states] \
                 + 2 * h[rows_to_flip] * best_state_temp[rows_to_flip, idx_states]
                 
        energy_cumsum = numpy.cumsum(energy_diff, axis=0)
        idx_min_energy = numpy.argmin(energy_cumsum, axis=0)

        # Reflip to get best state
        for s in idx_states:  
            for i in range(1, 2 * n_lasers - idx_min_energy[s]): 
                best_state_temp[ind_sort[i % n_lasers, s], s] = -1 * best_state_temp[ind_sort[i % n_lasers, s], s]
    
    else:
        print("state and probmat do not match in dimensions")    

    best_energy = numpy.min(energy_cumsum, axis=0)
    return best_state_temp, best_energy 

def calc_ising_energy_from_states(probmat_ising: numpy.ndarray, states: numpy.ndarray) -> numpy.ndarray:
    """
    ### Calculate the energies of a given Ising model and phasor states

    ## Parameters:
    - `probmat_ising` - numpy.ndarray of size N*N representing the Ising problem (N - Number of lasers)
    - `states` - numpy.ndarray of the laser states of size N * n_states
    
    ## Returns:
    - `states_ising` - numpy.array of size (N, n_states) consisting of the Ising states with the best energy
    - `energy` - numpy.array of size (n_states, ) consisting of the best Ising energy for each state
    """
    if len(states.shape) == 1:
        states = states.reshape(-1, 1)
        
    n_states = states.shape[1]
    
    # Find state with best energy for each of the states
    states_ising, energy = zip(*[best_energy_search_xy(states[:, idx_state], probmat_ising) for idx_state in range(n_states)])
    
    states_ising = numpy.array(states_ising).T
    energy = numpy.array(energy)
    return states_ising, energy


class XYModelParams():

    """
    ## Parameters for converting an Ising model to the XY model.

    ### Members:
    - `alphaR` - Reference lasers self coupling strength - default: 0.7
    - `alphaI` - Vortex lasers self coupling strength - default: 0.7
    - `coupAmp` - Coupling amplitude, default: 0.3
    - `exFieldCoup` - Coupling of external field, default : 0.3
    """
    def __init__(self, alphaR = 0.7, alphaI = 0.7, coupAmp = 0.3, exFieldCoup = 0.3):
        self.alphaI = alphaI
        self.alphaR = alphaR
        self.coupAmp = coupAmp
        self.exFieldCoup = exFieldCoup

    def __eq__(self, other: object) -> bool:
        return isinstance(other, XYModelParams) and \
            self.alphaI == other.alphaI and \
            self.alphaR == other.alphaR and \
            self.coupAmp == other.coupAmp and \
            self.exFieldCoup == other.exFieldCoup
    

def coupling_matrix_xy(problem_matrix:numpy.ndarray, modelParamsXY:XYModelParams, external_field=False):
    """
    ## Generate the coupling matrix for the XY model.

    ### Parameters:
    - `problem_matrix` - The input problem matrix
    - `modelParamsXY` - The parameters for the XY model
    - `external_field` - Whether to include an external field in the model

    ### Returns:
    - `coupling_matrix` - The generated coupling matrix for the XY model
    """
    N = problem_matrix.shape[0]
    max_sum_rows = numpy.max(numpy.sum(numpy.abs(problem_matrix), axis=1))

    if external_field:
        coupling_matrix = numpy.zeros((N + 1, N + 1), dtype=numpy.float32)
        coupling_matrix[1:, 1:] = - (problem_matrix / max_sum_rows) * modelParamsXY.coupAmp
        numpy.fill_diagonal(coupling_matrix[1:, 1:], modelParamsXY.alphaI)
        coupling_matrix[1:, 0] = - modelParamsXY.exFieldCoup * (problem_matrix.diagonal() / max_sum_rows)
        coupling_matrix[0, 0] = modelParamsXY.alphaR
    
    else:
        coupling_matrix = numpy.zeros((N, N), dtype=numpy.float32)
        coupling_matrix = - (problem_matrix / max_sum_rows) * modelParamsXY.coupAmp
        numpy.fill_diagonal(coupling_matrix, modelParamsXY.alphaI)

    return coupling_matrix