

import numpy as np
import scipy.linalg


def GMRES(A, b, x_0, n_rst, resid_relative_tolerance=1e-8, max_restarts=600, print_progress=True):
    
    """An implementation of the GMRES algorithm to solve a linear system
    of equations given by Ax = b. We will solve this by building a Krylov subspace
    starting with the initial residual. That is, we get the Krylov subspace given as
    
        K_n(A, r_0) = <r_0, A*r_0, A^2 * r_0, ..., A^{n-1} * r_0>
    
    We build this subspace using Arnoldi's method and then look for the solution that
    minimizes the L2 norm of the residual. 
    
    NOTE: This method has not been optimized in implementation for computational time. We
        can get extra savings by using Housholder's method to get a running QR factorization
        of our Hessnberg matrix as we proceed, however for simplicity this has not been 
        impelemented.
    
    Args:
        A (numpy array-like): LHS Matrix. Should be of dimension [m x m]
        b (numpy array-like): Description. Should be of dimension [m x 1]
        x_0 (numpy array-like, optional): The initial guess
        n_rst (int): Number of iterations per restart cycle
        resid_relative_tolerance (float, optional): The tolerance at which to exit the solver
            measured as ||r_n||_2/||r_0||_2.
        max_restarts (int, optional): Maximum number of restart iterations
        print_progress (bool, optional): Whether or not to print the progress
    
    Returns:
        Tuple: rel_res_convergence (numpy array-like) = Holds the residual norm at the ith restart step.
                The 0th index holds the initial values
            x_n (numpy array-like) = Of dimension [m x 1] for the estimated solution at
                the end of the GMRES process
            extra_data_dict (dict) = Dictionary of extra parameters for testing
    
    Raises:
        ValueError: Description
    """


    # =========================================
    #                  Setup
    # =========================================

    # The dimension of the problem
    m = A.shape[0]

    # Matrix structure to hold the orthonormal vectors for the Krylov space
    # formed using the Arnoldi algorithm. This holds the RHS in the relationships
    #
    #       A * V_n = V_nPlus1 * H_tilde_n
    # 
    # Here, we look to store the [m x (n+1)] matrix V_nPlus1.

    V_nPlus1 = np.zeros((m, n_rst+1))

    # The Hessenberg matrix of dimension [(n+1) x n] generated using Arnoldi's Method
    H_tilde_n = np.zeros((n_rst + 1, n_rst))

    # Initial residual
    r_init = b - np.dot(A,x_0)
    r_init_L2_norm = np.linalg.norm(r_init)

    # =========================================
    #               Main Solver
    # =========================================

    restart_iteration_count = 0

    rel_res_convergence = [1.]

    res_vecs_rst_convergence = [r_init]

    while True:

        # ------------------------------------
        #           Arnoldi's Method
        # ------------------------------------

        # Use the Arnoldi algorithm now to get an orthonormal basis for the
        # Krylov subspace of dimension n_rst

        r0 = b - np.dot(A,x_0)
        beta = np.linalg.norm(r0)
        v0 = r0/beta

        # Setup the first iteration by loading in the initial data to use
        V_nPlus1[:,0] = v0[:,0]

        for j in range(n_rst):

            w_j = np.dot(A, V_nPlus1[:,j])

            # Modified Gram-Schmidt Process

            for i in range(j+1):
                h_ij = np.dot(w_j, V_nPlus1[:,i])
                H_tilde_n[i,j] = h_ij

                w_j = w_j - h_ij*V_nPlus1[:,i]
            

            h_jPlus1j = np.linalg.norm(w_j)

            if h_jPlus1j < 1e-13:

                # We found a w_j that is in the space of the previous orthonormal
                # vectors in the Krylov space so far. 

                raise ValueError('To Be Implemented')

            H_tilde_n[j+1,j] = h_jPlus1j
            V_nPlus1[:,j+1] = w_j/h_jPlus1j


        # ------------------------------------
        #         Least Squares Solve
        # ------------------------------------

        # Obtain the esimate of the solution by solving the least squares sytem

        # Create the e1 vector
        e1 = np.zeros((n_rst + 1, 1))
        e1[0,0] = 1

        # Setup the least square system solve given by
        #
        #       min_x || b_lsq - A_lsq * x ||_2
        #
        # where A_lsq is the Hessenberg matrix
 
        b_lsq = beta*e1

        # Perform the least squares solve
        lsq_solve_results = scipy.linalg.lstsq(H_tilde_n, b_lsq)

        y_lsq = lsq_solve_results[0]

        # Compute the estimate up until now for the optimal solution in 
        # the Krylov subspace

        x_n = x_0 + np.dot(V_nPlus1[:,0:-1], y_lsq)


        # ------------------------------------
        #           Exit Conditions
        # ------------------------------------

        r_n = b - np.dot(A, x_n)
        r_n_L2_norm = np.linalg.norm(r_n)

        # Store the relative residual at the end of this restart iteration
        rel_res_convergence.append(r_n_L2_norm/r_init_L2_norm)  

        res_vecs_rst_convergence.append(r_n)

        if r_n_L2_norm/r_init_L2_norm < resid_relative_tolerance:
            break

        if print_progress:
            if restart_iteration_count % 25 == 0:
                print("Iter : %d,  Relative Residual Norm : %e" % (restart_iteration_count, r_n_L2_norm/r_init_L2_norm))

        restart_iteration_count += 1

        if restart_iteration_count >= max_restarts:
            break

        # ------------------------------------
        #           Restart Iteration
        # ------------------------------------
        x_0 = x_n

    rel_res_convergence = np.array(rel_res_convergence)

    return (rel_res_convergence, x_n, 
        {'res_vecs_rst_convergence' : res_vecs_rst_convergence,
         'r_init_L2_norm' : r_init_L2_norm})



def LGMRES(A, b, x_0, n_rst, k_err, resid_relative_tolerance=1e-8, max_restarts=600, print_progress=True):
    
    """An implementation of the LGMRES algorithm to solve a linear system
    of equations given by Ax = b. This method accelerates the convergence of GMRES in 
    cases where every second residual in the standard GMRES algorithm is approximately
    in the same direction.
    
    NOTE: This method has not been optimized in implementation for computational time. We
        can get extra savings by using Housholder's method to get a running QR factorization
        of our Hessnberg matrix as we proceed, however for simplicity this has not been 
        impelemented.
    
    Args:
        A (numpy array-like): LHS Matrix. Should be of dimension [m x m]
        b (numpy array-like): Description. Should be of dimension [m x 1]
        x_0 (numpy array-like, optional): The initial guess
        n_rst (int): Number of iterations per restart cycle
        k_err (int): The number of residual difference terms to append to the Krylov space
        resid_relative_tolerance (float, optional): The tolerance at which to exit the solver
            measured as ||r_n||_2/||r_0||_2.
        max_restarts (int, optional): Maximum number of restart iterations
        print_progress (bool, optional): Whether or not to print the progress
    
    Returns:
        Tuple: rel_res_convergence (numpy array-like) = Holds the residual norm at the ith restart step.
                The 0th index holds the initial values
            x_n (numpy array-like) = Of dimension [m x 1] for the estimated solution at
                the end of the GMRES process 
    
    Raises:
        ValueError: Description
    """

    # =========================================
    #                  Setup
    # =========================================

    # The dimension of the problem
    m = A.shape[0]

    # The augmented Krylov space will be of dimension n_rst + k_err
    s = n_rst + k_err

    # Matrix structure to hold the orthonormal vectors for the Krylov space
    # formed using the Arnoldi algorithm. This holds the RHS in the relationships
    #
    #       A * W_s = V_sPlus1 * H_tilde_s
    # 
    # NOTE: This also includes the augmented residual difference terms added to the space.


    V_sPlus1 = np.zeros((m, s+1))
    W_s = np.zeros((m, s))

    # The Hessenberg matrix of dimension [(s+1) x s] generated using Arnoldi's Method
    H_tilde_s = np.zeros((s + 1, s))

    # List to hold the differences (we will access the k latest ones in the augentation process)
    Z_vals = []

    # Initial residual
    r_init = b - np.dot(A,x_0)
    r_init_L2_norm = np.linalg.norm(r_init)

    # =========================================
    #               Main Solver
    # =========================================

    i_rst_count = 0

    rel_res_convergence = [1.]

    res_vecs_rst_convergence = [r_init]

    while True:

        # ------------------------------------
        #           Arnoldi's Method
        # ------------------------------------

        # Use the Arnoldi algorithm now to get an orthonormal basis for the
        # Krylov subspace of dimension n_rst

        r0 = b - np.dot(A,x_0)
        beta = np.linalg.norm(r0)
        v0 = r0/beta

        # The effective dimension of the system. Used as we will only
        # take a subset of the basis if we fill the subspace early
        s_effective = 0

        # Setup the first iteration by loading in the initial data to use
        V_sPlus1[:,0] = v0[:,0]

        for j in range(s):

            # Build the augmented Krylov subspace
            if j < n_rst:
                # Save the vector for the W
                W_s[:,j] = V_sPlus1[:,j]
                
                # Create the vector to start the Arnoldi method
                w_j = np.dot(A, V_sPlus1[:,j])
                s_effective += 1

            else:
                k = i_rst_count - (j - n_rst) - 1

                if k < 0:
                    # If we do not have enough vectors yet, then we are done
                    w_j = None
                
                else:
                    # Save the vector for the W
                    z_j = Z_vals[k]
                    W_s[:,j] = z_j[:,0]

                    # Create the vector to start the Arnoldi method
                    w_j = np.dot(A, z_j[:,0])
                    s_effective += 1

            if w_j is None:
                # No more vectors to add to the Krylov space, so we are done 
                # and now must do the least squares solve
                break

                      
            # Modified Gram-Schmidt Process

            for i in range(j+1):
                h_ij = np.dot(w_j, V_sPlus1[:,i])
                H_tilde_s[i,j] = h_ij

                w_j = w_j - h_ij*V_sPlus1[:,i]
            

            h_jPlus1j = np.linalg.norm(w_j)

            if h_jPlus1j < 1e-13:

                # We found a w_j that is in the space of the previous orthonormal
                # vectors in the Krylov space so far. 

                raise ValueError('To Be Implemented')

            H_tilde_s[j+1,j] = h_jPlus1j
            V_sPlus1[:,j+1] = w_j/h_jPlus1j


        # ------------------------------------
        #         Least Squares Solve
        # ------------------------------------

        if s_effective == s:
            
            # No need to truncate the spaces as we filled the whole
            # s dimensional space

            V_hat_sPlus1 = V_sPlus1
            H_hat_tilde_s = H_tilde_s
            W_hat_tilde_s = W_s

        else:

            # Get the relevant subset of the matrices to work with
            
            V_hat_sPlus1 = V_sPlus1[:, 0:(s_effective+1)]
            H_hat_tilde_s = H_tilde_s[0:(s_effective+1), 0:s_effective]
            W_hat_tilde_s = W_s[:, 0:s_effective]


        # Obtain the esimate of the solution by solving the least squares sytem

        # Create the e1 vector
        e1 = np.zeros((s_effective + 1, 1))
        e1[0,0] = 1

        # Setup the least square system solve given by
        #
        #       min_x || b_lsq - H_lsq * x ||_2
        #
        # where A_lsq is the Hessenberg matrix
 
        b_lsq = beta*e1

        # Perform the least squares solve
        lsq_solve_results = scipy.linalg.lstsq(H_hat_tilde_s, b_lsq)

        y_lsq = lsq_solve_results[0]

        # Compute the estimate up until now for the optimal solution in 
        # the Krylov subspace

        z_n = np.dot(W_hat_tilde_s, y_lsq)
        x_n = x_0 + z_n

        Z_vals.append(z_n)

        # ------------------------------------
        #           Exit Conditions
        # ------------------------------------

        r_n = b - np.dot(A, x_n)
        r_n_L2_norm = np.linalg.norm(r_n)

        # Store the relative residual at the end of this restart iteration
        rel_res_convergence.append(r_n_L2_norm/r_init_L2_norm)  

        res_vecs_rst_convergence.append(r_n)

        if r_n_L2_norm/r_init_L2_norm < resid_relative_tolerance:
            break

        if print_progress:
            if i_rst_count % 25 == 0:
                print("Iter : %d,  Relative Residual Norm : %e" % (i_rst_count, r_n_L2_norm/r_init_L2_norm))

        i_rst_count += 1

        if i_rst_count >= max_restarts:
            break

        # ------------------------------------
        #           Restart Iteration
        # ------------------------------------
        x_0 = x_n

    rel_res_convergence = np.array(rel_res_convergence)

    return (rel_res_convergence, x_n, {'res_vecs_rst_convergence' : res_vecs_rst_convergence})



def GMRES_E(A, b, x_0, n_rst, k_eig, resid_relative_tolerance=1e-8, max_restarts=600, print_progress=True):
    
    """An implementation of the GMRES-E algorithm to solve a linear system
    of equations given by Ax = b. This method accelerates the convergence of GMRES
    for some cases by appending approximate eigenvectors to the subspace 
    
    
    NOTE: This method has not been optimized in implementation for computational time. We
        can get extra savings by using Housholder's method to get a running QR factorization
        of our Hessnberg matrix as we proceed, however for simplicity this has not been 
        impelemented.
    
    Args:
        A (numpy array-like): LHS Matrix. Should be of dimension [m x m]
        b (numpy array-like): Description. Should be of dimension [m x 1]
        x_0 (numpy array-like, optional): The initial guess
        n_rst (int): Number of iterations per restart cycle
        k_eig (int): The number of eigenvector approximations to append to the Krylov space
        resid_relative_tolerance (float, optional): The tolerance at which to exit the solver
            measured as ||r_n||_2/||r_0||_2.
        max_restarts (int, optional): Maximum number of restart iterations
        print_progress (bool, optional): Whether or not to print the progress
    
    Returns:
        Tuple: rel_res_convergence (numpy array-like) = Holds the residual norm at the ith restart step.
                The 0th index holds the initial values
            x_n (numpy array-like) = Of dimension [m x 1] for the estimated solution at
                the end of the GMRES process 
    
    Raises:
        ValueError: Description
    """

    # =========================================
    #                  Setup
    # =========================================

    # The dimension of the problem
    m = A.shape[0]

    # The augmented Krylov space will be of dimension n_rst + k_eig (the max size)
    s = n_rst + k_eig

    # Matrix structure to hold the orthonormal vectors for the Krylov space
    # formed using the Arnoldi algorithm. This holds the RHS in the relationships
    #
    #       A * W_s = V_sPlus1 * H_tilde_s
    # 
    # NOTE: These matrices will be created to be able to store the max dimension possible.
    #   In the first iteration however, we will run just a simple GMRES solve though so
    #   when that happens we will take relevant sub-matrices to work with.

    V_sPlus1 = np.zeros((m, s+1))
    W_s = np.zeros((m, s))

    # The Hessenberg matrix of dimension [(s+1) x s] generated using Arnoldi's Method
    H_tilde_s = np.zeros((s + 1, s))

    # List to hold the approximations to use to augment the space. Only k_eig eigenvectors
    # will be taken from here at a time and, if it is empty, no eigenvector will be used.
    eigenvecs_approx_vals = []

    # Initial residual
    r_init = b - np.dot(A,x_0)
    r_init_L2_norm = np.linalg.norm(r_init)

    # =========================================
    #               Main Solver
    # =========================================

    i_rst_count = 0

    rel_res_convergence = [1.]

    res_vecs_rst_convergence = [r_init]

    eigenvals_convergence = []

    eigenvecs_convergence = []

    while True:

        # ------------------------------------
        #           Arnoldi's Method
        # ------------------------------------

        # Use the Arnoldi algorithm now to get an orthonormal basis for the
        # Krylov subspace of dimension n_rst

        r0 = b - np.dot(A,x_0)
        beta = np.linalg.norm(r0)
        v0 = r0/beta

        # The effective dimension of the system. Used to determine what the dimension
        # of the space is based on how many eigenvectors were appended.
        s_effective = 0

        # Setup the first iteration by loading in the initial data to use
        V_sPlus1[:,0] = v0[:,0]

        for j in range(s):
            # Note that we loop over a max of s vectors (this allows n_rst
            # Krylov subspace vectors and a max of k_eig eigenvectors). At
            # a minimum the space will be of dimension n_rst (which should only
            # happen on the first iteration).

            # Build the augmented Krylov subspace
            if j < n_rst:
                
                # Save the vector for the W
                W_s[:,j] = V_sPlus1[:,j]
                
                # Create the vector to start the Arnoldi method
                w_j = np.dot(A, V_sPlus1[:,j])
                s_effective += 1

            else:

                # Augment the space using the eigenvectors
                if len(eigenvecs_approx_vals) != 0:

                    y_j = eigenvecs_approx_vals.pop()

                    # The eigenvectors are stored as arrays so reshape into
                    # column vectors to be consistent with the other formulation
                    y_j = y_j.reshape((-1, 1))

                    W_s[:,j] = y_j[:,0]

                    # Create the vector to start the Arnoldi method
                    w_j = np.dot(A, y_j[:,0])
                    s_effective += 1

                else:
                    # No more eigenvectors to append
                    w_j = None
                    

            if w_j is None:
                # No more vectors to add to the Krylov space, so we are done 
                # and now must do the least squares solve
                break

                      
            # Modified Gram-Schmidt Process

            for i in range(j+1):
                h_ij = np.dot(w_j, V_sPlus1[:,i])
                H_tilde_s[i,j] = h_ij

                w_j = w_j - h_ij*V_sPlus1[:,i]
            

            h_jPlus1j = np.linalg.norm(w_j)

            if h_jPlus1j < 1e-13:

                # We found a w_j that is in the space of the previous orthonormal
                # vectors in the Krylov space so far. In this case, we don't want
                # to use w_j in the subspace generation process and can stop the Arnoldi
                # iteration (as we have a full description of the space)

                import pdb
                pdb.set_trace()

                raise ValueError("To Be Implemented")

                s_effective -= 1
                break
                print("Small h_jPlus1j : %e" % (h_jPlus1j))


            H_tilde_s[j+1,j] = h_jPlus1j
            V_sPlus1[:,j+1] = w_j/h_jPlus1j


        # ------------------------------------
        #         Least Squares Solve
        # ------------------------------------

        if s_effective == 0:
            raise ValueError("Empty Krylov Space")

        if s_effective == s:
            
            # No need to truncate the spaces as we filled the whole
            # s dimensional space

            V_hat_sPlus1 = V_sPlus1
            H_hat_tilde_s = H_tilde_s
            W_hat_tilde_s = W_s

        else:

            # Get the relevant subset of the matrices to work with
            
            V_hat_sPlus1 = V_sPlus1[:, 0:(s_effective+1)]
            H_hat_tilde_s = H_tilde_s[0:(s_effective+1), 0:s_effective]
            W_hat_tilde_s = W_s[:, 0:s_effective]


        # Obtain the esimate of the solution by solving the least squares sytem

        # Create the e1 vector
        e1 = np.zeros((s_effective + 1, 1))
        e1[0,0] = 1

        # Setup the least square system solve given by
        #
        #       min_x || b_lsq - H_lsq * x ||_2
        #
        # where A_lsq is the Hessenberg matrix
 
        b_lsq = beta*e1

        # Perform the least squares solve
        lsq_solve_results = scipy.linalg.lstsq(H_hat_tilde_s, b_lsq)

        y_lsq = lsq_solve_results[0]

        # Compute the estimate up until now for the optimal solution in 
        # the Krylov subspace
        x_n = x_0 + np.dot(W_hat_tilde_s, y_lsq)


        # ------------------------------------
        #         Eigenvector Solve
        # ------------------------------------

        # Compute the F and G matrix (note we do the full computation here for now
        # rather than the shortened version using the QR decomposition)        
        F = (W_hat_tilde_s.T).dot(A).dot(W_hat_tilde_s)

        G = (W_hat_tilde_s.T).dot(W_hat_tilde_s)

        # Solve the generalized eigenvalue problem given by
        #      G * g_i = theta_i * F * g_i =

        (eig_vals_approx, eig_vecs_approx) = scipy.linalg.eig(G, F)

        eigenvals_convergence.append(1./eig_vals_approx)

        # Get the k_eig eigenvectors to use to append to the Krylov space for the 
        # future iterations
        eigenvecs_approx_vals = []

        for i in range(s_effective):
            v_i = eig_vecs_approx[:,i]

            # Check if v_i is complex

            if np.sum(np.iscomplex(v_i)) == 0:                
                # We have a real vector
                eigenvecs_approx_vals.append(np.dot(W_hat_tilde_s, v_i.real))

            else:
                # We have a complex vector, so add the real and complex
                # part separately
                eigenvecs_approx_vals.append(np.dot(W_hat_tilde_s, v_i.real))
                eigenvecs_approx_vals.append(np.dot(W_hat_tilde_s, v_i.imag))

            # Stop the process if we have enough eigenvectors
            if len(eigenvecs_approx_vals) >= k_eig:
                break

        eigenvecs_convergence.append(eigenvecs_approx_vals.copy())

        # ------------------------------------
        #           Exit Conditions
        # ------------------------------------

        r_n = b - np.dot(A, x_n)
        r_n_L2_norm = np.linalg.norm(r_n)

        # Store the relative residual at the end of this restart iteration
        rel_res_convergence.append(r_n_L2_norm/r_init_L2_norm)  

        res_vecs_rst_convergence.append(r_n)

        if r_n_L2_norm/r_init_L2_norm < resid_relative_tolerance:
            break

        if print_progress:
            if i_rst_count % 25 == 0:
                print("Iter : %d,  Relative Residual Norm : %e" % (i_rst_count, r_n_L2_norm/r_init_L2_norm))

        i_rst_count += 1

        if i_rst_count >= max_restarts:
            break

        # ------------------------------------
        #           Restart Iteration
        # ------------------------------------
        x_0 = x_n

    rel_res_convergence = np.array(rel_res_convergence)

    return (rel_res_convergence, x_n, 
        {'res_vecs_rst_convergence' : res_vecs_rst_convergence,
         'r_init_L2_norm' : r_init_L2_norm,
         'eigenvecs_approx_vals' : eigenvecs_approx_vals, 
         'eig_vals_approx' : eig_vals_approx,
         'eigenvals_convergence' : eigenvals_convergence,
         'eigenvecs_convergence' : eigenvecs_convergence})



def matrix_normality_metric(A):
    
    """A metric for how normal a matrix is. We will simply get the L2 norm
    and then find effectively what the average squared value in the matrix is
    of the difference 
    
        D = A_T*T - A*A_T

    We will compute the Frobenius norm of the matrix D as ||D||_F. The average
    matrix entry will then be found as

        l = sqrt{ ||D||_F^2 / size(D) }
    
    l the average value in the matrix. 

    Args:
        A (numpy array-like): The matrix of interest that we want to check
            the normality property.
    
    Returns:
        float: The metric for normality
    """

    diff_mat = A.dot(A.T) - (A.T).dot(A)
    mat_norm = np.linalg.norm(diff_mat)

    return np.sqrt((mat_norm**2.)/A.size)


def residual_angle_convergence(res_vec_conv_list):
    
    """Compute the sequential and skip angles using the residual vectors from
    convergence.
    
    Args:
        res_vec_conv_list (list): List of residual vectors (which are numpy arrays)
    
    Returns:
        (seq_angles, skip_angles): A tuple of numpy arrays, where each array is the
            sequence of sequential angles and skip angles of the residuals.
    """
    

    # Sequential Angle Values
    seq_angles = []

    for i in range(1, len(res_vec_conv_list)):
        
        r_i = res_vec_conv_list[i]
        r_iMin1 = res_vec_conv_list[i-1]
        
        # Compute the cosine of the angle using the dot product
        cos_theta = np.dot(r_i[:,0], r_iMin1[:,0])/(np.linalg.norm(r_i) * np.linalg.norm(r_iMin1))
        
        # Compute the angle between the vectors    
        theta_rad = np.arccos(cos_theta)
        theta_deg = (180./np.pi)*theta_rad
        
        seq_angles.append(theta_deg)
        
    seq_angles_np = np.array(seq_angles)


    # Skip Angle Values
    skip_angles = []

    for i in range(2, len(res_vec_conv_list)):
        
        r_i = res_vec_conv_list[i]
        r_iMin2 = res_vec_conv_list[i-2]
        
        # Compute the cosine of the angle using the dot product
        cos_theta = np.dot(r_i[:,0], r_iMin2[:,0])/(np.linalg.norm(r_i) * np.linalg.norm(r_iMin2))
        
        # Compute the angle between the vectors
        theta_rad = np.arccos(cos_theta)
        theta_deg = (180./np.pi)*theta_rad
        
        skip_angles.append(theta_deg)

    skip_angles_np = np.array(skip_angles)


    return (seq_angles_np, skip_angles_np)
















