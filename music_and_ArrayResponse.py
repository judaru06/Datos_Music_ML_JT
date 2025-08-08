A_res = lambda D, ang: np.exp(-1j*(D*2*np.pi)* np.cos(np.deg2rad(ang))).T

def MUSIC (Rxx, signal_dim, sv):
    M = np.size(Rxx, 0)
    sigmai, vi = lin.eig(Rxx)
    eig_array = []
    for i in range(M):
            eig_array.append([np.abs(sigmai[i]),vi[:,i]])
    eig_array = sorted(eig_array, key=lambda eig_array: eig_array[0], reverse=False)

    signal_dim = D
    noise_dim = M - signal_dim 
    ## The eigenvectors corresponding to the positive eigenvalues span the signal subspace

    E = np.zeros((M,noise_dim),dtype=complex)
    #E = []

    for i in range(int(noise_dim)):
        E[:,i] = np.array(eig_array[i][1].flatten()) #eig_array[1][1]
    #np.array(E).reshape(10,8)
    E[:,i] = np.array(eig_array[1][1].flatten())
    E = np.matrix(E)
    #print(E.shape)

    P_music = np.zeros(np.size(sv, 1),dtype=complex)
    theta_i=0
    for i in range(np.size(sv, 1)):
        S_theta_ = sv[:, i]
        S_theta_  = np.matrix(S_theta_)
        P_music[theta_i]=  (1/np.abs((S_theta_.getH()*E*E.getH()*S_theta_)))[0,0]
        #print(theta_i)
        theta_i += 1
        
    return P_music