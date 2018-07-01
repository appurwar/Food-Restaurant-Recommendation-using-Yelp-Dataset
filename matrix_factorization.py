#Non-negative matrix factorization function
def matrix_factorization(b, b_u, b_i, P, Q, K, steps, alpha, beta, samples, samples_test):
    error_arr = []
    error_min = 1000000000
    for step in range(steps):
        for i, j, r in samples: 
#eij is the predicted value for ith user and jth movie
            eij = r - b - b_u[i] - b_i[j] - np.dot(P[i,:],Q[j,:])
            b_u[i] += alpha * (eij - beta * b_u[i])
            b_i[j] += alpha * (eij - beta * b_i[j])
            P[i,:] += alpha * (eij * Q[j,:] - beta * P[i,:])
            Q[j,:] += alpha * (eij * P[i,:] - beta * Q[j,:])
#         print('step : '+str(step)+' complete') 
#RMSE Calculation below
        e = 0
        for i, j, r in samples_test:
            e = e + pow(r - np.dot(P[i,:],Q[j,:]) - b - b_i[j] - b_u[i], 2)
#         print('Root mean squared error on test : ' + str(np.sqrt(e/len(samples_test))))
        error_arr.append(e)      
        if e > error_min:
            break
        else:
            error_min = e
    return b, b_u, b_i, P, Q, np.sqrt(e/len(samples_test))