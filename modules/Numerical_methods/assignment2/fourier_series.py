def fourier_series(funct_to_expand, m, eval_point):

    from math import sin, cos, pi, pow #TODO: check how to do this imports just once!

    #function to evaluate coefficients
    def series_coefficients(order_coeff):
        sub_tot = 0
        for j in range(0, m):
            left_val = (pi)/float(m)
            right_val = (pi)/float(m)
            #print right_val*j*order_coeff
            sub_tot += funct_to_expand(left_val*float(j))*cos(right_val*float(j)*float(order_coeff))
        #print (2/float(m))*sub_tot
        return (2/float(m))*sub_tot

    #N: order of expansion, N=[m/2]
    #m: sub_order of expansion
    #eval_point: evaluation point

    N = int(float(m)/2.0)

    approx_value = 0
    #first and last coefficients
    approx_value += series_coefficients(0)/2.0
    approx_value += (series_coefficients(N)/2.0)*cos(N*eval_point)
    #and then the rest of terms
    for i in range(1, N):
        approx_value += series_coefficients(i)*cos(i*eval_point)

    return approx_value
