def bisection_func(f, a, b, tol, file_name):
    #output file
    file_output = open(file_name, 'w')

    c = (a+b)/2.0
    steps_count = 0
    while (b-a)/2.0 > tol:
        previous_diff = b-a
        if f(c) == 0:
            return c
        elif f(a)*f(c) < 0:
            b = c
        else :
            a = c
        c = (a+b)/2.0
        steps_count += 1
        file_output.write(str(steps_count)+"\t"+str("{0:.6f}".format(c))+"\t"+str("{0:.6f}".format(previous_diff))+"\t"+str("{0:.6f}".format(a))+"\t"+str("{0:.6f}".format(b))+"\n")

    file_output.close()
    return (c, steps_count)
