def secant_func(f, x0, x1, tol, root_val, file_name):
    #output file
    file_output = open(file_name, 'w')

    #the function 'abs' is built-in

    n=1
    steps_count = 0
    while True:
        x2 = x1 - f(x1)*((x1-x0)/(f(x1)-f(x0)))
        if abs(x2-x1) < tol:
            return (x2, steps_count)
        else:
            x0 = x1
            x1 = x2
        steps_count += 1
        file_output.write(str(steps_count)+"\t"+str("{0:.6f}".format(x1))+"\t"+str("{0:.6f}".format(root_val-x1))+"\t"+"\n")
    return (x2, steps_count)
