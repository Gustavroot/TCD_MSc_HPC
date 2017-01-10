def newton_func(f, f_der, x, tol, root_val, file_name):

    from math import log10

    #output file
    file_output = open(file_name, 'w')

    steps_count = 0
    x1 = x
    while True:
        x2 = x1 - f(x1)/f_der(x1)
        for k in range(20, 120, 20):
            if k == 0:
                continue
            buff_p = 2+1/float(k)
            buff_rn = abs((root_val-x2))/pow(abs((root_val-x1)), buff_p)
            print "\tR_"+str(steps_count)+"(p="+str("{0:.5f}".format(buff_p))+")= "+str("{0:.8f}".format(buff_rn))
        print "\tR_"+str(steps_count)+"(p="+str(2)+")= "+str("{0:.8f}".format(abs((root_val-x2))/pow(abs((root_val-x1)), 2.0)))
        if abs(f(x2)/f_der(x2)) < tol/5:
            file_output.close()
            return (x2, steps_count)
        x1 = x2
        steps_count += 1
        file_output.write(str(steps_count)+"\t"+str("{0:.6f}".format(x1))+"\t"+str("{0:.6f}".format(root_val-x1))+"\t"+str("{0:.6f}".format(log10(abs(root_val-x1))))+"\n")
