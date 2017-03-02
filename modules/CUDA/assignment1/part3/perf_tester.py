from subprocess import Popen, PIPE
import StringIO


#Execute as:
#	$ python perf_tester.py



if __name__ == '__main__':

    mat_sizes = [100, 1000, 10000, 20000, 30000]
    thread_sizes = [16, 64, 256, 1024]

    file_out = open("out.dat", "w")

    #form: [ [s,t], [8 norm values], [8 exec times] ]
    num_results = list()

    for s in mat_sizes:
        for t in thread_sizes:

            buff_results = list()
            buff_results.append([s, t])

            process = Popen(['./mat_norms_parallel', '-n', str(s), '-m',
			str(s), '-t', '-T', str(t)], stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            #extracting values of norms
            #appended array of 12 norm values: [max, max_cu, frobenius,
            #frobenius_cu, one, one_cu, infinite, infinite_cu]
            buff_results.append(list())
            buff_results.append(list())
            for x in stdout.split('\n'):
                if len(x) != 0:
                    if x.split()[1][0:4] == "norm":
                        buff_results[len(buff_results)-2].append(float(x.split()[len(x.split())-1]))
                    if x[0:3] == "***":
                        buff_results[len(buff_results)-1].append(float(x.split()[len(x.split())-1]))
            print "Case: s = " + str(s) + " and t = " + str(t) + " processed"
            num_results.append(buff_results)

    for line in num_results:
        file_out.write(str(line) + "\n")
    file_out.close()
