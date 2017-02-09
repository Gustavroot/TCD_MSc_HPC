#include <stdio.h>	/* printf, perror */
#include <sys/ptrace.h> /* ptrace */
#include <sys/wait.h>	/* wait */
#include <sys/user.h>	/* registers struct */
#include <sys/reg.h>	/* register MACROS */
#include <unistd.h>	/* fork, execl */


int main() {
	struct user_regs_struct regs;
        long long counter = 0;
	long val;
        int wait_val;
        int pid;

        switch (pid = fork()) {
        case -1:
                perror("fork");
                break;
        case 0: 
                ptrace(PTRACE_TRACEME, 0, 0, 0);
                execl("/bin/ls", "ls", NULL);
                break;
        default:
                wait(&wait_val); 

		val = ptrace(PTRACE_PEEKUSER, pid, ORIG_RAX * 8, NULL);
		printf("ORIG RAX = %ld\n", val);

                while (!WIFEXITED(wait_val) ) {
                        counter++;
                        // if (ptrace(PTRACE_SINGLESTEP, pid, 0, 0) != 0)
                        if (ptrace(PTRACE_SYSCALL, pid, 0, 0) != 0)
                                perror("ptrace");
			ptrace(PTRACE_GETREGS, pid, NULL, &regs);
			printf("ORIG_RAX = %ld\n", regs.rax);
			// printf("RAX = %ld\n", regs.rax);
                        wait(&wait_val);
                }
        }

        printf("Number of system calls: %lld\n", counter);
        return 0;
}

