#include <unistd.h>

int main() {

	execl("/bin/ls", "ls", "-l", "/etc", NULL);
}
