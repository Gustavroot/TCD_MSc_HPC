#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <pwd.h>
#include <dirent.h>

#include <sys/stat.h>
#include <sys/ioctl.h>
#include  <sys/types.h>
#include <sys/wait.h>

#include <fcntl.h>

#include <readline/readline.h>
#include <readline/history.h>

//Aux functions
char* substring(char*, char, int);
void strip(char*, int);
void output_file(char*);
void get_current_dir(char*, char*, int);
int redirection(char*);
void command_parse();
int file_in_dir(char*, char*);
//char* rl_gets();


//CORE functions
void modular_shell(char*, char*, char*, char*, char*);
void own_shell(char*, char*, char*, char*, char*, FILE*);
void cd(char*, char*, char*, char*, char*, char*);
void ls(char*, FILE*, char*);
