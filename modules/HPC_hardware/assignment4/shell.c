#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <pwd.h>
#include <dirent.h>

#include <sys/stat.h>
#include <sys/ioctl.h>

#include <fcntl.h>

#include "shell_utils.c"



int main(){

  int i;
  char* buff;

  //use of PATH_MAX?

  //Var 'cwd_abs' refers to the actual current dir, and with added
  //'_current' means the virtual current dir
  char cwd_abs[254], cwd_rel[32], cwd_rel_previous[32], cwd_abs_previous[254], cwd_abs_current[286];
  //'command_root' is the command without params
  char command[254];

  get_current_dir(cwd_abs, cwd_rel, sizeof(cwd_abs));
  strcpy(cwd_abs_current, cwd_abs);
  strcpy(cwd_abs_previous, cwd_abs);
  strcpy(cwd_rel_previous, cwd_rel);

  fprintf(stdout, "\nOwn Bash v1.0.0\n\n");

  while(1){
    fprintf(stdout, "[user@computer %s]$ ", cwd_rel);
    
    //scanf("%s", command);
    fgets(command, 254, stdin);

    for(i=0; i<strlen(command); i++){
      if(command[i] == '\t'){
        //printf("%s\n", command);
        command[i] = ' ';
      }
    }
    //after the user inputs command, strip
    strip(command, strlen(command));
    //command[strlen(command)] = '\0';
    if(command[strlen(command)-1] != '\n'){
      i = strlen(command);
      command[i] = '\n';
      command[i+1] = '\0';
    }

    modular_shell(command, cwd_rel, cwd_abs_current, cwd_abs_previous, cwd_rel_previous);
  }

  return 0;
}
