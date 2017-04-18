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

#include "shell_aux.c"



//#include <sys/types.h>




int main(){

  int i;
  char* buff;

  //use of PATH_MAX?

  //Var 'cwd_abs' refers to the actual current dir, and with added
  //'_current' means the virtual current dir
  char cwd_abs[254], cwd_rel[32], cwd_rel_previous[32], cwd_abs_previous[254], cwd_abs_current[286];
  //'command_root' is the command without params
  char command[254];
  char filename[254];
  int size;
  
  //bin directories
  char bin1[32];
  strcpy(bin1, "/bin");
  char bin2[32];
  strcpy(bin2, "/usr/bin");
  char bin3[32];
  struct passwd *pw = getpwuid(getuid());
  const char *homedir = pw->pw_dir;
  strcpy(bin3, homedir);
  strcat(bin3, "/bin");

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

    FILE* out;

    i = redirection(command);

    //if no redirection
    if(i == 0){
      out = stdout;
      own_shell(command, cwd_rel, cwd_abs_current, cwd_abs_previous, cwd_rel_previous, out);
    }
    //depending on the redirection case, create the appropriate file first
    else if(i == 1 || i == 2){
      //create file with filename given
      buff = substring(command, '>', strlen(command))+2;
      if(i == 2){buff += 1;}
      strcpy(filename, buff);
      filename[strlen(filename)-1] = '\0';
      if(i == 1){
        out = fopen(filename, "w");
      }
      else{
        out = fopen(filename, "a");
      }
      //cut 'command' variable to only execute the command
      if(*(buff-1) == ' '){
        *(buff-2) = '\n';
        *(buff-1) = '\0';
      }
      else{
        *(buff-1) = '\n';
        *(buff) = '\0';
      }
      //printf("%s\n", filename);
      //printf("%s\n", command);
      own_shell(command, cwd_rel, cwd_abs_current, cwd_abs_previous, cwd_rel_previous, out);
      fclose(out);
      continue;
    }
    else{
      printf("redirection wants to be used!\n");
      //continue;
      out = stdout;
      own_shell(command, cwd_rel, cwd_abs_current, cwd_abs_previous, cwd_rel_previous, out);
    }

    //own_shell(command, cwd_rel, cwd_abs_current, cwd_abs_previous, cwd_rel_previous, out);
  }


  return 0;
}
