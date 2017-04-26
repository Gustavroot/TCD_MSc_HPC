#include "shell_utils.c"


//Compilation instructions:
//	$ gcc shell.c -lreadline


//read and write labels used in pipes
#define READ 0
#define WRITE 1

/* A static variable for holding the line. */
static char* command = (char *)NULL;

void rl_gets(char**);

int main(int argc, char** argv){

  int i, j;
  char* buff;
  //data for pipe
  int count_pipes, pipe1[2], pipe2[2];

  pid_t pid;

  //general pipe
  int pd[2];

  //use of PATH_MAX?

  //Var 'cwd_abs' refers to the actual current dir, and with added
  //'_current' means the virtual current dir
  char cwd_abs[254], cwd_rel[32], cwd_rel_previous[32], cwd_abs_previous[254], cwd_abs_current[286];
  //'command_root' is the command without params
  //char command[254];
  //char* command = (char*)(-1);
  char command_buff[254];

  //buffer for pipes
  int bytes_read;
  char stdin_buff[254];

  char** partial_commands;

  get_current_dir(cwd_abs, cwd_rel, sizeof(cwd_abs));
  strcpy(cwd_abs_current, cwd_abs);
  strcpy(cwd_abs_previous, cwd_abs);
  strcpy(cwd_rel_previous, cwd_rel);

  printf("\nOwn Bash v1.0.0\n\n");

  int k = 0;

  while(1){
    //if no previous call to readline()
    printf("[user@computer %s]$ ", cwd_rel);
    fflush(stdin);
    //get a line from the user
    if(command != (char*)NULL){free(command);}
    command = readline("");
    add_history(command);

    for(i=0; i<strlen(command); i++){
      if(command[i] == '\t'){
        command[i] = ' ';
      }
    }

    //after the user inputs command, strip
    strip(command, strlen(command));
    if(command[strlen(command)-1] != '\n'){
      i = strlen(command);
      command[i] = '\n';
      command[i+1] = '\0';
    }

    //count number of pipes
    count_pipes = 0;
    buff = command;
    for(; *buff; count_pipes += (*buff++ == '|'));

    //creating a copy of command
    strcpy(command_buff, command);
    command_buff[strlen(command_buff)-1] = '\0';
    //From the number of commands to be executed due to pipes
    partial_commands = (char**)malloc((count_pipes+1)*sizeof(char*));
    j = count_pipes;
    if(strlen(command_buff) == 0){
      partial_commands[0] = (char*)malloc(sizeof(char));
      strcpy(partial_commands[0], "\n");
    }
    else{
      //split in sub-commands due to pipes
      for(i=strlen(command_buff)-1; i>=0; i--){
        //for each pipe symbol, create a command
        if(command_buff[i] == '|'){
          partial_commands[j] = (char*)malloc((strlen(command_buff+i+2)+1)*sizeof(char));
          strcpy(partial_commands[j], command_buff+i+2);
          buff = partial_commands[j] + strlen(partial_commands[j]);
          *buff = '\n';
          *(buff+1) = '\0';
          command_buff[i] = '\0';
          strip(command_buff, strlen(command_buff));
          j--;
        }
      }
      //copy first command, left from last for loop
      partial_commands[0] = (char*)malloc(strlen(command_buff)*sizeof(char));
      strcpy(partial_commands[0], command_buff);
      buff = partial_commands[0] + strlen(partial_commands[0]);
      *buff = '\n';
      *(buff+1) = '\0';
    }

    //don't let the child to execute the last command
    strcpy(command_buff, partial_commands[0]);
    for(i=0; i<count_pipes; i++){
      pipe(pd);

      /* implement cd data here
      if(command_buff[0] == 'c' && command_buff[1] == 'd'){
      }
      */

      pid = fork();
      if(pid == -1){
        printf("fork failed!\n");
        exit(1);
      }
      //child execution
      else if(pid == 0){
        //send output to parent through pipe
        dup2(pd[1], 1);
        close(pd[0]);
        modular_shell(command_buff, cwd_rel, cwd_abs_current, cwd_abs_previous, cwd_rel_previous);
        exit(0);
      }
      else{
        //the parent sends output from child to STDIN for next execution
        dup2(pd[0], 0);
        close(pd[1]);
        //attach output from child to next execution
        strcpy(command_buff, partial_commands[i+1]);
        wait(NULL);
      }
    }

    //if cd, execute by parent process
    if(!(command_buff[0] == 'c' && command_buff[1] == 'd')){
      pid = fork();
      if(pid == -1){
        printf("fork failed!\n");
      }
      else if(pid == 0){
        modular_shell(command_buff, cwd_rel, cwd_abs_current, cwd_abs_previous, cwd_rel_previous);
        exit(0);
      }
      else{
        wait(NULL);
      }
    }
    else{
      modular_shell(command_buff, cwd_rel, cwd_abs_current, cwd_abs_previous, cwd_rel_previous);
    }

    //release memory
    for(j = 0; j<count_pipes+1; j++){
      free(partial_commands[j]);
    }
    free(partial_commands);

    //restore stdin
    freopen("/dev/tty","rb",stdin);
    k++;
  }
  free(command);

  return 0;
}
