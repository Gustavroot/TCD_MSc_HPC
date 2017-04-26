#include "shell_utils.h"


//----------------------CORE FUNCTIONS---------------------------------------------


void modular_shell(char* command, char* cwd_rel, char* cwd_abs_current, char* cwd_abs_previous, char* cwd_rel_previous){

  //All this code will be called by multiple modularizations
  //of the named pipes

  FILE* out;

  int i;
  char* buff;
  int size;

  char filename[254], filename_buff[254];

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
    //check if relative or absolute path
    if(filename[0] != '/'){
      strcpy(filename_buff, filename);
      strcpy(filename, cwd_abs_current);
      strcat(filename, "/");
      strcat(filename, filename_buff);
    }
    //overwrite or not
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
    own_shell(command, cwd_rel, cwd_abs_current, cwd_abs_previous, cwd_rel_previous, out);
    fclose(out);
    //continue;
  }
  else{
    //TODO: implement pending redirections. For now, redirection is set into stdout
    out = stdout;
    own_shell(command, cwd_rel, cwd_abs_current, cwd_abs_previous, cwd_rel_previous, out);
  }

  //own_shell(command, cwd_rel, cwd_abs_current, cwd_abs_previous, cwd_rel_previous, out);
}



//TODO: for following function own_shell(),
//use recursivity to solve the implementation of pipe()
void own_shell(char* command, char* cwd_rel, char* cwd_abs_current, char* cwd_abs_previous, char* cwd_rel_previous, FILE* out){

  int i;

  //bin directories
  char* bin_buff;
  int in_bin;
  char bin1[32];
  strcpy(bin1, "/bin");
  char bin2[32];
  strcpy(bin2, "/usr/bin");
  char bin3[32];
  struct passwd *pw = getpwuid(getuid());
  const char *homedir = pw->pw_dir;
  strcpy(bin3, homedir);
  strcat(bin3, "/bin");

  struct winsize terminal_size;

  char* str_ptr;
  char command_root[16];
  //file to which output will be sent.. can be pipe, stdout, etc.
  //FILE* out;
  //Extract root of inserted command.. NULL if no
  //separation by ' '
  str_ptr = substring(command, ' ', strlen(command));

  //if there is an empty space after the root of the command
  if(str_ptr != NULL){
    strncpy(command_root, command, str_ptr-command);
    command_root[str_ptr-command] = '\0';
  }
  else{
    //'-1' due to '\n' at the end of info in variable "command"
    strncpy(command_root, command, strlen(command)-1);
    command_root[strlen(command)-1] = '\0';
  }

  //printf("<root command: %s>\n", command_root);

  //TODO: construct file to which output will be sent
  //out = output_file(command);
  //output_file(command);

  //Check in advance if command_root exists in any bin/
  in_bin = 0;
  if(file_in_dir(bin1, command_root)){
    bin_buff = bin1;
    in_bin = 1;
  }
  else if(file_in_dir(bin2, command_root)){
    bin_buff = bin2;
    in_bin = 1;
  }
  else if(file_in_dir(bin3, command_root)){
    bin_buff = bin3;
    in_bin = 1;
  }

  //implement according to given command
  if(strcmp(command, "clear\n") == 0){
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &terminal_size);
    //printf("%d\n", terminal_size.ws_row);
    for(i=0; i<terminal_size.ws_row; i++){
      printf("\n");
    }
    printf("\033[%dA", terminal_size.ws_row);
  }
  //if ENTER, do nothing
  else if(strcmp(command, "\n") == 0){}
  //cd to dir, abs or rel
  else if(strcmp(command_root, "cd") == 0){
    cd(command_root, command, cwd_rel, cwd_abs_current, cwd_abs_previous, cwd_rel_previous);
  }
  else if(strcmp(command_root, "ls") == 0){

  //printf("works!\n");

    pid_t pid;
    //because printf() is buffered, flush
    fflush(stdout);
    pid = fork();
    if(pid == -1){
      printf("failed to fork!\n");
      exit(1);
    }
    else if(pid == 0){
      //the child is in charge of executing the execution of commands
      ls(cwd_abs_current, out, command);
      exit(0);
    }
    //the parent persists to exist
    else{
      wait(NULL);
      return;
    }
  }
  //for general purpose executables
  else if(in_bin){
    int j;
    //str_ptr++;
    char** arguments;
    int count_char = 0;
    if(str_ptr == NULL){
      arguments = (char**)malloc((count_char+2)*sizeof(char*));
      arguments[0] = (char*)malloc(strlen(command_root)*sizeof(char));
      strcpy(arguments[0], command_root);
    }
    else{
      char args_buff[254];
      strcpy(args_buff, str_ptr);
      char* buff_ptr = args_buff;
      args_buff[strlen(args_buff)-1] = '\0';
      //The child is in charge of executing the commands
      //First, construct array of params
      for(; *buff_ptr; count_char += (*buff_ptr++ == ' '));
      arguments = (char**)malloc((count_char+2)*sizeof(char*));
      arguments[0] = (char*)malloc(strlen(command_root)*sizeof(char));
      strcpy(arguments[0], command_root);
      //Next line is due to last argument being NULL in arguments
      j = count_char;
      for(i=strlen(args_buff)-1; i>=0; i--){
        //for each space, create a argument
        if(args_buff[i] == ' '){
          arguments[j] = (char*)malloc(strlen(args_buff+i+1)*sizeof(char));
          strcpy(arguments[j], args_buff+i+1);
          args_buff[i] = '\0';
          j--;
        }
      }
      //arguments[count_char+1] = (char*)malloc(sizeof(char));
    }
    arguments[count_char+1] = NULL;

    //fork()?
    //execute program
    if(execvp(arguments[0], arguments)){
      printf("ERROR: %s\n", strerror(errno));
    }

    //following code to execvp() won't execute... will dissapear...

    //release memory
    for(j = 0; j<count_char+1; j++){
      free(arguments[j]);
    }
    free(arguments);
  }
  else{
    printf("%s: command not found\n", command_root);
  }
}


void cd(char* command_root, char* command, char* cwd_rel, char* cwd_abs_current, char* cwd_abs_previous, char* cwd_rel_previous){

  //if no param given, go to home dir
  char* buff;
  int i;

  char tmp_cwd[254];

  //'< 4' due to '\n'
  if(strlen(command) < 4){
    strcpy(cwd_rel, "~\0");
    struct passwd *pw = getpwuid(getuid());
    const char *homedir = pw->pw_dir;
    strcpy(cwd_abs_current, homedir);
  }
  else{

    strip(command+3, strlen(command+3));
    if(command[strlen(command)-1] != '\n'){
      i = strlen(command);
      command[i] = '\n';
      command[i+1] = '\0';
    }

    i = *(command+4) == ' ' || *(command+4) == '\n';
    if(strlen(command) > 3 && *(command+3) == '-' && i){
      strcpy(tmp_cwd, cwd_abs_current);
      strcpy(cwd_abs_current, cwd_abs_previous);
      strcpy(cwd_abs_previous, tmp_cwd);

      strcpy(tmp_cwd, cwd_rel);
      strcpy(cwd_rel, cwd_rel_previous);
      strcpy(cwd_rel_previous, tmp_cwd);

      return;
    }

    //check if there are spaces within string for path
    buff = substring(command+3, ' ', strlen(command+3));
    if(buff != NULL){
      if(*(buff+1) != '\n'){
        printf("%s: usage: %s [dir]\n", command_root, command_root);
        return;
      }
    }
    
    char abs_dir[254], abs_dir_reduced[254];
    strcpy(abs_dir, cwd_abs_current);

    //in case path is relative
    if(command[3] != '/'){
      //printf("rel path!\n");
      if(cwd_abs_current[strlen(cwd_abs_current)-1] != '/'){
        strcat(abs_dir, "/\0");
      }
      strcat(abs_dir, command+3);
    }
    //in case path is absolute
    else{
      strcpy(abs_dir, command+3);
    }

    abs_dir[strlen(abs_dir)-1] = '\0';
    realpath(abs_dir, abs_dir_reduced);

    struct stat path_stat;
    stat(abs_dir, &path_stat);
    if(S_ISREG(path_stat.st_mode) == 1){
      printf("%s is a file\n", abs_dir_reduced);
      return;
    }
    else if(S_ISDIR(path_stat.st_mode) != 1){
      printf("%s is not a directory or file\n", abs_dir_reduced);
      return;
    }

    DIR* dir = opendir(abs_dir);
    //if directory exists
    if(dir){
      //
      strcpy(cwd_abs_previous, cwd_abs_current);
      strcpy(cwd_rel_previous, cwd_rel);
      strcpy(cwd_rel, strrchr(abs_dir_reduced, '/')+1);
      strcpy(cwd_abs_current, abs_dir_reduced);
      closedir(dir);
    }
    //if directory does not exist
    else if(ENOENT == errno){
      printf("directory does not exist\n");
    }
    else{
      //opendir() failed for some other reason
    }
  }
  chdir(cwd_abs_current);
}


void ls(char* cwd_abs_current, FILE* out, char* command){
  DIR* d;
  struct dirent* dir;
  char path_i[254];
  char inpts[254];
  char *buff2;
  strcpy(inpts, command);
  inpts[strlen(inpts)-1] = '\0';
  buff2 = substring(inpts, ' ', strlen(inpts));
  //Implementation of absolute and relative paths with ls
  //if a param has been passed to ls
  if(buff2 != NULL){
    strcpy(inpts, buff2+1);
    buff2 = substring(inpts, ' ', strlen(inpts));
    if(buff2 != NULL){
      buff2[0] = '\0';
    }
    //absolute or relative
    if(inpts[0] == '/'){
      strcpy(path_i, inpts);
    }
    else{
      strcpy(path_i, cwd_abs_current);
      strcat(path_i, "/");
      strcat(path_i, inpts);
    }
    strcpy(inpts, path_i);
    realpath(inpts, path_i);
    //matching with files
    d = opendir(path_i);
    if(d){
      strcpy(inpts, "");
    }
    else{
      //if file, then put that name in 'inpts'
      buff2 = strrchr(path_i, '/');
      strcpy(inpts, buff2+1);
      buff2[0] = '\0';
    }
    closedir(d);
  }
  else{
    strcpy(path_i, cwd_abs_current);
    strcpy(inpts, "");
  }

  d = opendir(path_i);
  if(d){
    while((dir = readdir(d)) != NULL){
      if((dir->d_name)[0] == '.'){continue;}
      if(strcmp(inpts, "") == 0){
        fprintf(out, "%s\n", dir->d_name);
      }
      else{
        if(strcmp(dir->d_name, inpts) == 0){
          fprintf(out, "%s\n", dir->d_name);
        }
      }
    }
    closedir(d);
  }
}







//----------------------AUX FUNCTIONS---------------------------------------------


void get_current_dir(char* dir_abs, char* dir_rel, const int size){

  int i = 0;
  char dir_buff;

  if(getcwd(dir_abs, size) != NULL){

    struct passwd *pw = getpwuid(getuid());
    const char *homedir = pw->pw_dir;

    if(strcmp(dir_abs, homedir) == 0){
      strcpy(dir_rel, "~\0");
    }
    else{
      strcpy(dir_rel, strrchr(dir_abs, '/')+1);
    }
  }
  else{
    perror("getcwd() error");
  }

}


//return address of first appearance of 'split_char'
//alternative: use strchr()
char* substring(char* dir,  char split_char, int size){

  char* position;
  int i;
  
  for(i=0; i<size; i++){
    if(*(dir+i) == split_char){
      return dir+i;
    }
  }

  return NULL;
}


void strip(char* command, int size){
  int i;

  //'-2' because of '\n' at the end
  //tail
  if(command[size-1] == '\n'){
    command[size-1] = '\0';
    i = size-2;
  }
  else{
    i = size-1;
  }
  for(; i>=0; i--){
    if(command[i] != ' '){
      break;
    }
    else{
      command[i] = '\0';
      //i--;
    }
  }
  size = i+1;

  //head
  for(i=0; i<size; i++){
    if(command[i] != ' '){break;}
  }
  strcpy(command, command+i);

  i = substring(command, ' ', strlen(command)) != NULL;
  if(i){
    strip(substring(command, ' ', strlen(command))+1, strlen(command));
  }
}


//types:
//	1: ">" or "1>"
//	2: ">>" or "1>>"
//	3: "2>"
//	4: "2>>"
//	5: "<"
//	6: "<<"
int redirection(char* command){
  if(strstr(command, "2>>") != NULL){
    return 4;
  }
  else if(strstr(command, ">>") != NULL || strstr(command, "1>>") != NULL){
    return 2;
  }
  else if(strstr(command, "2>") != NULL){
    return 3;
  }
  else if(strstr(command, ">") != NULL || strstr(command, "1>") != NULL){
    //stdout redirection
    return 1;
  }
  else if(strstr(command, "<<") != NULL){
    return 6;
  }
  else if(strstr(command, "<") != NULL){
    return 5;
  }
  else{
    return 0;
  }
}


int file_in_dir(char* path_i, char* filename){
  DIR* d;
  struct dirent* dir;
  d = opendir(path_i);
  if(d){
    while((dir = readdir(d)) != NULL){
      if(strcmp(dir->d_name, filename) == 0){
        //fprintf(stdout, "%s\n", dir->d_name);
        closedir(d);
        return 1;
      }
    }
    closedir(d);
  }
}
