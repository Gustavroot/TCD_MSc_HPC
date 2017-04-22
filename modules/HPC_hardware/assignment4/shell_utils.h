//Aux functions
char* substring(char*, char, int);
void strip(char*, int);
void output_file(char*);
void get_current_dir(char*, char*, int);
int redirection(char*);
void command_parse();


//CORE functions
void modular_shell(char*, char*, char*, char*, char*);
void own_shell(char*, char*, char*, char*, char*, FILE*);
void cd(char*, char*, char*, char*, char*, char*);
void ls(char*, FILE*, char*);
