#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#include <pthread.h>

#include <unistd.h>
#include <limits.h>

#define NUM_THREADS     3
#define NUM_INSERTIONS     30000
#define NUM_EXTRACTIONS     30000

//mili-seconds
#define POISSON_AVG     0.5
#define POISSON_AVG_THREAD_2     5

//change following flag to 1 if debug prints
//want to be enabled
#define PRINTS_FLAG     0

//variable to block tree when modifying/accesing it
pthread_mutex_t mutextree;


//Compilation:
//	$ gcc bbtree-parallel.c -o bbtree-parallel -lm -lpthread

//Execution (where N is an integer, the number of nodes):
//	$ ./bbtree-parallel

//main reference:
//https://computing.llnl.gov/tutorials/pthreads/#Mutexes



//Binary tree structure
struct node{
  int key_value;
  struct node *left;
  struct node *right;
};


//*******
//creating structures to pass arguments to threads

//threads 0 and 1: insert
struct data_threads_0_1{
  int  thread_id;
  //int key;
  struct node** leaf;
};

//thread 2: balance
struct data_threads_2{
  int  thread_id;
  struct node** tree;
};

//*******


//CORE functions

//handlers for threads
void* inserts( void* );
void* extract_elems(void*);
void* balance_parallel( void* );

//serial functions
void insert_serial(int key, struct node**);
void extract_elem_serial(int, struct node**);
void balance(struct node**);
void build_balanced_tree(struct node**, int*, int, int, int);
void print_tree(struct node*);
void destroy_tree(struct node*);

//auxiliary functions
struct node** smallest(struct node**);
int max_depth(struct node*);
void nr_nodes(struct node*, int*);
void list_inorder(struct node*, int*, int*);

//EXTRA functions
void print_usage(){
  printf("./bbtree-parallel\n");
}

int poisson_random(double expectedValue) {
  int n = 0; //counter of iteration
  double limit;
  double x;  //pseudo random number
  limit = exp(-expectedValue);
  x = rand() / (double) INT_MAX;
  while (x > limit){
    n++;
    x *= rand() / (double) INT_MAX;
  }
  return n;
}



//main code
int main(int argc, char** argv){

  int rand_buff;
  //giving seed to random nrs generator
  srand(time(NULL));
  
  //general purpose counter
  int i;
  
  /* rnd Poisson tests
  
  for(i = 0; i<17; i++){
    printf("%d\n", poisson_random(50) );
  }
  
  */
  
  //return 0;

  //time measurement diff
  double d_t;
  struct timeval begin, end;

  int *tree_depth, t_depth;
  
  //vars to track threads errors
  int rc;
  long t;
  void *status;

  //root node
  struct node *tree = 0;
  //at this point, the tree doesn't exist

  printf("\nParallel binary tree.\n\n");
  
  //init mutex variable
  pthread_mutex_init(&mutextree, NULL);
  
  //set up for threads:
  pthread_t threads[NUM_THREADS];
  pthread_attr_t attr;

  //setup args passed to threads
  struct data_threads_0_1 thread_data_0_1[2];
  struct data_threads_2 thread_data_2;
  thread_data_0_1[0].thread_id = 0;
  thread_data_0_1[0].leaf = &tree;
  thread_data_0_1[1].thread_id = 1;
  thread_data_0_1[1].leaf = &tree;
  
  //initialize and set thread detached attribute
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);  
  
  //initial time measurement
  gettimeofday(&begin, NULL);
  
  //thread 2 to balance on the background
  thread_data_2.thread_id = 2;
  thread_data_2.tree = &tree;
  
  //calling 'balance_parallel' handler - performed by thread 0
  rc = pthread_create(&threads[2], &attr, balance_parallel, (void *) &thread_data_2 );
  if (rc){
    printf("ERROR: return code from pthread_create() is %d\n", rc);
    exit(-1);
  }

  printf("Inserting data in tree...");

  tree_depth = 0;
  
  //calling 'insert' handler - performed by thread 0
  rc = pthread_create(&threads[0], &attr, inserts, (void *) &thread_data_0_1[0] );
  if (rc){
    printf("ERROR: return code from pthread_create() is %d\n", rc);
    exit(-1);
  }

  //printing tree after total insertion
  printf("... insertions assigned.\n");
  
  i = 0;
  nr_nodes(tree, &i);
  printf("\t*** nodes at this point: %d\n", i);

  printf("\nExtracting data from tree...");
  
  //calling 'extract_elems' handler - performed by thread 0
  rc = pthread_create(&threads[1], &attr, extract_elems, (void *) &thread_data_0_1[1] );
  if (rc){
    printf("ERROR: return code from pthread_create() is %d\n", rc);
    exit(-1);
  }

  //printing tree after some extractions
  printf("... extractions assigned.\n");
  
  i = 0;
  nr_nodes(tree, &i);
  printf("\t*** nodes at this point: %d\n", i);
  
  /* Free attribute and wait for the other threads */
  pthread_attr_destroy(&attr);
  
  for(t=0; t<NUM_THREADS; t++) {
    rc = pthread_join(threads[t], &status);
    if(rc){
      printf("ERROR: return code from pthread_join() is %d\n", rc);
      exit(-1);
    }
    printf("Main: completed join with thread %ld having a status of %ld\n", t, (long)status);
  }

  //end time measurement
  gettimeofday(&end, NULL);

  d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
  printf("Main: program completed. Total exec time: %f\n", d_t);
  
  i = 0;
  nr_nodes(tree, &i);
  printf("\nNodes at this point: %d\n", i);
  
  //pthread_cancel(threads[2]);
  
  //destroy tree
  destroy_tree(tree);
  printf("\n... tree destroyed.\n\n");
  
  pthread_mutex_destroy(&mutextree);
  pthread_exit(NULL);  
}


//CORE functions

//destroy all tree
void destroy_tree(struct node *leaf){
  if( leaf != 0 )
  {
    destroy_tree(leaf->left);
    destroy_tree(leaf->right);
    free( leaf );
  }
}


//inserting new node to the tree
void* inserts(void* thread_args){

  int rnd_buff;

  //extracting args passed to thread
  struct data_threads_0_1* t_data;
  t_data = (struct data_threads_0_1*) thread_args;
  //int key = t_data->key;
  struct node** leaf = t_data->leaf;
  int task_id = t_data->thread_id;
    
  //for loop for extractions
  int i;
  for(i=0; i<NUM_EXTRACTIONS; i++){
  
    rnd_buff = ((int)rand())%NUM_EXTRACTIONS;
      
    //lock
    pthread_mutex_lock (&mutextree);
    insert_serial(rnd_buff, leaf);
    pthread_mutex_unlock (&mutextree);
    
    if(PRINTS_FLAG){
      print_tree(*leaf);
      printf("tried to insert nr %d\n", rnd_buff);
    }
    
    sleep( (double) poisson_random(POISSON_AVG) / 1000 );
  }
  
  printf("done inserting!\n");
  pthread_exit((void*) 0);
}


//inserting new node to the tree
void insert_serial(int key, struct node **leaf){
  //in case there is no tree yet
  if( *leaf == 0 ){
    //if root pointing to NULL, then allocate memory
    *leaf = (struct node*) malloc( sizeof( struct node ) );
    (*leaf)->key_value = key;
    //point the children nodes to NULL
    (*leaf)->left = 0;    
    (*leaf)->right = 0;  
  }
  
  //if value must be at left
  else if(key < (*leaf)->key_value){
    insert_serial( key, &(*leaf)->left );
  }
  
  //if key must be at right
  else if(key > (*leaf)->key_value){
    insert_serial( key, &(*leaf)->right );
  }
  //if key already in tree, do nothing
}


//find smallest element within tree
struct node** smallest(struct node** leaf){
  if( (**leaf).left != 0 ){
    return smallest( &((**leaf).left) );
  }
  else{
    return leaf;
  }
}


void* extract_elems( void* thread_args ){

  int rnd_buff;

  //extracting args passed to thread
  struct data_threads_0_1* t_data;
  t_data = (struct data_threads_0_1*) thread_args;
  struct node** leaf = t_data->leaf;
  int task_id = t_data->thread_id;
  
  //for loop for extractions
  int i;
  for(i=0; i<NUM_EXTRACTIONS; i++){

    rnd_buff = ((int)rand())%NUM_EXTRACTIONS;
  
    //lock
    pthread_mutex_lock (&mutextree);
    extract_elem_serial(rnd_buff, leaf);
    pthread_mutex_unlock (&mutextree);
    
    if(PRINTS_FLAG){
      print_tree(*leaf);
      printf("tried to extract nr %d\n", rnd_buff);
    }
    
    sleep( (double) poisson_random(POISSON_AVG) / 1000 );
  }
  
  printf("done extracting!\n");
  pthread_exit((void*) 0);
  
}

//extracting node with key_value = 'key'
void extract_elem_serial(int key, struct node** leaf){
  if( *leaf != 0 ){
    if(key == (*leaf)->key_value){

      //multiple cases when removing elem:
      //http://www.algolist.net/Data_structures/Binary_search_tree/Removal
      
      //no child nodes
      if((*leaf)->left==0 && (*leaf)->right==0){
        free( *leaf );
        *leaf = 0;
      }
      
      //just one child for this node
      //left child 'inherits'
      else if((*leaf)->right==0){
        //pointer buffer
        struct node* buff;
        //releasing and re-directing
        buff = (*leaf)->left;
        free(*leaf);
        *leaf = buff;
      }
      //right child 'inherits'
      else if((*leaf)->left==0){
        //pointer buffer
        struct node* buff;
        //releasing and re-directing
        buff = (*leaf)->right;
        free(*leaf);
        *leaf = buff;
      }
      
      //if node has two children
      else{
        //look for node with smallest value on right wing
        struct node** buff;
        buff = smallest( &((**leaf).right) );
        
        //replace value in 'to delete' node
        (**leaf).key_value = (**buff).key_value;
        
        //delete smallest
        //..before deleting, check if it has no right child:
        if( (**buff).right != 0 ){
          //pointer buffer
          struct node* buff2;
          //releasing and re-directing
          buff2 = (*buff)->right;
          free(*buff);
          *buff = buff2;
        }
        else{
          free(*buff);
          *buff = 0;
        }
      }
      
    }
    else if(key < (*leaf)->key_value){
      extract_elem_serial( key, &(*leaf)->left );
    }
    else{
      extract_elem_serial( key, &(*leaf)->right );
    }
  }
  //in case key not in tree, do nothing
}


int max_depth(struct node* node){
 if (node == NULL)
   return 0;
 else{
   /* compute the depth of each subtree */
   int lDepth = max_depth(node->left);
   int rDepth = max_depth(node->right);
   /* use the larger one */
   if (lDepth > rDepth) 
     return(lDepth+1);
   else return(rDepth+1);
 }
}


//pending: add padding here for better display
void print_tree(struct node* tree){

  int max_printable = 6;

  int depth = 1;
  //counters within layer
  int i, j;
  int nr_nulls;
  
  int max_d = max_depth(tree);
  if(max_d > max_printable){
    max_d = max_printable;
  }
  
  struct node** nodes_ptr = (struct node**) malloc( 1 * sizeof(struct node*) );
  nodes_ptr[0] = tree;
  struct node** nodes_ptr_buff;

  while(1){
  
    //print current layer
    for(i=0; i< pow(2, depth-1) ; i++){
      //print left spaces
      for(j=0; j< pow(2, max_d-1)-1 ; j++){
        printf(" ");
      }
      //print node key
      if(nodes_ptr[i] != 0){
        printf("%d", (*nodes_ptr[i]).key_value );
      }
      else{
        printf("-");
      }
      //print right spaces
      for(j=0; j< pow(2, max_d-1)-1 ; j++){
        printf(" ");
      }
      printf(" ");
    }
    printf("\n");
    
    //print link symbols
    if(max_d == 2){
      for(i=0; i< pow(2, depth) ; i++){
        //print left spaces
        for(j=0; j< pow(2, max_d-3)-1 ; j++){
          printf(" ");
        }
        //print node key
        if(i%2 != 0){
          printf("\\");
        }
        else{
          printf("/");
        }
        //print right spaces
        for(j=0; j< pow(2, max_d-3)-1 ; j++){
          printf(" ");
        }
        printf(" ");
      }
      printf("\n");
    }
    else if(max_d > 1){
      for(i=0; i< pow(2, depth+1)-1 ; i++){
        //print left spaces
        for(j=0; j< pow(2, max_d-3)-1 ; j++){
          printf(" ");
        }
        //print node key
        if(i == 0){
          printf(" ");
        }
        else if( (i-2)%4 == 0 ){
          printf("\\");
        }
        else if( (i-1)%4 == 0 ){
          printf("/");
        }
        else{
          printf(" ");
        }
        //print right spaces
        for(j=0; j< pow(2, max_d-3)-1 ; j++){
          printf(" ");
        }
        printf(" ");
      }
      printf("\n");
    }

    //allocate the nodes of current layer
    nodes_ptr_buff = (struct node**) malloc( pow(2, depth) * sizeof(struct node*) );
    
    
    nr_nulls = 0;
    for(i=0; i< pow(2, depth-1) ; i++){
    
      if( nodes_ptr[i] == 0 ){
        nodes_ptr_buff[2*i] = 0;
        nodes_ptr_buff[2*i+1] = 0;
        nr_nulls += 2;
      }
      else{
      
        if( ( *(nodes_ptr[i]) ).left != 0 ){
          nodes_ptr_buff[2*i] =  ( *(nodes_ptr[i]) ).left;
        }
        else{
          nodes_ptr_buff[2*i] = 0;
          nr_nulls++;
        }
        if( ( *(nodes_ptr[i]) ).right != 0 ){
          nodes_ptr_buff[2*i+1] =  ( *(nodes_ptr[i]) ).right;
        }
        else{
          nodes_ptr_buff[2*i+1] = 0;
          nr_nulls++;
        }
        
      }
    }
    if(nr_nulls == pow(2, depth) || depth==max_printable){
      free(nodes_ptr);
      free(nodes_ptr_buff);
      break;
    }
    
    //free previous layer
    free(nodes_ptr);
    //set initial pointers of next iteration, equal to current pointers
    nodes_ptr = (struct node**) malloc( pow(2, depth) * sizeof(struct node*) );
    for(i=0; i< pow(2, depth) ; i++){
      nodes_ptr[i] = nodes_ptr_buff[i];
    }
    
    free(nodes_ptr_buff);

    max_d--;
    depth++;
  }
}


//determine nr of nodes in tree
void nr_nodes(struct node* leaf, int* ctr){
  if( leaf != 0 ){

    if( (*leaf).left == 0 ){
      (*ctr)++;
    }
    else{
      nr_nodes( (*leaf).left , ctr);
    }
  
    if( (*leaf).right == 0 ){
      (*ctr)++;
    }
    else{
      nr_nodes( (*leaf).right , ctr);
    }
    
  }
}


//extracting nodes inorder
void list_inorder(struct node* leaf, int* arr, int* ctr){

   if(leaf == 0)
     return;
   list_inorder(leaf->left, arr, ctr);
   arr[*ctr] = leaf->key_value;
   (*ctr)++;
   list_inorder(leaf->right, arr, ctr);
}


void* balance_parallel( void* thread_args ){

  //extracting args passed to thread
  struct data_threads_2* t_data;
  t_data = (struct data_threads_2*) thread_args;
  struct node** tree = t_data->tree;
  int task_id = t_data->thread_id;
  
  //when general_counter > 1000, then exit this thread
  int general_counter = 0;
  
  //loop for balancing
  int i = 1, i_ctr = 0;
  int tree_depth = 0;
  printf("\nThread %d balancing... ", task_id);
  while(i){
  
    if(general_counter > 1000){
      printf("Inactivity detected, balance thread stopped.\n");
      pthread_exit((void*) 0);
    }
  
    if(tree_depth != max_depth( *tree )){
      //lock
      pthread_mutex_lock (&mutextree);
      balance(tree);
      pthread_mutex_unlock (&mutextree);
      
      if(PRINTS_FLAG){
        printf("b(%d), ", i_ctr);
      }
      
      tree_depth = max_depth( *tree );
      i_ctr++;
      
      general_counter = 0;
    }
    else{
      general_counter++;
    }

    sleep( (double) POISSON_AVG_THREAD_2 / 1000.0 );
    
  }
}


//balance tree
void balance(struct node** tree){

  if( *tree == 0 ){
    //printf("empty tree!\n");
    return;
  }

  int i;

  int* array_nodes;
  
  int tot_nodes = 0;
  int* ptr_tot_nodes = &tot_nodes;
  //need to know number of nodes in advance, to allocate
  //array that will contain the key_values
  nr_nodes(*tree, ptr_tot_nodes);
  tot_nodes--;
  
  array_nodes = (int*) malloc( tot_nodes * sizeof(int) );

  tot_nodes = 0;
  //array of ordered key_values in nodes
  list_inorder(*tree, array_nodes, ptr_tot_nodes);
  
  //finally, creating a balanced tree from the array
  //destroy old tree
  destroy_tree(*tree);
  
  //and re-point root node to null
  *tree = 0;
  
  build_balanced_tree(tree, array_nodes, 0, tot_nodes-1, tot_nodes);
}


//building balanced tree
void build_balanced_tree(struct node** new_tree, int* nodes, int start, int end, int tot_nodes){

  //create balanced tree
  if(start>end){return;}

  //calling recursively to create the tree
  
  int mid = (start+end)/2;
  
  if(mid >= tot_nodes){return;}

  insert_serial(nodes[mid], new_tree);
  
  build_balanced_tree(new_tree, nodes, start, mid-1, tot_nodes);
  build_balanced_tree(new_tree, nodes, mid+1, end, tot_nodes);
  
}
