#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <time.h>
#include <math.h>


//Compilation:
//	$ gcc bbtree-serial.c -o bbtree-serial -lm

//Execution (where N is an integer, the number of nodes):
//	$ ./bbtree-serial -n N


//Binary tree structure
struct node{
  int key_value;
  struct node *left;
  struct node *right;
};


//EXTRA functions
void print_usage();

//CORE functions
//destroy all tree
void destroy_tree(struct node*);
//inserting new node to the tree
//<unique value tree>
void insert(int key, struct node**);
//function to print tree structure
void print_tree(struct node*);
//balancing tree
void balance(struct node**);
//function to extract element from tree
void extract_elem(int, struct node**);
//look for smallest elem within tree
struct node** smallest(struct node**);

int max_depth(struct node*);



//main code
int main(int argc, char** argv){

  int option, N, rand_buff;
  int i;
  int *tree_depth, t_depth;

  //giving seed to random nrs generator
  srand(time(NULL));

  printf("\nSerial binary tree.\n\n");

  //root node
  struct node *tree = 0;
  //at this point, the tree doesn't exist

  if(argc != 3){
    printf("ERROR: wrong number of input params.\n");
    return 0;
  }

  //getting value of N with getopt
  while ((option = getopt(argc, argv,"n:")) != -1) {
    switch (option) {
      case 'n' :
        N = atoi(optarg);
        break;
      default: print_usage();
        printf("ERROR: incorrect input params.\n");
        return 0;
    }
  }
  
  //check that N val is a positive integer
  if(N <= 0){
    printf("ERROR: N must be > 0.\n");
    return 0;
  }
  
  printf("Inserting data in tree (done when 5 dots)");

  //entering N random elements into tree
  for(i=0; i<N; i++){
  
    tree_depth = 0;
  
    //entering a node
    insert(((int)rand())%N, &tree);
    
    //TODO: call balancing function

    if(i%(N/5) == 0){printf(".");}
  }
  printf("\n\n");
  
  //printing tree after total insertion
  printf("After insertions:\n");
  print_tree(tree);
  printf("\n");

  printf("Extracting data from tree.\n");

  //extracting N random elements from tree
  for(i=0; i<N; i++){
    //entering a node
    extract_elem(((int)rand())%N, &tree);
    
    //TODO: call balancing function
  }
  printf("\n\n");

  //printing tree after some extractions
  printf("After extractions:\n");
  print_tree(tree);
  printf("\n");

  printf("\n");
  //releasing memory
  destroy_tree(tree);

  return 0;
}







//EXTRA functions
void print_usage(){
  printf("./bbtree -n N\n");
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
//<unique value tree>
void insert(int key, struct node **leaf){

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
    insert( key, &(*leaf)->left );
  }
  
  //if key must be at right
  else if(key > (*leaf)->key_value){
    insert( key, &(*leaf)->right );
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


//extracting node with key_value = 'key'
void extract_elem(int key, struct node** leaf){
  if( *leaf != 0 ){
    if(key == (*leaf)->key_value){

      printf("...elem %d extracted\n", key);
    
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
      return extract_elem(key, &((*leaf)->left));
    }
    else{
      return extract_elem(key, &((*leaf)->right));
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


//TODO: add padding
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
  
    //nothing to release on 1st layer
    //if(depth != 1){free(nodes_ptr);}
    
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

    //printf("\n");
    max_d--;
    depth++;
  }
}


//TODO: implement
void balance(struct node** tree){}
