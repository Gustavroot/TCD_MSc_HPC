#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "bst.h"


/*
 ******************************************************************************
 * Create a new binary search tree.
 * Return the new pointer.
 ******************************************************************************
 */
bst *bst_create() {
	bst *my_tree;

	if (! (my_tree = (bst *)malloc(sizeof(bst)))) return NULL;

	my_tree->root = NULL;
	my_tree->size = 0;

	return my_tree;
}


/*
 * Helper function:
 * Destroy the nodes in the tree recursively.
 */
void bst_destroy_recursive(struct bstnode_s *node) {
	if (node == NULL) {
		// empty child - do nothing
		return;
	} else if (node->left == NULL && node->right == NULL) {
		// no children, so free this child
		free(node);
		return;
	} 


	// ok, so we have some children, at least on one side, but possibly
	// on both

	if (node->left) {
		// children on one side - free them
		bst_destroy_recursive(node->left);
	}

	if (node->right) {
		// children on one side - free them
		bst_destroy_recursive(node->right);
	}

	// then the parent node itself
	free(node);
}


/*
 ******************************************************************************
 * Destroy the binary search tree cleanly, freeing all memory.
 * If the tree is empty, then nothing to do. Otherwise
 * call the recursive destroy function.
 ******************************************************************************
 */
void bst_destroy(bst *my_tree) {
	if (my_tree == NULL) {
		// nothing to do!
		return;
	}

	// recursively free the nodes
	bst_destroy_recursive(my_tree->root);

	// and finally the binary search tree struct itself
	free(my_tree);
}


/*
 ******************************************************************************
 * Return the current size.
 ******************************************************************************
 */
int bst_size(bst *my_tree) {
	return my_tree->size;
}


/*
 * Helper function:
 * Perform the search recursively - binary search.
 * Return 1 for sucessful search, 0 for failure.
 */
int bst_search_recursive(struct bstnode_s *node, int data) {

	if(node == NULL){
		return 0;
	}
	else if(node->data == data){
		return 1;
	}
	else if(data < node->data){
		return bst_search_recursive(node->left, data);
	}
	else{
		return bst_search_recursive(node->right, data);
	}
}


/*
 ******************************************************************************
 * Search for a value recursively - binary search.
 * If the tree is empty, then not found. Otherwise search
 * recursively.
 * Return 1 for sucessful search, 0 for failure.
 ******************************************************************************
 */
int bst_search(bst *my_tree, int data) {
	if (my_tree->root == NULL) {
		return 0;	// empty tree - return false
	} else {
		return bst_search_recursive(my_tree->root, data);
	}
}


/*
 * Helper function:
 * Recursive insert.
 * Do not insert duplicate values.
 * Return 1 for sucessful insert, 0 for failure.
 */
int bst_insert_recursive(struct bstnode_s **node, int data) {
	//in case there is no tree yet
	if(*node == NULL){
  
		//if root pointing to NULL, then allocate memory
		*node = (struct bstnode_s *)malloc(sizeof(struct bstnode_s));
		(*node)->data = data;

		//point the children nodes to NULL
		(*node)->left = NULL;
		(*node)->right = NULL;
		
		return 1;
	}

	//if value must be at left
	else if(data < (*node)->data){
		return bst_insert_recursive(&(*node)->left, data);
	}

	//if key must be at right
	else if(data > (*node)->data){
		return bst_insert_recursive(&(*node)->right, data);
	}
	//if key already in tree, don't insert
	else{
		return 0;
	}
}


/*
 ******************************************************************************
 * Insert a (unique) value into the tree.
 * Do not insert duplicate values.
 * If the tree is empty, it's the new root. Otherwise,
 * do a recursive insert.
 * Return 1 for sucessful insert, 0 for failure.
 ******************************************************************************
 */
int bst_insert(bst *my_tree, int data) {
	struct bstnode_s *node;

	// empty tree?
	if (my_tree->root == NULL) {
		if (! (node=(struct bstnode_s *)malloc(sizeof(struct bstnode_s))) ) return 0; // return fail
		node->left  = NULL;
		node->right = NULL;
		node->data  = data;

		my_tree->root = node;
		my_tree->size = 1;

		return 1;
	} else {
		if (bst_insert_recursive(&my_tree->root, data)) {
			my_tree->size++;
			return 1;
		} else {
			return 0;
		}
	}
}


/*
 * Helper function:
 * Recursive search of minimum.
 * Return the value of the minimum.
 */
int bst_find_min_recursive(struct bstnode_s *node) {
	if(node->left == NULL){
		return node->data;
	}
	else{
		return bst_find_min_recursive(node->left);
	}
}


/*
 ******************************************************************************
 * Search for min value within tree.
 * If the tree is empty, return 0.
 ******************************************************************************
 */
int bst_find_min(bst *my_tree) {
	// empty tree?
	if (my_tree->root == NULL) {
		return 0;
	} else {
		return bst_find_min_recursive(my_tree->root);
	}
}


/*
 * Helper function:
 * Recursive search of maximum.
 * Return the value of the maximum.
 */
int bst_find_max_recursive(struct bstnode_s *node) {
	if(node->right == NULL){
		return node->data;
	}
	else{
		return bst_find_max_recursive(node->right);
	}
}


/*
 ******************************************************************************
 * Search for max value within tree.
 * If the tree is empty, return 0.
 ******************************************************************************
 */
int bst_find_max(bst *my_tree) {
	// empty tree?
	if (my_tree->root == NULL) {
		return 0;
	} else {
		return bst_find_max_recursive(my_tree->root);
	}
}


/*
 * Helper function:
 * Return the address of the pointer to the node
 * with the smallest values of 'data'.
 */
struct bstnode_s** node_min(struct bstnode_s** node){
  if( (**node).left != 0 ){
    return node_min( &((**node).left) );
  }
  else{
    return node;
  }
}


/*
 * Helper function:
 * Recursive removal of entry.
 * Return 1 if success, 0 if entry did not exist.
 */
int bst_remove_recursive(struct bstnode_s** node, int data) {
	if( *node != NULL ){
		if(data == (*node)->data){

			//multiple cases when removing elem:
			//http://www.algolist.net/Data_structures/Binary_search_tree/Removal

			//no child nodes
			if((*node)->left == NULL && (*node)->right == NULL){
				free(*node);
				*node = NULL;
				return 1;
			}
			//if just one child for this node
			//left child 'inherits'
			else if((*node)->right == NULL){
				//pointer buffer
				struct bstnode_s* buff;
				//releasing and re-directing
				buff = (*node)->left;
				free(*node);
				*node = buff;
			}
			//or right child 'inherits'
			else if((*node)->left == NULL){
				//pointer buffer
				struct bstnode_s* buff;
				//releasing and re-directing
				buff = (*node)->right;
				free(*node);
				*node = buff;
			}
			//if node has two children
			else{
				//look for node with smallest value on right wing
				struct bstnode_s** buff;
				buff = node_min( &((**node).right) );
        
				//replace value in 'to delete' node
				(**node).data = (**buff).data;

				//delete smallest node
				//..before deleting, check if it has no right child:
				if( (**buff).right != 0 ){
					//pointer buffer
					struct bstnode_s* buff2;
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
			//as one node has been removed
			return 1;
		}
		else if(data < (*node)->data){
			return bst_remove_recursive(&((*node)->left), data);
		}
		else{
			return bst_remove_recursive(&((*node)->right), data);
		}
	}
	//in case key not in tree, do nothing
	return 0;
}


/*
 ******************************************************************************
 * Removal of entry.
 * If the tree is empty, return 0, as the entry did not exist.
 ******************************************************************************
 */
int bst_remove(bst *my_tree, int data) {
	// empty tree?
	if (my_tree->root == NULL) {
		return 0;
	} else {
		return bst_remove_recursive(&my_tree->root, data);
	}
}


/*
 * Helper function:
 * Traverse the tree in-order (left, root, right).
 * Append each value to the string 'str'.
 * Assumes that the string has been allocated and has enough
 * space to hold all the values.
 */
void bst_inorder_tostring_recursive(struct bstnode_s *node, char *str) {
	char buf[10];

	if (node == NULL) {
		return;
	} else {
		bst_inorder_tostring_recursive(node->left, str);

		sprintf(buf, "%d ", node->data);
		strcat(str, buf);

		bst_inorder_tostring_recursive(node->right, str);
	}
}


/*
 ******************************************************************************
 * Traverse the tree in-order (left, root, right).
 * Append each value to the string 'str'.
 * Assumes that the string has been allocated and has enough
 * space to hold all the values.
 ******************************************************************************
 */
void bst_inorder_tostring(bst *my_tree, char *str) {
	char buf[10];

	// initialise the string (assumes it has already been allocated)
	*str = '\0';

	// empty tree?
	if (my_tree->root == NULL) {
		return;
	} else {
		bst_inorder_tostring_recursive(my_tree->root->left, str);

		sprintf(buf, "%d ", my_tree->root->data);
		strcat(str, buf);

		bst_inorder_tostring_recursive(my_tree->root->right, str);
	}
}


/*
 * Helper function:
 * Traverse the tree pre-order (root, left, right).
 * Append each value to the string 'str'.
 * Assumes that the string has been allocated and has enough
 * space to hold all the values.
 */
void bst_preorder_tostring_recursive(struct bstnode_s *node, char *str) {
	char buf[10];

	if (node == NULL) {
		return;
	} else {
		sprintf(buf, "%d ", node->data);
		strcat(str, buf);

		bst_preorder_tostring_recursive(node->left, str);
		bst_preorder_tostring_recursive(node->right, str);
	}
}


/*
 ******************************************************************************
 * Traverse the tree pre-order (root, left, right).
 * Append each value to the string 'str'.
 * Assumes that the string has been allocated and has enough
 * space to hold all the values.
 ******************************************************************************
 */
void bst_preorder_tostring(bst *my_tree, char *str) {
	char buf[10];

	// initialise the string (assumes it has already been allocated)
	*str = '\0';

	// empty tree?
	if (my_tree->root == NULL) {
		return;
	} else {
		sprintf(buf, "%d ", my_tree->root->data);
		strcat(str, buf);

		bst_preorder_tostring_recursive(my_tree->root->left, str);
		bst_preorder_tostring_recursive(my_tree->root->right, str);
	}
}


/*
 ******************************************************************************
 * Print the tree to stdout.
 * Use the bst_inorder_tostring() to traverse the tree
 * in-order and print the resultant string
 * to stdout.
 ******************************************************************************
 */
void bst_display(bst *my_tree) {
	/* temp string for display */
	char *str = NULL;

	// malloc 4 chars per entry (' 100'), and enough for twice the current size
	str = (char *) malloc(bst_size(my_tree) * 2 * 4 * sizeof(char));

	bst_inorder_tostring(my_tree, str);

	printf("BST contains: %s\n", str);

	free(str);
}


/*
 * vim:ts=4:sw=4
 */
