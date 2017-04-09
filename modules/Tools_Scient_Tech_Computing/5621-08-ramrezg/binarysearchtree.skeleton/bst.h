#ifndef __BST_H
#define __BST_H

#include<stdlib.h>

/* a node within the binary search tree, storing an int */
struct bstnode_s {
	int data;	// potentially could be any other data type
	struct bstnode_s *left, *right;	// the left and right children
};

/* the tree itself - stores the root pointer and the size */
typedef struct bst_s {
    struct bstnode_s *root;
    int size;
} bst;



bst *bst_create();
void bst_destroy(bst *my_tree);

int bst_insert(bst *my_tree, int data);

int bst_search(bst *my_tree, int data);
int bst_size(bst *my_tree);

void bst_display(bst *my_tree);

void bst_inorder_tostring(bst *my_tree, char *str);
void bst_preorder_tostring(bst *my_tree, char *str);

#endif

/*
 * vim:ts=4:sw=4
 */
