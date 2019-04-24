#include <iostream>
#include <string>
using namespace std;

struct node{
	int value;
	string name;
	node *next;
};

node *head = NULL;

void insertHead(int value, string name){
	node *curr = new(node);
	curr->value = value;
	curr->name = name;
	
	if(head==NULL){
		head = curr;
		curr->next= NULL;
	}else{
		curr->next=head;
		head = curr;
	}
}

void insertTail(int value, string name){
	node *curr = new(node);
	curr->value = value;
	curr->name = name;
	
	if(head==NULL){
		head = curr;
		curr->next= NULL;
	}else{
		node *temp = head;
		while(temp->next!=NULL){
			temp = temp->next;
		}
		temp->next = curr;
		curr->next = NULL;
	}
}

void delHead(){
	if(head==NULL){
		cout << "Data Still empty";
	}
	else{
		node *curr = head;
		head = head->next;;
		delete(curr);
	}
};

void delTail(){
	if(head==NULL){
		cout << "Data Still empty";
	}
	else{
		if(head->next=NULL){
			node *curr = head;
			head = NULL;
			delete(curr);
		}
		else{
			node *curr = head;
			node *temp = head->next;
			while(temp->next!=NULL){
				temp = temp->next;
				curr = curr->next;
			}
			curr->next = NULL;
			delete(temp);
		}
	}
};


void view(){
	node *curr =head;
	while(curr!=NULL){
		cout<< curr->value << " " << curr->name <<endl;
		curr = curr->next;
	}
};

int main(){
//	insertHead(11, "A");
//	insertHead(12, "B");
//	insertHead(13, "C");
//	insertHead(14, "D");
	insertTail(10, "AA");
	insertTail(12, "BB");
	insertTail(13, "CC");
	insertTail(14, "DD");
	view();
	return 0;
	
}
