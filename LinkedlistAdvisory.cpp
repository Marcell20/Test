#include <iostream>
#include <string>
using namespace std;

struct node{
	int value;
	string name;
	node *next,*prev;
};

node *head = NULL;
node *tail = NULL;

void insertHead(int value, string name){
	node *curr = new(node);
	curr->value = value;
	curr->name = name;
	
	if(head==NULL){
		head = tail = curr;
		curr->next= NULL;
		curr->prev= NULL;
	}else{
		curr->next=head;
		curr->prev= NULL;
		head->prev = curr;
		head = curr;
		
	}
}

void insertTail(int value, string name){
	node *curr = new(node);
	curr->value = value;
	curr->name = name;
	
	if(head==NULL){
		head = tail = curr;
		curr->next= NULL;
	}else{
//		node *temp = head;
//		while(temp->next!=NULL){
//			temp = temp->next;
//		}
//		temp->next = curr;
//		curr->next = NULL;
		tail->next = curr;
		curr->next = NULL;
		curr->prev = tail;
		tail = curr;
	}
}

void delHead(){
	if(head==NULL){
		cout << "Data Still empty";
	}
	else if(head==tail){
		node *curr = head;
		head = tail = NULL;
		delete(curr);
	}
	else{
		node *curr = head;
		head = head->next;
		head->prev = NULL;
		
		delete(curr);
	}
};

void delTail(){
	if(head==NULL){
		cout << "Data Still empty";
	}
	else if(head==tail){
		node *curr = head;
		head = tail = NULL;
		delete(curr);
	}
	else{
		node *curr = tail;
		node *temp = tail->prev;
		temp->next = NULL;
		delete(curr);
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
	delTail();
	delTail();
	view();
	return 0;
	
}
