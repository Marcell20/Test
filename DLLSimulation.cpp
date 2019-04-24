#include <iostream>
#include <string>
using namespace std;

struct node{
	string name;
	int quantity;
	node *next, *prev;
};

	node *head = NULL;
	node *tail = NULL;
class Function{
	string name;
	int quantity;

public:
	
void insertOrder(){
	string name;
	int quantity;
	node *curr = new(node);
//	curr->name = name;
//	curr->quantity = quantity;
	cout << "Input name : \n";
	cin >> name;
	curr->name = name;
	cout << "Input quantity :  \n";
	cin >> quantity;
	curr->quantity = quantity;
	if(head==NULL){
		head = tail = curr;
		curr->next = NULL;
	}
	else{
		tail->next = curr;
		curr->next = NULL;
		curr->prev = tail;
		tail = curr;
		}
}

void view(){
	node *curr = head;
	while(curr!=NULL){
		cout << curr->name << " " << curr->quantity << endl;
		curr = curr->next;
	}
}

void delFunc(){
	int num;
	node *curr=head;
	cout << "delete from index : \n";
	cin >> num;
	int index = 1;
	node *prev = curr;
	while(curr!=NULL){
		if(index==num){
			prev->next = curr->next;
			delete(curr);
//			curr = curr->next;
			break;
		}
		else{
			index++;
			prev = curr;
			curr = curr->next;
		}
//		curr = head->next;
//		head = curr->next;
//		tail = NULL;
//		tail = curr->prev;	
	}
}
//void delHead(){
//	
//	if(head==NULL){
//		cout << "Data still empty \n";
//	}else if(head==tail){
//		node *curr = head;
//		head = tail = NULL;
//		delete(curr);
//	}
//	else{
//		node *curr = head;
//		head = head->next;
//		head->prev = NULL;
//		delete(curr);
//	}
//}
//
//void delTail(){
//	if(head==NULL){
//		cout << "Data Still empty";
//	}
//	else if(head==tail){
//		node *curr = head;
//		head = tail = NULL;
//		delete(curr);
//	}
//	else{
//		node *curr = tail;
//		node *temp = tail->prev;
//		temp->next = NULL;
//		delete(curr);
//	}
//};
};



int main(){
	int x;
	int y;
	Function func;
	while(true){
	cout << "========================= \n";
	cout << "  BLUE MOTORCYCLE PARTS \n";
	cout << "========================= \n";
	cout << "1. View order list \n";
	cout << "2. Add a new order \n";
	cout << "3. Take Order \n";
	cout << "4. Exit \n";
	cout << "Choose numbers : \n";
	cin >> x;
	
		if(x==1){
			func.view();
			cout << "View order \n";
		}
		else if(x==2){
			func.insertOrder();
			
			cout << "Item added \n";
//			cout<<"1. Input item name \n";
//			cout<<"2. Qty \n";
//			 if(y==1){
//			 	cout << "Input name: \n";
//				func.insertOrder();
//			}
//			else if(y==2){
//				cout << "Input quantity : \n";
//				cin >> y;
//			}
//	
		}
		else if(x==3){
			func.delFunc();
			cout << "Take order \n";
		}
		else if(x==4){
			break;
		}
	}
}
