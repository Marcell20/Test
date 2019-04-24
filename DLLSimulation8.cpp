#include <iostream>
#include <string>
using namespace std;

struct node{
	string name;
	int quantity;
	node *next,*prev;
};

node *head = NULL;
node *tail = NULL;

class Function {
	string name;
	int quantity;
	
	public:
		void InsertOrd(){
			string name;
			int quantity;
			node *curr = new (node);
			cout << "Input name : \n";
			cin >> name;
			curr->name = name;
			cout << "Input quantity : \n";
			cin >> quantity;
			curr->quantity = quantity;
			if(head==NULL){
				head = tail = curr;
				curr = curr->next;
			}
			else{
				tail->next = curr;
				curr->next = NULL;
				curr->prev = tail;
				tail = curr;
			}
			
		}
		void ViewOrd(){
			node *curr = head;
			while(curr!=NULL){
				cout << curr->name << " " << curr->quantity << endl;
				curr = curr->next;
			}
		}
		
		void DelOrd(){
			int num;
			node *curr = head;
			cout << "Delete from index ? \n";
			cin >> num;
			int index = 1;
			node *prev = curr;
			while(curr!=NULL){
				if(index==num){
					prev->next = curr->next;
					delete(curr);
					break;
				}
				else{
					index++;
					prev = curr;
					curr = curr->next;
				}
			}
		}
};


int main(){
	int x;
	Function func;
	while(true){
		cout << "MOTORCYCLE PARTS \n";
		cout << "[1] View order \n";
		cout << "[2] Add order \n";
		cout << "[3] Take order \n";
		cout << "[4] Exit \n";
		cout << "INPUT NUMBER : \n";
		cin >> x;
		if(x==1){
			func.ViewOrd();
			cout << "View order \n";
		}
		else if(x==2){
			func.InsertOrd();
			cout << "Add order \n";
		}
		else if(x==3){
			func.DelOrd();
			cout << "Take order \n";
		}
		else if(x==4){
			break;
		}
		
	}
}
