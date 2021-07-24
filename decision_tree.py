# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.

# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import graphviz
from sklearn import tree


# Driver code
def main():
	
	print ('--------- Đọc dữ liệu khảo sát -----------\n')
	data_train = pd.read_csv('balance-scale.data',sep= ',', header = None)
	x_train = data_train.values[:, 1:5]
	y_train = data_train.values[:,0]

	print ('--------- Xây dựng cây quyết định theo dữ liệu khảo sát -----------\n')	
	clf_entropy = DecisionTreeClassifier(criterion = "entropy",random_state = 100,max_depth=3, min_samples_leaf=5)
	clf_entropy = clf_entropy.fit(x_train, y_train)

	print ('--------- Đọc dữ liệu test để đánh giá độ chính xác của cây quyết định -----------\n')
	data_test = pd.read_csv('balance-scale-test.data',sep= ',', header = None)
	x_test = data_test.values[:, 1:5]
	y_test = data_test.values[:,0]

	print ('--------- Đánh giá độ chính xác -----------\n')
	y_predict = clf_entropy.predict(x_test)
	print ("Confusion Matrix entropy: \n", confusion_matrix(y_test, y_predict))
	print ("Accuracy entropy: \n", accuracy_score(y_test,y_predict)*100)
	print("Report entropy: \n ",classification_report(y_test, y_predict, labels=np.unique(y_predict)))
	
	print ('--------- Vẽ đồ họa cây quyết định -----------\n')
	feature_cols = [ '[Ngoại hình]', '[Học vấn]', '[Tài sản]','[Quan hệ]']
	dot_data = tree.export_graphviz(clf_entropy, out_file=None, 
                            	feature_names=feature_cols,  
                            	class_names=['[Chưa xác định]','[Đã có người yêu]','[Chưa có người yêu]'],
                                filled=True)
	graph = graphviz.Source(dot_data, format="png") 
	graph.render("decision_tree_have_lover")

# Calling main function
if __name__=="__main__":
	main()
