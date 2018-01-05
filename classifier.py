
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn import preprocessing

from sklearn.cross_validation import KFold
from sklearn.utils import shuffle
from IPython.display import Image
from sklearn.metrics import f1_score, accuracy_score

#from IPython.display import Image

dataMatTrain = pd.read_csv("Output/article_word_freq2.csv")    # Read the data
#print dataMatTrainOrg.head()
#df = pd.read_csv("train.csv")

# dataMatTrain = shuffle(dataMatTrain)
# dataMatTrain = dataMatTrain.reset_index(drop=True)

#print dataMatTrain.head()
#print("* iris types:", df["Sex"].unique())
print("* Category Types:", dataMatTrain["Category"].unique())

#features = list(dataMatTrain.columns[1:5])
#print("* features:", features)

# #print dataMatTrain
# # Impute median Age for NA Age values
# # new_age_var = np.where(titanic_train["Age"].isnull(), # Logical check
# #                        28,                       # Value if check is true
# #                        titanic_train["Age"])     # Value if check is false

# # titanic_train["Age"] = new_age_var

#Initialize label encoder
label_encoder = preprocessing.LabelEncoder()

#Convert Sex variable to numeric
encoded_category = label_encoder.fit_transform(dataMatTrain["Category"])
#print (" Encoded categories " , encoded_category)

# Initialize model
tree_model = tree.DecisionTreeClassifier()

# Train the model
#tree_model.fit(X,Y)

#Save tree as dot file
# with open("tree2.txt", 'w') as f:
#      f = tree.export_graphviz(tree_model, 
#                               feature_names=features, 
#                               out_file=f)


# Image("tree2.png")               # Display image*

# Get survival probability
#preds = tree_model.predict_proba(X)

# #mat = pd.crosstab(preds[:,0], titanic_train["died"])

fold_accuracy = []

#titanic_train["died"] = encoded_sex

cv = KFold(n=len(dataMatTrain),  # Number of elements
           n_folds=5            # Desired number of cv folds
          )       # Set a random seed

#outputStr = ["", ""]
totalAccuracy = [0,0,0,0,0]
totalFScore = [0,0,0,0,0]
count = 1
accuracyArr = []
fScoreArr = []

arr = [10,20,30,40,50]



for train_index, valid_index in cv:
	train = dataMatTrain.loc[train_index] # Extract train data with cv indices
 	valid = dataMatTrain.loc[valid_index] # Extract valid data with cv indices

 	#features = list(train.columns[1:5])
 	#encoded_category = label_encoder.fit_transform(train["Category"])
 	totalAccuracy = [0,0,0,0,0]
	totalFScore = [0,0,0,0,0]

 	for index in range(len(arr)):

	 	ytrain = label_encoder.fit_transform(train["Category"])
	 	Xtrain = train[list(train.columns[1: arr[index]])]
	 	
	 	model = tree_model.fit(Xtrain , ytrain)

	 	#features = list(valid.columns[1:5])
	 	#encoded_category = label_encoder.fit_transform(valid["Category"])
	 	
	 	ytest = label_encoder.fit_transform(valid["Category"])
	 	Xtest = valid[list(valid.columns[1:arr[index]])]
	 	
	 	#valid_acc = model.score(Xtest , ytest)
	 	#fold_accuracy.append(valid_acc)  
	 	y_pred = tree_model.predict(Xtest) 

	 	ac_score = accuracy_score(ytest, y_pred)
	 	fscore = f1_score(ytest, y_pred, average = "weighted")
	 	
	 	#outputStr.append(str(arr[index])+ " " + str(ac_score) +" "+str(fscore)+"\n")
	 	totalAccuracy[index] = ac_score
	 	totalFScore[index] = fscore

	 	# print ac_score
	 	# print fscore
	 	# print "\n"
	#print totalAccuracy
	accuracyArr.append(totalAccuracy)
	fScoreArr.append(totalFScore)



for num in range(5):
	#print accuracyArr[num]
	with open("Output/newTree/treeClassifier_unShuffled_%s.txt" % num , 'w') as f: 
		for index in range(len(arr)):
			
			outputStr = str(arr[index])+ " " + str(accuracyArr[num][index]) +" " + str(fScoreArr[num][index]) +" \n"
			f.write(outputStr)


with open("Output/newTree/treeClassifier_unShuffled_avg.txt" , 'w') as f: 
	for index in range(len(arr)):
		avgAccuracy = 0;
		avgFScore = 0;
		for num in range(5):
			print str(num) + " " + str(arr[index])+" " + str(accuracyArr[num][index])
			avgAccuracy += accuracyArr[num][index];
			avgFScore += fScoreArr[num][index];

		avgAccuracy = avgAccuracy/5;
		avgFScore = avgFScore/5;
		outputStr = str(arr[index])+" "+ str(avgAccuracy) +" " + str(avgFScore) +" \n"
		f.write(outputStr)


		#outputStr[index] += str(totalAccuracy[index]) +" " + str(totalFScore[index]) +"\n"
		
			

# for train_fold, valid_fold in cv:
# 	train = dataMatTrain.loc[train_fold] # Extract train data with cv indices
#  	valid = dataMatTrain.loc[valid_fold] # Extract valid data with cv indices

#  	#train.sample(frac=1)
 	
#  	#print train_fold
#  	#print valid_fold 
#  	#print train.columns[0:1]
#  	#print valid[u'SN  ']

#  	features = list(train.columns[1:5])
#  	encoded_category = label_encoder.fit_transform(train["Category"])
#  	y = encoded_category
#  	X = train[features]

#  	model = tree_model.fit(X , y)

#  	features = list(valid.columns[1:5])
#  	encoded_category = label_encoder.fit_transform(valid["Category"])
#  	y = encoded_category
#  	X = valid[features]

#  	valid_acc = model.score(X , y)
#  	fold_accuracy.append(valid_acc)    

# with open("Output/treeClassifier_200_shuffled_5.txt", 'w') as f: 
# 	for valid_acc in fold_accuracy:
# 		f.write(str(valid_acc) + "\n")
# 	f.write( str(sum(fold_accuracy)/len(fold_accuracy)))

# print("Accuracy per fold: ", fold_accuracy, "\n")
# print("Average accuracy: ", sum(fold_accuracy)/len(fold_accuracy))





