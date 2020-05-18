import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from tensorflow.keras.models import load_model

np.random.seed(7)


def Deep_model():
	X_test11 = pd.read_json("snli/snli_1.0_test.jsonl", lines = True)
	X_train11 = pd.read_json("snli/snli_1.0_dev.jsonl", lines = True)

	X_testfm = X_test11[['sentence1', 'sentence2', 'gold_label']]
	X_trainfm = X_train11[['sentence1', 'sentence2', 'gold_label']]


	X_testf = X_testfm
	X_testf['sentence'] = X_testfm['sentence1'] +" "+X_testfm['sentence2']
	X_testf = X_testf[['sentence', 'gold_label']]
	X_testf.head()


	X_trainf = X_trainfm
	X_trainf['sentence'] = X_trainfm['sentence1'] +" "+X_trainfm['sentence2']
	X_trainf = X_trainf[['sentence', 'gold_label']]
	X_trainf.head()

	t = X_trainf[['sentence']]
	print(len(t))

	s = X_testf[['sentence']]
	print(len(t))

	#Create an empty list 

	hashmap = {}

	def get_dict(t):
	  
	    
	  # Iterate over each row 
	  Row_list = []

	  for rows in t.itertuples(): 
	      # Create list for the current row 
	      my_list = rows.sentence.split()
	        
	      # append the list to the final list
	      for i in my_list:
	        Row_list.append(i) 
	    
	  # Print the list 
	  print(Row_list)
	  for i in Row_list:
	    if i not in hashmap:
	      hashmap[i] = 1
	    else:
	      hashmap[i] += 1



	# t['data'] = X_trainf[['sentence']]
	print(len(t))
	get_dict(t)
	# s['data'] = X_testf[['sentence']]
	print(len(s))
	get_dict(s)
	t

	{k: v for k, v in sorted(hashmap.items(), key=lambda item: item[1], reverse = True)}


	#sorting and assighing hashmap new value according to freq
	print(hashmap)
	j = 1
	for key in hashmap:
	  hashmap[key] = j 
	  j += 1
	print(hashmap)

	def get_list_dataframe(t):
	  
	  Row_list = []

	  for rows in t.itertuples(): 
	    my_list = rows.sentence.split()
	    l = []
	    for i in my_list:
	      if i in hashmap:
	        l.append(hashmap[i])
	      else:
	        l.append(0)
	    Row_list.append(l)
	  print(Row_list)
	  print(len(Row_list))
	  # X_trainfff = pd.DataFrame(Row_list)
	  # return X_trainfff
	  return Row_list




	t = X_trainf[['sentence']]
	print(t.head())
	x_tr = get_list_dataframe(t)

	s = X_testf[['sentence']]
	print(len(s))
	x_te = get_list_dataframe(s)

	# y_f = X_trainf
	# # x_f_train['sentence'] = x_tr['sentence']

	# y_f = X_testf
	# x_f_test['sentence'] = x_te['sentence']

	example_dict1 = {'neutral':'0','entailment':'1', 'contradiction':'2', '-':'3'}


	y_f_train = X_trainf['gold_label'].apply(lambda value: example_dict1[value])


	y_f_test = X_testf['gold_label'].apply(lambda value: example_dict1[value])

	# print(y_f_train)

	# print(y_f_test)

	# fix random seed for reproducibility
	




	# truncate and/or pad input sequences
	max_review_length = 100
	X_train = sequence.pad_sequences(x_tr, maxlen=max_review_length)
	X_test = sequence.pad_sequences(x_te, maxlen=max_review_length)

	print(X_train.shape)
	print(X_train[1])

	# create the model
	top_words = len(hashmap)


	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	#Refer: https://datascience.stackexchange.com/questions/10615/number-of-parameters-in-an-lstm-model

	# model.fit(X_train, y_f_train, nb_epoch=10, batch_size=64)
	# Final evaluation of the model

	# model.save_weights("SNLI_weights.hdf5")
	# model.save("SNLI_model.h5")

	model = load_model("model/my_model.h5")
	model.load_weights('model/my_weights.hdf5')



	scores = model.evaluate(X_test, y_f_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	y_pred=model.predict_classes(X_test)
	f = open("deepmodel.txt", "w+")
	# f.write("Loss on Test Data : "+str(score[0])+"\n")
	# f.write("Accuracy on Test Data : "+str(score[1])+"\n")
	# f.write("gt_label,pred_label\n") 
	for i in range(10000):
		if y_pred[i] == 0:
			f.write("neutral"+"\n")
		if y_pred[i] == 1:
			f.write("entailment"+"\n")
		if y_pred[i] == 2:
			f.write("contradiction"+"\n")
		if y_pred[i] == 3:
			f.write("-"+"\n")






Deep_model()




def Tfidf_model():
	import pandas as pd
	import numpy as np

	X_test = pd.read_json("snli/snli_1.0_test.jsonl", lines = True)
	X_train = pd.read_json("snli/snli_1.0_train.jsonl", lines = True)

	X_test.head()
	X_test = X_test[['sentence1', 'sentence2', 'gold_label']]
	X_train = X_train[['sentence1', 'sentence2', 'gold_label']]

	X_train.gold_label = pd.factorize(X_train.gold_label)[0]
	X_test.gold_label = pd.factorize(X_test.gold_label)[0]
	print(X_train.head());

	from sklearn.feature_extraction.text import TfidfVectorizer
	from nltk.corpus import words

	tv = TfidfVectorizer(
			ngram_range = (1,2),
	                    sublinear_tf = True,
	                    max_features = 40000)

	train_tv1 = tv.fit_transform(X_train['sentence1'] +" "+ X_train['sentence2'])
	test_tv1 = tv.transform(X_test['sentence1'] +" "+ X_test['sentence2'])


	vocab = tv.get_feature_names()
	print(vocab[:5])

	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve

	train_tv = train_tv1.todense()
	test_tv = test_tv1.todense()

	import pickle




	filename = 'finalized_model.sav'

	loaded_model = pickle.load(open(model/filename, 'rb'))
	result = loaded_model.score(test_tv, X_test['gold_label'])
	print(result)



	y_pred=loaded_model.predict(test_tv)




	# y_pred=model.predict_classes(X_test)
	f = open("”tfidf.txt", "w+")
	# f.write("Loss on Test Data : "+str(score[0])+"\n")
	# f.write("Accuracy on Test Data : "+str(score[1])+"\n")
	# f.write("gt_label,pred_label\n") 
	for i in range(10000):
		if y_pred[i] == 0:
			f.write("neutral"+"\n")
		if y_pred[i] == 1:
			f.write("entailment"+"\n")
		if y_pred[i] == 2:
			f.write("contradiction"+"\n")
		if y_pred[i] == 3:
			f.write("-"+"\n")
	


	from sklearn import metrics

	print("Accuracy:",metrics.accuracy_score(X_test['gold_label'], y_pred))


Tfidf_model()