import string
import sys 
import os 
import re
import math
import random 
import json 
def readData(path):
	neg_dec_path = path+ "\\negative_polarity\\deceptive_from_MTurk"
	neg_tru_path = path+"\\negative_polarity\\truthful_from_Web"
	pos_dec_path = path+ "\\positive_polarity\\deceptive_from_MTurk"
	pos_tru_path = path+ "\\positive_polarity\\truthful_from_TripAdvisor"

	neg_dec = []
	neg_tru = []
	pos_dec = []
	pos_tru = []

	neg_dec_train = []
	neg_tru_train = []
	pos_dec_train = []
	pos_tru_train = []
	neg_dec_dev = []
	neg_tru_dev = []
	pos_dec_dev = []
	pos_tru_dev = []
	for folder in os.listdir(neg_dec_path):
		if(folder[0]!="f"):
			continue
		else:
			path_to_folder = neg_dec_path+"\\"+folder
			for review in os.listdir(path_to_folder):
				path_to_txt = path_to_folder+"\\"+review
				neg_dec.append(readtext(path_to_txt))
				if(folder!="fold1"):
					neg_dec_train.append(readtext(path_to_txt))
				else:
					neg_dec_dev.append(readtext(path_to_txt))
	for folder in os.listdir(neg_tru_path):
		if(folder[0]!="f"):
			continue
		else:
			path_to_folder = neg_tru_path+"\\"+folder
			for review in os.listdir(path_to_folder):
				path_to_txt = path_to_folder+"\\"+review
				neg_tru.append(readtext(path_to_txt))
				if(folder!="fold1"):
					neg_tru_train.append(readtext(path_to_txt))
				else:
					neg_tru_dev.append(readtext(path_to_txt))
	for folder in os.listdir(pos_dec_path):
		if(folder[0]!="f"):
			continue
		else:
			path_to_folder = pos_dec_path+"\\"+folder
			for review in os.listdir(path_to_folder):
				path_to_txt = path_to_folder+"\\"+review
				pos_dec.append(readtext(path_to_txt))
				if(folder!="fold1"):
					pos_dec_train.append(readtext(path_to_txt))
				else:
					pos_dec_dev.append(readtext(path_to_txt))
	for folder in os.listdir(pos_tru_path):
		if(folder[0]!="f"):
			continue
		else:
			path_to_folder = pos_tru_path+"\\"+folder
			for review in os.listdir(path_to_folder):
				path_to_txt = path_to_folder+"\\"+review
				pos_tru.append(readtext(path_to_txt))
				if(folder!="fold1"):
					pos_tru_train.append(readtext(path_to_txt))
				else:
					pos_tru_dev.append(readtext(path_to_txt))
	num_feature = [i for i in range(10,501,20)]
#### positive and negtive classification ####################
	# neg_dev =  neg_dec_dev+neg_tru_dev
	# pos_dev =  pos_dec_dev+pos_tru_dev
	# neg_dev_cleaned = list(map(mapper,neg_dev))
	# pos_dev_cleaned = list(map(mapper,pos_dev))
	# data_dev_cleaned = list(map(lambda x:(x,"neg"),neg_dev_cleaned)) + list(map(lambda x:(x,"pos"),pos_dev_cleaned))
	# for num in num_feature:
	# 	train_result,prior,feature_map,accuracy = pos_neg_classification(neg_dec_train,neg_tru_train,pos_dec_train,pos_tru_train,num)
	# 	dev_accuracy = predict(data_dev_cleaned,train_result,prior,feature_map,["pos","neg"])
	# 	print("training error rate is : "+str(accuracy) +". with " + str(num)+" features.")
	# 	print("development error rate is: "+str(dev_accuracy)+". with " + str(num)+" features.")
#####
	output = {}
	train_result,prior,feature_map,accuracy = pos_neg_classification(neg_dec,neg_tru,pos_dec,pos_tru,310)
	train_result["prior"] = prior
	train_result["feature"] = feature_map
	output["pos_neg_classification"] = train_result
	# writefile(train_result,prior,feature_map,"pos_neg_classification.json")
#### truth and deciet classification ########################
	# num_feature = [i for i in range(2000,5000,300)]
	# dec_dev =  pos_dec_dev+neg_dec_dev
	# tru_dev =  pos_tru_dev+neg_tru_dev
	# dec_dev_cleaned = list(map(mapper,dec_dev))
	# tru_dev_cleaned = list(map(mapper,tru_dev))
	# data_dev_cleaned = list(map(lambda x:(x,"dec"),dec_dev_cleaned)) + list(map(lambda x:(x,"tru"),tru_dev_cleaned))
	# for num in num_feature:
	# 	train_result,prior,feature_map,accuracy = tru_dec_classification(neg_dec_train,neg_tru_train,pos_dec_train,pos_tru_train,num)
	# 	dev_accuracy = predict(data_dev_cleaned,train_result,prior,feature_map,["tru","dec"])
	# 	print("training error rate is : "+str(accuracy) +". with " + str(num)+" features.")
	# 	print("development error rate is: "+str(dev_accuracy)+". with " + str(num)+" features.")
####
	train_result,prior,feature_map,accuracy = tru_dec_classification(neg_dec,neg_tru,pos_dec,pos_tru,1100)
	# writefile(train_result,prior,feature_map,"tru_dec_classification.json")
	train_result["prior"] = prior
	train_result["feature"] = feature_map
	output["tru_dec_classification"] = train_result
	with open("nbmodel.txt","w") as file:
		json.dump(output,file)
def tru_dec_classification(neg_dec,neg_tru,pos_dec,pos_tru,n):
	tru = pos_tru+neg_tru
	dec = pos_dec+neg_dec
	tru_cleaned = list(map(mapper,tru))
	dec_cleaned = list(map(mapper,dec))
	unique_word = set()
	for review in dec_cleaned:
		for word in review:
			unique_word.add(word)
	for review in tru_cleaned:
		for word in review:
			unique_word.add(word)
	unique_word = sorted(unique_word)
	MI_word = sorted(MutualInfo(tru_cleaned,dec_cleaned,unique_word),key = lambda x :x[1],reverse = True)[0:n]
	feature = sorted(MI_word,key = lambda x:x[0])
	feature_map = {}#### feature map that holds the hash value #####
	for i in range(len(feature)):
		feature_map[feature[i][0]] = i
	data_feature = buildFeature(feature_map,tru_cleaned,dec_cleaned,"tru","dec")
	# print(data_feature)
	# data_feature = buildFeature_bernoulli(feature_map,tru_cleaned,dec_cleaned,"tru","dec")
	# print(data_feature)
	train_result,prior = train(data_feature,feature_map)
	# train_result,prior = train_bernoulli(data_feature,feature_map)
	# print(train_result)
	data_cleaned = list(map(lambda x:(x,"dec"),dec_cleaned)) + list(map(lambda x:(x,"tru"),tru_cleaned))
	# print(data_cleaned)
	accuracy = predict(data_cleaned,train_result,prior,feature_map,["tru","dec"])
	# print(accuracy)
	return train_result,prior,feature_map,accuracy
def buildFeature_bernoulli(feature_map,pos_cleaned,neg_cleaned,class1,class2):
	data_num = []
	for review in pos_cleaned:
		feat = [0 for i in range(len(feature_map))]
		for word in review:
			if(word in feature_map):
				feat[feature_map[word]] = 1
		data_num.append((feat,class1))
	for review in neg_cleaned:
		feat = [0 for i in range(len(feature_map))]
		for word in review:
			if(word in feature_map):
				feat[feature_map[word]] = 1
		data_num.append((feat,class2))
	return data_num	
def train_bernoulli(data_feature,feature_map):
	train_result = {}
	Sum = {}
	prior = {}
	for feat in data_feature:
		label = feat[-1]
		prior[label] = prior.get(label,0)+1
		data = feat[0]
		Sum[label] = Sum.get(label,0)+sum(data)
		for index in range(len(data)):
			content = train_result.get(label,[0 for i in range(len(data))])
			content[index]+=data[index]
			train_result[label] = content
	for key in prior.keys():
		prior[key] = prior[key]/len(data_feature)
	for key in train_result.keys():
		feature = train_result[key]
		N = Sum[key]
		B = len(feature)
		for i in range(B):
			feature[i] = (feature[i]+1)/(N+len(Sum))
		train_result[key] = feature
	return train_result,prior
def pos_neg_classification(neg_dec,neg_tru,pos_dec,pos_tru,n):
	neg =  neg_dec+neg_tru
	pos =  pos_dec+pos_tru
	neg_cleaned = list(map(mapper,neg))
	pos_cleaned = list(map(mapper,pos))

	unique_word = set() ### set to store unique words
	for review in neg_cleaned:
		for word in review:
			unique_word.add(word)
	for review in pos_cleaned:
		for word in review:
			unique_word.add(word)
	unique_word = sorted(unique_word)
	MI_word = sorted(MutualInfo(pos_cleaned,neg_cleaned,unique_word),key = lambda x :x[1],reverse = True)[0:n]

	feature = sorted(MI_word,key = lambda x:x[0])
	feature_map = {}#### feature map that holds the hash value #####
	for i in range(len(feature)):
		feature_map[feature[i][0]] = i
	data_feature = buildFeature(feature_map,pos_cleaned,neg_cleaned,"pos","neg")
	train_result,prior = train(data_feature,feature_map)
	data_cleaned = list(map(lambda x:(x,"neg"),neg_cleaned)) + list(map(lambda x:(x,"pos"),pos_cleaned))
	accuracy = predict(data_cleaned,train_result,prior,feature_map,["pos","neg"])
	return train_result,prior,feature_map,accuracy
def train(data_feature,feature_map):
	train_result = {}
	Sum = {}
	prior = {}
	for feat in data_feature:
		label = feat[-1]
		prior[label] = prior.get(label,0)+1
		data = feat[0]
		Sum[label] = Sum.get(label,0)+sum(data)
		for index in range(len(data)):
			content = train_result.get(label,[0 for i in range(len(data))])
			content[index]+=data[index]
			train_result[label] = content
	for key in prior.keys():
		prior[key] = prior[key]/len(data_feature)
	for key in train_result.keys():
		feature = train_result[key]
		N = Sum[key]
		B = len(feature)
		for i in range(B):
			feature[i] = (feature[i]+1)/(N+B)
		train_result[key] = feature
	return train_result,prior

def predict(data_cleaned,train_result,prior,feature_map,label_list):
	#data_cleaned: ((featrues),label)
	mis = 0
	for review in data_cleaned:
		p_0 = 0
		p_1 = 0
		data = review[0]

		for token in data:
			if(token in feature_map):
				p_0 += math.log(train_result[label_list[0]][feature_map[token]])
				p_1 += math.log(train_result[label_list[1]][feature_map[token]])
		if(p_0>=p_1):
			label = label_list[0]
		else:
			label = label_list[1]
		if(label!=review[-1]):
			mis+=1
	return mis/len(data_cleaned)



def buildFeature(feature_map,pos_cleaned,neg_cleaned,class1,class2):
	data_num = []
	for review in pos_cleaned:
		feat = [0 for i in range(len(feature_map))]
		for word in review:
			if(word in feature_map):
				feat[feature_map[word]]+=1
		data_num.append((feat,class1))
	for review in neg_cleaned:
		feat = [0 for i in range(len(feature_map))]
		for word in review:
			if(word in feature_map):
				feat[feature_map[word]]+=1
		data_num.append((feat,class2))
	return data_num

	##### train the pos and neg classifier #####


def MutualInfo(pos_cleaned,neg_cleaned,unique_word):
	result = []
	for word in unique_word:
		N11 = 0
		for review in pos_cleaned:
			if word in review:
				N11+=1
		N01 = len(pos_cleaned)-N11

		N10 = 0 
		for review in neg_cleaned:
			if(word in review):
				N10+=1
		N00 = len(neg_cleaned)-N10

		N = N11+N01+N10+N00
		N_1 = N11 + N01
		N1_ = N11 + N10
		N_0 = N10 + N00
		N0_ = N01 + N00

		a = (N*N11)/(N1_*N_1) if (N*N11)/(N1_*N_1)!=0 else 1
		b = (N*N01)/(N0_*N_1) if (N*N01)/(N0_*N_1)!=0 else 1
		c = (N*N10)/(N1_*N_0) if (N*N10)/(N1_*N_0)!=0 else 1 
		d = (N*N00)/(N0_*N_0) if (N*N00)/(N0_*N_0)!=0 else 1 
		# print(a,b,c,d)
		I = (N11/N)*math.log2 (a) + (N01/N)*math.log2(b) + (N10/N)*math.log2(c) + (N00/N)*math.log2(d)
		result.append((word,I))
	return result

def mapper(review):
	result = []
	review = re.sub('[0-9]','',review)
	review_no_pun = removePun(review).lower().split(" ")
	stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
	for word in review_no_pun:
		if(word in stopwords):
			continue
		if(word != ""):
			result.append(word)
	return result

def removePun(s):
	trans = str.maketrans(string.punctuation," "*len(string.punctuation))
	return s.translate(trans)

def readtext(path):
	result = ""
	with open(path,"r") as file:
		result = file.read()
	return result

def writefile(train_result,prior,feature_map,path):
	train_result["prior"] = prior
	train_result["feature"] = feature_map
	with open(path,'w') as file:
		json.dump(train_result,file)


path = sys.argv[1]
readData(path)