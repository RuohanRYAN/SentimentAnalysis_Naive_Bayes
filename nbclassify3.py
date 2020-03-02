import string
import sys 
import os 
import re
import math
import random 
import json 
def main(path):
	# data_clean = readData(path)
	out_path = "nboutput.txt"
	with open("nbmodel.txt") as file:
		model = json.load(file)
	train_result_1 = model["pos_neg_classification"]
	train_result_2 = model["tru_dec_classification"]

	# with open("pos_neg_classification.json") as file:
	# 	train_result_1 = json.load(file)
	train_result_1,prior_1,feature_map_1 = unpack(train_result_1)
	# with open("tru_dec_classification.json") as file:
	# 	train_result_2 = json.load(file)
	train_result_2,prior_2,feature_map_2 = unpack(train_result_2)
	result = []
	for comment in os.listdir(path):
		review_path = path+"\\"+comment
		if(not valid_dir(review_path)):
			continue
		for level1 in os.listdir(review_path):
			review_path_level1 = review_path + "\\" + level1
			if(not valid_dir(review_path_level1)):
				continue
			for level2 in os.listdir(review_path_level1):
				review_path_level2 = review_path_level1 + "\\" + level2
				if(not valid_dir(review_path_level2)):
					continue
				for comments in os.listdir(review_path_level2):
					comment_path = review_path_level2 + "\\" + comments
					if(not valid_file(comment_path)):
						continue
					with open(comment_path,'r') as file:
						review = file.read()
						label_1 = predict(review,train_result_1,prior_1,feature_map_1,["pos","neg"])
						label_2 = predict(review,train_result_2,prior_2,feature_map_2,["tru","dec"])
						
						label_1 = "positive" if label_1=="pos" else "negative"
						label_2 = "truthful" if label_2=="tru" else "deceptive"
						result.append(label_2+" "+label_1+" "+comment_path+"\n")
						writeOutpue(out_path,result)
			# print(label_1,label_2)
			# break
def valid_dir(filename):
	return os.path.isdir(filename)
def valid_file(filename):
	return os.path.isfile(filename)
def writeOutpue(out_path,result):
	with open(out_path,"w+") as file:
		for res in result:
			file.write(res)


def predict(review,train_result,prior,feature_map,label_list):
	# print(train_result)
	review_cleaned = list(map(mapper,[review]))[0]
	p_0 = 0
	p_1 = 0
	for token in review_cleaned:
		if(token in feature_map):
			p_0+=math.log(train_result[label_list[0]][feature_map[token]])
			p_1+=math.log(train_result[label_list[1]][feature_map[token]])
	if(p_0>=p_1):
		return label_list[0]
	else:
		return label_list[1]
	# print(review_cleaned)
	# data_clean = list(map(lambda x:(x,"neg"),data_clean))
	# error_rate = predict(data_clean,train_result,prior,feature_map,["pos","neg"])
	# print(error_rate)
	# print(error_rate)


def unpack(train_result):
	feature_map = train_result["feature"]
	prior = train_result["prior"]
	del train_result["feature"]
	del train_result["prior"]
	return train_result,prior,feature_map


def readData(path):
	# print(os.listdir(path))
	data = []
	for comment in os.listdir(path):
		# print(comment)
		review_path = path+"\\"+comment
		with open(review_path,'r') as file:
			data.append(file.read())
	data_clean = list(map(mapper,data))
	return data_clean

# def predict(data_cleaned,train_result,prior,feature_map,label_list):
# 	#data_cleaned: ((featrues),label)
# 	mis = 0
# 	for review in data_cleaned:
# 		p_0 = 0
# 		p_1 = 0
# 		data = review[0]

# 		for token in data:
# 			if(token in feature_map):
# 				p_0 += math.log(train_result[label_list[0]][feature_map[token]])
# 				p_1 += math.log(train_result[label_list[1]][feature_map[token]])
# 		if(p_0>=p_1):
# 			label = label_list[0]
# 		else:
# 			label = label_list[1]
# 		if(label!=review[-1]):
# 			mis+=1
# 	return mis/len(data_cleaned)

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
path = sys.argv[1]
main(path)