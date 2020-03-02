# SentimentAnalysis
A generative model that classifies hotel reviews the into positive, negative, and truthful, deceitful 

This model trains on the hotel review in chicago area under folder op_spam_training_data. 

Run nblearn3.py path/to/op_spam_training_data will train a Naive Bayes model on the training and write the model parameters to a file named nbmodel.txt

Run nbclassify3.py path/to/test_set will test the model on the test folder that has the same structure as the op_spam_training_data. This program will also write the result into a file named nboutput.txt in the below format: 
deceptive/truthful positive/negative path/to/test_reivew 
