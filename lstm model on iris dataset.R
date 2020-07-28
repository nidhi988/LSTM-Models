#LSTM using the iris dataset
#The program requires mxnet, which needs to be installed.

cran <- getOption("repos") cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/" 
options(repos = cran) 
install.packages("mxnet") 

library(mxnet)

data(iris)
#We asked R to select from the iris dataset, which consists of 150 lines and five columns, only lines one to four, leaving out the fifth. This procedure will also be performed for multiples of five, so in the end, we will omit every multiple row of five from our selection. We will also omit the fifth column. At the end, we will get 120 rows and four columns. 
x = iris[1:5!=5,-5] 
y = as.integer(iris$Species)[1:5!=5] 

train.x = data.matrix(x) 
train.y = y 

test.x = data.matrix(iris[1:5==5,-5]) 
test.y = as.integer(iris$Species)[1:5==5] 

#The mx.lstm function is called with the input and output values so that the model is trained with the LSTM on the RNN with the dataset
model <- mx.lstm(train.x, train.y, hidden_node=10, out_node=3, out_activation="softmax",                num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9,                eval.metric=mx.metric.accuracy) 
preds = predict(model, test.x) 
pred.label = max.col(t(preds)) 

test.y 
pred.label