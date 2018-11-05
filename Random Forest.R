#install.packages("adabag")
#randomForest
#install.packages("randomForest")
library(randomForest)

data<-read.table("/Users/chenchuandong/Desktop/german.csv",
                 header = TRUE, sep = ",", dec=".")
#View(data)
attach(data)
summary(data)
colnames(data)
nrow(data)
head(data)
prop.table(table(data$target))

target<-ifelse(data[,"target"]=="0","bad","good")
tmp<-data[,-21]
data<-cbind(tmp,target)
#attach(data)
#View(data)
#Simple Splitting Based on the Outcome
#Details can be seen:http://topepo.github.io/caret/splitting.html

set.seed(4543)#randomization

splitIndex<-createDataPartition(
  data$target,## the outcome data are needed
  p=0.75,## The percentage of data in the training set
  list=FALSE## The format of the results
)

## The output is a set of integers for the rows of data
## that belong in the training set.

training<-data[splitIndex,]
testing<-data[-splitIndex,]
nrow(training);nrow(testing)
prop.table(table(training$target))
prop.table(table(testing$target))
str(training);str(testing)

#build the model
rf<-randomForest(target ~.,data= training,importance=T)
print(rf)

#importance
varImpPlot(rf,main="random")

#prediction
rf.pr<-predict(rf, testing)

predType <-c()
for (i in 1:length(rf.pr))
{
  if (rf.pr[i]=="good")
  {
    predType[i]<-1
  }
  if (rf.pr[i]=="bad")
  {
    predType[i]<-0
  }
}

trclass<-testing$target
trueType <-c()
for (i in 1:length(trclass))
{
  if (trclass[i]=="good")
  {
    trueType[i]<-1
  }
  if (trclass[i]=="bad")
  {
    trueType[i]<-0
  }
}

# Model evaluation based on confusion matrix
#trueType:true variable. The type of the data is 0 and 1. 
#predType:predict result. The type of the data is 0 and 1. 
#return-->list(...)

ppe<-function(trueType,predType)
{
  #build confusion matrix
  confusionMatrix<-as.matrix(table(trueType,predType))
  
  #get TP、FN、FP、TN
  TP<-confusionMatrix[2,2] #true positive
  FN<-confusionMatrix[2,1] #false negative
  FP<-confusionMatrix[1,2] #false positive
  TN<-confusionMatrix[1,1] #true negative
  
  
  #1.Accuracy
  e.A<-TP/(TP+FP)
  
  #2.Negtive Accuracy
  e.NA<-TN/(TN+FN)
  
  #3.Total Accuracy
  e.TA<-(TP+TN)/(TP+FN+FP+TN)
  
  #4.Error Rate
  e.ER<-FP/(TP+FP)
  
  #5.Negtive Error Rate
  e.NER<-FN/(FN+TN)
  
  #6.Total Error Rate
  e.TER<-1-e.TA
  
  #7.Coverage Rate
  e.CR<-TP/(TP+FN)
  
  #8.Negtive Coverage Rate
  e.NCR<-TN/(FP+TN)
  
  #9.FP Rate
  e.FPR<-FP/(FP+TN)
  
  #10.FN Rate
  e.FNR<-FN/(TP+FN)
  
  #11.F value
  e.F<-2*e.A*e.CR/(e.A+e.CR)
  
  #12.Lift Value
  e.LV<-e.A/((TP+FN)/(TP+FN+FP+TN))
  
  #13.correlation coefficient 
  e.phi<-(TP*TN-FP*FN)/sqrt((TP+FN)*(TN+FP)*(TP+FP)*(TN+FN))
  
  #14.Kappa
  pe<-((TP+FN)*(TP+FP)+(FP+TN)*(FN+TN))/(TP+FN+FP+TN)^2
  e.Kappa<-(e.TA-pe)/(1-pe)
  
  return(list(e.A=e.A,e.NA=e.NA,e.TA=e.TA,e.ER=e.ER,e.NER=e.NER,e.TER=e.TER,
              e.CR=e.CR,e.NCR=e.NCR,e.FPR=e.FPR,e.FNR=e.FNR,e.F=e.F,e.LV=e.LV,
              e.phi=e.phi,e.Kappa=e.Kappa))
}
ppe(trueType, predType)

#plot roc
#install.packages("gplots")
library(ROCR)
#score test data set
pred<-prediction(trueType, predType)
perf <- performance(pred,"tpr","fpr")
plot(perf)

#calculate auc
auc.obj <- performance(pred,"auc")
auc <- auc.obj@y.values[[1]]
auc
