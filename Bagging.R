#install.packages("adabag")
library(adabag)
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
View(data)
#Simple Splitting Based on the Outcome
#Details can be seen:http://topepo.github.io/caret/splitting.html

set.seed(8517)#randomization

splitIndex<-createDataPartition(
  data$target,## the outcome data are needed
  p=0.8,## The percentage of data in the training set
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
typeof(target)
bag<-bagging(target~., training)

bag.pred <- predict.bagging(bag,newdata=testing, newmfinal=10) 
bag.pred$confusion
bag.pred$error
probofgood<-bag.pred$prob[,2]

prclass<-bag.pred$class

predType <-c()

for (i in 1:length(prclass))
{
  if (prclass[i]=="good")
  {
    predType[i]<-1
  }
  if (prclass[i]=="bad")
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

#predType
cbind(pr,testing$target)

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

#plot Kolmogorov–Smirnov(ks) curve 
#prob：probability of predicted result where the result is positive
#labels：the actual result classification 
#n:the number of parts of probability interval for bisection 

ks.plot<-function(prob,labels,n=100)
{
  L<-NROW(prob)
  if(n>L){n<-L}
  
  #build data.frame
  tmpdata<-data.frame(prob,labels)
  
  tmpdata<-tmpdata[order(tmpdata$prob,decreasing=T),]
  tmpdata$rowno=1:L
  
  #Split the probability interval into n parts 
  qus<-quantile(1:L,probs=seq(0,1,1/n))
  culList1<-culList0<-NULL
  out<-mapply(function(i){
    sublab<-tmpdata[tmpdata$rowno>=1 & 
                      tmpdata$rowno<ifelse(i==n,qus[i+1]+0.001,qus[i+1]),]
    culList1<<-c(culList1,sum(sublab$labels==1))
    culList0<<-c(culList0,sum(sublab$labels==0)) 
  },1:n)
  culList1<-culList1/sum(labels==1)
  culList0<-culList0/sum(labels==0)
  
  #Plot KS curve
  plot(1:n,culList1,col='white',xlab=paste(n,"fractile",sep=""),ylab="cumulative percentage",
       xlim=c(0,n),ylim=c(0,1))
  lines(1:n,culList1,col='blue',lwd=2)
  lines(1:n,culList0,col='red',lwd=2)
  legend(0,1,legend=c("cumulative percentage-1","cumulative percentage-0"),col=c("blue","red"),lty=2)
  
  #calculate the value of KS and mark it in the graph
  ks.value<-max(culList1-culList0)
  x<-(1:n)[which.max(culList1-culList0)]
  abline(v=x,lty=2,col='gray',lwd=2)
  return(ks.value)
}

ks.plot(probofgood,trueType)
