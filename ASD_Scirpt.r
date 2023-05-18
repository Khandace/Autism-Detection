#-----ML coursework-------------------------------------------

# set working directory
setwd(dirname(file.choose()))
getwd()

# read in data from csv file
asd.data <- read.csv("Autism.csv", stringsAsFactors = TRUE)
head(asd.data)
str(asd.data)

#----------------Data Exploration------------------------------------
# check for missing data
apply(asd.data, MARGIN = 2, FUN = function(x) sum(is.na(x)))
library(Amelia)
missmap(asd.data, col = c("black", "grey"), legend = TRUE)
#asd.data <- na.omit(asd.data)

# summarise the numerical variables - see ranges, outliers? approx normal?
summary(asd.data)
boxplot(asd.data[2:21])

# table of asd.data
table(asd.data$asd.data)

# recode asd.data as a factor, indicate all possible levels and label
asd.data$Class.ASD <- factor(asd.data$Class.ASD)

#----------------------BOXPLOTS-------------------------------------------------
#table(asd.data$Age)
#barplot(table(asd.data$Age))

Agegrp<-cut(asd.data$Age, c(17,29,49,64),
            labels=c("17-29", "30-49", "50-64"))
head(Agegrp)
table(Agegrp)
barplot(table(Agegrp),col = "lightblue4",
        border = "dark blue",ylim = c(0,350),
        xlab = "Age Group", main = "Bar Plot of Age Groups")
#-------------------------------------------------------------------------------
boxplot(Age~Gender, data=asd.data,
        horizontal= FALSE,
        names=c("f","m"),
        col=c("darkred","navy"))

boxplot(Result~Gender, data=asd.data,
        horizontal= FALSE,
        names=c("f","m"),
        col=c("turquoise","tomato"))

#-------------------------BARPLOT-----------------------------------------------
barplot(table(asd.data$Class.ASD),
        col=c("darkslategrey","darkslategray4"))


barplot(table(asd.data$used_app_before),
        col=c("darkslategray1","dodgerblue4"))

#---------------------------PIE CHART-------------------------------------------
library(scales)
# install.packages("lessR")
library(lessR)

# Categorical data
Ethnicity<- factor(c(rep("Asian", 124),
                    rep("Black", 44),
                    rep("Hispanic", 14),
                    rep("Latino", 21),
                    rep("Middle-Eastern", 93),
                    rep("Others", 32),
                    rep("Pasifika", 13),
                    rep("South-Asian", 37),
                    rep("Turkish", 7),
                    rep("White-European", 234)))

# Store the variable as data frame
cat <- data.frame(Ethnicity)

# Pie
cols <-  hcl.colors(length(levels(Ethnicity)), "Fall")
PieChart(Ethnicity, data = cat, hole = 0,
         fill = cols,
         labels_cex = 0.6)


table(asd.data$Relation)
# Simple Pie Chart from Proportions with more informative labels - Gender
slices <- c(4, 5, 50, 28, 522)
lbls <- c( "Health care professional", "Others","Parent", "Relative", "Self")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels 
pie(slices, labels = lbls, main="Pie Chart of Relation")

library(plotrix)
slices <- c(4, 5, 50, 28, 522)
lbls <- c("Health care professional", "Others","Parent", "Relative", "Self")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels 
pie3D(slices, labels = lbls, explode=0.1, main="Pie Chart of Relation")

#--------------------------Encoding---------------------------------------------
#Convert dependent and independent attributes to numerical values of 1&0
asd.data$Class.ASD <- ifelse(asd.data$Class.ASD=="YES",1,0)
asd.data$Jaundice <- ifelse(asd.data$Jaundice=="yes",1,0)
asd.data$Autism <- ifelse(asd.data$Autism=="yes",1,0)
asd.data$used_app_before <- ifelse(asd.data$used_app_before=="yes",1,0)
asd.data$Gender <- ifelse(asd.data$Gender=="m",2,1)
head(asd.data)
#------------------------Summary STATS of CLASS.ASD-----------------------------
# Mean, Std. deviation, Maximum and Minimum of "Class.ASD"
summary(asd.data$Class.ASD)
summary(asd.data)

# Summary statistics printed separately
mean(asd.data$Class.ASD)
sd (asd.data$Class.ASD)
max(asd.data$Class.ASD)
min(asd.data$Class.ASD)


# table or proportions with more informative labels
round(prop.table(table(asd.data$Class.ASD)) * 100, digits = 1)


#---------------------Correlation-----------------------------------------------

library(corrplot)
df.asd = subset(asd.data, select = -c(13,16,17,18,19,20))
library(tidyverse)
library(dplyr)
# check structure, row and column number with: glimpse(df)
str(df.asd)
# convert to numeric e.g. from 2nd column to 10th column
df1.asd <- df.asd %>% 
  mutate_at(c(1:15), as.numeric)

#library(ggcorrplot)
#data(df1.asd)
#corr <- round(cor(df1.asd), 1)
#head(corr[, 1:15])

M = cor(df1.asd)
#Order by alphabet and arranged in order of correlation
corrplot(M, method = 'number', order= "alphabet", diag=FALSE)

#correlationMatrix <- cor(df1.asd[,1:15])
library(corrplot)
corrplot(correlationMatrix, method="color", 
         # Show the coefficient value
         addCoef.col="Black", 
         # Cluster based on coeeficient
         order="FPC", 
         # Show only the matrix bottom and avoid the diagonal of ones.
         type="lower", diag=FALSE, 
         # Cross the values that are not significant
         sig.level=0.05)

#library(corrplot)
#corrplot(cor(df1.asd), method = "circle")

#------------------------Modelling----------------------------------------------
# create training (80%) and test data (20%) (data already in random order)

asd_train <- df1.asd[1:487, ]
asd_test <- df1.asd[488:609, ]

# create labels (from first column) for training and test data
asd_train_labels <- df1.asd[1:487, 15]
asd_test_labels <- df1.asd[488:609, 15]

#-------------------------------Training----------------------------------------
# training a model on the data

# load the "class" library
library(class)
# look at help for class and knn

# perform kNN, use k=21 as starting point because SQR(455)
asd_test_pred <- knn(train = asd_train, test = asd_test,
                      cl = asd_train_labels, k=5)

# inspect results for 114 (20%) test observations
asd_test_pred

#-------------------------------------------------------------------------------
# evaluating model performance

# load the "gmodels" library
library(gmodels)
# look at help for gmodels and CrossTable

# Create the cross tabulation of predicted vs. actual
CrossTable(x = asd_test_labels, y = asd_test_pred, prop.chisq=FALSE)
# Inspect FP (0) and FN (2)

library(caret)
confusionMatrix(as.factor(asd_test_labels), asd_test_pred)

library(ROCR)
#Applying ROC and AUC on boosted model of 100 Trials
df1.asd_probability<-predict(df1.asd, df1.asd_test_labels, type="prob")
#bind with test and earlier predicted data
df1.asd_BIND<-cbind(df1.asd_test_labels,df1.asd_test_pred, df1.asd_probability)
head(df1.asd_BIND)
#Create a prediction object
df1.asd_BIND_PRED<-prediction(predictions =df1.asd_BIND$`YES`,
                          labels = df1.asd_BIND$Class.ASD )
#Plot the ROC Curve
at2_BIND_PERF<-performance(at2_BIND_PRED, measure = "tpr", x.measure = "fpr")
plot(at2_BIND_PERF, main="ROC Curve of Decision Tree",
     col="red", print.auc=TRUE, lwd=4)
abline(a=0, b=1, lty=2)

#--------------------Logistic regression----------------------------------------

#df1.asd <- read.csv("Autism.csv", stringsAsFactors = FALSE)

attach(df1.asd)
head(df1.asd)    # The top of the data
str(df1.asd)


# check for missing data
apply(df1.asd, MARGIN = 2, FUN = function(x) sum(is.na(x)))
library(Amelia)
missmap(df1.asd, col = c("black", "grey"), legend = FALSE)
df1.asd <- na.omit(df1.asd)   # remove any missing data

# Change categorical variable to factors
df1.asd$Class.ASD = factor (df1.asd$Class.ASD)
df1.asd$ï..A1_Score  = factor (df1.asd$ï..A1_Score)
df1.asd$A2_Score = factor (df1.asd$A2_Score)
df1.asd$A3_Score = factor (df1.asd$A3_Score)
df1.asd$A4_Score = factor (df1.asd$A4_Score)
df1.asd$A5_Score = factor (df1.asd$A5_Score)
df1.asd$A6_Score = factor (df1.asd$A6_Score)
df1.asd$A7_Score = factor (df1.asd$A7_Score)
df1.asd$A8_Score = factor (df1.asd$A8_Score)
df1.asd$A9_Score = factor (df1.asd$A9_Score)
df1.asd$A10_Score = factor (df1.asd$A10_Score)
df1.asd$Age = factor (df1.asd$Age)
df1.asd$Gender = factor (df1.asd$Gender)
df1.asd$Autism = factor (df1.asd$Autism)
df1.asd$Jaundice = factor (df1.asd$Jaundice)
str(df1.asd)

# check correlations between variables
library(polycor)
df1.asd.cor <- hetcor(df1.asd[-1])
df1.asd.cor$type
round(df1.asd.cor$correlations, 2)

# Split into train and test (at mother ID = 120)
df1.asd_train <- df1.asd[1:487, ]
df1.asd_test <- df1.asd[488:609, ]

Class.ASD_test <- df1.asd_test$Class.ASD
df1.asd_test2 <- df1.asd_test[-15]

# Check proportions of low bith weight in train and test
round(prop.table(table(df1.asd_train$Class.ASD))*100,1)
round(prop.table(table(Class.ASD_test))*100,1)

#-------------------------------------------------------------------------------
# First round use all variables

mylogit1 = glm(Class.ASD ~ ï..A1_Score + A2_Score + A3_Score + A4_Score + A5_Score 
               + A6_Score + A7_Score + A8_Score + A9_Score + A10_Score + Age + Gender
               + Autism +Jaundice,
               data = df1.asd_train, family = "binomial")
summary(mylogit1)

# Calculate Odds Ratio - Exp(b) with 95% confidence intervals (2 tail)
exp(cbind(OR = coef(mylogit1), confint(mylogit1)))

# Assess accuracy
library(gmodels)
CrossTable(x = Class.ASD_test, y = Class.ASD_pred, prop.chisq = FALSE)

library(caret)
confusionMatrix(Class.ASD_pred, Class.ASD_test, positive = "0")

#-------------------------------------------------------------------------------
# Second round excluding variable Age and all the other scores except A9_Score

mylogit2 = glm(Class.ASD ~ A9_Score + Autism + Jaundice + Gender, 
               data = df1.asd_train, family = "binomial")
summary(mylogit2)
# Calculate Odds Ratio - Exp(b) with 95% confidence intervals (2 tail)
exp(cbind(OR = coef(mylogit2), confint(mylogit2)))

#-------------------------------------------------------------------------------
# Predict with mylogit2

df1.asd_te2 <- df1.asd_test[c("A9_Score", "Autism", "Jaundice",  "Gender")]
Class.ASD_pred <- predict.glm(mylogit2, df1.asd_te2)
summary(Class.ASD_pred)
Class.ASD_pred <- ifelse(exp(Class.ASD_pred) > 0.5,1,0)
Class.ASD_pred <- as.factor(Class.ASD_pred)

# Assess accuracy
library(gmodels)
CrossTable(x = Class.ASD_test, y = Class.ASD_pred, prop.chisq = FALSE)

library(caret)
confusionMatrix(Class.ASD_pred, Class.ASD_test, positive = "0")

#----------------------------- support vector machine --------------------------

library(MASS)
library(DMwR2)
library(polycor)
library(kernlab)
library(caret)

#-------------------------------------------------------------------------------
# run support vector machine algorithms
str(df.asd)
#Scale of "Age" attribute to standardize the dataframe within 0-1.
maxs <- apply(df.asd[c(11)], 2, max)
mins <- apply(df.asd[c(11)], 2, min)
df1.asd[c(11)] <- scale(df.asd[c(11)], center = mins, scale = maxs - mins)
str(df1.asd[c(11)])


# Split into train and test (at mother ID = 120)
df1.asd_train <- df1.asd[1:487, ]
df1.asd_test <- df1.asd[488:609, ]
Class.ASD_test <- df1.asd_test$Class.ASD
df1.asd_test <- df1.asd_test[-15]
# run initial model
set.seed(12345)
svm0 <- ksvm(Class.ASD ~ , data = df1.asd_train, kernel = "vanilladot", type = "C-svc")
# vanilladot is a Linear kernel; -- WARNING -- some kernels take a long time

# look at basic information about the model
svm0

# evaluating the model
df1.asd_train.pred0 <- predict(svm0, df1.asd_test)
table(df1.asd_train.pred0, Class.ASD_test)

#-------------------------------------------------------------------------------
#the $ sign can only be used to select variable in dataframe but not atomic vector
is.atomic(Class.ASD)
is.atomic(Class.ASD_test)
is.recursive(Class.ASD)
is.recursive(Class.ASD_test)
Class.ASD_<- data.frame(as.data.frame(Class.ASD)) ######## Convert named vector to data.frame
Class.ASD_test_<- data.frame(as.data.frame(Class.ASD_test)) 
Class.ASD_test_
Class.ASD_
#-------------------------------------------------------------------------------

round(prop.table(table(df1.asd_train.pred0, Class.ASD_test_$Class.ASD_))*100,1)
# sum diagonal for accuracy
sum(diag(round(prop.table(table(df1.asd_train.pred0,Class.ASD_test_$Class.ASD_))*100,1)))

library(gmodels)
CrossTable(x = Class.ASD_test_$Class.ASD_, y = df1.asd_train.pred0, prop.chisq = FALSE)
library(caret)
confusionMatrix(round(prop.table(table(df1.asd_train.pred0, Class.ASD_test_$Class.ASD_))*100,1))


library(rminer)
#svm0.imp <- Importance(svm0, data = df1.asd_train)
#svm0.imp

#-------------------------------------------------------------------------------
# explore improvements of the model by changing the kernel

set.seed(12345)
svm1 <- ksvm(Class.ASD ~ ., data = df1.asd_train, kernel = "rbfdot", type = "C-svc")
# radial basis - Gaussian

# look at basic information about the model
svm1
#-------------------------------------------------------------------------------
# evaluate
df1.asd_train.pred1 <- predict(svm1, df1.asd_test)
table(df1.asd_train.pred1,Class.ASD_test_$Class.ASD_)
round(prop.table(table(df1.asd_train.pred1, Class.ASD_test_$Class.ASD_))*100,1)
# sum diagonal for accuracy
sum(diag(round(prop.table(table(df1.asd_train.pred1, Class.ASD_test_$Class.ASD_))*100,1)))

library(gmodels)
CrossTable(x = Class.ASD_test_$Class.ASD_, y = df1.asd_train.pred1, prop.chisq = FALSE)
library(caret)
confusionMatrix(round(prop.table(table(df1.asd_train.pred1,Class.ASD_test_$Class.ASD_))*100,1))

#RBFDot Kernel Plot
library(ROCR)
library(pROC)
par(pty="s")
ROCPlot<-roc(Class.ASD_test_$Class.ASD ~ predict(svm1,df1.asd_test, type="response"),
             plot=TRUE, print.auc=TRUE,
             col="red", lwd=4, legacy.axes=TRUE,main="ROC Curve")


pred_SVM1 <- predict(svm1, Class.ASD_test, type = "response")
ROCPlot <- roc(Class.ASD_test$Class.ASD ~ as.numeric(pred_SVM1))


#-------------------------------DECISION TREE-----------------------------------
# get the data
# set working directory
setwd(dirname(file.choose()))
getwd()

#-------------------------------------------------------------------------------
# import data file bank.csv and put relevant variables in a data frame
at <- read.csv("Autism.csv", stringsAsFactors = TRUE)

#-------------------------------------------------------------------------------
# examining the structure of the autism data frame
str(at)

# checking for missing values
apply(at, MARGIN = 2, FUN = function(x) sum(is.na(x)))
library(Amelia)
missmap(at, col = c("black", "blue"), legend = FALSE)

at2 <- at
at2 <-at2[-16]
at2 <-at2[-16]
at2 <-at2[-17]
at2 <-at2[-17]
at2 <-at2[-16]
at2 <-at2[-13]
at3 <-at2
# summarizing the variables - see ranges, outliers? approx normal?

summary(at2)

library(tidyverse)
library(caret)
library(C50)
library(plyr)
library(gmodels)
library(ROCR)
library(RWeka)
library(rpart)
library(rpart.plot)
library(dplyr)
library(data.tree)
library(caTools)

#-------------------------------------------------------------------------------
#converting to factors
at2$Class.ASD <- factor (at2$Class.ASD, levels = c("NO", "YES"))
str(at2$Class.ASD)
at2$Gender <- factor(at2$Gender, levels = c("f", "m"))
at2$Ethnicity <- factor(at2$Ethnicity, levels = c("Asian", "Black", "hispanic", "Latino",
                                                  "Middle-Eastern", "others", "Pasifika",
                                                  "South-Asian", "Turkish", "White-European"))
at2$Jaundice <- factor(at2$Jaundice, levels = c("no", "yes"))
at2$Autism <- factor(at2$Autism, levels = c("no", "yes"))


#randomizing, creating and testing datasets
set.seed(1234)
at2_random <- at2[order(runif(609)), ]

# Splitting dataset
split <- sample.split(at2_random, SplitRatio = 0.8)
split

train_sample <- subset(at2_random, split == "TRUE") 
Class.ASD <- subset(at2_random, split == "FALSE") 


# check for the proportion of target variable
round(prop.table(table(train_sample$Class.ASD)) *100,1)
round(prop.table(table(Class.ASD$Class.ASD)) *100,1)


at2_mod<- C5.0(train_sample[-15], train_sample$Class.ASD)
at2_mod
summary(at2_mod)
plot(at2_mod)


#-------------------------------------------------------------------------------
# improving model performance

# pruning the tree to simplify and/or avoid over-fitting
?C5.0Control

set.seed(12345)
at2_prune <- C5.0(train_sample[-15], train_sample$Class.ASD,
                  control = C5.0Control(minCases = 15)) # 1% training obs.
at2_prune
summary(at2_prune)
at2_prune_pred <- predict(at2_prune, Class.ASD)
CrossTable(Class.ASD$Class.ASD, at2_prune_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Class.ASD', 'predicted Class.ASD'))

confusionMatrix(at2_prune_pred, Class.ASD$Class.ASD, positive = "YES")

# boosted decision tree with 100 trials

set.seed(12345)
at2_boost100 <- C5.0(train_sample[-15], train_sample$Class.ASD, control = C5.0Control(minCases = 15), trials = 100)
at2_boost100

at2_boost_pred100 <- predict(at2_boost100, test_sample)
CrossTable(test_sample$Class.ASD, at2_boost_pred100,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Class.ASD', 'predicted Class.ASD'))

confusionMatrix(at2_boost_pred100, test_sample$Class.ASD, positive = "YES")
#-------------------------------------------------------------------------------
library(ROCR)
#Applying ROC and AUC on boosted model of 100 Trials
at2_probability<-predict(at2, test_sample, type="prob")
#bind with test and earlier predicted data
at2_BIND<-cbind(test_sample,at2_prune_pred, at2_probability)
head(at2_BIND)
#Create a prediction object
at2_BIND_PRED<-prediction(predictions =at2_BIND$`YES`,
                          labels = at2_BIND$Class.ASD )
#Plot the ROC Curve
at2_BIND_PERF<-performance(at2_BIND_PRED, measure = "tpr", x.measure = "fpr")
plot(at2_BIND_PERF, main="ROC Curve of Decision Tree",
     col="red", print.auc=TRUE, lwd=4)
abline(a=0, b=1, lty=2)

#---------------------------------AUC-------------------------------------------
#Calculating the AUC VALUE (Area Under the Curve)
at2_AUC<-performance(at2_BIND_PRED, measure = "auc")
at2_AUC@y.values

#-------------------------------------------------------------------------------
# remove all variables from the environment
rm(list=ls())


                                                            








