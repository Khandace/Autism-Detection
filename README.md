# Aims and Objectives

The main aim of this paper is to use my understanding of machine learning to transform data on autism screening for adults into actionable intelligence, to build a model capable of detecting individuals who are most likely to suffer from ASD by highlighting.
*	Individuals with a relative born with PDD (Pervasive Developmental Disorder)
* Individuals born with jaundice and/or have a relative with PDD.
Based on the objective of this paper the intention is to find the best model for classifying or predicting autism
This entire process is practically otherwise known as advanced decision-making- predictive analytics and machine learning.  This will be done using R-Studio.


# METHODOLOGY

The methodology can simply be described as the approach taken to ensure all necessary steps needed to prepare and explore the data critically are carefully taken. This ensures that proper analysis, forecasting, and predictions are drawn with much accuracy and with little to no errors.  The data is first collected, prepared, and explored to see the impact of the independent variables on the dependent variable such as CLASS/ASD. 

# Data Acquisition & Preparation

The data for this project was taken from the UCI Machine Learning Repository website (UCI website) as shown below:
Source of Data: Autism Screening Adult Dataset 
 
# Data Analysis

This engulfs the entire project because it is the process of acquiring, preparing, cleaning, exploring, visualizing, and modeling data to gain meaningful insight for decision-making. 

# Description of Dataset

The dataset is all the data collected on a subject matter usually in a tabulated format with several instances and objects, attributes, or features. The dataset is the autism screening of adults both male and female who are between the age of 17years and more, born with jaundice, or have a relative who has PDD and falls within the White-European, Latino, Black, Middle Eastern, Asian, and others. It also recorded whether the user has used the screening app before and who completed the test whether it was completed by a health care professional, parent, relative, or by an individual.

# Sample Size

The dataset has 704 instances and 21 features, in this data 10 behavioral characteristics were recorded and were classified into A1-A10_score, and any individual that showed 7 or more of these behaviors was classified as autistic.

# Data Cleaning

It is the process of identifying and correcting (if possible) or deleting all inaccurate, incomplete, or corrupt records and illegal characters in a dataset.
There are a few errors within the ASD dataset that needs to be corrected such as:
•	Misspelt words like 'jundice' should be jaundice, 'austim' to autism, and 'contry_of_res' to country.
•	Duplicates: There are 2 levels of others under Ethnicity  ' Others' and 'Others' need to be corrected to just one level, say Other.
•	Illegal characters: Correct Class/ASD to Class.ASD this because R-programming does not recognize the symbol ''/''.
•	Also, the countries and some other observations have quotes, this needs to be erased.

# DATA EXPLORATION

It can be termed as a basic data analysis approach that helps in understanding the dataset at full length in other to carry out further analysis. Exploring the data helps to understand each feature or object in the dataset and how they impact the dependent variable. Also, helps to highlight which areas or patterns within the dataset to further explore.

# Head of Dataset

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/974602fb-59af-4954-abd3-a15a44f8e587)

# Structure of Dataset

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/397fc22d-089a-4916-b497-c3433096ebf4)

# Statistical Summary of the Dataset

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/c85455be-c8ad-4cb3-9e13-c54da865be36)

The summary of data above shows that under the feature Class.ASD, 515 NO represents the individuals without ASD and 189 YES represent individuals with autism, under gender, there are 337 females and 367 males, 69 of these people were born with jaundice and 635 were not while 91 of them have relatives with PDD/autism and 613 without. Only 12 people used the screening app and 690 did not. Taking a closer look at the data relation and ethnicity have 95 observations that are not recorded this could be because of missing data, and oversight during the collation and collection of data or the data is just unknown.

# Missing data

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/37cc33da-86af-418f-b5dc-70e2a048d7b6)

The missingness map above shows that 2 observations of Age are missing, and all others are not even though the main data showed there were other observations like Ethnicity and Relation missing some values this was not captured by the missingness map. Both Relation and Ethnicity have 95 missing observations.

# Deleted Rows

The data had several empty rows or missing observations which could not be filled using my discretion, therefore it was necessary to delete those rows and deal with the rows with complete or no missing values. Initially, the data contained 704 rows but after deleting the rows with missing values we are left with 609, most of these rows were relation and ethnicity (95 rows each) including the 2 rows for age with missing values were also deleted.

# Outlier

While going through my data to understand its characteristics and how best to explore it to give me a critical overview of what the data entails, it was noticed in the age column, apart from 2 missings values there was an extreme value under age which was 383 since most people or no one lives that long, also the data had several errors which needed to be cleaned it was reasonable to ascertain that this age could be a result of typing mistake and replace with age 38.

# Summary Data after Deletion of Rows

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/486bc078-cfc6-4e85-8ec8-dec6657b9387)

# Overview of Dependent Variable

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/f0085010-9696-419f-8b71-284ed91b49ec)

The statistics of the dependent variable show that most individuals tested negative (NO) which is a bit biased towards the project since the main aim of this paper is to predict the presence of Autism. Also, the difference between the median and mean shows that the dependent variable is not evenly distributed.

# Implementation and Data Visualisation 

# Boxplots

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/963c8ea5-fef4-4f21-a4c2-8105c77ebd37)


![image](https://github.com/Khandace/Autism-Detection/assets/95150377/c84f6db3-bae0-4df3-9ad4-de9456f73a82)

The male and female distributions, as well as their ages, are depicted in the boxplots above. The participants range in age from late teens to early sixties. Because the median is not in the centre of the box and the whiskers are not the same on both sides, neither boxplot is normally distributed. Outliers in boxplots are extreme values rather than errors.

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/9ee8c360-6ff3-46ce-a4ea-66ecf8512265)

The distribution of screening findings for both genders is shown in the boxplots above, and the distributions are symmetric. The whiskers are all the same length, and the median appears to be in the middle of the box. During the screening, both genders appear to have equal representation.

# Pie Chart and Bar plot

# Pie Chart on Ethnicity

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/000a1ad0-e3f4-44b8-bd26-267ca710e7d2)

The percentage of each ethnicity present during the screening procedure is depicted in the pie chart above. White Europeans account for 38% of the population, followed by Asians and Middle Easterners with 20% and 15%, respectively. Other forms of ethnicity were in the minority.

# Bar plot on Class.ASD

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/1cd640d4-cfd1-4522-b931-ccb960cdf002)

The distribution of Class.ASD is highlighted in the barplot above. It reveals that most people do not fit into Class.ASD, indicating that their scores are below 7. Individuals without autism are represented by the "No" barplot, whereas those with autism are represented by the "YES" barplot.

# Pie Chart on Relation

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/a90d19b4-02b3-4039-ab6e-1023f5d92246)

The data above shows that 86 percent of people (self) completed the screening, followed by parents and relatives with 8% and 5%, respectively, and healthcare professionals and others with 1%.

# Bar plot on Used_app_before

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/2957a960-b3c6-4453-8a21-d35eb9274760)

The bar plot shows that most people did use the app for their screening.

# Bar plot for Age (Age Group)

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/60935d36-6636-4c6c-b3ca-5348f11f752f)

The above barplot denotes the distribution of ages. From the dataset, we have ages from 17 -64 and these were grouped into 3 to highlight the spread between the ages from late teens to late twenties, thirties to late forties, and fifties to late sixties. It shows that based on the data most of the individuals were between the ages of 17-29 and the minority were between 50 to 64 years.

# Correlation 

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/c6a8dbdd-3b8c-41aa-8713-c160a57c392e)

The correlation matrix shows which independent variable has the highest correlation with the dependent variable. It can be observed the A9_Score has the highest correlation of 0.64 and it is positive while gender had the lowest and negative correlation. The scores seemed to have the strongest correlation with the dependent variable. 

# Data Preparation for Modelling

# Dropping Columns

This process involves reducing the number of variables within the dataset, especially the ones that do not offer any significance to the prediction or do not wish to carry on with them; this can be termed variable reduction. Finally, 15 variables were selected.

Use_app_before: The usage of the app by an individual or relation does not have any direct impact on the outcome of the dependent. It cannot be used to determine whether an individual has ASD or not, therefore would play a significant role in the prediction.

Country of Residence: There are too many levels under this variable and will be too complex to work. This includes Ethnicity as well.

Relation: Who completed the test does not play a significant role in the prediction of ASD in an individual and due to this, it dropped because it has no impact.

Age Description: Age description only tells us the age range of the individuals who took the test, and it has only one level and cannot be useful in the classification process.

Result: Result was dropped based on my discretion that A1-A10_Score all add up to get Result which further translates to Class.ASD, I do not need all these variables since they are kind of the same. Also, the result seems more of a result for the prediction, and it is what I am expecting the machine to predict for me.

# One-Hot Encoding
In one-hot encoding, the categorical variables within a column are assigned 1 or 0 (true or false) as their new values. For this data set, the categorical variables are Yes or No, where Yes is equal to 1 and No represents 0.

The encoding was done on the new data frame created from the main dataset.

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/3d08d043-21f4-4943-a55b-c322eed7af9b)

# Training

During this step and with the reaming 15 variables were used for the modeling by partitioning the data into two sets, that’s the training set and the testing set. 80% of the data was used for training and testing 20% reaming the first 487 rows in the data will be trained while the rest will be tested. 

# Testing
The next step is to test the train model against the test model to determine whether the model can make accurate predictions.

# Model Performance 

# K Nearest Neighbour

# Cross Table  

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/b1af0fca-7d6f-472e-8179-fa559cb5cf1c)

# Confusion Matrix  

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/fd7f7056-7b9d-4dfd-9ce4-eb10bbb97dd1)

# Best Model

# perform kNN, use k=5 as starting point because SQR(455)
asd_test_pred <- knn(train = asd_train, test = asd_test,
                      cl = asd_train_labels, k=5)
                      

# Logistic Regression

# Cross Table 

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/01d2d6a9-6009-4e3e-a792-a0ded31147a2)

# Confusion Matrix 

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/6e498ac7-4bf9-4636-a6ef-61a585cb914f)

# Best Model

# Run the final model with all instances
df1.asd2 <- df1.asd[c("A9_Score", "Class.ASD")]
mylogit3 = glm(Class.ASD ~ A9_Score, data = df1.asd2, family = "binomial")
summary(mylogit3)
exp(cbind(OR = coef(mylogit3), confint(mylogit3)))

# Support Vector Machine

# Cross Table

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/cfc0693b-9fae-4901-b799-c1166807475c)

# Confusion Matrix 

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/d5b5c222-5bcb-4ac3-9f70-d621fd9c2538)

# Best Model

svm1 <- ksvm(Class.ASD ~ ., data = df1.asd_train, kernel = "rbfdot", type = "C-svc")

# Decision Tree

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/75bafe52-a575-4a84-aba5-90e0e5591e98)

# Best Model

at2_prune <- C5.0(train_sample[-16], train_sample$Class.ASD,
C5.0Control (minCases = 15, trials = 100) # 1% training obs.
plot(at2_prune)
summary(at2_prune)
at2_prune_pred <- predict(at2_prune, test_sample)
library(gmodels)
CrossTable(test_sample$Class.ASD, at2_prune_pred, prop.chisq = FALSE, prop.c = FALSE,
           dnn = c("actual class", "predicted class"))
           
# ROC Curve 

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/a1ef5672-2b09-4188-ad97-ceea96fcd656)


at2_AUC@y.values


0.9516002

From the decision tree, it can be seen that 2 of the variables that highly correlated to the dependent variable formed the root nodes. The model has an accuracy of approximately 95% with significantly higher specificity and sensitivity. It has a Kappa of 0.89% which means there is a strong agreement.

# Discussion 

The classification process was carried out using KNN, Logistic Regression, SVM, and Decision Tree for each model 15 variables were used in building the model. 
Out of all 4 models, KNN predicted an accuracy of 98% and a kappa of 96% with an error rate of 2%, this means that KNN has an excellent agreement in terms of truth values compared to the rest of the models. It can be said that KNN has proven to be the best model for detecting autism in adults in this paper.
The decision tree selected 2 of the most correlated features of the dependent variable as the important variables for classifying Class.ASD. A6_score(being able to know if someone is bored) and A5_score(No difficulty in reading between the lines when communicating with someone).
Also, since the given data is a bit biased towards NO it would be ideal to use the Kappa as a metric rather than accuracy and though SVM(vanilladot) has a Kappa of 1 it seems to be an overfitted model and since KNN has the 2 highest Kappa of 96% and not overfitted makes it a better model.

![image](https://github.com/Khandace/Autism-Detection/assets/95150377/d38ad410-2af1-488f-8d64-adb164f4cf70)

# Conclusion

The idea of autism screening is to create a broad perspective on the extent of the traits and how best to treat them at the early stages. According to (Thabtah and Peebles, 2020)existing screening systems, such as AQ, Q-CHAT, and a slew of others, rely on simple math, with scoring functions that sum the scores of individual responses. These score functions were used during the screening technique. Because it was designed using handcrafted criteria, it can be accused of being subjective. Therefore, improving the assessment process is one of the most critical considerations in ASD screening research. Individuals and their families will benefit from a more efficient and accurate service. The models in this paper will most likely aid in the prediction of autism though it did not take into consideration toddlers, the models can be further modified to predict autism in toddlers.
A personal reflection on the analysis indicated that most individuals are most likely to fall under ASD if they have these traits A9_score, A6_score and A5_score rather than inheriting it from a relative.

