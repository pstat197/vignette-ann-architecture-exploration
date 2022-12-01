#giselle code (will delete later)

library(neuralnet)
library(tidyverse)
library(tidymodels)
library(AppliedPredictiveModeling)
# abalone dataset
data(abalone)
abalone['age'] <- abalone['Rings'] + 1.5

#apply class
sapply(abalone, class)
head(abalone)

#normalize (min-max 0-1)
#abalone$age = (abalone$)



# install.packages("AppliedPredictiveModeling")
library(neuralnet)
library(tidyverse)
library(tidymodels)
library(AppliedPredictiveModeling)
# abalone dataset
data(abalone)

##binary classification (student mental health)

#change factor???
abalone_mine <- abalone
#levels(abalone_mine$Type)[1] <- 1 #1
#levels(abalone_mine$Type)[2] <- 2 #2
#levels(abalone_mine$Type)[3] <- 3 #3
testing1 <- LETTERS[1:4]
testing2 <- c("Female", "Male", "Infant")
factor1 <- factor(sample(testing2, count(abalone_mine), replace = TRUE))
abalone_mine <- data.frame(factor1)
library(qdapTools)
mtabulate(abalone_mine$factor1)


# Normalize the data
maxs <- apply(abalone[2:8], 2, max) 
mins <- apply(abalone[2:8], 2, min)
scaled <- as.data.frame(scale(abalone[2:8], center = mins, 
                              scale = maxs - mins))
abalone['age'] <- abalone['Rings'] + 1.5
age <- abalone['age']

# Split the data into training and testing set
set.seed(12312001)
partitions <- scaled %>% initial_split(prop = 0.8)
partitions2 <- age %>% initial_split(prop = 0.8)
test_ <- testing(partitions)
train_ <- training(partitions)
train_age <- training(partitions2)
test_age <- testing(partitions2)
train2 <- cbind(train_, train_age)
test2 <- cbind(test_, test_age)
# Build Neural Network
nn <- neuralnet(age ~ LongestShell + Diameter + 
                  Height + WholeWeight + ShuckedWeight + VisceraWeight + 
                  ShellWeight, data = train2, hidden = c(2, 2), 
                linear.output = TRUE, stepmax = 500000, learningrate = 500)
write_rds(nn, file = "neural_net_take_1.rds")
nn1 <- read_rds(file = "neural_net_take_1.rds")

###example below
set.seed(500)
library(neuralnet)
library(MASS)
data4 <- Boston
maxs4 <- apply(data4, 2, max)
mins4 <- apply(data4, 2, min)
