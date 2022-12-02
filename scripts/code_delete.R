#giselle code (will delete later)

library(neuralnet)
library(tidyverse)
library(tidymodels)
library(AppliedPredictiveModeling)
# abalone dataset
data(abalone)
abalone['age'] <- abalone['Rings'] + 1.5

#normalize (min and max to each individual variable)
abalone_mine <- abalone #made a copy()
abalone_mine$LongestShell <- (abalone_mine$LongestShell - 
                                min(abalone_mine$LongestShell)) / 
  (max(abalone_mine$LongestShell) - min(abalone_mine$LongestShell))
abalone_mine$Diameter <- (abalone_mine$Diameter - 
                                min(abalone_mine$Diameter)) / 
  (max(abalone_mine$Diameter) - min(abalone_mine$Diameter))
abalone_mine$Height <- (abalone_mine$Height - 
                                min(abalone_mine$Height)) / 
  (max(abalone_mine$Height) - min(abalone_mine$Height))
abalone_mine$WholeWeight <- (abalone_mine$WholeWeight - 
                                min(abalone_mine$WholeWeight)) / 
  (max(abalone_mine$WholeWeight) - min(abalone_mine$WholeWeight))
abalone_mine$ShuckedWeight <- (abalone_mine$ShuckedWeight - 
                                min(abalone_mine$ShuckedWeight)) / 
  (max(abalone_mine$ShuckedWeight) - min(abalone_mine$ShuckedWeight))
abalone_mine$VisceraWeight <- (abalone_mine$VisceraWeight - 
                                min(abalone_mine$VisceraWeight)) / 
  (max(abalone_mine$VisceraWeight) - min(abalone_mine$VisceraWeight))
abalone_mine$ShellWeight <- (abalone_mine$ShellWeight - 
                                min(abalone_mine$ShellWeight)) / 
  (max(abalone_mine$ShellWeight) - min(abalone_mine$ShellWeight))
abalone_mine$Rings <- (abalone_mine$Rings - min(abalone_mine$Rings)) / 
  (max(abalone_mine$Rings) - min(abalone_mine$Rings))
abalone_mine$age <- (abalone_mine$age - min(abalone_mine$age)) / 
  (max(abalone_mine$age) - min(abalone_mine$age))

#factor variables to dummy variables: Type
table(abalone_mine$Type)
levels(abalone_mine$Type)
head(model.matrix(~Type, data = abalone_mine))
abalone_mine$Type <- relevel(abalone_mine$Type, ref = "I")
head(model.matrix(~Type, data = abalone_mine))
abalone_matrix <- model.matrix(~Type+LongestShell+Diameter+Height+
                                 WholeWeight+ShuckedWeight+
                                 VisceraWeight+ShellWeight+age, 
                               data = abalone_mine)
colnames(abalone_matrix)
col_list <- paste(c(colnames(abalone_matrix[, -
                                              c(1, 11)])), collapse = "+")
col_list <- paste(c("age~", col_list), collapse="")
f <- formula(col_list)

#one hidden node
library(neuralnet)
set.seed(1234)
nmodel1 <- neuralnet(f, data = abalone_matrix, hidden = 1, 
                    threshold = 0.01, 
                    learningrate.limit = NULL, 
                    learningrate.factor = list(minus = 0.5, plus = 1.2),
                    algorithm = "rprop+")

# Predict on test data
pr.nn2 <- neuralnet::compute(nmodel, abalone_matrix)

# Compute mean squared error
pr.nn_2 <- pr.nn2$net.result * (max(age) - min(age)) 
+ min(age)
test.r2 <- (abalone_matrix$age) * (max(abalone_matrix$age) 
                                   - min(abalone_matrix$age)) + 
  min(abalone_matrix$age)
MSE.nn <- sum((test.r - pr.nn_)^2) / nrow(test_)

# Plot the neural network
plot(nmodel1)

#normalize (min-max 0-1)
#abalone$age = (abalone$)



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
