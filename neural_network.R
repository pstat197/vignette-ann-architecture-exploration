# Import Required packages
#install.packages("AppliedPredictiveModeling")
library(neuralnet)
library(tidyverse)
library(tidymodels)
library(AppliedPredictiveModeling)
# abalone dataset
data(abalone)

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

# Predict on test data
pr.nn <- neuralnet::compute(nn1, test2)

# Compute mean squared error
pr.nn_ <- pr.nn$net.result * (max(age) - min(age)) 
+ min(age)
test.r <- (test2$age) * (max(test2$age) - min(test2$age)) + 
  min(test2$age)
MSE.nn <- sum((test.r - pr.nn_)^2) / nrow(test_)

# Plot the neural network
plot(nn)

# Plot regression line
plot(test2$age, pr.nn_, col = "red", 
     main = 'Real vs Predicted')
abline(0, 1, lwd = 2)

