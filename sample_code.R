# Import Required packages
set.seed(500)
#install.packages("AppliedPredictiveModeling")
library(neuralnet)
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)
library(AppliedPredictiveModeling)
# abalone dataset
data(abalone)

# Normalize the data
maxs <- apply(abalone[2:8], 2, max) 
mins <- apply(abalone[2:8], 2, min)
scaled <- as.data.frame(scale(abalone[2:8], center = mins, 
                              scale = maxs - mins))

# Split the data into training and testing set
set.seed(102722)
partitions <- scaled %>% initial_split(prop = 0.8)
test_ <- testing(partitions)
train_ <- training(partitions)

# Build Neural Network
nn <- neuralnet(medv ~ crim + zn + indus + chas + nox 
                + rm + age + dis + rad + tax + 
                  ptratio + black + lstat, 
                data = train_, hidden = c(5, 3), 
                linear.output = TRUE)

# Predict on test data
pr.nn <- compute(nn, test_[,1:13])

# Compute mean squared error
pr.nn_ <- pr.nn$net.result * (max(data$medv) - min(data$medv)) 
+ min(data$medv)
test.r <- (test_$medv) * (max(data$medv) - min(data$medv)) + 
  min(data$medv)
MSE.nn <- sum((test.r - pr.nn_)^2) / nrow(test_)

# Plot the neural network
plot(nn)

# Plot regression line
plot(test$medv, pr.nn_, col = "red", 
     main = 'Real vs Predicted')
abline(0, 1, lwd = 2)

