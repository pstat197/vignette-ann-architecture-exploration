#giselle code (will delete later)

library(neuralnet)
library(tidyverse)
library(tidymodels)
library(AppliedPredictiveModeling)
library(cobalt)
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

#another way: factor variables to dummy variables: Type
to_split <- list(letters[1:ncol(abalone_mine)], 
                 1:ncol(abalone_mine))
abalone_dumm <- splitfactor(abalone_mine, split.with = to_split, 
                            drop.first = FALSE)
abalone_dumm

#split to training and testing
set.seed(1234)
partitions_all <- abalone_dumm %>% initial_split(prop = 0.8)
test_all <- testing(partitions_all)
train_all <- training(partitions_all)

#build one hidden node with 3 split type
###website example
nmodel_1 <- neuralnet(age ~ Type_F + Type_I + Type_M + LongestShell + 
                        Diameter + Height + WholeWeight + ShuckedWeight + 
                        VisceraWeight + ShellWeight, data = train_all, 
                      hidden = 1, threshold = 0.01, 
                      learningrate.limit = NULL, 
                      learningrate.factor = list(minus = 0.5, plus = 1.2),
                      algorithm = "rprop+")
###sammy example
nmodel_1b <- neuralnet(age ~ Type_F + Type_I + Type_M + LongestShell + 
                         Diameter + Height + WholeWeight + ShuckedWeight + 
                         VisceraWeight + ShellWeight, data = train_all, 
                       hidden = 1, linear.output = TRUE, stepmax = 500000, 
                       learningrate = 500)
write_rds(nmodel_1, file = "single_nn_1.rds")
single_nn1 <- read_rds(file = "single_nn_1.rds")

write_rds(nmodel_1b, file = "single_nn_2.rds")
single_nn2 <- read_rds(file = "single_nn_2.rds")

#predict on test data
pr.nn1 <- neuralnet::compute(single_nn1, test_all)
pr.nn2 <- neuralnet::compute(single_nn2, test_all)

#compute mean square error
pr_nn1 <- pr.nn1$net.result * (max(age) - min(age))
+ min(age)
test_r1 <- (test_all$age) * (max(test_all$age) - 
                              min(test_all$age)) + 
  min(test_all$age)
mse_nn1 <- sum((test_r1 - pr_nn1)^2) / nrow(test_all)
#79.00748

pr_nn2 <- pr.nn2$net.result * (max(age) - min(age))
+ min(age)
test_r2 <- (test_all$age) * (max(test_all$age) - 
                               min(test_all$age)) + 
  min(test_all$age)
mse_nn2 <- sum((test_r2 - pr_nn2)^2) / nrow(test_all)
#79.04422

#plot
plot(single_nn1) #10.119157 error, 4729? steps

plot(single_nn2) #10.12522 error, 69883 steps

#plot regression line
plot(test_all$age, pr_nn1, col = "red", 
     main = "Real vs. Predicted for Single node")
lm(pr_nn1 ~ test_all$age)
#intercept: 3.972, slope: 15.571
abline(3.972, 15.571)

plot(test_all$age, pr_nn2, col = "red", 
     main = "Real vs. Predicted for Single node pt.2")
lm(pr_nn2 ~ test_all$age)
#intercept: 3.966, slope: 15.587
abline(3.966, 15.587)

#multiple hidden nodes
set.seed(1234)
#website example
nn_multi1 <- neuralnet(age ~ Type_F + Type_I + Type_M + LongestShell + 
                        Diameter + Height + WholeWeight + ShuckedWeight + 
                        VisceraWeight + ShellWeight, data = train_all,
                       algorithm = "rprop+",
                      hidden = c(2,2), threshold = 0.01, 
                      stepmax = 500000)
#sammy example
nn_multi2 <- neuralnet(age ~ Type_F + Type_I + Type_M + LongestShell + 
                        Diameter + Height + WholeWeight + ShuckedWeight + 
                        VisceraWeight + ShellWeight, data = train_all,
                      hidden = c(2,2), linear.output = TRUE, stepmax = 500000, 
                      learningrate = 500)
write_rds(nn_multi1, file = "multi_nn_1.rds")
multi_nn1 <- read_rds(file = "multi_nn_1.rds")

write_rds(nn_multi2, file = "multi_nn_2.rds")
multi_nn2 <- read_rds(file = "multi_nn_2.rds")

###fix code below
#predict on test data
pr.nn3 <- neuralnet::compute(multi_nn1, test_all)
pr.nn4 <- neuralnet::compute(multi_nn2, test_all)

#compute mean square error
pr_nn3 <- pr.nn3$net.result * (max(age) - min(age))
+ min(age)
test_r3 <- (test_all$age) * (max(test_all$age) - 
                               min(test_all$age)) + 
  min(test_all$age)
mse_nn3 <- sum((test_r3 - pr_nn3)^2) / nrow(test_all)
#79.50398

pr_nn4 <- pr.nn4$net.result * (max(age) - min(age))
+ min(age)
test_r4 <- (test_all$age) * (max(test_all$age) - 
                               min(test_all$age)) + 
  min(test_all$age)
mse_nn4 <- sum((test_r4 - pr_nn4)^2) / nrow(test_all)
#79.73448

#plot
plot(multi_nn1) #9.031362 error, 12087 steps

plot(multi_nn2) #9.072675 error, 9675 steps

#plot regression line
plot(test_all$age, pr_nn3, col = "red", 
     main = "Real vs. Predicted for Multi Class")
lm(pr_nn3 ~ test_all$age)
#intercept: 17.083, slope: 3.504
abline(3.502, 17.083)

#fit multi regression
fit1 <- lm(pr_nn3 ~ test_all$age)


#multi line regression
fit1 <- lm(pr_nn3~test_all$agee)

plot(test_all$age, pr_nn2, col = "red", 
     main = "Real vs. Predicted for Single node pt.2")
lm(pr_nn2 ~ test_all$age)
#intercept: 3.966, slope: 15.587
abline(3.966, 15.587)

####testing code below

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
