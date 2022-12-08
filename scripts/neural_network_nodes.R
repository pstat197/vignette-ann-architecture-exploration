#load libraries
library(neuralnet)
library(tidyverse)
library(tidymodels)
library(AppliedPredictiveModeling)
library(cobalt)

# abalone dataset
data(abalone)
abalone['age'] <- abalone['Rings'] + 1.5

#normalize (min and max to each individual variable)
#long way
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
to_split <- list(letters[1:ncol(abalone_mine)], 
                 1:ncol(abalone_mine))
abalone_dumm <- splitfactor(abalone_mine, split.with = to_split, 
                            drop.first = FALSE)

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
write_rds(nmodel_1, file = "../models/single_nn_1.rds")
single_nn1 <- read_rds(file = "../models/single_nn_1.rds")

write_rds(nmodel_1b, file = "../models/single_nn_2.rds")
single_nn2 <- read_rds(file = "../models/single_nn_2.rds")

#predict on test data
pr.nn1 <- neuralnet::compute(single_nn1, test_all)
pr.nn2 <- neuralnet::compute(single_nn2, test_all)

#compute mean square error
pr_nn1 <- pr.nn1$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r1 <- (test_all$age) * (max(abalone$age) - 
                              min(abalone$age)) + 
  min(abalone$age)
mse_nn1 <- sum((test_r1 - pr_nn1)^2) / nrow(test_all)
#79.00748 or 10.78287

pr_nn2 <- pr.nn2$net.result * (max(age) - min(age))
+ min(age)
test_r2 <- (test_all$age) * (max(test_all$age) - 
                               min(test_all$age)) + 
  min(test_all$age)
mse_nn2 <- sum((test_r2 - pr_nn2)^2) / nrow(test_all)
#79.04422

#plot
plot(single_nn1) #10.119157 error, 4729 steps

plot(single_nn2) #10.12522 error, 69883 steps

#merge age and pr_nn1 together
plot_new <- cbind(test_all, pr_nn1) %>%
  select(age, pr_nn1)

#plot real vs. predicted values

#plot regression line
plot(test_r1, pr_nn1, col = "red", 
     main = "Real vs. Predicted for Single node") 
abline(0, 1, lwd = 2)

#r squared
r2_single1 <- function(test_r1, pr_nn1){
  cor(test_r1, pr_nn1)^2 #0.5169197
}

#other below

lm(pr_nn1 ~ test_all$age)
#intercept: 3.972, slope: 15.571
abline(3.972, 15.571)

#fit polynomial regression
fit1a <- lm(pr_nn1 ~ test_all$age)
fit2a <- lm(pr_nn1~poly(test_all$age, 2, raw = TRUE))
fit3a <- lm(pr_nn1~poly(test_all$age, 3, raw = TRUE))
fit4a <- lm(pr_nn1~poly(test_all$age, 4, raw = TRUE))
fit5a <- lm(pr_nn1~poly(test_all$age, 5, raw = TRUE))

plot(test_all$age, pr_nn1, col = "red", 
     main = "Real vs. Predicted for Single node")

x_axis1 <- seq(0, 1)
lines(x_axis1, predict(fit1a, col = 'green'))
abline(fit1a, col = "blue")
abline(fit2a, col = "green")
abline(fit3a, col = "purple")
abline(fit4a, col = "pink")
abline(fit5a, col = 'black')

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
write_rds(nn_multi1, file = "models/multi_nn_1.rds")
multi_nn1 <- read_rds(file = "models/multi_nn_1.rds")

write_rds(nn_multi2, file = "models/multi_nn_2.rds")
multi_nn2 <- read_rds(file = "models/multi_nn_2.rds")

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
#intercept: 16.962, slope: 3.561
abline(lm(pr_nn3 ~ test_all$age))

plot(test_all$age, pr_nn4, col = "red", 
     main = "Real vs. Predicted for Multi Class pt.2")
lm(pr_nn4 ~ test_all$age)
#intercept: 3.562, slope: 16.950
abline(3.652, 16.950)
