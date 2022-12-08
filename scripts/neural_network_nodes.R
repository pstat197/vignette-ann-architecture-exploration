#load libraries
library(neuralnet)
library(tidyverse)
library(tidymodels)
library(AppliedPredictiveModeling)
library(cobalt)
library(car)

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

#build one hidden node
###HIDDEN = 1
nmodel_1 <- neuralnet(age ~ Type_F + Type_I + Type_M + LongestShell + 
                        Diameter + Height + WholeWeight + ShuckedWeight + 
                        VisceraWeight + ShellWeight, data = train_all, 
                      hidden = 1, threshold = 0.01, 
                      learningrate.limit = NULL, 
                      learningrate.factor = list(minus = 0.5, plus = 1.2), 
                      act.fct = "logistic", algorithm = "rprop+")

write_rds(nmodel_1, file = "models/single_nn_1.rds")
single_nn1 <- read_rds(file = "models/single_nn_1.rds")

#predict on test data
pr.nn1 <- neuralnet::compute(single_nn1, test_all)

#compute mean square error
pr_nn1 <- pr.nn1$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r1 <- (test_all$age) * (max(abalone$age) - 
                              min(abalone$age)) + 
  min(abalone$age)
mse_nn1 <- sum((test_r1 - pr_nn1)^2) / nrow(test_all)
#10.8165

#plot
plot(single_nn1) #10.12676 error, 3621 steps

#merge age and pr_nn1 together
plot_new <- cbind(test_r1, pr_nn1) %>%
  as.data.frame()
colnames(plot_new)[2] <- "pr_nn1"

#plot real vs. predicted values
res <- avPlots(lm(pr_nn1 ~ test_r1, data = plot_new), 
               main = "Real vs. Predicted for Single node")
fit1 <- lsfit(res$test_r1[,1], res$test_r1[,2])
fit1$coefficients

#r squared
r2_single1 <-  cor(res$test_r1[,1], res$test_r1[,2])^2 #0.5138825

###HIDDEN = 5
nmodel_2 <- neuralnet(age ~ Type_F + Type_I + Type_M + LongestShell + 
                          Diameter + Height + WholeWeight + ShuckedWeight + 
                          VisceraWeight + ShellWeight, data = train_all,
                        algorithm = "rprop+", act.fct = "logistic",
                        hidden = 5, threshold = 0.01, learningrate.limit = NULL,
                      learningrate.factor = list(minus = 0.5, plus = 1.2))

write_rds(nmodel_2, file = "models/single_nn_2.rds")
single_nn2 <- read_rds(file = "models/single_nn_2.rds")

#predict on test data
pr.nn2 <- neuralnet::compute(single_nn2, test_all)

#compute mean square error
pr_nn2 <- pr.nn2$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r2 <- (test_all$age) * (max(abalone$age) - 
                               min(abalone$age)) + min(abalone$age)
mse_nn2 <- sum((test_r2 - pr_nn2)^2) / nrow(test_all)
#10.3677

#plot
plot(single_nn2) #8.834897 error, 12863 steps

#merge age and pr_nn2 together
plot_new2 <- cbind(test_r2, pr_nn2) %>%
  as.data.frame()
colnames(plot_new2)[2] <- "pr_nn2"

#plot real vs. predicted values
res2 <- avPlots(lm(pr_nn2 ~ test_r2, data = plot_new2), 
               main = "Real vs. Predicted for Single node (5)")
fit2 <- lsfit(res2$test_r2[,1], res2$test_r2[,2])
fit2$coefficients

#r squared
r2_single2 <-  cor(res2$test_r2[,1], res2$test_r2[,2])^2 #0.5672913

###HIDDEN = 10?
nmodel_3 <- neuralnet(age ~ Type_F + Type_I + Type_M + LongestShell + 
                        Diameter + Height + WholeWeight + ShuckedWeight + 
                        VisceraWeight + ShellWeight, data = train_all, 
                      hidden = 15, threshold = 0.01, 
                      learningrate.limit = NULL, #act.fct = "logistic", 
                      learningrate.factor = list(minus = 0.5, plus = 1.2), 
                      algorithm = "rprop+")

write_rds(nmodel_3, file = "models/single_nn_3.rds")
single_nn3 <- read_rds(file = "models/single_nn_3.rds")

#predict on test data
pr.nn3 <- neuralnet::compute(single_nn3, test_all)

#compute mean square error
pr_nn3 <- pr.nn3$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r3 <- (test_all$age) * (max(abalone$age) - 
                               min(abalone$age)) + min(abalone$age)
mse_nn3 <- sum((test_r3 - pr_nn3)^2) / nrow(test_all)
#??

#plot
plot(single_nn3) #??? error, ??? steps

#merge age and pr_nn3 together
plot_new3 <- cbind(test_r3, pr_nn3) %>%
  as.data.frame()
colnames(plot_new3)[2] <- "pr_nn3"

#plot real vs. predicted values weighted
res3 <- avPlots(lm(pr_nn3 ~ test_r3, data = plot_new3), 
                main = "Real vs. Predicted for Single node (??)")
fit3 <- lsfit(res3$test_r3[,1], res3$test_r3[,2])
fit3$coefficients

#r squared
r2_single3 <-  cor(res3$test_r3[,1], res3$test_r3[,2])^2 #0.5672913

#multiple hidden nodes
set.seed(1234)
###HIDDEN = (2,2)
nn_multi1 <- neuralnet(age ~ Type_F + Type_I + Type_M + LongestShell + 
                        Diameter + Height + WholeWeight + ShuckedWeight + 
                        VisceraWeight + ShellWeight, data = train_all,
                       algorithm = "rprop+", act.fct = "logistic",
                      hidden = c(2,2), threshold = 0.01, 
                      stepmax = 500000)

write_rds(nn_multi1, file = "models/multi_nn_1.rds")
nn_multi1 <- read_rds(file = "models/multi_nn_1.rds")

#predict on test data
pr.nn4 <- neuralnet::compute(nn_multi1, test_all)

#compute mean square error
pr_nn4 <- pr.nn4$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r4 <- (test_all$age) * (max(abalone$age) - 
                               min(abalone$age)) + min(abalone$age)
mse_nn4 <- sum((test_r4 - pr_nn4)^2) / nrow(test_all)
#?????

#plot
plot(nn_multi1) #???? error, ??? steps

#merge age and pr_nn4 together
plot_new4 <- cbind(test_r4, pr_nn4) %>%
  as.data.frame()
colnames(plot_new4)[2] <- "pr_nn4"

#plot real vs. predicted values
res4 <- avPlots(lm(pr_nn4 ~ test_r4, data = plot_new4), 
                main = "Real vs. Predicted for Multi node (2,2)")
fit4 <- lsfit(res4$test_r4[,1], res4$test_r4[,2])
fit4$coefficients

#r squared
r2_single4 <-  cor(res4$test_r4[,1], res4$test_r4[,2])^2 #????

###HIDDEN(5,3)
nn_multi2 <- neuralnet(age ~ Type_F + Type_I + Type_M + LongestShell + 
                         Diameter + Height + WholeWeight + ShuckedWeight + 
                         VisceraWeight + ShellWeight, data = train_all,
                       algorithm = "rprop+", act.fct = "logistic",
                       hidden = c(5,3), threshold = 0.01, 
                       stepmax = 500000)
write_rds(nn_multi2, file = "models/multi_nn_2.rds")
nn_multi2 <- read_rds(file = "models/multi_nn_2.rds")

#predict on test data
pr.nn5 <- neuralnet::compute(nn_multi2, test_all)

#compute mean square error
pr_nn5 <- pr.nn5$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r5 <- (test_all$age) * (max(abalone$age) - 
                               
                               min(abalone$age)) + min(abalone$age)
mse_nn5 <- sum((test_r5 - pr_nn5)^2) / nrow(test_all)
mse_nn5
#?????

#plot
plot(nn_multi2) #???? error, ??? steps

#merge age and pr_nn2 together
plot_new5 <- cbind(test_r5, pr_nn5) %>%
  as.data.frame()
colnames(plot_new5)[2] <- "pr_nn5"

#plot real vs. predicted values
res5 <- avPlots(lm(pr_nn5 ~ test_r5, data = plot_new5), 
                main = "Real vs. Predicted for Multi node (5,3)")
fit5 <- lsfit(res5$test_r5[,1], res5$test_r5[,2])
fit5$coefficients

#r squared
r2_single5 <-  cor(res5$test_r5[,1], res5$test_r5[,2])^2 #????

###HIDDEN()
nn_multi3 <- neuralnet(age ~ Type_F + Type_I + Type_M + LongestShell + 
                         Diameter + Height + WholeWeight + ShuckedWeight + 
                         VisceraWeight + ShellWeight, data = train_all,
                       algorithm = "rprop+", act.fct = "logistic",
                       hidden = c(8, 6), threshold = 0.01, 
                       stepmax = 500000)

write_rds(nn_multi3, file = "models/multi_nn_3.rds")
nn_multi3 <- read_rds(file = "models/multi_nn_3.rds")

#predict on test data
pr.nn6 <- neuralnet::compute(nn_multi3, test_all)

#compute mean square error
pr_nn6 <- pr.nn6$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r6 <- (test_all$age) * (max(abalone$age) - 
                               min(abalone$age)) + min(abalone$age)
mse_nn6 <- sum((test_r6 - pr_nn6)^2) / nrow(test_all)
#?????

#plot
plot(nn_multi3) #???? error, ??? steps

#merge age and pr_nn2 together
plot_new6 <- cbind(test_r6, pr_nn6) %>%
  as.data.frame()
colnames(plot_new6)[2] <- "pr_nn4"

#plot real vs. predicted values
res6 <- avPlots(lm(pr_nn6 ~ test_r6, data = plot_new6), 
                main = "Real vs. Predicted for Multi node (??,?)")
fit6 <- lsfit(res6$test_r6[,1], res6$test_r6[,2])
fit6$coefficients

#r squared
r2_single6 <-  cor(res6$test_r6[,1], res6$test_r6[,2])^2 #????




###extra:

#predict on test data
pr.nn3 <- neuralnet::compute(multi_nn1, test_all)

#compute mean square error
pr_nn3 <- pr.nn3$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r3 <- (test_all$age) * (max(abalone$age) - 
                               min(abalone$age)) + 
  min(abalone$age)
mse_nn3 <- sum((test_r3 - pr_nn3)^2) / nrow(test_all)
#79.50398


###new (5,3)
pr.nn4 <- neuralnet::compute(multi_nn2, test_all)

#compute mean square error
pr_nn4 <- pr.nn4$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r4 <- (test_all$age) * (max(abalone$age) - 
                               min(abalone$age)) + 
  min(abalone$age)
mse_nn4 <- sum((test_r4 - pr_nn4)^2) / nrow(test_all)
#10.14382



####

#plot
plot(multi_nn1) #9.031362 error, 12087 steps
plot(multi_nn2)

#plot regression line
plot(test_all$age, pr_nn3, col = "red", 
     main = "Real vs. Predicted for Multi Class")
lm(pr_nn3 ~ test_all$age)
#intercept: 16.962, slope: 3.561
abline(lm(pr_nn3 ~ test_all$age))
