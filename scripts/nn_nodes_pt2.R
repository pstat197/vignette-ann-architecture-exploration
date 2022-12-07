pr.nn2 <- neuralnet::compute(single_nn2, test_all)
pr.nn3 <- neuralnet::compute(single_nn3, test_all)
pr.nn4 <- neuralnet::compute(single_nn4, test_all)


###hidden = 5
#compute mean square error
pr_nn2 <- pr.nn2$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r2 <- (test_all$age) * (max(abalone$age) - 
                               min(abalone$age)) + 
  min(abalone$age)
mse_nn2 <- sum((test_r2 - pr_nn2)^2) / nrow(test_all)
#10.82502

###hidden = 10

#compute mean square error
pr_nn3 <- pr.nn3$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r3 <- (test_all$age) * (max(abalone$age) - 
                               min(abalone$age)) + 
  min(abalone$age)
mse_nn3 <- sum((test_r3 - pr_nn3)^2) / nrow(test_all)
#10.14796

##hidden = 100
#compute mean square error
pr_nn3 <- pr.nn3$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r3 <- (test_all$age) * (max(abalone$age) - 
                               min(abalone$age)) + 
  min(abalone$age)
mse_nn3 <- sum((test_r3 - pr_nn3)^2) / nrow(test_all)
#10.14796

##third multi

write_rds(nn_multi3, file = "models/multi_nn_3.rds")
multi_nn3 <- read_rds(file = "models/multi_nn_3.rds")

pr.nn5 <- neuralnet::compute(multi_nn3, test_all)

#compute mean square error
pr_nn5 <- pr.nn5$net.result * (max(abalone$age) - min(abalone$age))
+ min(abalone$age)
test_r5 <- (test_all$age) * (max(abalone$age) - 
                               min(abalone$age)) + 
  min(abalone$age)
mse_nn5 <- sum((test_r5 - pr_nn5)^2) / nrow(test_all)

#????


