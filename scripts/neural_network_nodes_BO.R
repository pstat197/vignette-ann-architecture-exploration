#load libraries
library(neuralnet)
library(tidyverse)
library(tidymodels)
library(AppliedPredictiveModeling)

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

#Bayesian Optimization

set.seed(1234) #for reproducibility

#set folds to cross-validate
folds <- vfold_cv(train_all, v=5, strata = age) 

#create keras regression model and set hidden_units and epochs for tuning
model_tune1 <- mlp(hidden_units = tune(), epochs = tune()) %>% 
  set_engine("keras") %>% 
  set_mode("regression")

#create workflow, and add model
nn_bo_wf1 <- workflow() %>%
  add_model(model_tune1) %>%
  add_formula(age ~ Type_F + Type_I + Type_M + LongestShell + 
                Diameter + Height + WholeWeight + ShuckedWeight + 
                VisceraWeight + ShellWeight)

#Using random search to determine best hyperparameters
# tuned_random1 <- nn_bo_wf1 %>% 
#   tune_grid(resamples = folds, 
#             grid = grid_random(hidden_units(range = c(5L,15L)), epochs(range = c(10L,100L)), size = 10), metrics=metric_set(rmse))
# df_r1 <- tuned_random1 %>% collect_metrics()

#write_csv(df_r1, "data/nn_R1.csv")
df_r1 <- read_csv("data/nn_R1.csv")
df_r1 %>% arrange(mean)

#Using bayesian optimization UCB to determine best hyperparameters
# tuned_UCB1 <- nn_bo_wf1 %>% 
#   tune_bayes(resamples = folds,
#              param_info=parameters(hidden_units(range = c(5L,15L)), 
#              epochs(range = c(10L,100L))),
#              metrics=metric_set(rmse),
#              objective=conf_bound(kappa = 2))
# 
# df_UCB1 <- tuned_UCB1 %>% collect_metrics()

#write_csv(df_UCB1, "data/nn_UCB1.csv")
df_UCB1 <- read_csv("data/nn_UCB1.csv")
df_UCB1 %>% arrange(-mean)

#Using bayesian optimization EI to determine best hyperparameters
# tuned_EI1 <- nn_bo_wf1 %>% 
#   tune_bayes(resamples = folds,
#              param_info=parameters(hidden_units(range = c(5L,15L)), 
#              epochs(range = c(10L,100L))),
#              metrics=metric_set(rmse),
#              objective=exp_improve(trade_off = 0.01))
# 
# df_EI1 <- tuned_EI1 %>% collect_metrics()

#write_csv(df_EI1, "data/nn_EI1.csv")
df_EI1 <- read_csv("data/nn_EI1.csv")
df_EI1 %>% arrange(-mean)

#Using bayesian optimization PI to determine best hyperparameters
# tuned_PI1 <- nn_bo_wf1 %>% 
#   tune_bayes(resamples = folds,
#              param_info=parameters(hidden_units(range = c(5L,15L)), 
#                                    epochs(range = c(10L,100L))),
#              metrics=metric_set(rmse),
#              objective=prob_improve(trade_off = 0.01))
# 
# df_PI1 <- tuned_PI1 %>% collect_metrics()

#write_csv(df_PI1, "data/nn_PI1.csv")
df_PI1 <- read_csv("data/nn_PI1.csv")
df_PI1 %>% arrange(-mean)

#Find best parameters

df_deep <- tibble(names=c("random", "UCB", "PI", "EI"),
                  hidden_units=c(df_r1[df_r1$mean==min(df_r1$mean),1, drop=TRUE],
                                 df_UCB1[df_UCB1$mean==min(df_UCB1$mean),1, drop=TRUE],
                                 df_PI1[df_PI1$mean==min(df_PI1$mean),1, drop=TRUE],
                                 df_EI1[df_EI1$mean==min(df_EI1$mean),1, drop=TRUE]),
                  epochs=c(df_r1[df_r1$mean==min(df_r1$mean),2, drop=TRUE],
                           df_UCB1[df_UCB1$mean==min(df_UCB1$mean),2, drop=TRUE],
                           df_PI1[df_PI1$mean==min(df_PI1$mean),2, drop=TRUE],
                           df_EI1[df_EI1$mean==min(df_EI1$mean),2, drop=TRUE]),
                  rmse=c(min(df_r1$mean), min(df_UCB1$mean),
                            min(df_PI1$mean),min(df_EI1$mean)),
                  std_err=c(df_r1[df_r1$mean==min(df_r1$mean),ncol(df_r1), drop=TRUE],
                            df_UCB1[df_UCB1$mean==min(df_UCB1$mean),ncol(df_UCB1), drop=TRUE],
                            df_PI1[df_PI1$mean==min(df_PI1$mean),ncol(df_PI1), drop=TRUE],
                            df_EI1[df_EI1$mean==min(df_EI1$mean),ncol(df_EI1), drop=TRUE]))

#add MSE by RMSE^2
df_deep <- df_deep %>% mutate(mse = rmse^2)

#write_csv(df_deep, 'data/nn_final_deep.csv')
#credit to https://www.r-bloggers.com/2020/05/bayesian-hyperparameters-optimization/