# packages
library(tidyverse)
library(tidymodels)
library(tidytext)
library(keras)
library(tensorflow)


x_train <- train_all %>% # train_ 
  ungroup() %>%
  as.matrix()
y_train <- train_all$age %>% # train_age
  as.matrix()

x_test <- test_all %>% # test_
  ungroup() %>%
  as.matrix()
y_test <- test_all$age %>% # test_age
  as.matrix()

# neural network - 100 epochs
model <- keras_model_sequential(input_shape = 7)
model <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  layer_dense(10) %>%
  layer_dense(1) %>%
  layer_dense(100) %>%
  layer_dense(1) %>%
  layer_dense(10) %>%
  layer_activation(activation = 'relu')


model %>%
  compile(
    loss = 'MeanAbsoluteError',
    optimizer = 'adam',
    metrics = 'MeanSquaredError'
  )

history <- model %>%
  fit(x = x_train, 
      y = y_train,
      epochs = 100,
      validation_split = 0.3)

summary(model)
plot(history)

# retrieve weights
get_weights(model)

# evaluate on specified data
evaluate(model, x_test, y_test)

write_rds(model, file = "models/nn_100_epochs.rds")
nn_100_epochs <- read_rds(file = "models/nn_100_epochs.rds")


## neural network - 50 epochs
model2 <- keras_model_sequential(input_shape = 7)
model2 <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  layer_dense(10) %>%
  layer_dense(1) %>%
  layer_dense(100) %>%
  layer_dense(1) %>%
  layer_dense(10) %>%
  layer_activation(activation = 'relu')


model2 %>%
  compile(
    loss = 'MeanAbsoluteError',
    optimizer = 'adam',
    metrics = 'MeanSquaredError'
  )

history <- model2 %>%
  fit(x = x_train, 
      y = y_train,
      epochs = 50,
      validation_split = 0.3)

summary(model2)
plot(history)

# retrieve weights
get_weights(model2)

# evaluate on specified data
evaluate(model2, x_test, y_test)

write_rds(model2, file = "models/nn_50_epochs.rds")
nn_50_epochs <- read_rds(file = "models/nn_50_epochs.rds")

## neural network - 10 epochs
model3 <- keras_model_sequential(input_shape = 7)
model3 <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  layer_dense(10) %>%
  layer_dense(1) %>%
  layer_dense(100) %>%
  layer_dense(1) %>%
  layer_dense(10) %>%
  layer_activation(activation = 'relu')


model3 %>%
  compile(
    loss = 'MeanAbsoluteError',
    optimizer = 'adam',
    metrics = 'MeanSquaredError'
  )

history <- model3 %>%
  fit(x = x_train, 
      y = y_train,
      epochs = 10,
      validation_split = 0.3)

summary(model3)
plot(history)

# retrieve weights
get_weights(model3)

# evaluate on specified data
evaluate(model3, x_test, y_test)

write_rds(model3, file = "models/nn_10_epochs.rds")
nn_10_epochs <- read_rds(file = "models/nn_10_epochs.rds")