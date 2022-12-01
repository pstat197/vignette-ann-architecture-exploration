# packages
library(tidyverse)
library(tidymodels)
library(tidytext)
library(keras)
library(tensorflow)


x_train <- train_ %>%
  ungroup() %>%
  as.matrix()
y_train <- train_age %>%
  as.matrix()

x_test <- test_ %>%
  ungroup() %>%
  as.matrix()
y_test <- test_age %>%
  as.matrix()

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
