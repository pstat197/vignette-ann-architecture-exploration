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
model <- keras_model_sequential(input_shape = 7)
# add output layer
model <- model %>% layer_dense(1) 
model <- model %>% 
  layer_activation(activation = 'relu')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

history <- model %>%
  fit(x = x_train, 
      y = y_train,
      epochs = 10)

# retrieve weights
get_weights(model)

# evaluate on specified data
evaluate(model, x_train, y_train)

model <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  layer_dense(10) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'relu')

summary(model)

model %>%
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = 'binary_accuracy'
  )

history <- model %>%
  fit(x = x_train,
      y = y_train,
      epochs = 50)

plot(history)

# redefine model
model <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  layer_dense(10) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

model %>%
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = 'binary_accuracy'
  )

# train with validation split
history <- model %>%
  fit(x = x_train,
      y = y_train,
      epochs = 20,
      validation_split = 0.2)

plot(history)
