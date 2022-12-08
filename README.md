### Description:

Vignette on Neural Network architecture using abalone data; created as a class project for PSTAT197A in Fall 2022.

### Contributors:

Alexis Navarra, Giselle Ramirez, Nealson Setiawan, Sammy Suliman

### Vignette Abstract:

For our PSTAT 197A Final vignette, we wanted to take a deep dive into Artificial Neural Network Architecture and how different architectures affect the neural network model outcomes.

Our example data is the abalone dataset built into R, containing data on 4177 abalones. The dataset include predictor variables such as type (male, female, or infant), diameter, height, longest shell measurements, number of rings, and various weights. Our response variable, and what we are trying to predict, is the age of the abalone. This response variable can be calculated by adding 1.5 to the number of rings, meaning we are working with a supervised dataset.

The outcomes of our exploration of the different aspects of neural network architecture (layers, nodes, activation functions, and epochs) allowed us to conclude that blah blah blah.

### Repository Contents:

Our repository is structured as follows:

-   An .rmd file containing the report of our exploration (vignette-ann-architecture-exploration/vignette.rmd).

-   A folder containing our data (vignette-ann-architecture-exploration/data).

-   A folder containing our R scripts (vignette-ann-architecture-exploration/scripts). This folder contains a folder for our coding drafts, along with R scripts containing the code used for each aspect of our exploration, titled accordingly.

-   A folder containing our saved neural net models from our exploration (vignette-ann-architecture-exploration/models), titled accordingly.

-   A folder containing plots from our exploration (vignette-ann-architecture-exploration/img), also contained in our vignette.rmd file, titled accordingly.

### References:

["Activation Functions in Neural Networks" - V7 Labs](https://www.v7labs.com/blog/neural-networks-activation-functions#:~:text=drive%20V7's%20tools.-,What%20is%20a%20Neural%20Network%20Activation%20Function%3F,prediction%20using%20simpler%20mathematical%20operations)

["Deep Neural Network - Parameter Search (R)" - Kaggle](https://www.kaggle.com/code/wti200/deep-neural-network-parameter-search-r/script)

["Everything you need to know about Neural Networksand Backpropagation" - Towards Data Science](https://towardsdatascience.com/everything-you-need-to-know-about-neural-networks-and-backpropagation-machine-learning-made-easy-e5285bc2be3a)

["How Neural Networks are used for Regression in R Programming" - GeeksforGeeks](https://www.geeksforgeeks.org/how-neural-networks-are-used-for-regression-in-r-programming/)

["Regression Analysis Using Artificial Neural Networks" - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/08/a-walk-through-of-regression-analysis-using-artificial-neural-networks-in-tensorflow/)

### Instructions:

In order to use the models we have built in our exploration, you can access the code we used by navigating to our scripts folder. The dataset and training/testing sets can be accessed by running lines 1-52 in the neural_network_nodes.R script, as it was the first script that we created. The rest of the scripts can be run after that.

If you are interested in contributing to the repository, you can do so by following the repository structure we have laid out for our exploration, discussed above.

## Conclusion:
As the number of nodes increase for single layer ANN and the number of nodes decrease for multiple layer ANN, we see a lower MSE and higher weighted R-squared. We see that as the number approaches to 50 epochs, the more accurate our model becomes. 
