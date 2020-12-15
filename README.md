# Introducing LisaNetA simple implementation of a deep neural network to answer the most important question of all...<p align="center">  <img src="https://github.com/tjards/deep-neural-network-play/blob/master/images/russian_cat.jpg" width="65%" /></p><div align="center"> **Is this a cat?**<div align="left"> The cat's name is Lisa. She is adorbable and belongs to my girlfriend's mother. ## Results The network correctly predicts the above photo as a 'cat' picture. Example output:<div align="center"> `y = 1.0, your model predicts a "cat" picture"`<div align="left"> The network uses batch learning. Here's an illustration of the performance over repeated iterations:<p align="center">  <img src="https://github.com/tjards/deep-neural-network-play/blob/master/results/progress.png" width="45%" />  <img src="https://github.com/tjards/deep-neural-network-play/blob/master/results/progress_mse.png" width="45%" /></p><div align="center"> **Reduction in cost** (a) when using logistic regression; (b) when using least squares regression<div align="left"> The network makes some mistakes. Here is a print out of the where images incorrectly predicted:<p align="center">  <img src="https://github.com/tjards/deep-neural-network-play/blob/master/results/mistakes.png" width="90%" /></p><div align="center"> <div align="left"> *Note: this code was developed in partial fufillment of the Deep Learning Specialization program under Andrew Ng et al.*