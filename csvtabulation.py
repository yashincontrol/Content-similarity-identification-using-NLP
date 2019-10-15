from pandas import DataFrame
from nltk.tokenize import sent_tokenize, word_tokenize
text=["Crowdsourcing has gained immense popularity in machine learning applications for obtaining large amounts of labeled data. Crowdsourcing is cheap and fast, but suffers from the problem of low-quality data.",
("Convex potential minimisation is the de facto approach to binary classification. However, Long and Servedio [2010] proved that under symmetric label noise (SLN), minimisation of any convex potential over a linear function class can result in classification performance equivalent to random guessing. This ostensibly"),
("We develop a sequential low-complexity inference procedure for Dirichlet process mixtures of Gaussians for online clustering and parameter estimation when the number of clusters are unknown a-priori. We present an easily computable, closed form parametric expression for the conditional likelihood, in which hyperparameters are recursively updated as a function of the streaming data assuming"),
("Monte Carlo sampling for Bayesian posterior inference is a common approach used in machine learning. The Markov chain Monte Carlo procedures that are used are often discrete-time analogues of associated stochastic differential equations (SDEs). These SDEs are guaranteed to leave invariant the required posterior distribution."),
("We study the problem of hierarchical clustering on planar graphs. We formulate this in terms of finding the closest ultrametric to a specified set of distances and solve it using an LP relaxation that leverages minimum cost perfect matching as a subroutine to efficiently explore the space of planar partitions. ")]

C = {1: text }
df = DataFrame(C, columns= [1])
export_csv = df.to_csv (r'G:\IIIT Hyderabad Megathon\NLP\abstract1.csv', index = None, header=True) # here you have to write path, where result file will be stored
