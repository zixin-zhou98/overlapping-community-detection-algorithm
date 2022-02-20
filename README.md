# Overview
This is my graduate thesis project. This project proposed an overlapping community detection algorithm with prior information, exploring its denoising effect.

Contribution of this study is three-fold. First, the overlapping community detection problem is modeled as a multi-objective optimization problem. The optimiazation objectives are maximizing modularity and the function that encodes prior information.

In solving this multi-objective optimization problem, this study utilizes the combination of the NSGA-II algorithm and local search strategy to enhance the efficiency of finding solutions. The NSGA-II algorithm is responsible for global search, which adopts the elitist non-dominated sorting genetic algorithm to preserve the population elites, and maintains the population diversity using crowding distance. Local search strategy can explore the promising regions to improve the efficiency of community detection.

In addition, this study improves an existing denoising method for solving conflicting pairwise constraints. This method defines the dissimilarity index as the shortest distance between two points. When two points are in the same community, the difference index is small; otherwise, the dissimilarity index is large. If the dissimilarity index of a cannot-link constraint is too small or if the dissimilarity index of a must-link constraint is too large, it is discarded.

Experiments show that the improved algorithm can produce better community detection results on noisy networks with a small amount of prior information. Moreover, by comparing with existing methods, this study verifies the denoising effect of the dissimilarity index-based noise reduction method on noisy prior information.