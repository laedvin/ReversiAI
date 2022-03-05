# Determining fitness

Simulate a population of agents, each with their own starting Elo. They play games against each other so that their Elos can be determined.

Might need a way to compare Elos between generations.

## Current implementation

Currently each agent play a series of placement matches against a random policy AI. The random policy AI has a fixed 500 Elo.

Matches among the populations are then played in a round-robin format, meaning that each agent will play against every other agent.

After that the agents play against the random policy AI again.


## Possible changes

Keep a pool of past agents. For example, the top 3 agents per generation for the past 5 generations. This pool will then have at most 15 old agents which the current generation of agents can play against. This will hopefully make the Elos between generations more comparable.

Another factor is how the Elo is initialized. A new agent's Elo is currently set to (500 + avg(Elo of last generation))/2. This is because a new generation of agents should be comparable in skill to the past generation of agents. Is this a good assumption?

Instead of playing round-robin, what if a ladder population is simulated instead? E.g. 1000 matches are played each generation, and the agents in these generations are randomly selected from a pool containing all the current generation agents and some past generation agents.

In the same vein, maybe the top 3 agents from the last generation should simply be kept when creating a new generation. Their Elos might need to be fixed (but give double Elo reward/penalty to the opposing agent) in order to keep Elos meaningful between generation.

Can the Elos diverge?

# Genetics

The agents use neural networks whose parameters (weights and biases) are determined by their genes.

## Mapping genes to parameters

Currently the genes of a particular agent is just a list of all its weights and biases.

## Reproduction
Reproduction is the process where two agents mate and produce a new agent that inherits some features from its parents but also exhibits some mutations. Each reproduction produces two children.

### Selecting mating pairs
The mating pairs (i.e. parents) are selected randomly from a weighed distribution. The first agent in a mating pair is selected from a Boltzmann distribution where the energy factor is the agent's Elo and the temperature factor includes an "Elo attractiveness" hyperparameter: `β = ln(10) / 400 * Elo attractiveness`, `z = Σ_i exp(β * Elo_i)` and `p(i) = exp(β * Elo_i) / z`, where `p(i)` is the probability that the `i`th agent is selected. The second parent is also selected in the same way but in a distribution that leaves out the first parent.

### Crossover
After having selected the parents their genomes undergo a so-called crossover, wherein the genomes of the parents get mixed into two new genomes. In a n-crossover process, n crossover points are randomly chosen along the genome, thus creating n+1 sections in the genome. Each of the two child genomes are made up of alternating sections from their parents.

For example, in a 2-crossover point reproduction a parent A might have the genome AAA (each letter represents a section) and parent B might have the genome BBB. Their two children would then have the genomes ABA and BAB. 

### Mutation
For each gene in the new genomes there is a small chance that a mutation will occur. This is controlled with the `mutation_rate` hyperparameter. A mutation is modelled as a Gaussian noise that gets added to the gene with a variation (i.e. the square of its standard deviation) that is configurable with the `mutation_var` hyperparameter.

## Possible changes
A big worry is that the model parameter space is too large for our relatively simple reproduction scheme to optimize. One way of downsizing this is by discretizing the genes; instead of any float they could be restricted to a linspace of floats between two small but not constricting values. This way mutations could have more tangible effects at the risk of the parameter space being too small.