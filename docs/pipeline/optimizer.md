### Sub-section `optimizer`

The type of an optimizer is defined by the parameter `algorithm` and we 
currently support values `adam`, `sgd`, and `rmsprop`, which can be provided
either with the learning rate (parameter `init_lr`) or with a 
scheduler description (see below). 

Optimizers can have additional arguments specified via `add_arguments`.
These arguments are passed directly to the construtor of a respective Pytorch optimizer.

In terms of schedulers, the most useful so far has been `one_cycle`, which starts
with a zero learning rate and increases it for a given number of steps.
Then, the learning rate goes back to zero.

1. The peak/maximum learning rate is defined by `max_lr`

2. The length of the warm up cycle is defined by `pct_start` (from 0 to 1). 

3. The type of the learning rate strategy is defined by `anneal_strategy`, 
which can be either `linear` or `cos`.