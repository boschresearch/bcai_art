### Sub-section `attack`

An attack is defined by the following parameters:

1. `attack_type` a type of attack (see below): `pgd`, `fgsm`, `patch`, `univ_perturb`, `frame_saliency`).

2. `norm`: a type of the p-norm to use: `l1`, `l2`, `l3`, `l4`, and `linf`.

3. `targets`: a list of targets for a targeted attack. 
It is currently used **only** when we train attacks in the mode `independent`.

4. `weights_file` an attack weights file (for `patch` or `univ_perturb`).

5. `add_arguments` attack-specific additional arguments.

### Attacks and their specific parameters

Most attacks are stateless, however, the patch attack and the universal perturbation attack
have trainable parameters.

#### PGD attack

A [projected (multi-step) gradient descent attack](https://arxiv.org/abs/1706.06083) has
code `pgd` and the following attack-specific parameters:

1. `alpha` PGD step size (can be seen as a learning rate)

2. `num_iters` a number of PGD steps

3. `restarts` a number of PGD restarts

4. `start` a type of the starting point:  `random` and `zero` for random- and zero-initialization, respectively.

5. `patch_width`: a width of the patch for the masked PGD. If specified, a PGD attack is applied only to pixels 
                in a random square patch whose side is specified by this parameter.
                


#### FGSM attack


A [fast gradient sign attack](https://arxiv.org/abs/1412.6572) has code `fgsm' and
only one attack-specific parameter:

1. `start` a type of the starting point:  `random` and `zero` for random- and zero-initialization, respectively.


#### Patch attack

A [patch attack](https://arxiv.org/abs/1712.09665) has the code `univ_perturb` and
the following attack-specific parameters:

1. `alpha` an attack step size (can be seen as a learning rate)

2. `mask_kind` a type of the mask: `circle`, `rectangle`.

3. `scale_min` a minimum scaling factor of the patch.

4. `scale_max` a maximum scaling factor of the patch.

5. `rotation` maximum rotation (in degrees) of the patch.

6. `aspect_ratio` an aspect ratio for rectangular patches.

7. `start` a type of the starting point: `random` and `zero` for random- and zero-initialization, 
respectively.


#### Universal perturbation attack

Universal adversarial perturbation has the code `univ_perturb`. 
It is based **loosely** on [the paper by Moosavi-Dezfooli et al](https://arxiv.org/abs/1610.08401).
However, the paper is unclear about the details of the  a loss-maximizing step.
Here, we follow an approach proposed by Anit Kumar Sahu. Here is the list 
of attack-specific parameters:

1. `alpha`   a step size (can be seen as a learning rate)

2. `start` a type of the starting point: `random` and `zero` for random- and zero-initialization, 
respectively.


#### Frame saliency attack

A frame-saliency attack (code) works only for videos. [The original paper](https://arxiv.org/abs/1811.11875) 
describes only a single-step L-INF attack, which is basically an FGSM attack.
We support frame-specific PGD-like attacks as well as other norms. It has the following attack-specific parameters:

1. `attack_subtype` one shot (`one_shot`), iterative (`iter`) without updating gradients after each frame, 
   iterative with refreshing gradients after attack a frame (`iter_rg`).
        
2. `alpha` PGD step size (can be seen as a learning rate)

3. `num_iters` a number of PGD steps

4. `restarts` a number of PGD restarts



  




