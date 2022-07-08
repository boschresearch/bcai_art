### Section `training`

The section `training` is an array of elements each of which describes a single 
training step. A training setup has the following key parameters:

1. `trainer_type` is a type of the training procedure, the most useful types are
`normal` and `adv_plug_attack`, which represent standard and adversarial training training.
In the case of adversarial training, we define a ``pluggable`` adversarial attack,
which maybe either stateless (such as projected gradient descent),
or trainable with a state (e.g., a universal patch)

2. Boolean flags `train_model` and `train_attack` define if we train a model and/or attack,
respectively.

3. Optimization parameters:

  i. `optimizer`   a nested definition of an optimizer, see [more details here](optimizer.md).
  
  ii. `train_batch_size` an overall batch size for training. If we train using multiple GPUs
  using `para_type` set to `data`,, this is the total batch size that will 
  be divided among GPUs, not a batch size per GPU!
  
  iii. `epoch_qty` a number of training epochs
  
  iv. `snapshot_dir` a directory to save trained models (and attacks)
  
  v. `log_dir` a directory to store logs in the Tensorboard format.
  
  vi. `para_type` is a parallelization type, which can be: `data`, `attack`, or `independent`.
  In the case of `data` parallelization, each batch is split among GPUs. 
  In the case of `attack` parallelization the batch is not split, but the attack generation is.
  After gathering attacks from all the GPUs, we apply them to identical copies of the same model.
  In the case of `independent`, we train only a give number of attacks (parameter `instance_qty`)
  
  vii. `batch_sync_step` a number of minibatches after we which we synchronize model parameters across GPUs.
  
  
4. Attack parameters. In the case of adversarial training, we can specify one or more attack.
A single attack is specified as a nested definition with the key `attack`. See [this document for more details](attack.md).
Multiple attacks are specified as an array `attack_list` where each attack definition has the same
structure as in the case of a single atack. The `attack_list` is useful only for `para_type` equal to
`attack` in which case each GPU may host its own version of the attack. The current limitation is
that the system will not use more attacks than there are GPUs specified using the parameter `device_qty`
in the section `general`.
 