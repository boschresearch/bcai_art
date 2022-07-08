### Section `general`

The `general` section defines the following key parameters:

1. `device` which is a nested configuration with the following options:

    i. `device_qty` number of GPU devices used for distributed training
    
    ii. `device_name` the name of the device used for training, e.g., `cpu` or `cuda:1` when
    there is only a single device used. 
    
2. `dist_backend` a type of the Pytorch distributed backend, see 
[Pytorch documentation](https://pytorch.org/tutorials/intermediate/dist_tuto.html).
  
2. `num_data_workers` number of processes used in a data loader

3. `master_port` a master port used for distributed training

4. `base_seed` the base/main seed value 