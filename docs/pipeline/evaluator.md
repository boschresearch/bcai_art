### Section `evaluator`

The section `evaluator` is an array of elements each of which describes a single  evaluation step. 
An evaluation setup has the following key parameters:

1. `evaluator_name` an evaluator name/identifier: it must be **unique**!

2. `evaluator_type` a type of the evaluator. Most useful are `normal` and `attack` (adversarial evaluation).

3. For the advesarial evaluation, one should define `attack` (See [this document for more details](attack.md)).

4. Additional evaluator parameters are provided via `add_arguments`.  