# Vector-to-Vector experiment

The goal of this experiment is test how well an neural network (NN) can reproduce a vector (3-components).  The vector is implemented using ROOT's TLorentzVector so that you can easily change from (pt,eta,phi) representation to (px,py,pz).

If you want to see what options are available for any script in this directory, just try calling the script with `-h`.

## Generate Vectors

This script generates a file with random vectors.  The file is stored in compressed numpy format.

```
./toy_mc.py 1000000 train1M.npz
```

The command above generates 1 million random vectors for training.

## Train NN

This script trains the NN and saves the model for later use.

```
./train.py train1M.npz -o test1
./train.py train1M.npz -l 10 -l 5 -o two_layer
```

The first option trains a basic NN with 1 hidden layer (3 nodes).  The second option trains a neural network with two hidden layers with 10 nodes in the first and 5 in the second.

## Validate NN

This script compares the performance of the NN on an independent sample you supply.  (Don't use your training file for this!)

```
./toy_mc.py 10000 validate10k.npz
./validate.py testv1_N10_b256_l3_frac0.900000.tgz validate10k.npz
```

The first line generates a validation sample.  The second line runs the validation on the simple model (single layer with 3 hidden nodes).  The output is saved to a file that has the same name as the training file but the `.tgz` is replaced by `.root`.

## Print NN

This is a very simple (and not well designed) script that prints out the NN weights and verifies if the solution makes sense.  It **only** works for the simple one layer, 3 node network.  It won't produce sensible output for anything else.

```
./print.py testv1_N10_b256_l3_frac0.900000.tgz
```



