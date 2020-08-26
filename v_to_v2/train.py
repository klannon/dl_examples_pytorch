#! /usr/bin/env python


import numpy as np
import torch
import argparse, tarfile, os, tempfile, shutil, math, pickle, copy, sys, shlex
import pandas as pd

def run(argv,out_stream=None):
    
    parser = argparse.ArgumentParser(description='Train ANN auto-encoder.')
    parser.add_argument('infile',
                        help='File name for reading events.')

    parser.add_argument('-o','--out', dest='outbase', metavar='OUTBASE',
                        default='model',
                        help='File name base (no extension) ' +
                        'for saving model structure and weights (two separate ' +
                        'files).')
    
    parser.add_argument('-N','--num-epochs',
                        default=10, type=int,
                        help='Number of epochs')

    parser.add_argument('-b','--batch-size',
                        default=250, type=int,
                        help='Minibatch size')

    parser.add_argument('--learning-rate',default = 1e-4,type=float,
                        help='Learning rate')

    parser.add_argument('-a','--activation', default='relu',
                        choices=['relu','relu6','elu','leakyrelu','prelu','selu','sigmoid','tanh','softplus','linear'],
                        help='The non-linear activation function')

    parser.add_argument('-l','--layer', dest='layers',
                        metavar = 'NH', action='append',
                        type=int,
                        help='Specify a layer with %(metavar)s hidden nodes.  ' +
                        'Multiple layers can be specified')

    parser.add_argument('-i','--init-weights', dest='weight_initialization',
                        default='default',
                        choices=['default','kaiming_normal','kaiming_uniform'],
                        help='Scheme used to initialize weights')

    def restricted_float(x):
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
        return x

    parser.add_argument('--train-fraction',type=restricted_float,
                        default = 0.9,
                        help='Fraction (between 0. and 1.) of the examples in '+
                        'the input file to use for training.  The rest is used '+
                        'for testing.')

    args = parser.parse_args(argv)

    # Keep track of all the output files generate so they can be
    # stuffed into a tar file.  (Yes, a tarfile.  I'm old, OK?)
    outFileList = []
    tmpDirName = tempfile.mkdtemp()

    # Load the data
    npfile = np.load(args.infile)

    inputs = npfile['inputs']
    outputs = npfile['outputs']

    # Standardize the input so that it has mean 0 and std dev. of 1.  This helps
    # tremendously with training performance.
    splitIndex = int(args.train_fraction*inputs.shape[0])
    inputMeans = inputs[0:splitIndex,:].mean(axis=0)
    inputStdDevs = inputs[0:splitIndex,:].std(axis=0)
    inputs = (inputs-inputMeans)/inputStdDevs
    outputMeans = outputs[0:splitIndex,:].mean(axis=0)
    outputStdDevs = outputs[0:splitIndex,:].std(axis=0)
    outputs = (outputs-outputMeans)/outputStdDevs

    npFileName = 'std.npz'
    outFileList.append(npFileName)
    np.savez_compressed(os.path.join(tmpDirName,npFileName),
                        inputMeans=inputMeans,
                        inputStdDevs=inputStdDevs,
                        outputMeans=outputMeans,
                        outputStdDevs=outputStdDevs)


    # Check the requested layers.  If none, make the simplest
    # possible: 1 layer with number of nodes equal to the size of the
    # input.
    if hasattr(args,'layers') and args.layers != None:
        layers = args.layers
    else:
        layers = [inputs.shape[1]]

    
    # List of layers
    model_layers = []

    # First layer
    linear = torch.nn.Linear(inputs.shape[1],layers[0])
    if args.weight_initialization.startswith('kaiming') :
        activation = args.activation
        slope = 0
        if activation == 'relu6':
            activation = 'relu6'
        elif activation == 'elu':
            activation = 'relu'
        elif activation == 'softplus':
            activation = 'relu'
        elif activation == 'leakyrelu':
            activation = 'leaky_relu'
            slope = 0.1
        elif activation == 'prelu':
            activation = 'leaky_relu'
            slope = 0.25
        if args.weight_initialization == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(linear.weight,a=slope, nonlinearity=activation)
        elif args.weight_initialization == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(linear.weight,a=slope, nonlinearity=activation)
        torch.nn.init.constant_(linear.bias,0)
        
    model_layers.append(linear)
    if args.activation == 'relu':
        model_layers.append(torch.nn.ReLU())
    elif args.activation == 'relu6':
        model_layers.append(torch.nn.ReLU6())
    elif args.activation == 'elu':
        model_layers.append(torch.nn.ELU())
    elif args.activation == 'leakyrelu':
        model_layers.append(torch.nn.LeakyReLU(0.1))
    elif args.activation == 'prelu':
        model_layers.append(torch.nn.PReLU())
    elif args.activation == 'selu':
        model_layers.append(torch.nn.SELU())
    elif args.activation == 'sigmoid':
        model_layers.append(torch.nn.Sigmoid())
    elif args.activation == 'softplus':
        model_layers.append(torch.nn.Softplus())
    elif args.activation == 'tanh':
        model_layers.append(torch.nn.Tanh())
    elif args.activation == 'linear':
        # Nothing to add for a linear activiation
        pass

    for l in range(1,len(layers)):
        linear = torch.nn.Linear(layers[l-1],layers[l])
        if args.weight_initialization == 'kaiming_normal':
            activation = args.activation
            slope = 0
            if activation == 'relu6':
                activation = 'relu6'
            elif activation == 'elu':
                activation = 'relu'
            elif activation == 'softplus':
                activation = 'relu'
            elif activation == 'leakyrelu':
                activation = 'leaky_relu'
                slope = 0.1
            elif activation == 'prelu':
                activation = 'leaky_relu'
                slope = 0.25
            if args.weight_initialization == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(linear.weight,a=slope, nonlinearity=activation)
            elif args.weight_initialization == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(linear.weight,a=slope, nonlinearity=activation)
            torch.nn.init.constant_(linear.bias,0)
        model_layers.append(linear)
        if args.activation == 'relu':
            model_layers.append(torch.nn.ReLU())
        elif args.activation == 'relu6':
            model_layers.append(torch.nn.ReLU6())
        elif args.activation == 'elu':
            model_layers.append(torch.nn.ELU())
        elif args.activation == 'leakyrelu':
            model_layers.append(torch.nn.LeakyReLU(0.1))
        elif args.activation == 'prelu':
            model_layers.append(torch.nn.PReLU())
        elif args.activation == 'selu':
            model_layers.append(torch.nn.SELU())
        elif args.activation == 'sigmoid':
            model_layers.append(torch.nn.Sigmoid())
        elif args.activation == 'softplus':
            model_layers.append(torch.nn.Softplus())
        elif args.activation == 'tanh':
            model_layers.append(torch.nn.Tanh())
        elif args.activation == 'linear':
            # Nothing to add for a linear activiation
            pass

    linear = torch.nn.Linear(layers[-1],outputs.shape[1])
    if args.weight_initialization == 'kaiming_normal':
        activation = args.activation
        slope = 0
        if activation == 'relu6':
            activation = 'relu6'
        elif activation == 'elu':
            activation = 'relu'
        elif activation == 'softplus':
            activation = 'relu'
        elif activation == 'leakyrelu':
            activation = 'leaky_relu'
            slope = 0.1
        elif activation == 'prelu':
            activation = 'leaky_relu'
            slope = 0.25
        if args.weight_initialization == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(linear.weight,a=slope, nonlinearity=activation)
        elif args.weight_initialization == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(linear.weight,a=slope, nonlinearity=activation)
        torch.nn.init.constant_(linear.bias,0)
    model_layers.append(linear)

    # Build them model
    model = torch.nn.Sequential(*model_layers)

    # Give some feeback about the model constructed
    print(model,file=out_stream)

    #Let's do the training now.

    # We need some tensors
    inputsTrainT = torch.from_numpy(inputs[:splitIndex,:])
    inputsValT = torch.from_numpy(inputs[splitIndex:,:])
    outputsTrainT = torch.from_numpy(outputs[:splitIndex,:])
    outputsValT = torch.from_numpy(outputs[splitIndex:,:])

    # Break up the training tensors into mini-batches
    trainSize = inputsTrainT.shape[0]
    numMiniBatch = int(math.ceil(trainSize/float(args.batch_size)))
    inputMiniBatches = inputsTrainT.chunk(numMiniBatch)
    outputMiniBatches = outputsTrainT.chunk(numMiniBatch)
    if trainSize % args.batch_size != 0:
        print ('Warning: Training set size ({}) does not divide evenly into batches of {}'.format(trainSize,args.batch_size),file=out_stream)
        print ('-->Discarding the remaider, {} examples'.format(trainSize % args.batch_size),file=out_stream)
        numMiniBatch -= 1


    lossFunc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
        
    # Training loop
    bestModel = None
    minValLoss = 9e20
    saveEpoch = 0

    # Let's save some training history too
    epochs = []
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    
    for epoch in range(args.num_epochs):

        loss_epoch = 0
        for miniBatch in range(numMiniBatch):

            # Do a training step
            out = model(inputMiniBatches[miniBatch])
            loss = lossFunc(out,outputMiniBatches[miniBatch])
            loss_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Now do the validation loss
        with torch.no_grad():
            outVal = model(inputsValT)
            lossVal = lossFunc(outVal,outputsValT)

        saved = ''
        if lossVal.item() < minValLoss:
            bestModel = copy.deepcopy(model)
            minValLoss = lossVal.item()
            saved = '[S]'
            bestEpoch = epoch

        train_loss = loss_epoch/numMiniBatch
        print('Epoch: {:4d}, Train Loss: {:10.4f}, Val Loss: {:10.4f} {}'.format(epoch,train_loss,lossVal.item(),saved),file=out_stream)

        # Save the per epoch performance numbers
        epochs.append(epoch)
        train_loss_per_epoch.append(train_loss)
        val_loss_per_epoch.append(lossVal.item())
            
    print('Saving: Min Loss, Best Epoch: {:12.8f} {:4d}'.format(minValLoss,bestEpoch),file=out_stream)
    filepath = 'best_model_info.pkl'
    best_model_info = {'min_val_loss':minValLoss,
                       'best_epoch':bestEpoch,
                       }
    outFileList.append(filepath)
    with open(os.path.join(tmpDirName,filepath),'wb') as out:
        pickle.dump(best_model_info,out,-1)


    print('Saving model structure and parameters:',file=out_stream)
    filepath = 'model.pkl'
    outFileList.append(filepath)
    torch.save(bestModel, os.path.join(tmpDirName,filepath))

    print('Saving training history:',file=out_stream)

    # create dataframe of epochs, losses
    filepath = 'loss_data.pkl'
    outFileList.append(filepath)
    d = {'training_loss':train_loss_per_epoch, 'validation_loss':val_loss_per_epoch}
    df = pd.DataFrame(d, index=epochs)
    df.to_pickle(os.path.join(tmpDirName,filepath))
    
    
    print('Tarring outfiles...',file=out_stream)
    weight_init = ''
    if args.weight_initialization == 'kaiming_normal':
        weight_init = '_kaiming_normal'
    elif args.weight_initialization == 'kaiming_uniform':
        weight_init = '_kaiming_uniform'
        
    outfile_name = '{}_N{}_b{}_l{}_frac{:f}_lr{:f}_{}{}'.format(args.outbase,
                                                                args.num_epochs,
                                                                args.batch_size,
                                                                '_'.join([str(l) for l in layers]),
                                                                args.train_fraction,
                                                                args.learning_rate,
                                                                args.activation,
                                                                weight_init)
    outfile_name += '.tgz'                                               
                                                          
    with tarfile.open(outfile_name,'w:gz') as tar:
        for f in outFileList:
            tar.add(os.path.join(tmpDirName,f),f)

        shutil.rmtree(tmpDirName)

    print('Done.',file=out_stream)

def run_from_string(s,out_stream=None):
    run(shlex.split(s),out_stream)
    
if __name__ == "__main__":
    run(sys.argv[1:])
