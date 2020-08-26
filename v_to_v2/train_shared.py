#! /usr/bin/env python


import numpy as np
import torch
import argparse, tarfile, os, tempfile, shutil, math, pickle, copy, sys, shlex
import pandas as pd

import kpl_models

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

    parser.add_argument('-H','--hidden-nodes',
                        default=3, type=int,
                        help='Number of hidden layers')
    
    parser.add_argument('-b','--batch-size',
                        default=250, type=int,
                        help='Minibatch size')

    parser.add_argument('--learning-rate',default = 1e-4,type=float,
                        help='Learning rate')

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

    parser.add_argument('-s','--standardization',
                        default='global',
                        choices=['channel','global','learned'],
                        help='Scheme used to initialize weights')

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
    if args.standardization == 'channel' or args.standardization == 'learned':
        inputMeans = inputs[0:splitIndex,:].mean(axis=0)
        inputStdDevs = inputs[0:splitIndex,:].std(axis=0)
        if args.standardization == 'channel':
            inputs = (inputs-inputMeans)/inputStdDevs
        else:
            inWeights = 1./inputStdDevs
            inOffsets = -inputMeans
            inputMeans = np.full(inputs.shape[1],0.)
            inputStdDevs = np.full(inputs.shape[1],1.)            
        outputMeans = outputs[0:splitIndex,:].mean(axis=0)
        outputStdDevs = outputs[0:splitIndex,:].std(axis=0)
        if args.standardization == 'channel':
            outputs = (outputs-outputMeans)/outputStdDevs
        else:
            outWeights = outputStdDevs
            outOffsets = outputMeans/outputStdDevs
            outputMeans = np.full(inputs.shape[1],0.)
            outputStdDevs = np.full(inputs.shape[1],1.)
    elif args.standardization == 'global':
        inputMeans = inputs[0:splitIndex,:].mean()
        inputMeans = np.full(inputs.shape[1],inputMeans)
        inputStdDevs = inputs[0:splitIndex,:].std()
        inputStdDevs = np.full(inputs.shape[1],inputStdDevs)
        inputs = (inputs-inputMeans)/inputStdDevs
        outputMeans = outputs[0:splitIndex,:].mean()
        outputMeans = np.full(outputs.shape[1],outputMeans)
        outputStdDevs = outputs[0:splitIndex,:].std()
        outputStdDevs = np.full(outputs.shape[1],outputStdDevs)
        outputs = (outputs-outputMeans)/outputStdDevs

    npFileName = 'std.npz'
    outFileList.append(npFileName)
    np.savez_compressed(os.path.join(tmpDirName,npFileName),
                        inputMeans=inputMeans,
                        inputStdDevs=inputStdDevs,
                        outputMeans=outputMeans,
                        outputStdDevs=outputStdDevs)

    # Build them model
    if args.standardization == 'learned':
        stdLayerIn = kpl_models.StandardizationLayer(3)
        stdLayerIn.initializeWeights(inWeights,inOffsets)
        shared = kpl_models.SharedModel(args.hidden_nodes)
        stdLayerOut = kpl_models.StandardizationLayer(3)
        stdLayerOut.initializeWeights(outWeights,outOffsets)
        model = torch.nn.Sequential(stdLayerIn, shared, stdLayerOut)
        with torch.no_grad():
            inTest = torch.from_numpy(inputs[:5,:])
            stdInLay = stdLayerIn(inTest)
            shr = shared(stdInLay)
            stdOutLay = stdLayerOut(shr)
            outTest = torch.from_numpy(outputs[:5,:])
            for i in range(5):
                print('------')
                print(inTest[i,:])
                print(stdInLay[i,:])
                print(shr[i,:])
                print(stdOutLay[i,:])
                print(outTest[i,:])   
    else:
        model = kpl_models.SharedModel(args.hidden_nodes)

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
    bestModel = copy.deepcopy(model)
    out = model(inputsValT)
    loss = lossFunc(out,outputsValT)
    minValLoss = loss.item()
    saveEpoch = 0
    bestEpoch = 0
    
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
    outfile_name = '{}_std{}_N{}_b{}_shared_H{}_frac{:f}_lr{:f}'.format(args.outbase,
                                                                        args.standardization,
                                                                        args.num_epochs,
                                                                        args.batch_size,
                                                                        args.hidden_nodes,
                                                                        args.train_fraction,
                                                                        args.learning_rate)
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
