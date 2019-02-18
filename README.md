# junctionPredictionFromGeneSequence
Splice junction prediction from gene sequences using recurrent neural networks

# Repository description
The dataset used for this project has been provided in the /data/spliceData.txt file. The data was made available at UCI machine learning repository by:  
> G. Towell, M. Noordewier, and J. Shavlik,   
> {towell,shavlik}@cs.wisc.edu, noordewi '@' cs.rutgers.edu  

# Encoding the sequences
The sequences can be encoded for the model by using the Encoder.py file.   

The user is required to store the value returned by the get_all_posible_codon_list() method.  
This method requires the following parameters:  
- **filePath** : path to the sequence file  
- **seqSt**: the common starting posistion of the sequences in the file. For the dataset we use seqSt=39  
- **seqEd**: the common ending position of the sequences in the file + 1. For the dataset we use seqEd=99  
- **labSt**: the common starting position of the labels in the file. End of the labels are detected automatically for this dataset(commas are used to detect the end).  
- **padvectorLen**: is equal to the length of the one hot encoded vector(default: 64) 

This method returns the 3 possible encoded codon sequences generated from a single DNA sequence.

# Generating test and train data
We use 5 fold cross validation to train and validate our model. The training and testing data in this case can be obtained from the CrossValDataPreparation.py file. To encode the sequences one must first obtain the lists returned by the get_all_possible_codon_list().  

User must create an object of the CrossValDataPreparation class specifying the number of cross validation folds desired. The method stratified_crossVal_split() method can then be used. The aforementioned method requires the following parameters:  
- **X**: Contains the sequences 
- **y**: Contains the labels

This method will return the possible sets of cross validation data, each training set being represented as X_train[i] and its correponding validation set being represented as X_test[i]. Hence, the method returns:
- **X_train**: All the sets of training data obtained
- **y_train**: Corresponding sets of labels
- **X_test**: Respective validation sets
- **y_test**:Corresponding validation labels

# Building a model
To use the proposed model, one must create an object of the class SpliceClassificationModel present in the SpliceClassification.py file specifying the the parameters listed below with their following definitions.
- **n_units**: Number of hidden reccurent units in a singe layer(default: 90)
- **n_layers**: Number of layers in a single stack of the model(default: 3)
- **n_classes**: Number of classification categories(default: 3)
- **n_seq**: Number of shift sequences 0-shift, 1-shift and 2-shift. i.e 3 in our case(default: 3)
- **seq_len**: Length of each sequence(default: 20)
- **word_size**: Vocabulary length(default: 64) 

Users need to call the rnn_base() method to create the model. To define the model optimizations, one needs to call the model_optimizer_define() method with the required lrate as the parameter of the function. 
The model can be trained using the model_train() method with the following parameters mentioned below.
- **X_train and y_train**: Sets of cross validation training sets
- **X_test and y_test**: Sets of corresponding test sets
- **train_steps**: Number of training steps(default: 10000)
- **weight_path**: Path to save the weights(default: "")
- **n_folds**: the number of cross validation folds(default: 5)
- **esPatience**: patience for early stopping(default: 20)
- **lrPatience**: patience for learning rate reduction(default: 15)
- **epsilon**: number of places after decimal to which the loss is scalled(default: 3)
- **lr_decay**: learning rate decay factor(default: 0.1)
- **per_process_gpu_memory_fraction**: percentage of gpu memory allowed(default: 0.925)
- **log_path_train**: path to which log files are saved(default: "") 

The model visualization can be performed by calling model_roc_visualize() method with the parameters defined below. The function will return the ROC curves for the 3 individual classes along with their AUROC scores. It will also print the corresponding accuracy and loss. 
- **X**: Training encoded codon lists
- **y**: Respective labels
- **model_path**: path to the saved best model parameters 

We used 5-fold cross validation for our contribution  
i.e., X_train and X_test contain 5 sets of training and validation sets representing every possible combination of the 5 folds.
