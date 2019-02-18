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
- **padvectorLen**: is equal to the length of the one hot encoded vector  

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


- **n_units**: Number of hidden reccurent units in a singe layer
- **n_layers**: Number of layers in a single stack of the model
- **n_classes**: Number of classification categories
- **n_seq**: Number of shift sequences 0-shift, 1-shift and 2-shift. i.e 3 in our case
- **seq_len**: Length of each sequence
- **word_size**: Vocabulary length


- **X_train and y_train**: Sets of cross validation training sets
- **X_test and y_test**: Sets of corresponding test sets
- **train_steps**: Number of training steps
- **weight_path**: Path to save the weights
- **n_folds**: the number of cross validation folds
- **esPatience**: patience for early stopping
- **lrPatience**: patience for learning rate reduction
- **epsilon**: number of places after decimal to which the loss is scalled
- **lr_decay**: learning rate decay factor
- **per_process_gpu_memory_fraction**: percentage of gpu memory allowed
- **log_path_train**: path to which log files are saved

We used 5-fold cross validation for our contribution  
i.e., X_train and X_test contain 5 sets of training and validation sets representing every possible combination of the 5 folds
- **
