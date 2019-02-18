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
