####Importing necessary libraries####
import tensorflow as tf
import numpy as np
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

class SpliceClassificationModel:
    
    def __init__(self, n_units=90, n_layers=3, n_classes=3,
                n_seq=3, seq_len=20, word_size=64):
        '''
        n_units: number of hidden recurrent units in a single layer
        n_layers: number of layers in a single stack of the model
        n_classes: number of classifiation categories
        n_seq: number of shift sequences 0-shift, 1-shift and 2-shift, i.e., 3 in our case
        seq_len: length of the encoded sequences, in terms of states
        word_size: size of vocabulary
        '''
        self.n_units = n_units
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_seq = n_seq
        self.seq_len = seq_len
        self.word_size = word_size
        
    def get_a_cell(self, cell_size, keep_prob=1):
        cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return drop
    
    def rnn_base(self):
        tf.reset_default_graph()
        
        self.input_data = tf.placeholder(tf.float32, [None, self.n_seq, self.seq_len, self.word_size])
        self.target = tf.placeholder(tf.float32, [None, self.n_classes])
        
        with tf.name_scope('RNN_Base'):
            cell0 = tf.nn.rnn_cell.MultiRNNCell(
             [self.get_a_cell(self.n_units, 1) for _ in range(self.n_layers)]
             )
            
            cell1 = tf.nn.rnn_cell.MultiRNNCell(
             [self.get_a_cell(self.n_units, 1) for _ in range(self.n_layers)]
             )

            cell2 = tf.nn.rnn_cell.MultiRNNCell(
             [self.get_a_cell(self.n_units, 1) for _ in range(self.n_layers)]
             )
        with tf.variable_scope("RNNOutput", reuse = tf.AUTO_REUSE):
            outputs0, self.states0 = tf.nn.dynamic_rnn(cell0, self.input_data[:, 0, :, :], dtype=tf.float32)
            outputs1, self.states1 = tf.nn.dynamic_rnn(cell1, self.input_data[:, 1, :, :], dtype=tf.float32)
            outputs2, self.states2 = tf.nn.dynamic_rnn(cell2, self.input_data[:, 2, :, :], dtype=tf.float32)
        
        weights0 = {"linear_layer":tf.Variable(tf.truncated_normal([self.n_units, self.n_classes], mean =0, stddev=0.01))}
        biases0 = {"linear_layer":tf.Variable(tf.truncated_normal([self.n_classes], mean =0, stddev=0.01))}

        weights1 = {"linear_layer":tf.Variable(tf.truncated_normal([self.n_units, self.n_classes], mean =0, stddev=0.01))}
        biases1 = {"linear_layer":tf.Variable(tf.truncated_normal([self.n_classes], mean =0, stddev=0.01))}

        weights2 = {"linear_layer":tf.Variable(tf.truncated_normal([self.n_units, self.n_classes], mean =0, stddev=0.01))}
        biases2 = {"linear_layer":tf.Variable(tf.truncated_normal([self.n_classes], mean =0, stddev=0.01))}
        
        self.final_output0 = tf.matmul(outputs0[:,-1,:], weights0["linear_layer"]) + biases0["linear_layer"]
        self.final_output1 = tf.matmul(outputs1[:,-1,:], weights1["linear_layer"]) + biases1["linear_layer"]
        self.final_output2 = tf.matmul(outputs2[:,-1,:], weights2["linear_layer"]) + biases2["linear_layer"]
        
    def model_optimizer_define(self, lrate=0.001):
        '''
        lrate: learning rate
        '''
        self.lrate = lrate
        softmax0 = tf.nn.softmax_cross_entropy_with_logits(logits = self.final_output0, labels = self.target)
        cross_entropy0 = tf.reduce_mean(softmax0)

        softmax1 = tf.nn.softmax_cross_entropy_with_logits(logits = self.final_output1, labels = self.target)
        cross_entropy1 = tf.reduce_mean(softmax1)

        softmax2 = tf.nn.softmax_cross_entropy_with_logits(logits = self.final_output2, labels = self.target)
        cross_entropy2 = tf.reduce_mean(softmax2)
        
        self.final_output = tf.reduce_mean([self.final_output0, self.final_output1, self.final_output2],0)
        self.cross_entropy = tf.reduce_mean([cross_entropy0, cross_entropy1, cross_entropy2],0)
        
        self.train_step = tf.train.RMSPropOptimizer(self.lrate).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.target, 1), tf.argmax(self.final_output,1))
        self.accuracy = (tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32)))
        
    def model_train(self, X_train, y_train, X_test, y_test, train_steps=10000, weight_path="", 
                    n_folds=5, esPatience=15, lrPatience=10, epsilon=3, lr_decay = 0.1,
                    per_process_gpu_memory_fraction=0.925, log_path_train=""):
        '''
        X_train and y_train: Sets of cross validation training sets
        X_test and y_test: Sets of corresponding cross validation test sets
        
        train_steps: number of training epochs
        weight_path: path to save the weights
        n_folds: the number of cross validation folds
        esPatience: patience for early stopping
        lrPatience: patience for learning rate reduction
        epsilon: number of places after decimal to which the loss is scalled
        lr_decay: learning rate decay factor
        per_process_gpu_memory_fraction: percentage of gpu memory allowed
        log_path_train: path to which log files are saved
        
        We used 5-fold cross validation for our contribution 
        i.e., X_train and X_test contain 5 sets of training and validation sets representing every possible combination of the 5 folds
        '''
        taccList = []
        tlossList = []
        
        self.gpuOpt = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
        self.saver = tf.train.Saver()
        
        loss_summary = tf.summary.scalar(name='loss', tensor=self.cross_entropy)
        accuracy_summary = tf.summary.scalar(name="accuracy", tensor=self.accuracy)
        
        with tf.Session(config = tf.ConfigProto(gpu_options = self.gpuOpt)) as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(log_path_train, sess.graph)
            summaries_train = tf.summary.merge_all()
            ##Dummy initial accuracy and loss values##
            mAcc=0
            lLoss = 999
            ##Patience counter##
            lCounter=0
            for step in range(train_steps):
                ##validation loss and accuracy lists##
                lList=[]
                aList=[]
                ##training loss and accuracy list##
                ta = []
                tl = []
                
                for fold in range(0,n_folds):
                    _, tacc, tloss = sess.run([self.train_step, self.accuracy, self.cross_entropy], feed_dict = {self.input_data:X_train[fold],
                                                          self.target: y_train[fold]})
                    ta.append(tacc)
                    tl.append(tloss)
                    summary_str, acc, loss = sess.run([summaries_train, self.accuracy, self.cross_entropy], 
                                                feed_dict = {self.input_data:X_test[fold], self.target: y_test[fold]})
                    aList.append(acc)
                    lList.append(loss)
                taccList.append(np.mean(ta))
                tlossList.append(np.mean(tl))
                train_writer.add_summary(summary_str, global_step=step)
                if np.mean(aList) > mAcc:
                    mAcc = np.mean(aList)
                    self.saver.save(sess, os.path.join(weight_path, str(step), "modelParam.ckpt"), global_step=step)
                    print("Accuracy and loss at %d: %f and %f" % (step,np.mean(aList),np.mean(lList)))
                if round(np.mean(lList),epsilon) != lLoss:
                    lLoss = round(np.mean(lList),epsilon)
                    lCounter = 0
                else:
                    lCounter += 1
                if lCounter >= esPatience:
                    break   
                if lCounter >= lrPatience:
                    self.lrate *= lr_decay
            train_writer.close()
        self.taccList = np.asarray(taccList, dtype="float32")
        self.tlossList = np.asarray(tlossList, dtype="float32")
        np.save("accuracy.npy",self.taccList)
        np.save("loss.npy",self.tlossList)
    
    def model_roc_visualize(self, X, y, model_path, per_process_gpu_memory_fraction=0.925):
        '''
        X: training encoded codone lists
        y: respective labels
        model_path: path to the saved best model parameters 
        acc_path: path to the saved training accuracy list 
        loss_path: path to the saved training loss list
        '''
        self.gpuOpt = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
        with tf.Session(config = tf.ConfigProto(gpu_options = self.gpuOpt)) as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, model_path)

            ####Getting prediction scores####
            y_score = sess.run(self.final_output, feed_dict={self.input_data:X})
            acc, loss = sess.run([self.accuracy, self.cross_entropy], feed_dict={self.input_data:X, self.target:y})
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig = plt.figure(figsize=(8,4))
        plt.axes([0.00,0.00,1.90,0.90])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('ROC curves for our multi-class classification model',fontsize=18)
        ax1 = plt.axes([0.10, 0.10, 0.5, 0.7])
        ax2 = plt.axes([0.70, 0.10, 0.5, 0.7])
        ax3 = plt.axes([1.30, 0.10, 0.5, 0.7])

        colors = ['blue','orange','green']
        classes = ['Exon-Intron','Intron-Exon','Neither']
        ax1.plot(fpr[0], tpr[0], color=colors[0], linewidth=2,
                    label='(area = {1:0.4f})'
                    ''.format(classes[0], roc_auc[0]), linestyle="--")
        l1 = ax1.legend(loc = "lower right", prop={'size':16})
        l1.set_title(classes[0],prop={'size':16})
        ax2.plot(fpr[1], tpr[1], color=colors[1], linewidth=3,
                    label='(area = {1:0.4f})'
                    ''.format(classes[1], roc_auc[1]), linestyle=":")
        l2 = ax2.legend(loc = "lower right", prop={'size':16})
        l2.set_title(classes[1],prop={'size':16})
        ax3.plot(fpr[2], tpr[2], color=colors[2], linewidth=2,
                    label='(area = {1:0.4f})'
                    ''.format(classes[2], roc_auc[2]))
        l3 = ax3.legend(loc = "lower right", prop={'size':16})
        l3.set_title(classes[2],prop={'size':16})
        if not os.path.exists('visualization'):
            os.makedirs('visualization')
        fig.savefig(os.path.join("visualization/ROC.png"), transparent=True, dpi=fig.dpi, bbox_inches='tight')
        print("Accuracy: %f, Loss: %f" % (acc,loss))
