__author__ = 'shreyarajpal'

from csv import reader
import numpy as np
from math import log
import Queue
import matplotlib.pyplot as plt
from copy import deepcopy

class DecisionTree:

    def __init__(self):
        cabin_row = {'0':'0', 'A':'1', 'B':'2', 'C':'3', 'D':'4', 'E':'5', 'F':'6', 'G':'7', 'T':'8'}
        embarked = {'0':'0', 'C':'1', 'Q':'2', 'S':'3'}
        sex = {'male':'0', 'female':'1'}

        train_data = []
        test_data = []
        valid_data = []

        with open('train.csv') as f:
            raw_trainData = reader(f)
            count = True
            for row in raw_trainData:
                if count:
                    count = False
                    continue
                row[2] = sex[row[2]]
                row[8] = embarked[row[8]]
                row[9] = cabin_row[row[9]]
                train_data.append(row)

        with open('validation.csv') as f:
            raw_valiData = reader(f)
            count = True
            for row in raw_valiData:
                if count:
                    count = False
                    continue
                row[2] = sex[row[2]]
                row[8] = embarked[row[8]]
                row[9] = cabin_row[row[9]]
                valid_data.append(row)

        with open('test.csv') as f:
            raw_testData = reader(f)
            count = True
            for row in raw_testData:
                if count:
                    count = False
                    continue
                row[2] = sex[row[2]]
                row[8] = embarked[row[8]]
                row[9] = cabin_row[row[9]]
                test_data.append(row)



        self.train_data = np.array(train_data[1:]).astype(float)
        self.test_data = np.array(test_data[1:]).astype(float)
        self.valid_data = np.array(valid_data[1:]).astype(float)

        self.train_raw = deepcopy(self.train_data)
        self.valid_raw = deepcopy(self.valid_data)
        self.test_raw = deepcopy(self.test_data)

        self.numerical_attributes = [3, 6, 7, 10]

        for i in self.numerical_attributes:
            median_val = np.median(self.train_data[:,i])
            self.train_data[:,i] = (self.train_data[:,i] >= median_val)
            self.test_data[:,i] = (self.test_data[:,i] >= median_val)
            self.valid_data[:,i] = (self.valid_data[:,i] >= median_val)


        self.node = {}
        self.children = {}
        self.branch = {}
        self.parent = {}
        self.majorityLabel = {}

        self.attributes = {1,2,3,4,5,6,7,8,9,10}
        self.attributeVals = {1:[1,2,3], 2:[0,1], 3:[0,1], 4:[0,1,2,3,4,5,6,7,8], 5:[0,1,2,3,4,5,6], 6:[0,1], 7:[0,1], 8:[0,1,2,3], 9:[0,1,2,3,4,5,6,7,8], 10:[0,1]}

        self.validationAccuracy = 0

        self.nodeMedianList = {}

        self.tree_accuracy = []

    def get_entropy(self, pos, neg):
        total = pos + neg
        #print total
        if total == 0:
            return 0
            #print 'LOL\n\n\n'
        prob_pos = float(pos)/total
        if prob_pos==0:
            prob_pos_log = 0.000001
        else:
            prob_pos_log = prob_pos
        prob_neg = 1 - prob_pos
        if prob_neg==0:
            prob_neg_log = 0.000001
        else:
            prob_neg_log = prob_neg
        return -1*(prob_pos*log(prob_pos_log) + prob_neg*log(prob_neg_log))

    def classBalance(self):
        print 'TRAINING BALANCE'
        print np.sum([self.train_data[:,0] == 0]), np.sum([self.train_data[:,0] == 1])
        print 'VALIDATION BALANCE'
        print np.sum([self.valid_data[:,0] == 0]), np.sum([self.valid_data[:,0] == 1])
        print 'TESTING BALANCE'
        print np.sum([self.test_data[:,0] == 0]), np.sum([self.test_data[:,0] == 1])
        return

    def find_most_informative_feature(self, data, attributes):

        most_informative_val = 20000
        most_informative_feature = -1

        for i in attributes:
            vals_attribute = self.attributeVals[i]
            neg_sum = 0
            for j in vals_attribute:
                temp_entropy = 0
                if data[data[:,i] == j].size:
                    training_data_split = data[data[:,i] == j]
                    neg = 0
                    pos = 0

                    if len((training_data_split[:,0])==0):
                        neg = (training_data_split[(training_data_split[:,0])==0]).shape[0]
                    if len(training_data_split[:,0]==1):
                        pos = training_data_split[training_data_split[:,0]==1].shape[0]

                    temp_entropy= float(training_data_split.shape[0])*self.get_entropy(pos, neg)/data.shape[0]
                neg_sum += temp_entropy

            if neg_sum < most_informative_val:
                most_informative_val = neg_sum
                most_informative_feature = i

        root_neg = np.sum(data[:, 0] == 0)
        root_pos = np.sum(data[:, 0] == 1)

        parent_entropy = self.get_entropy(root_pos, root_neg)

        if abs(most_informative_val - parent_entropy) > 0:
            return most_informative_feature
        else:
            return -1

    def find_most_informative_feature_continuous(self, data, attributes):

        most_informative_val = 20000
        most_informative_feature = -1

        for i in attributes:
            vals_attribute = self.attributeVals[i]
            neg_sum = 0

            if i in self.numerical_attributes:
                median_val = np.median(data[:, i])
                #neg_0 = np.sum((data[data[:, i] < median_val])[:, 0] == 0)
                #pos_0 = np.sum((data[data[:, i] < median_val])[:, 0] == 1)
                #neg_1 = np.sum((data[data[:, i] >= median_val])[:, 0] == 0)
                #pos_1 = np.sum((data[data[:, i] >= median_val])[:, 0] == 1)

                data_0 = data[data[:, i] < median_val]
                data_1 = data[data[:, i] >= median_val]
                neg_0 = np.sum(data_0[:, 0] == 0)
                pos_0 = np.sum(data_0[:, 0] == 1)
                neg_1 = np.sum(data_1[:, 0] == 0)
                pos_1 = np.sum(data_1[:, 0] == 1)

                #print neg_0, pos_0, neg_1, pos_1, data.shape[0], i

                neg_sum += (((neg_0 + pos_0)*self.get_entropy(pos_0, neg_0) + (neg_1 + pos_1)*self.get_entropy(pos_1, neg_1))/float(data.shape[0]))

            else:
                for j in vals_attribute:
                    temp_entropy = 0
                    if data[data[:,i] == j].size:
                        training_data_split = data[data[:,i] == j]
                        neg = 0
                        pos = 0

                        if len((training_data_split[:,0])==0):
                            neg = (training_data_split[(training_data_split[:,0])==0]).shape[0]
                        if len(training_data_split[:,0]==1):
                            pos = training_data_split[training_data_split[:,0]==1].shape[0]

                        temp_entropy= float(training_data_split.shape[0])*self.get_entropy(pos, neg)/data.shape[0]
                    neg_sum += temp_entropy

            if neg_sum < most_informative_val:
                most_informative_val = neg_sum
                most_informative_feature = i

        root_neg = np.sum(data[:, 0] == 0)
        root_pos = np.sum(data[:, 0] == 1)

        parent_entropy = self.get_entropy(root_pos, root_neg)

        if abs(most_informative_val - parent_entropy) > 0.005:
            if most_informative_feature in self.numerical_attributes:
                median_val = np.median(data[:,most_informative_feature])
            else:
                median_val = -1
            return most_informative_feature, median_val
        else:
            return -1, -1

    def predict_label(self):

        predictions_train = []
        predictions_test = []
        predictions_valid = []

        for i in self.train_data:
            currentNode = 0

            while(True):
                attributeToBeTested = self.node[currentNode]
                attributeValueOfInstance = i[attributeToBeTested]
                branchToGo = self.attributeVals[attributeToBeTested].index(attributeValueOfInstance)
                nextNode = self.children[currentNode][branchToGo]
                if self.node[nextNode]==-10:
                    predictions_train.append(0)
                    break
                elif self.node[nextNode]==-11:
                    predictions_train.append(1)
                    break
                else:
                    currentNode = nextNode



        count = 0
        for i in self.test_data:
            currentNode = 0
            count += 1
            while(True):
                attributeToBeTested = self.node[currentNode]
                attributeValueOfInstance = i[attributeToBeTested]
                branchToGo = self.attributeVals[attributeToBeTested].index(attributeValueOfInstance)
                nextNode = self.children[currentNode][branchToGo]

                if self.node[nextNode]==-10:
                    predictions_test.append(0)
                    break
                elif self.node[nextNode]==-11:
                    predictions_test.append(1)
                    break
                else:
                    currentNode = nextNode


        for i in self.valid_data:
            currentNode = 0

            while(True):
                attributeToBeTested = self.node[currentNode]
                attributeValueOfInstance = i[attributeToBeTested]
                branchToGo = self.attributeVals[attributeToBeTested].index(attributeValueOfInstance)
                nextNode = self.children[currentNode][branchToGo]

                if self.node[nextNode]==-10:
                    predictions_valid.append(0)
                    break
                elif self.node[nextNode]==-11:
                    predictions_valid.append(1)
                    break
                else:
                    currentNode = nextNode

        predictions_test = np.array(predictions_test)
        predictions_train = np.array(predictions_train)
        predictions_valid = np.array(predictions_valid)

        labels_test = self.test_data[:,0]
        labels_train = self.train_data[:,0]
        labels_valid = self.valid_data[:,0]

        correct_test = labels_test[labels_test == predictions_test].size/float(labels_test.size)
        correct_train = labels_train[labels_train == predictions_train].size/float(labels_train.size)
        correct_valid = labels_valid[labels_valid == predictions_valid].size/float(labels_valid.size)

        return correct_train, correct_valid, correct_test

    def predict_label_continuous(self):

        predictions_train = []
        predictions_test = []
        predictions_valid = []

        #for i in self.train_data:
        for i in self.train_raw:
            currentNode = 0

            while(True):
                attributeToBeTested = self.node[currentNode]
                attributeValueOfInstance = i[attributeToBeTested]
                if attributeToBeTested not in self.numerical_attributes:
                    branchToGo = self.attributeVals[attributeToBeTested].index(attributeValueOfInstance)

                else:
                    median_val = self.nodeMedianList[currentNode]
                    if attributeValueOfInstance >= median_val:
                        branchToGo = 1
                    else:
                        branchToGo = 0
                nextNode = self.children[currentNode][branchToGo]
                if self.node[nextNode]==-10:
                    predictions_train.append(0)
                    break
                elif self.node[nextNode]==-11:
                    predictions_train.append(1)
                    break
                else:
                    currentNode = nextNode


        count = 0
        #for i in self.test_data:
        for i in self.test_raw:
            currentNode = 0
            count += 1
            while(True):
                attributeToBeTested = self.node[currentNode]
                attributeValueOfInstance = i[attributeToBeTested]

                if attributeToBeTested not in self.numerical_attributes:
                    branchToGo = self.attributeVals[attributeToBeTested].index(attributeValueOfInstance)

                else:
                    median_val = self.nodeMedianList[currentNode]
                    if attributeValueOfInstance >= median_val:
                        branchToGo = 1
                    else:
                        branchToGo = 0
                #branchToGo = self.attributeVals[attributeToBeTested].index(attributeValueOfInstance)
                nextNode = self.children[currentNode][branchToGo]

                if self.node[nextNode]==-10:
                    predictions_test.append(0)
                    break
                elif self.node[nextNode]==-11:
                    predictions_test.append(1)
                    break
                else:
                    currentNode = nextNode


        #for i in self.valid_data:
        for i in self.valid_raw:
            currentNode = 0

            while(True):
                attributeToBeTested = self.node[currentNode]
                attributeValueOfInstance = i[attributeToBeTested]
                if attributeToBeTested not in self.numerical_attributes:
                    branchToGo = self.attributeVals[attributeToBeTested].index(attributeValueOfInstance)

                else:
                    median_val = self.nodeMedianList[currentNode]
                    if attributeValueOfInstance >= median_val:
                        branchToGo = 1
                    else:
                        branchToGo = 0
                #branchToGo = self.attributeVals[attributeToBeTested].index(attributeValueOfInstance)
                nextNode = self.children[currentNode][branchToGo]

                if self.node[nextNode]==-10:
                    predictions_valid.append(0)
                    break
                elif self.node[nextNode]==-11:
                    predictions_valid.append(1)
                    break
                else:
                    currentNode = nextNode

        predictions_test = np.array(predictions_test)
        predictions_train = np.array(predictions_train)
        predictions_valid = np.array(predictions_valid)

        labels_test = self.test_data[:,0]
        labels_train = self.train_data[:,0]
        labels_valid = self.valid_data[:,0]

        correct_test = labels_test[labels_test == predictions_test].size/float(labels_test.size)
        correct_train = labels_train[labels_train == predictions_train].size/float(labels_train.size)
        correct_valid = labels_valid[labels_valid == predictions_valid].size/float(labels_valid.size)

        return correct_train, correct_valid, correct_test

    def grow_tree(self, mode = 0):

        counter = 0

        self.node[counter] = 0
        self.children[counter] = 0
        self.parent[counter] = -1

        stackOfNodes = Queue.Queue()
        relevantDataForNodes = Queue.Queue()
        relevantAttributes = Queue.Queue()
        stackOfNodes._put(0)
        relevantDataForNodes._put(self.train_data)
        relevantAttributes._put(self.attributes)
        numOfNodes = 0

        main_neg = np.sum(self.train_data[:, 0] == 0)
        main_pos = np.sum(self.train_data[:, 0] == 1)
        if main_neg > main_pos:
            self.majorityLabel[counter] = -10
        else:
            self.majorityLabel[counter] = -11

        while not stackOfNodes.empty():

            currentNode = stackOfNodes._get()
            currentData = relevantDataForNodes._get()
            currentAttributes = relevantAttributes._get()


            if currentData.size==0:
                pass

            elif np.sum(currentData[:, 0] == 0) == currentData[:, 0].size:
                self.node[currentNode] = -10

            elif np.sum(currentData[:, 0] == 1) == currentData[:, 0].size:
                self.node[currentNode] = -11

            elif len(currentAttributes)==0:
                neg = np.sum([currentData[:, 0] == 0])
                pos = np.sum([currentData[:, 0] == 1])
                if neg>pos:
                    self.node[currentNode] = -10
                else:
                    self.node[currentNode] = -11

            else:
                attribute_index = self.find_most_informative_feature(currentData, currentAttributes)
                if attribute_index==-1:
                    neg = np.sum(currentData[:, 0] == 0)
                    pos = np.sum(currentData[:, 0] == 1)
                    if neg > pos:
                        self.node[currentNode] = -10
                    else:
                        self.node[currentNode] = -11
                    continue

                self.node[currentNode] = attribute_index

                self.children[currentNode] = range(len(self.node), len(self.node) + len(self.attributeVals[attribute_index]), 1)

                relevantAttributesForChild = {k for k in currentAttributes}
                relevantAttributesForChild.remove(attribute_index)

                for i in xrange(len(self.children[currentNode])):
                    nodeOfChild = self.children[currentNode][i]
                    dataOfChild = currentData[currentData[:, attribute_index] == self.attributeVals[attribute_index][i]]

                    if dataOfChild.size == 0:
                        root_neg = np.sum([currentData[:, 0] == 0])
                        root_pos = np.sum([currentData[:, 0] == 1])
                        if root_neg > root_pos:
                            self.node[nodeOfChild] = -10
                            self.majorityLabel[nodeOfChild] = -10
                        else:
                            self.node[nodeOfChild] = -11
                            self.majorityLabel[nodeOfChild] = -11
                    else:
                        neg = dataOfChild[:, 0][dataOfChild[:, 0] == 0].size
                        pos = dataOfChild[:, 0][dataOfChild[:, 0] == 1].size
                        if neg>pos:
                            self.node[nodeOfChild] = -10
                            self.majorityLabel[nodeOfChild] = -10
                        else:
                            self.node[nodeOfChild] = -11
                            self.majorityLabel[nodeOfChild] = -11
                    stackOfNodes._put(nodeOfChild)
                    relevantDataForNodes._put(dataOfChild)
                    relevantAttributes._put(relevantAttributesForChild)


            trainAccuracy, validAccuracy, testAccuracy = self.predict_label()
            #self.tree_accuracy.append([numOfNodes, trainAccuracy, validAccuracy, testAccuracy])
            self.tree_accuracy.append([len(self.node), trainAccuracy, validAccuracy, testAccuracy])
            numOfNodes += 1

        self.tree_accuracy = np.array(self.tree_accuracy)

        valid_line = self.tree_accuracy[:, 2]
        x_index = self.tree_accuracy[:, 0]
        train_line = self.tree_accuracy[:, 1]

        test_line = self.tree_accuracy[:, 3]

        if mode == 1:

            fig = plt.figure()
            ax = plt.subplot(111)

            training_plot, = ax.plot(x_index, 100*train_line, 'b')
            testing_plot, = ax.plot(x_index, 100*test_line, 'r')
            validation_plot, = ax.plot(x_index, 100*valid_line, 'g')

            plt.title('Decision Trees: Performance vs. Size')
            plt.xlabel('Size of Decision Tree (Nodes)')
            plt.ylabel('Accuracy (Percentage)')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            #ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax.legend([training_plot, validation_plot, testing_plot], ['Training', 'Validation', 'Testing'], loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=3)
            #ax.legend([training_plot, validation_plot, testing_plot], ['Training', 'Validation', 'Testing'], loc='center left', bbox_to_anchor=(1.0, 0.5))

            plt.show()

        self.validationAccuracy = valid_line[-1]

        finalTestAccuracy = test_line[-1]
        finalTrainAccuracy = train_line[-1]

        print '\nTree Stats\n===================================='
        print 'Training Accuracy', finalTrainAccuracy*100
        print 'Validation Accuracy', self.validationAccuracy*100
        print 'Testing Accuracy', finalTestAccuracy*100
        print 'Size of Tree', len(self.node)
        print 'Internal Nodes', np.sum(np.array(self.node.values()) > 0)

        return

    def prune_subtree(self, index):

        childrenToBePruned = Queue.Queue()

        relevantChildren = self.children[index]
        for c in relevantChildren:
            childrenToBePruned._put(c)

        while not childrenToBePruned.empty():
            nextNode = childrenToBePruned._get()
            self.majorityLabel.pop(nextNode)

            if self.node[nextNode] < -1:
                self.node.pop(nextNode)

            else:
                childNodes = self.children.pop(nextNode)
                self.node.pop(nextNode)
                for x in childNodes:
                    childrenToBePruned._put(x)

        return

    def post_prune(self, mode = 0):

        #TODO: Print accuracies post pruning

        benchmark = self.validationAccuracy
        accuracyGraph = []
        accuracyGraph.append(self.tree_accuracy[-1].tolist())

        while(True):
            nodeToBePruned = -1

            for i in self.node.keys()[1:]:
                i_value = self.node[i]
                self.node[i] = self.majorityLabel[i]

                trainAcc, validAcc, testAcc = self.predict_label()
                self.node[i] = i_value

                if validAcc > benchmark:
                    benchmark = validAcc
                    nodeToBePruned = i

            if nodeToBePruned == -1:
                break

            else:
                self.prune_subtree(nodeToBePruned)
                self.node[nodeToBePruned] = self.majorityLabel[nodeToBePruned]
                trainAcc, validAcc, testAcc = self.predict_label()

                accuracyGraph.append([len(self.node.keys()), trainAcc, validAcc, testAcc])
                benchmark = validAcc

        accuracyGraph = np.array(accuracyGraph)

        x_index = accuracyGraph[:, 0]
        train_line = accuracyGraph[:, 1]
        valid_line = accuracyGraph[:, 2]
        test_line = accuracyGraph[:, 3]

        #print np.sum(np.array(self.node.values()) > 0)

        x_index_whole = self.tree_accuracy[:, 0]
        train_line_whole = self.tree_accuracy[:, 1]
        valid_line_whole = self.tree_accuracy[:, 2]
        test_line_whole= self.tree_accuracy[:, 3]

        if mode == 1:

            fig = plt.figure()
            ax = plt.subplot(111)

            training_plot, = ax.plot(x_index_whole, 100*train_line_whole, 'b')
            testing_plot, = ax.plot(x_index_whole, 100*test_line_whole, 'r')
            validation_plot, = ax.plot(x_index_whole, 100*valid_line_whole, 'g')

            plt.plot(x_index, 100*train_line, 'b--', x_index, 100*test_line, 'r--', x_index, 100*valid_line, 'g--')

            train_prune ,= ax.plot(x_index, 100*train_line, 'bo')
            test_prune ,= ax.plot(x_index, 100*test_line, 'ro')
            valid_line_prune ,= ax.plot(x_index, 100*valid_line, 'go')

            plt.title('Decision Trees: Performance vs. Size')
            plt.xlabel('Size of Decision Tree (Nodes)')
            plt.ylabel('Accuracy (Percentage)')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            #ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax.legend([training_plot, validation_plot, testing_plot, train_prune, valid_line_prune, test_prune], ['Training', 'Validation', 'Testing', 'Training (Pruned)', 'Validation (Pruned)', 'Testing (Pruned'], loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=6)
            #ax.legend([training_plot, validation_plot, testing_plot, train_prune, valid_line_prune, test_prune], ['Training', 'Validation', 'Testing', 'Training (Pruned)', 'Validation (Pruned)', 'Testing (Pruned'], loc='center left', bbox_to_anchor=(1.0, 0.5))

            plt.show()

        finalTrainAccuracy = train_line[-1]
        finalTestAccuracy = test_line[-1]
        finalValidAccuracy = valid_line[-1]


        print '\nTree Stats (Post Pruning)\n===================================='
        print 'Training Accuracy', finalTrainAccuracy*100
        print 'Validation Accuracy', finalValidAccuracy*100
        print 'Testing Accuracy', finalTestAccuracy*100
        print 'Size of Tree', len(self.node)
        print 'Internal Nodes', np.sum(np.array(self.node.values()) > 0)

        return

    def grow_tree_threshold(self, mode=0):

        counter = 0

        self.node[counter] = 0
        self.children[counter] = 0
        self.parent[counter] = -1

        stackOfNodes = Queue.Queue()
        relevantDataForNodes = Queue.Queue()
        relevantAttributes = Queue.Queue()
        stackOfNodes._put(0)
        relevantDataForNodes._put(self.train_data)
        relevantAttributes._put(self.attributes)
        numOfNodes = 0

        #main_neg = np.sum(self.train_data[:, 0] == 0)
        #main_pos = np.sum(self.train_data[:, 0] == 1)
        main_neg = np.sum(self.train_raw[:, 0] == 0)
        main_pos = np.sum(self.train_raw[:, 0] == 1)
        if main_neg > main_pos:
            self.majorityLabel[counter] = -10
        else:
            self.majorityLabel[counter] = -11

        while not stackOfNodes.empty():

            currentNode = stackOfNodes._get()
            currentData = relevantDataForNodes._get()
            currentAttributes = relevantAttributes._get()

            print currentNode

            if currentData.size==0:
                pass

            elif np.sum(currentData[:, 0] == 0) == currentData[:, 0].size:
                self.node[currentNode] = -10

            elif np.sum(currentData[:, 0] == 1) == currentData[:, 0].size:
                self.node[currentNode] = -11

            elif len(currentAttributes)==0:
                neg = np.sum([currentData[:, 0] == 0])
                pos = np.sum([currentData[:, 0] == 1])
                if neg>pos:
                    self.node[currentNode] = -10
                else:
                    self.node[currentNode] = -11

            else:
                attribute_index, median_val = self.find_most_informative_feature_continuous(currentData, currentAttributes)
                if attribute_index==-1:
                    neg = np.sum(currentData[:, 0] == 0)
                    pos = np.sum(currentData[:, 0] == 1)
                    if neg > pos:
                        self.node[currentNode] = -10
                    else:
                        self.node[currentNode] = -11
                    continue

                self.node[currentNode] = attribute_index
                self.children[currentNode] = range(len(self.node), len(self.node) + len(self.attributeVals[attribute_index]), 1)

                #relevantAttributesForChild = {k for k in currentAttributes}
                #relevantAttributesForChild.remove(attribute_index)

                relevantAttributesForChild = {k for k in currentAttributes}

                if median_val != -1:
                    self.nodeMedianList[currentNode] = median_val
                    #flag = True
                else:
                    relevantAttributesForChild.remove(attribute_index)

                for i in xrange(len(self.children[currentNode])):
                    nodeOfChild = self.children[currentNode][i]
                    if median_val!= -1:
                        if i==0:
                            dataOfChild = currentData[currentData[:, attribute_index] < median_val]
                        else:
                            dataOfChild = currentData[currentData[:, attribute_index] >= median_val]
                    else:
                        dataOfChild = currentData[currentData[:, attribute_index] == self.attributeVals[attribute_index][i]]



                    if dataOfChild.size == 0:
                        root_neg = np.sum([currentData[:, 0] == 0])
                        root_pos = np.sum([currentData[:, 0] == 1])
                        if root_neg > root_pos:
                            self.node[nodeOfChild] = -10
                            self.majorityLabel[nodeOfChild] = -10
                        else:
                            self.node[nodeOfChild] = -11
                            self.majorityLabel[nodeOfChild] = -11
                    else:
                        neg = dataOfChild[:, 0][dataOfChild[:, 0] == 0].size
                        pos = dataOfChild[:, 0][dataOfChild[:, 0] == 1].size
                        if neg>pos:
                            self.node[nodeOfChild] = -10
                            self.majorityLabel[nodeOfChild] = -10
                        else:
                            self.node[nodeOfChild] = -11
                            self.majorityLabel[nodeOfChild] = -11
                    stackOfNodes._put(nodeOfChild)
                    relevantDataForNodes._put(dataOfChild)
                    relevantAttributes._put(relevantAttributesForChild)


            trainAccuracy, validAccuracy, testAccuracy = self.predict_label_continuous()
            self.tree_accuracy.append([numOfNodes, trainAccuracy, validAccuracy, testAccuracy])
            numOfNodes += 1

        self.tree_accuracy = np.array(self.tree_accuracy)

        valid_line = self.tree_accuracy[:, 2]
        x_index = self.tree_accuracy[:, 0]
        train_line = self.tree_accuracy[:, 1]
        test_line = self.tree_accuracy[:, 3]

        if mode == 1:

            plt.plot(x_index, 100*train_line, x_index, 100*test_line, x_index, 100*valid_line)
            plt.xlabel('Number of nodes in the Decision Tree.')
            plt.ylabel('Accuracy (Percentage)')
            plt.show()


        #print self.tree_accuracy

        finalTrainAccuracy = train_line[-1]
        finalTestAccuracy = test_line[-1]
        finalValidAccuracy = valid_line[-1]


        print '\nTree Stats (Median Splitting)\n===================================='
        print 'Training Accuracy', finalTrainAccuracy*100
        print 'Validation Accuracy', finalValidAccuracy*100
        print 'Testing Accuracy', finalTestAccuracy*100
        print 'Size of Tree', len(self.node)
        print 'Internal Nodes', np.sum(np.array(self.node.values()) > 0)

        return






DT = DecisionTree()
#DT.classBalance()
#DT.grow_tree(mode=1)
#DT.post_prune(mode=1)
DT.grow_tree_threshold(mode=1)
