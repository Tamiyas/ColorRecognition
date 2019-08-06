import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score

def read_train_data():
    dataset = pd.read_csv('../data/train_hsv.csv', sep=',',
                          names=['H', 'S', 'V', 'target'])
    train = np.array(dataset[['H', 'S', 'V']])
    target = np.array(dataset['target'].astype(int))

    return (train, target)


def read_test_data():
    dataset = pd.read_csv('../data/test_hsv.csv', sep=',',
                          names=['H', 'S', 'V', 'target'])
    test = np.array(dataset[['H', 'S', 'V']])
    target = np.array(dataset['target'].astype(int))

    return (test, target)


def train(depth):
    (train, target) = read_train_data()

    # build a decision tree
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(train, target)

    return clf


def predict(clf):
    (test, target) = read_test_data()
    predicted = clf.predict(test)

    accuracy = sum(predicted == target) / len(target)

    return accuracy


def closs_validation(partition = 10):
    (train, target) = read_train_data()
    clf = optimize()
    return cross_val_score(clf, train, target, cv=partition)


def optimize():
    accuracy_list = []
    clf_list = []
    for i in range(10):
        clf = train(i + 1)
        accuracy_list.append(predict(clf))
        clf_list.append(clf)

    return clf_list[accuracy_list.index(max(accuracy_list)) + 1]


if __name__ == "__main__":
    print(predict(optimize()))
