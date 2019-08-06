from dtreeviz.trees import dtreeviz
import graphviz
from sklearn.tree import export_graphviz
from train import read_train_data
from train import optimize
from train import predict


def visualize(clf):
    (train, target) = read_train_data()
    viz = dtreeviz(clf, train, target, target_name='HSV', feature_names=['Hue', 'Saturation', 'Value'],
                   class_names=['Yellow', 'Green',
                                'Red', 'Blue', 'Black', 'White']
                   )

    viz.view()


def plot(clf):
    (train, target) = read_train_data()
    dot_data = export_graphviz(
        clf, class_names=['yellow', 'green', 'red', 'blue', 'black', 'white'], feature_names=['Hue', 'Saturation', 'Value'], filled=True, rounded=True, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render('../docs/tree', format='png')


if __name__ == "__main__":
    clf = optimize()
    print(predict(clf))
    plot(clf)
    visualize(clf)
