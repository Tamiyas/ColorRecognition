from dtreeviz.trees import dtreeviz
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


if __name__ == "__main__":
    clf = optimize()
    print(predict(clf))
    visualize(clf)
