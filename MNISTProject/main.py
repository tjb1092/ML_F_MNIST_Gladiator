import numpy as np
import fashion_mnist.utils.mnist_reader as m_reader

def main():
    # Orchestrate the retrival of data, training, and testing.
    data = get_data()

    # get classifier
    from sklearn.svm import SVC
    clf = SVC(probability=False,  # cache_size = 200,
                kernel="rbf", C=2.8, gamma=0.0073)

    print("Start fitting. This might take a bit")

    #  Take all of it - make that number lower for experiments
    examples = len(data['train']['X'])
    clf.fit(data['train']['X'][:examples],  data['train']['y'][:examples])
    analyze(clf, data)


def analyze(clf, data):

    """
    Analyse how well a classifier performs on data.

    Parameters
    ----------
    clf : classifier object
    data : dict
    """

    # Get confusion matrix
    from sklearn import metrics

    predicted = clf.predict(data['test']['X'])
    print("Confusion matrix: \n%s" % metrics.confusion_matrix(data['test']['y'],
                                                              predicted))

    print("Accuracy: %0.4f" % metrics.accuracy_score(data['test']['y'],
                                                     predicted))
    # Print example
    try_id = 1
    size = int(len(data['test']['X'][try_id]) ** (0.5))
    view_image(data['test']['X'][try_id].reshape((size, size)),
               data['test']['y'][try_id])

    out = clf.predict(data['test']['X'][try_id])
    print("out: %s" % out)


def view_image(image, label=""):
    """
    View a single image.

    Parameters
    ----------
    image : numpy array
    label : str
    """

    from matplotlib.pyplot import show, imshow, cm
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()


def get_data():
    """
    Get data ready to learn with.

    Returns
    -------
    dict
    """

    simple = False
    if simple:
        from sklearn.datasets import load_digits
        from sklearn.utils import shuffle
        digits = load_digits()
        x = [np.array(el).flatten() for el in digits.images]
        y = digits.target

        x = np.divide(x, 255.0) * 2 - 1

        x, y = shuffle(x, y, random_state=0)

        from sklearn.cross_validation import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        data = {'train': {'X': x_train,
                          'y': y_train},
                'test': {'X': x_test,
                         'y': y_test}}
    else:
        # from sklearn.datasets import fetch_mldata
        from sklearn.utils import shuffle
        # mnist = fetch_mldata("MNIST original")
        # x = mnist.data
        # y = mnist.target

        # x = x / 255.0 * 2 - 1
        #
        # x, y = shuffle(x, y, random_state=0)
        #
        # from sklearn.cross_validation import train_test_split
        #
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        X_train, y_train = m_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
        X_test, y_test = m_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

        X_test = X_test / 255.0 * 2 - 1
        X_train = X_train / 255.0 * 2 - 1

        data = {'train': {'X': X_train,
                          'y': y_train},
                'test': {'X': X_test,
                         'y': y_test}}
    return data

if __name__ == '__main__':
    main()
