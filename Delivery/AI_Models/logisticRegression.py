import numpy as np

class LogisticRegressionClassifier:
    def __init__(self, degree=2, max_iter=10000, tol=0.001):
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.classes = None
        self.cls_one_hot_encoding = None  # ground truth

    def predict(self, X):
        y_head = self._sigmoid(self.weights, X)
        prob = self._softmax(y_head.T)
        pred = np.argmax(prob, axis=1)  #Get Index (class) with highest score
        return pred

    def fit(self, X, y):
        # Train the model
        self.classes = np.sort(np.unique(y))  # ascending order
        self._init_weights(self.classes.shape[0], X.shape[1])  # Shape of (number of classes, number of features). Every class has his own weights vector
        self._create_label_matrix(y)

        # cls_one_hot_encoding (150,3) X.shape=(150,4)
        cost_list = []
        for _ in range(self.max_iter):
            y_head = self._sigmoid(self.weights, X)  # (3,150)
            prob = self._softmax(y_head.T)  # Row vector corresponds to the probability distribution for one sample (150,3)
            loss = self._log_loss(prob)

            self.weights = self.weights - self.tol * ((y_head - self.cls_one_hot_encoding.T) @ X) ### ??? Shapes... weights = (3,4) - (matrix with 3,4)
            cost_list.append(loss)
        return cost_list

    def _sigmoid(self, w, X):
        Z = np.float64(w @ X.T)
        y_head = 1 / (1 + np.exp(-Z))
        return y_head
    
    # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    def _softmax(self, y_head):
        assert len(y_head.shape) == 2
        s = np.max(y_head, axis=1)
        s = s[:, np.newaxis] # necessary step to do broadcasting. Transform array into next dimension as column vector
        e_x = np.exp(y_head - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        return e_x / div

    def _log_loss(self, prob):
        # label matrix 
        L = np.multiply(self.cls_one_hot_encoding, np.log(prob))
        log_loss = -(np.sum(L)) / len(self.cls_one_hot_encoding)
        return log_loss

    def _create_label_matrix(self, y):
        #print("Classes: ", self.classes)
        self.cls_one_hot_encoding = np.zeros((len(y), len(self.classes)))  # (150, 3)
        for i, label in enumerate(y):
            self.cls_one_hot_encoding[i][label] = 1

    def _init_weights(self, num_cls, num_feat):
        self.weights = abs(np.random.randn(num_cls, num_feat))
        #print("Init Weights: \n", self.weights)
        