import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 ('manhattan') or L2 ('euclidean') loss
    """
    def __init__(self, k=1, metric=None):
        self.k = k
        self.metric = metric
        self.eps = 1e-8

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict classes for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        if self.metric == 'manhattan':
            for i_test in range(num_test):
                for i_train in range(num_train):
                    # TODO: Fill dists[i_test][i_train]
                    
                    dists[i_test,i_train]= np.sum(np.abs(X[i_test]-self.train_X[i_train]))
            return dists
        elif self.metric == 'euclidean':
            for i_test in range(num_test):
                for i_train in range(num_train):
                    # TODO: Fill dists[i_test][i_train]

                    dists[i_test,i_train]= np.sqrt(np.sum((X[i_test]-self.train_X[i_train])**2))
            return dists

        # return dists


                    

    def compute_distances_one_loop(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        if self.metric == 'manhattan':
            for i_test in range(num_test):
                # TODO: Fill the whole row of dists[i_test]
                # without additional loops or list comprehensions
                dists[i_test] = np.sum(np.abs(X[i_test]- self.train_X),axis = 1)
            return dists
        elif self.metric == 'euclidean':
            for i_test in range(num_test):
                # TODO: Fill the whole row of dists[i_test]
                # without additional loops or list comprehensions
                dists[i_test] = np.sum(np.sqrt(X[i_test]-self.train_X)**2,axis = 1)
            return dists
    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        if self.metric == 'manhattan':
            # TODO: Implement computing all distances with no loops!
            # dists = np.sum(np.abs(X-self.train_X),axis =0)
            # dists  = np.abs(X[:, 0, None] - self.train_X[:, 0]) + np.abs(X[:, 1, None] - self.train_X[:, 1])
            # m = np.dot(X.sum(axis=1).reshape(-1, 1), self.train_X.sum(axis=1).reshape(-1, 1).T)
            # dists = m / self.train_X.sum(axis=1) - self.train_X.sum(axis=1)
            # print(dists.shape)
            # X_test_squared = np.sum(X ** 2, axis=1, keepdims=True)
            # X_train_squared = np.sum(self.train_X ** 2, axis=1, keepdims=True)
            # two_X_test_X_train = np.dot(X, self.train_X.T)

            # (Taking sqrt is not necessary: min distance won't change since sqrt is monotone)

            # dists = np.sqrt(
            #     self.eps + X_test_squared - 2 * two_X_test_X_train + X_train_squared.T
            # )
            # from scipy.spatial import distance_matrix
            # dists = distance_matrix(X,self.train_X,p=1)
            dists = np.abs(X[:,np.newaxis]- self.train_X).sum(axis = -1)
            return dists

        if self.metric == 'euclidean':
            # TODO: Implement computing all distances with no loops!
            # dists = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(self.train_X ** 2, axis=1) - 2 * np.matmul(X,
            # s1 = np.sum(X ** 2, axis=1)
            # s2 = np.sum(self.train_X ** 2, axis=1)
            # s = s1.reshape((num_test, 1)) + s2 - 2 * X.dot(self.train_X.T)
            # dists = np.sqrt(s)
            dists = np.sqrt((X[:, np.newaxis] - self.train_X)**2).sum(axis=-1)

            return dists
    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case

        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        # print("hi")
        print(dists  )
        num_test = dists.shape[0]
        pred = np.zeros(num_test)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            ## y_indices = np.argsort(dists[i,:])
            ##k_closest_classes = self.train_y[y_indices[:self.k]].astype(int)
            ## pred[i] = np.argmax(np.bincount(k_closest_classes))
            # closest_y = []
            # closest_y = self.train_y[np.argsort(dists[i])[:self.k]].astype(int)
            #
            # pred[i] = np.argmax(np.bincount(closest_y))
            closest_y = []
            k_nearest_index = np.argsort(dists[i, :], axis=0)
            closest_y = self.train_y[k_nearest_index[:self.k]]
            pred[i] = np.argmax(np.bincount(closest_y))
        return pred

    def predict_labels_multiclass(self, dists):
        import operator
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        # print("hi1")

        num_test = dists.shape[0]
        pred = np.zeros(num_test)

        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples

            ## y_indices = np.argsort(dists[i, :])
            ## k_closest_classes = self.train_y[y_indices[:,self.k]].astype(int)
            ## pred[i] = np.argmax(np.bincount(k_closest_classes))
            closest_y = []
            k_nearest_index = np.argsort(dists[i, :],axis =0)
            closest_y = self.train_y[k_nearest_index[:self.k]]
            pred[i] = np.argmax(np.bincount(closest_y))
            # labels_counts = {}
            # for label in closest_y:
            #     if label in labels_counts.keys():
            #         labels_counts[label] += 1
            #     else:
            #         labels_counts[label]x = 0
            # sorted_labels_counts = sorted(
            #     labels_counts.items(), key=operator.itemgetter(1), reverse=True)
            # pred[i] = sorted_labels_counts[0][0]

        return pred


