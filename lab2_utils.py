import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.linalg import norm
import matplotlib.pyplot as plt


class RBF:
    def __init__(self, num_nodes, sigma, mus=None):
        """ Radial Basis Functions in a Network
         Args:
              num_nodes (int): number of nodes (RBFs) in the network
        """

        self.num_nodes = num_nodes
        self.weights = None
        self.sigma = sigma
        self.mus = mus # None 

    def _kernel_function(self, mu, x, sigma):
        return np.exp(-((np.linalg.norm(np.array(x)- np.array(mu))) ** 2) / (2 * sigma ** 2)) # numpy.linalg.norm: euclidean dist

    def _compute_interpolation_matrix(self, x):
        """Calculates the interpolation matrix using the kernel_function
        Args:
            x(ndarray): data points
        
        """
        matrix = np.zeros((len(x), self.num_nodes))
        for n in range(len(x)):
            for m in range(self.num_nodes): 
                matrix[n, m] = self._kernel_function(self.mus[m], x[n], self.sigma)
        return matrix

    def _select_mus(self, x):
        """ Randomly selecting mus"""
        
        if len(x) < self.num_nodes:
            print("Warning: Number of nodes shouldn't be bigger than the input")
        
        # if mus not manually initialized, create random mus
        if self.mus is None: ######## eventuell if statement wieder löschen
            idx = np.random.choice(len(x), self.num_nodes) # out of all training samples choose randomly self.num_nodes mus
            self.mus = x[idx] 


    def fit(self, x, y):
        """ Fit weights according to the least square rule
        Args:
            x (ndarray): training data points
            y (ndarray): training targets
        """
        self._select_mus(x)
        #print("self.mus",self.mus)
        phi_matrix = self._compute_interpolation_matrix(x)
        self.weights = np.dot(np.linalg.pinv(phi_matrix), y) # w = phi^(-1) * f, from equation (5); np.linalg.pinv: inverse

    def fit_delta(self, x, y, learning_rate=1e-3, num_epochs=10, error_scores=False):
        """ Fit weights according to the delta rule
        Args:
            x (ndarray): training data points
            y (ndarray): training targets
        """
         
        self._select_mus(x)
        #print("self.mus",self.mus)
        self.weights = np.random.normal(0, 0.3, self.num_nodes)
        errors = []
        for n in range(num_epochs):
            for i in range(len(x)):
                phi_matrix = self._compute_interpolation_matrix(np.array([x[i]]))
                delta_weights = learning_rate * (y[i] - self.predict([x[i]])) * phi_matrix
                self.weights += delta_weights.reshape(self.weights.shape[0])
            if error_scores:
                mse = mean_squared_error(y,self.predict(x))
                errors.append(mse)
                    
        if error_scores: 
            return errors
        
    def predict(self, x):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        if self.weights is None:
            raise ValueError("Radial Basis Function Network is not yet fitted")

        phi_matrix = self._compute_interpolation_matrix(x)
        oputput = np.dot(phi_matrix, self.weights)
        return oputput


def cl_competitive(num_nodes, input_space, iterations=20, lr=0.1, num_winners=1):
    input_space = input_space.reshape(len(input_space), -1)
    idx = np.random.choice(len(input_space), num_nodes)
    nodes = input_space[idx] # nodes is shape (num_nodes, 1)
    for i in range(iterations):
        np.random.shuffle(input_space)
        for x in input_space:
            distances = norm(nodes - np.array([x]), axis = 1)
            winners = distances.argsort()[:num_winners]
            for winner in winners:
                nodes[winner] += lr * (x - nodes[winner])

    return nodes


class SOM():
    def __init__(self, num_nodes=100):
        self.num_nodes = num_nodes
        
    def fit(self, x, batch_size=1, epochs=20, neighbours_init=50, lr=0.2, two_d = None):
        """
        Args: 
            neighbours_init (int): number of nodes which are adjusted per sample for the first epoch
            x (pandas.DataFrame): (32x84) pandas.DataFrame with 32 animals (rows) and 84 attributes (columns)
        """
        
        self.weights = self.initialize_weights(x)
        #print("self.weights: ",self.weights)
        num_neighbours = neighbours_init
        
        for e in range(epochs):
            for sample in x: # one animal at a time
                rest_lower = None
                rest_upper = None
                winners = []

                #Calculate Winner
                dist = norm(sample - self.weights, axis=1)
                winner = dist.argsort()[:1] # find row of weight matrix with shortest distance to sample
                #print("winner",winner)
            
                #Calculate neighbourhood
                if two_d is None: 
                    upper_bound = winner + int((num_neighbours + 1)/2)
                    if upper_bound >= self.num_nodes: # either upper_bound > num_nodes or lower_bound < 0, but XOR
                        rest_lower = upper_bound - self.num_nodes
                        winners.extend(np.arange(0, rest_lower).tolist())
                        upper_bound = self.num_nodes
                    lower_bound = winner - int((num_neighbours + 1)/2)
                    if lower_bound < 0:
                        rest_upper = self.num_nodes + lower_bound
                        winners.extend(np.arange(rest_upper, self.num_nodes).tolist())
                        lower_bound = 0
                    winners.extend(np.arange(lower_bound, upper_bound).tolist())
                else: # two-dimensional case: get surrounding neighbours
                    winners.extend(winner)

                    if winner == 0:
                        winners.extend(self.get_neighbourhood(winner, up=False, down=True, left=False, right=True))
                    elif winner == 9:
                        winners.extend(self.get_neighbourhood(winner, up=False, down=True, left=True, right=False))
                    elif winner == 90:
                        winners.extend(self.get_neighbourhood(winner, up=True, down=False, left=False, right=True))
                    elif winner == 99:
                        winners.extend(self.get_neighbourhood(winner, up=True, down=False, left=True, right=False))
                    elif 1 <= winner <= 8:
                        winners.extend(self.get_neighbourhood(winner, up=False, down=True, left=True, right=True))
                    elif 91 <= winner <= 98:
                        winners.extend(self.get_neighbourhood(winner, up=True, down=False, left=True, right=True))
                    elif (winner%10) == 0:
                        winners.extend(self.get_neighbourhood(winner, up=True, down=True, left=False, right=True))
                    elif (winner%10) == 9:
                        winners.extend(self.get_neighbourhood(winner, up=True, down=True, left=True, right=False))
                    else:
                        winners.extend(self.get_neighbourhood(winner, up=True, down=True, left=True, right=True))
                    
                #print("winners:",winners)
                
                for weight_idx in winners:
                    self.weights[weight_idx] += lr * (sample - self.weights[weight_idx])
            num_neighbours -= (neighbours_init/epochs)
        print("self.weights: ",self.weights.shape)
        
    def initialize_weights(self, x):
        """Initialized weight matrix with random numbers between zero and one"""
        return abs(np.random.randn(self.num_nodes,x.shape[1])) # (100x84)
    
    def predict(self, x):
        """Calculate winner nodes for every data point"""
        pos = []
        for sample in x:
            dist = norm(sample - self.weights, axis=1)
            pos.append(dist.argsort()[:1])
        return pos
    
    def get_neighbourhood(self, x, up=False, down=False, left=False, right=False):
        add_winners = []
        
        if up == True:
            add_winners.extend(x-10)
        if down == True:
            add_winners.extend(x+10)
        if left == True:
            add_winners.extend(x-1)
        if right == True:
            add_winners.extend(x+1)
        
        return add_winners
        
