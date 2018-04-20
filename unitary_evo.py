from learning_unitary_matrices import *


class ULearn:
    def __init__(self, dim):
        self.n = dim
        self.params = np.random.randn(self.n**2)
        self.u = lie_algebra_element(self.n, self.params).flatten()
        self.sigma = 1.

    def set_params(self, params):
        self.params = params
        self.u = lie_algebra_element(self.n, self.params).flatten()

    def _objective(self, target):
        if len(target) == self.n:
            u = project_to_unitary(self.u, check_unitary=False)
            u_r = np.real(u)
            u_i = np.imag(u)
            t_r = np.real(target.flatten())
            t_i = np.imag(target.flatten())
            return -(np.linalg.norm(u_r-t_r)+np.linalg.norm(u_i-t_i))

    def nes(self, target, npop=50, alpha=0.01):
        sigma = self.sigma
        # mu
        N = np.random.randn(npop, self.n**2)*sigma + self.params
        R = np.zeros(npop)
        for i in range(npop):
            new_ul = ULearn(self.n)
            new_ul.set_params(N[i])
            R[i] = new_ul._objective(target)
        A = (R - np.mean(R))/np.std(R)
        #A = self.compute_centered_ranks(R)
        self.set_params(self.params + alpha/(npop*sigma) * np.dot(N.T, A))

    def train(self, target, epoch=10000):
        for i in range(epoch):
            self.nes(target)
            if i%100 == 0:
                print("Epoch ", i, self._objective(target))

    @staticmethod
    def compute_centered_ranks(x):
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        y = ranks.reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y

target = random_unitary_composition(2)
print(target)
print(np.dot(np.conj(target).T, target))

# ul = ULearn(2)
# ul.train(target)
# print(target)
# print(project_to_unitary(ul.u).reshape(2, 2))
# print(ul.params)