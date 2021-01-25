import numpy as np

#gradient(differentiation)
def gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val   #값 복원

    return grad


##Stochastic Gradient Descent

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, x, f):
            grads = gradient(f, x)
            x -= self.lr * grads


class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = 0

    def update(self, x, f):
        grads = gradient(f, x)
        self.v = self.momentum *self.v - self.lr * grads
        x += self.v


class NAG:
    def __init__(self, lr= 0.01,momentum = 0.9 ):
        self.lr = lr
        self.momentum = momentum
        self.v = 0

    def update(self, x, f):
        grads = gradient(f, x+self.momentum*self.v)
        self.v = self.momentum*self.v - self.lr * grads
        x += self.v


class Adagrad:   #1.5
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = 0

    def update(self, x, f):
        grads = gradient(f, x)
        self.h += grads*grads
        x -= self.lr* np.sqrt(1/self.h) * grads


class RMSP:
    def __init__(self, lr = 0.01, r = 0.8):
        self.lr = lr
        self.r = r
        self.h = 0

    def update(self, x, f):
        grads = gradient(f, x)
        self.h = self.r*self.h + (1-self.r)*grads*grads
        x -= self.lr*np.sqrt(1/(self.h + 1e-7)) * grads


class Adam:
    def __init__(self, lr = 0.01, B1 = 0.9 , B2 = 0.999 ):
        self.lr = lr
        self.B1 = B1
        self.B2 = B2
        self.m = 0
        self.v = 0
        self.iter = 0


    def update(self, x, f):
        self.iter += 1
        grads = gradient(f, x)
        self.m = self.B1*self.m + (1 - self.B1)*grads
        self.v = self.B2*self.v +(1 - self.B2)*grads*grads

        revised_lr = self.lr * np.sqrt(1-self.B2**self.iter) / (1- self.B1**self.iter)
        x -= revised_lr*self.m / (np.sqrt(self.v+1e-7))


#Test
def function_1(x):
    return x[0]**2/15 +x[1]**2



from collections import OrderedDict


optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.9)
optimizers["Momentum"] = Momentum(lr=0.07)
optimizers["NAG"] = NAG(lr = 0.05)
optimizers["AdaGrad"] = Adagrad(lr=1.5)
optimizers["RMsp"] = RMSP(lr = 0.5, r = 0.8)
optimizers["Adam"] = Adam(lr=0.3)

#graph comparision _1 (contour)
idx = 1

for key in optimizers:
    optimizer = optimizers[key]

    def process(f, init_x, I=100):
        x = init_x
        process = []

        for i in range(I):
            process.append(x.copy())
            optimizer.update(x, f)

        return x, np.array(process)


    x, x_process = process(function_1, init_x=np.array([-7.0, 2.0]), I=30)

    from matplotlib.pylab import plt

    X = np.arange(-10, 10, 0.01)
    Y = np.arange(-10, 10, 0.01)

    X, Y = np.meshgrid(X, Y)
    Z = function_1(np.array([X, Y]))

    # 외곽선 단순화
    mask = Z > 10
    Z[mask] = 0

    plt.subplot(2, 3, idx)
    idx += 1
    plt.plot(x_process[:, 0], x_process[:, 1], '.-', color="blue")
    plt.contour(X, Y, Z)

    plt.ylim(-10, 10)
    plt.xlim(-10, 10)

    plt.plot(0, 0, '+', color="red")  # 극소점
    plt.title(key)

plt.show()


#graph comparision _2 (2 dimensional)

idx = 1
for key in optimizers:
    optimizer = optimizers[key]

    def process(f, init_x, I=100):
        x = init_x
        process = []

        for i in range(I):
            process.append(x.copy())
            optimizer.update(x, f)

        return x, np.array(process)


    plt.subplot(2, 3, idx)
    idx += 1

    x, x_process = process(function_1, init_x=np.array([-7.0, 2.0]), I=30)

    plt.plot([-10, 10], [0, 0], '-b')  # x axis (color = blue)
    plt.plot([0, 0], [-10, 10], '-b')  # y axis (color = blue)
    plt.plot(x_process[:, 0], x_process[:, 1], '.-')  # x_process 의 첫번째 열, 두번째 열 circle marker
    plt.plot(0, 0, '+', color="red")

    plt.title(key)
plt.show()


#graph comparision _3 (3 dimensional)

idx = 1
fig = plt.figure()
for key in optimizers:
    optimizer = optimizers[key]

    def process(f, init_x, I=100):
        x = init_x
        process = []

        for i in range(I):
            process.append(x.copy())
            optimizer.update(x, f)

        return x, np.array(process)

    x, x_process = process(function_1, init_x=np.array([-7.0, 2.0]), I=30)

    X = np.arange(-5, 5, 0.01)
    Y = np.arange(-5, 5, 0.01)

    X, Y = np.meshgrid(X, Y)
    Z = function_1(np.array([X, Y]))


    ax = fig.add_subplot(2, 3, idx, projection='3d') #make each subplot by making 6 subplots in total.

    surf = ax.plot_wireframe(X, Y, Z, color='grey', alpha=0.1) #add axis to each plot
    ax.plot3D(x_process[:, 0], x_process[:, 1], np.array([function_1(x_process[i, :]) for i in range(len(x_process))]), color="blue") #draw the graph (how the optimization function works)
    ax.plot3D(0, 0, '+', color="red")  #the local minimum point
    idx +=1
    plt.title(key)

plt.show()



