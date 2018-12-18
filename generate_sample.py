import numpy as np
from typing import Optional, Tuple


def generate_sample(f: Optional[float] = 1.0, t0: Optional[float] = None, batch_size: int = 1,
                    predict: int = 1, samples: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    T = np.empty((batch_size, samples))
    Y = np.empty((batch_size, samples))
    FT = np.empty((batch_size, samples))
    FY = np.empty((batch_size, samples))

    _t0 = t0
    for i in range(batch_size):
        t = np.arange(0, samples + predict)
        if _t0 is None:
            t0 = np.random.randint(10,size=1) * 2 * np.pi
        else:
            t0 = _t0 + i/float(batch_size)

        freq = f
        y = np.sin(2 * np.pi * freq * (t + t0))

        T[i, :] = t[0:samples]
        Y[i, :] = y[0:samples]

        FT[i, :] = t[0:samples]
        FY[i, :] = y[1:samples + predict]

    return T, Y, FT, FY


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    t, y, t_next, y_next = generate_sample(f= 0.2, t0=None, batch_size=3, predict = 1, samples = 20)

    n_tests = t.shape[0]
    for i in range(0, n_tests):
        plt.subplot(n_tests, 1, i+1)
        plt.plot(t[i, :], y[i, :])
        plt.plot(t_next[i, :], y_next[i, :], color='red', linestyle=':')
        #plt.plot(np.append(t[i, -1], t_next[i, :]), np.append(y[i, -1], y_next[i, :]), color='red', linestyle=':')

    plt.xlabel('time [t]')
    plt.ylabel('signal')
    plt.show()
