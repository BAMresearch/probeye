"""
08.10.2021
atul.agrawal@tum.de
A method to inform pytorch about dy/\theta (JACOBIANS) from the forward solver and keep teh computational graph intact.
Resources:
- https://pytorch.org/docs/stable/_modules/torch/autograd/function.html#Function.backward
- https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- https://pytorch.org/docs/stable/notes/extending.html
- http://www.dolfin-adjoint.org/en/latest/documentation/maths/3-gradients.html

The implementation aims to check if we can overload autograd to infer A,b and noise in $y = Ax +b$ for a two different
experiment with A and b shared. No use of probEye here. Essentially providing manually J = [dy/dA,dy/db] ([x,1]) to pytorch.
So {dl/theta} = [J]^t {v} , with v = {dl/dy}

Investigation results:

The overloading returns a speed up of ~ 80x as compared to the case when gradient in grad based VI/MCMC is computed by FD under the hood.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch as th
import pyro
import pyro.distributions as dist
from numpy.core._multiarray_umath import ndarray
from pyro.infer import EmpiricalMarginal, Importance, NUTS, MCMC, HMC
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class forward_solver:
    def __init__(self, x):
        self.x = x

    def __call__(self, A, b):
        """
        :param A [np.array]
        :param b [np.array]
        :return
            y [np.array] [Nx1]
            jac [np.array] [MxN] with M being the number of parameters. Transpose is returned now
        """
        # this is only executed with overloading is not done
        if th.is_tensor(A):
            A = A.detach().numpy()
        if th.is_tensor(b):
            b = b.detach().numpy()

        y = A * self.x + b

        # Jacobian^T is returned dy/dA =x and dy/db = 1
        jac = np.vstack((self.x, np.ones(np.size(self.x))))
        return y, jac


# -- True values
b = 1.0
A = 2.0

# -- experiment 1 inputs
x1 = np.linspace(0, 1, 50)
sigma1 = 0.2
# -- experiment 2 inputs
x2 = np.linspace(0, 1, 100)
sigma2 = 0.4


##----------------------------------------------------------
# ----- wrapper for forward solver and torch autograd--------
##----------------------------------------------------------

def wrapper_autograd_overload():
    class Autograd(th.autograd.Function):
        """
        Every operation performed on :class:`Tensor` s creates a new function
        object, that performs the computation, and records that it happened.
        The history is retained in the form of a DAG of functions, with edges
        denoting data dependencies (``input <- output``). Then, when backward is
        called, the graph is processed in the topological ordering, by calling
        :func:`backward` methods of each :class:`Function` object, and passing
        returned gradients on to next :class:`Function` s
        """

        @staticmethod
        def forward(ctx, theta1, theta2, x):
            """
            This function is to be overridden by all subclasses.
            It must accept a context ctx as the first argument, followed by any
            number of arguments (tensors or other types).

            The context can be used to store arbitrary data that can be then
            retrieved during the backward pass.
            """
            if th.is_tensor(theta1):
                theta1 = theta1.detach().numpy()
            if th.is_tensor(theta2):
                theta2 = theta2.detach().numpy()

            # x=np.linspace(0,1,50)
            forward = forward_solver(x)

            y_hat, grad = forward(theta1, theta2)
            # y_hat,grad =forward_overloaded_autodiff(theta)
            Jacobian = th.from_numpy(grad)
            ctx.save_for_backward(Jacobian)

            return th.from_numpy(y_hat)

        @staticmethod
        def backward(ctx, grad_output):
            """
            It must accept a context :attr:`ctx` as the first argument, followed by
            as many outputs as the :func:`forward` returned (None will be passed in
            for non tensor outputs of the forward function),
            and it should return as many tensors, as there were inputs to
            :func:`forward`. Each argument is the gradient w.r.t the given output,
            and each returned value should be the gradient w.r.t. the
            corresponding input. If an input is not a Tensor or is a Tensor not
            requiring grads, you can just pass None as a gradient for that input.

            grad_output {dl/dy}
            returned [dl/dtheta_1], [dl/dtheta_2]...
            If the parameter is not latent (known), then None needs to be returned.
            """
            # -- J^T stored from the forward pass
            grad_forward = ctx.saved_tensors

            # dl/theta = [J]^T * {v}
            grad = th.matmul(grad_forward[0], grad_output)

            # three tensors needs to be returned as forward takes three agruments. None for x as it doenst need grad
            return grad[0], grad[1], None

    return Autograd.apply


# Generating noisy data

forward_1 = forward_solver(x1)
y_exp_1 = np.random.normal(loc=forward_1(A, b)[0], scale=sigma1)
plt.plot(x1, y_exp_1, '*', x1, forward_1(A, b)[0])

forward_2 = forward_solver(x2)
y_exp_2 = np.random.normal(loc=forward_2(A, b)[0], scale=sigma2)
plt.plot(x2, y_exp_2, '*', x2, forward_2(A, b)[0])
plt.show()


# posterior model using the overloaded autograd
def pos_model():
    A_tmp = pyro.sample('a', dist.Normal(1.8, 1.0))
    B = pyro.sample('b', dist.Normal(1.0, 1.0))

    sigma_smpl_1 = pyro.sample('sigma1', dist.Uniform(0, 1.0))
    sigma_smpl_2 = pyro.sample('sigma2', dist.Uniform(0, 1.0))

    # solver inputs.Unclean but works
    x1 = np.linspace(0, 1, 50)
    x2 = np.linspace(0, 1, 100)

    torch_forward = wrapper_autograd_overload()
    pyro.sample('lkl1', dist.Normal(torch_forward(A_tmp, B, x1), sigma_smpl_1), obs=th.from_numpy(y_exp_1))
    pyro.sample('lkl2', dist.Normal(torch_forward(A_tmp, B, x2), sigma_smpl_2), obs=th.from_numpy(y_exp_2))


# sampling with overloaded autograd
kernel = NUTS(pos_model)
mcmc_1 = MCMC(kernel, num_samples=100)
mcmc_1.run()
mcmc_1.summary()

sns.pairplot(pd.DataFrame(mcmc_1.get_samples()), kind='kde')
plt.show()
