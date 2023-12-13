import  sympy
def derivative(func, x):
    h = .001
    return (func(x+h)-func(x))/h
sympy.Derivative

def fun(x): return x ** 2
x=3
predicted_value = derivative(fun,x)
print(predicted_value)


# let code for diffrentitation

fx="3x^2"
def diff(fx):
    coff = fx.split('x')

    print(coff)

diff(fx)


