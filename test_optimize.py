from scipy.optimize import minimize

def f(params):
    print(params)
    a, b = params
    return a**2 + b**2

initial_guess = [1, 1]
result = minimize(f, initial_guess, options={'maxiter':100})
if result.success:
    print(result.x)
    print(result.fun)
else:
    raise ValueError(result.message)
