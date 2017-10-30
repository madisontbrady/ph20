# Physics 20 Set 3 Functions
# comment added for git version control
# Import the necessary packages.
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


# Numerical functions.
def spring_motion(h, x0, v0, Tmax):
    '''Generate a list of t, x, and v which describe the stepwise
    explicit Euler approximations of a spring's motion.  
    
    Args:
        h: Step size
        x0: Initial x value
        v0: Initial v value
        Tmax: Maximum value of t
        
    Returns:
        t_list: A list of values of t associated with the other returned values.
        x_list: A list of x-values derived by the explicit Euler's method
        v_list: A list of v-values derived by the explicit Euler's method
    '''
    
    # Generate a list of t's given h and Tmax.
    t_list = np.arange(0, Tmax, h)
    
    # Create numpy arrays to store the lists in.
    n_iter = len(t_list)
    x_list = np.zeros(n_iter)
    v_list = np.zeros(n_iter)
    
    # Begin our iteration at (x0, v0).
    current_x = x0
    current_v = v0
    
    # For each step, use Euler's explicit formula to calculate
    # the new x and v.  Store these in a list.
    for i in range(n_iter):
        x_list[i] = current_x
        v_list[i] = current_v
        
        current_x = x_list[i] + h * v_list[i]
        current_v = v_list[i] - h * x_list[i]
    
    # Return the lists of t, x, and v.
    return t_list, x_list, v_list


def analytic_spring_motion(h, x0, v0, Tmax):
    
    '''Generate a list of t, x, and v which describe the analytically 
    derived motion of a spring.  
    
    Args:
        h: Step size
        x0: Initial x value
        v0: Initial v value
        Tmax: Maximum value of t
        
    Returns:
        t_list: A list of values of t associated with the other returned values.
        x_list: A list of x-values derived analytically.
        v_list: A list of v-values derived analytically.
    '''
    
    # Generate a range of t's.
    t_list = np.arange(0, Tmax, h)
    
    # Calculate the list of x's and v's.
    x_list = x0 * np.cos(t_list) + v0 * np.sin(t_list)
    v_list = -x0 * np.sin(t_list) + v0 * np.cos(t_list)

    # Return a list of t, x, and v.
    return t_list, x_list, v_list

    def num_anal_err(h, x0, v0, Tmax):
    '''Generate a list of t, x, and v which describe the stepwise
    error of the explicit approximation of a spring's motion.  
    
    Args:
        h: Step size
        x0: Initial x value
        v0: Initial v value
        Tmax: Maximum value of t
        
    Returns:
        t_list: A list of values of t associated with the other returned values.
        x_err: A list of values corresponding to the explicit method's error on x.
        v_err: A list of values corresponding to the explicit method's error on v.
    '''
    
    # Generate the analytic and spring solutions.
    t_list, x_list, v_list  = spring_motion(h, x0, v0, Tmax)
    at_list, ax_list, av_list  = analytic_spring_motion(h, x0, v0, Tmax)
    
    # Generate a list of error by finding the difference between the
    # x and v values.
    x_err = ax_list - x_list
    v_err = av_list - v_list
    
    # Return the errors.
    return t_list, x_err, v_err

def max_err(h, x0, v0, Tmax):
    '''Given h, find the maximum possible value of the x error of the explicit
        method in an interval from 0 to Tmax.
    
    Args:
        h: Step size
        x0: Initial x value
        v0: Initial v value
        Tmax: Maximum value of t
        
    Returns:
        x_max: The maximum error on x.
    '''

    # Find the error on x by finding the difference between the analytic
    # solution and the explicit solution.
    t_list, x_list, v_list = spring_motion(h, x0, v0, Tmax)
    at_list, ax_list, av_list  = analytic_spring_motion(h, x0, v0, Tmax)
    x_err = ax_list - x_list
    
    # Return the maximum error.
    x_max = np.max(x_list)
    return x_max


def energy(h, x0, v0, Tmax):    
    '''Given h, find the maximum possible value of the x error of the explicit
        method in an interval from 0 to Tmax.
    
    Args:
        h: Step size
        x0: Initial x value
        v0: Initial v value
        Tmax: Maximum value of t
        
    Returns:
        t_list: List of t's.
        E_num: List representing the numerical energy.
        E_anal: List representing the analytic energy.
        E_error: List representing the numerical solution's energy error.
    '''
    
    # Find the explicit and analytic solutions.
    t_list, x_list, v_list  = spring_motion(h, x0, v0, Tmax)
    at_list, ax_list, av_list  = analytic_spring_motion(h, x0, v0, Tmax)
    
    # Calculate the energy associated with x and v.
    E_num = x_list**2. + v_list**2.
    E_anal = ax_list**2. + av_list**2.
    
    # Calculate the explicit method's error.
    E_err = E_num - E_anal
    
    # Return the t, energy, and error on energy.
    return t_list, E_num, E_anal, E_err


def implicit_spring_motion(h, x0, v0, Tmax):
    '''Generate a list of t, x, and v which describe the stepwise
    implicit Euler approximations of a spring's motion.  
    
    Args:
        h: Step size
        x0: Initial x value
        v0: Initial v value
        Tmax: Maximum value of t
        
    Returns:
        t_list: A list of values of t associated with the other returned values.
        x_list: A list of x-values derived by the implicit Euler's method
        v_list: A list of v-values derived by the implicit Euler's method
        x_err: List of implicit error on x
        v_err: List of implicit error on v
        e_list: List of implicit energy
    '''
        
    # Generate list of x and v to describe motion.
    t_list = np.arange(0, Tmax, h)
    n_iter = len(t_list)
    
    x_list = np.zeros(n_iter)
    v_list = np.zeros(n_iter)
    
    current_x = x0
    current_v = v0
    
    for i in range(n_iter):
        x_list[i] = current_x
        v_list[i] = current_v
        
        current_x = (x_list[i] + h * v_list[i]) / (1. + h**2.)
        current_v = (v_list[i] - h * x_list[i]) / (1. + h**2.)
        
    # Generate list of analytically derived x and v 
    # to describe the approximation error.
    at_list, ax_list, av_list = analytic_spring_motion(h, x0, v0, Tmax)
    
    x_err = ax_list - x_list
    v_err = av_list - v_list
    
    # Calculate the energy for the implicitly derived spring motion.
    e_list = x_list**2. + v_list**2.
    
    return t_list, x_list, v_list, x_err, v_err, e_list

def symplectic_spring_motion(h, x0, v0, Tmax):
    '''Generate a list of t, x, and v which describe the stepwise
    symplectic Euler approximations of a spring's motion.  
    
    Args:
        h: Step size
        x0: Initial x value
        v0: Initial v value
        Tmax: Maximum value of t
        
    Returns:
        t_list: A list of values of t associated with the other returned values.
        x_list: A list of x-values derived by the symplectic Euler's method
        v_list: A list of v-values derived by the symplectic Euler's method
        x_err: List of symplectic error on x
        v_err: List of symplectic error on v
        e_list: List of symplectic energy
    '''
    
    # Generate list of x and v to describe motion.   
    t_list = np.arange(0, Tmax, h)
    n_iter = len(t_list)
    
    x_list = np.zeros(n_iter)
    v_list = np.zeros(n_iter)
    
    current_x = x0
    current_v = v0
    
    for i in range(n_iter):
        x_list[i] = current_x
        v_list[i] = current_v
        
        current_x = current_x + h * current_v
        current_v = current_v - h * current_x       
                
    # Generate list of analytically derived x and v 
    # to describe the approximation error.
    at_list, ax_list, av_list = analytic_spring_motion(h, x0, v0, Tmax)
    
    x_err = ax_list - x_list
    v_err = av_list - v_list
    
    # Calculate the energy for the implicitly derived spring motion.
    e_list = x_list**2. + v_list**2.
    
    return t_list, x_list, v_list, x_err, v_err, e_list

# Plotting functions.

def xv_t_graph(h, x0, v0, Tmax, filename)
    '''Plot x and v against t.  Save a plot with the desired filename.'''

    # Generate a list of t, x, and v with the desired parameters.
    t_list, x_list, v_list  = spring_motion(h, x0, v0, Tmax)

    # Graph the t, x, and v.  Label the plot.
    plt.plot(t_list, x_list, label="X")
    plt.plot(t_list, v_list, label="V")
    plt.legend()
    plt.xlabel("T Coordinate")
    plt.ylabel("Value")
    plt.savefig(filename)
    plt.show()

def err_plot(h, x0, v0, Tmax, approx_type, filename):
    '''Generate a plot of the error on x and v for the desired method.'''

    # Generate the error lists based on the type of error.

    if approx_type == "explicit":
        plt.title("Error of Explicit Function")
        t_list, x_err, v_err = num_anal_err(h, x0, v0, Tmax)

    elif approx_type == "implicit":
        plt.title("Error of implicit Function")
        t_list, x_list, v_list, x_err, v_err, e_list \
        = implicit_spring_motion(h, x0, v0, Tmax)

    elif approx_type == "symplectic":
        plt.title("Error of symplectic Function")
        t_list, x_list, v_list, x_err, v_err, e_list \
        = symplectic_spring_motion(h, x0, v0, Tmax)

    # Plot the error against time.
    plt.plot(t_list, x_err, label="X Error")
    plt.plot(t_list, v_err, label="V Error")
    plt.legend()

    plt.xlabel("T Coordinate")
    plt.ylabel("Error")
    plt.savefig(filename)
    plt.show()

def h_err_plot(h, x0, v0, Tmax, approx_type, filename):
    '''Plot maximum error on x vs h for the given type of plot.'''
    # Create an array of h's to check the error on.
    h_array = np.linspace(h0 / 10, h0, 10)
    max_err_list = np.zeros(len(h_array))

    # For each h in the array, find the maximum error.
    for i in range(len(h_array)):
        max_err_list[i] = (max_err(h_array[i], x0, v0, Tmax))

    # Plot the maximum error against h.
    plt.xlabel("h")
    plt.ylabel("Maximum Error")
    plt.plot(h_array, max_err_list, label="Error")

    if approx_type == "explicit":
        plt.title("The Dependence of the Explicit Error on $h$")

    elif approx_type == "implicit":
        plt.title("The Dependence of the Implicit Error on $h$")

    elif approx_type == "symplectic":
        plt.title("The Dependence of the Symplectic Error on $h$")

    plt.title("The Dependence of the Explicit Error on $h$")
    plt.savefig(filename)
    plt.show()

def energy_plot(h, x0, v0, Tmax, approx_type, filename):
    '''Plot the energy of the given approximation vs. time.'''

    # Calculate the energy.
    if approx_type == "explicit":
        plt.title("Explicit Numerical Evolution of Energy")
        t_list, e_list, e_anal, e_err = energy(h, x0, v0, Tmax)

    elif approx_type == "implicit":
        plt.title("Implicit Numerical Evolution of Energy")
        t_list, x_list, v_list, x_err, v_err, e_list \
        = implicit_spring_motion(h, x0, v0, Tmax)

    elif approx_type == "symplectic":
        plt.title("Symplectic Numerical Evolution of Energy")
        t_list, x_list, v_list, x_err, v_err, e_list \
        = symplectic_spring_motion(h, x0, v0, Tmax)

    # Plot the implicit energy.
    plt.plot(t_list, e_list, label="Numerically Derived Energy")
    plt.plot([t_list[0], t_list[-1]], [1, 1], label="Analytically Derived Energy")
    plt.xlim([t_list[0], t_list[-1]])
    plt.xlabel("T Coordinate")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig(filename)
    plt.show()
    
def phase_plot(h, x0, v0, Tmax, approx_type, filename):
    '''Plot the phase-space geometry of the numerical approximations.'''

    ta_list, xa_list, va_list = analytic_spring_motion(h, x0, v0, Tmax)

    # Plot the phase-space geometry of the euler method.
    if approx_type == "explicit":
        plt.title("Phase-space Geometry of Explicit Euler Method")
        t_list, e_list, e_anal, e_err = energy(h, x0, v0, Tmax)

    elif approx_type == "implicit":
        plt.title("Phase-space Geometry of Implicit Euler Method")
        t_list, x_list, v_list, x_err, v_err, e_list \
        = implicit_spring_motion(h, x0, v0, Tmax)

    elif approx_type == "symplectic":
        plt.title("Phase-space Geometry of Symplectic Euler Method")
        t_list, x_list, v_list, x_err, v_err, e_list \
        = symplectic_spring_motion(h, x0, v0, Tmax)

    plt.xlabel("x(t)")
    plt.ylabel("v(t)")
    plt.plot(xa_list, va_list, ls='dotted', label="Analytic Method")
    plt.plot(x_list, v_list, label="Numerical Method")
    plt.legend(loc="upper left")
    plt.savefig(filename)
    plt.show()


# Another comment for git version control.