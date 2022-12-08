### N-Body simulation: Computational Physics Project

## The Code:
I've defined a particles class similar to Jon's code. The basic idea here is the same: I compute the softened potential of a single particle on a N-by-N grid and convolve it with the particle density matrix. This results in the full potential and its gradient serves to calculate the forces on each particle. For each timestep, the position and velocity of each particle is updated according to either the leapfrog method

$$x_i(t + dt) = x_i(t) + v_i(t + dt/2)dt$$



## Task 1: Single particle

## Task 2: Two rotating particles

## Task 3: 100k particles

## Task 4: Using rk4
