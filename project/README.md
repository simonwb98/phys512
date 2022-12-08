### N-Body simulation: Computational Physics Project

## The Code:
I've defined a particles class similar to Jon's code. The basic idea here is the same: I compute the softened potential of a single particle on a N-by-N grid and convolve it with the particle density matrix. This results in the full potential and its gradient serves to calculate the forces on each particle. For each timestep, the position and velocity of each particle is updated according to either the leapfrog method (in tasks 1-3) or the Runge-Kutta 4th order interpolation.

## Task 1: Single particle

The `single_particle.gif` shows that a single particle at the center of the grid does not move from it's original position as expected.

## Task 2: Two rotating particles

Giving two particles some initial velocity in equal and opposite directions makes them rotate about a common center of mass under certain conditions. I'm showing one of these cases in `two_particles.gif` - Here, the particles return to their original positions after one full rotation. 

## Task 3: 100k particles

## Task 4: Using rk4
