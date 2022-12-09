### N-Body simulation: Computational Physics Project

## The Code:
I've defined a particles class similar to Jon's code. The basic idea here is the same: I compute the softened potential of a single particle on a N-by-N grid and convolve it with the particle density matrix. This results in the full potential and its gradient serves to calculate the forces on each particle. For each timestep, the position and velocity of each particle is updated according to either the leapfrog method (in tasks 1-3) or the Runge-Kutta 4th order interpolation.

For each timestep, I compute the density matrix ``self.grid``, convolve it with the softened single particle potential using a helper function from the `numpy` package
```
self.grid, self.xedges, self.yedges = np.histogram2d(self.x, self.y, range = [[0, self.N], [0, self.N]], bins = self.N)
```
where `self.N` is the number of grid cells in each dimension. I've used `scipy`'s `signal.convolve2d` function to compute the convolution instead of using `numpy.fft.fftn`/`numpy.fft.rfftn`/etc., because I didn't want to mess with complications following wrap-around errors or other slightly non-intuitive behaviours of those functions. Additionally, `signal` allows one to skip several steps in the computation of the convolution (like zero-padding or shifting) by simply defining them in the function call. 

I compute the gradient of the full potential matrix using the `numpy.gradient` method and defining the acceleration for each particle by finding it's position on the grid. The values of the gradient matrix in the cell of the matrix then serve as the force on each. Thereby, particles in the same grid cell do not feel forces from each other.


## Task 1: Single particle

The `single_particle.gif` shows that a single particle at the center of the grid does not move from it's original position as expected.
![](https://github.com/simonwb98/phys512/blob/main/project/gifs/single_particle.gif)


## Task 2: Two rotating particles

Giving two particles some initial velocity in equal and opposite directions makes them rotate about a common center of mass under certain conditions. I'm showing one of these cases in `two_particles.gif` - Here, the particles return to their original positions after one full rotation. The trajectories are not perfectly round, but *slightly* elliptic. Thus the opposing particle does not reach the other's starting position after half a rotation. This effect is symmetric and the center of mass stays the same. 

![](https://github.com/simonwb98/phys512/blob/main/project/gifs/two_particles.gif)

## Task 3: Many, many, many particles (Wish computational death for my laptop)
With periodic boundary conditions, I get the following gif for 100k particles. For long time periods the clusters are seen to agglomerate close to the corners of the frame (which are of course the same position by periodicity) to form a mega-cluster. There could be a physical explanation, but I found this effect to be peculiar, because the corners should in principle not be *special* in that sense. 
![](https://github.com/simonwb98/phys512/blob/main/project/gifs/many_particles_100k.gif)
![](https://github.com/simonwb98/phys512/blob/main/project/figs/100k_particles_periodic_leapfrog.jpg)
## Task 4: Using rk4
