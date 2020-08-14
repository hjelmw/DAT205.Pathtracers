

# DAT205-Pathtracers

![alt text](https://github.com/hjelmw/DAT205-Pathtracer/blob/master/img/project_img.png)
Noisy             | Denoised
:-------------------------:|:-------------------------:
<img src="https://github.com/hjelmw/DAT205-Pathtracer/blob/master/img/optix_noisy.PNG">   |  <img src="https://github.com/hjelmw/DAT205-Pathtracer/blob/master/img/optix_denoised.PNG">


# Summary
Pathtracer implemented together with [@Type-0](https://github.com/type-o) in the course DAT205 Advanced Computer Graphics at Chalmers University of Technology.
In the project we first implemented a Monte Carlo pathtracer running on the CPU using [Embree](https://www.embree.org/) as the ray tracing kernel. It features a shading model composed of multiple linearly blended [BRDFs](https://en.wikipedia.org/wiki/Bidirectional_reflectance_distribution_function), reflections, shadows (no jaggies) and environment map sampling. This version was later *painstakingly* ported to the Nvidia Optix ray tracing engine so it would run on the GPU instead.
Finally we added support for image denoising to this version using the Nvidia AI Denoiser, a pretrained neural network built for *smoothing* out image artifacts common during pathtracing.


# Install
Not recommended but if you still want to try here are some pointers (also for future me: Hello future William!)

## Optix
If you want to run the project yourself the easiest way for the Optix version is probably the following

* Start by installing and setting up [Nvidia Optix 6.5](https://raytracing-docs.nvidia.com/optix6/index.html)
* After setting up optix, Place the folder `src\OptixPathtracer` under `<Path to Optix 6.5>\SDK` 
* Add an entry in the Cmakelist.txt and rebuild. The project **PathtracerGPU** should now appear and be executable along with the other samples for optix

## Embree
If you instead want to check out the Embree version you should start by
* Setup the project skeleton code by going to the TDA362 Introduction to Computer Graphics course page [link](http://www.cse.chalmers.se/edu/course/TDA362/tutorials/index.html)
* Replace the *contents* of the folder `project` with the contents of `src\EmbreePatchtracer`. 
* Set it as the startup project and run
