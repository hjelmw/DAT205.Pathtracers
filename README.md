# DAT205-Pathtracer
Pathtracer implemented as part of a project in the course DAT205 Advanced Computer Graphics at Chalmers University of Technology.
In the project we first implemented a monte carlo pathtracer with a complete BRDF running on the CPU using Embree. This version was later ported to the Nvidia Optix ray tracing engine. 
Finally we added image denoising using the Nvidia AI Denoiser, a pretrained neural network for *smoothing* out artifacts common during pathtracing

# Install

## Optix
If you want to run the project yourself the easiest way for Optix is probably the following

* Start by installing and setting up [Nvidia Optix 6.5](https://raytracing-docs.nvidia.com/optix6/index.html)
* After setting up optix, Place the folder `src\OptixPathtracer` under `<Path to Optix 6.5>\SDK` 
* Add an entry in the Cmakelist.txt and rebuild. The project should now appear and be executable along with the other samples for optix

## Embree
If you instead want to check out the embree version please follow the setup for the skeleton code on the TDA362 introduction to Computer Graphics course page [link](http://www.cse.chalmers.se/edu/course/TDA362/tutorials/index.html) since this version uses the skeleton code. After this is done, replace the *contents* of `project` with `EmbreePathtracer`
