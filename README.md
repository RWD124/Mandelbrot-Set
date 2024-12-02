# Mandelbrot-Set
Julia code that makes a video of the Mandelbrot Set being zoomed into using CUDA to run on GPU.

You can get Julia at https://julialang.org/

To run the code you need Julia and Colors, CUDA, ColorTypes, ProgressBars, Adapt, and VideoIO.
I think you might need FileIO too.
Theese can be installed with          
                                  import Pkg; Pkg.add(["Colors", "CUDA", "ColorTypes", "ProgressBars", "Adapt", "VideoIO", FileIO])

Then just julia Mandelbrot_Set_Cuda_Video.jl
And it will start to make the video using a Nvidia GPU asuming you have the right drivers for that card.

Current examples are not the best but we are working on it.
