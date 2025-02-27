using Colors
using CUDA  # Import the CUDA package
using ColorTypes: N0f8
using ProgressBars
import Adapt  # Import Adapt for GPU array conversion
import VideoIO

function mandelbrot_gpu_video(width, height, max_iter, zoom, x_offset, y_offset, zoom_rate, frames)
    imgstack = []
    for i in ProgressBar(0:frames)
        zoom = zoom - zoom * zoom_rate
        push!(imgstack, mandelbrot_gpu(width, height, max_iter, zoom , x_offset, y_offset))
    end
    println(zoom)
    return imgstack
end

function mandelbrot_gpu(width, height, max_iter, zoom::Float64, x_offset::Float64, y_offset::Float64)
    # Move parameters to the GPU
    zoom_gpu = Adapt.adapt(CuArray, zoom)
    x_offset_gpu = Adapt.adapt(CuArray, x_offset)
    y_offset_gpu = Adapt.adapt(CuArray, y_offset)

    # Pre-allocate the pixel array on the GPU
    pixels_gpu = CUDA.zeros(RGB{N0f8}, height, width)

    # Calculate indices for each pixel on the GPU
    threads = (32, 32)
    blocks = (ceil(Int, width / threads[1]), ceil(Int, height / threads[2]))
    aspect_ratio = Float64(width) / Float64(height)

    @cuda threads=threads blocks=blocks kernel_mandelbrot(
        pixels_gpu, width, height, max_iter, zoom_gpu, x_offset_gpu, y_offset_gpu, aspect_ratio
    )

    # Copy the result back to the CPU
    pixels = Array(pixels_gpu)
    return pixels
end

function kernel_mandelbrot(pixels, width, height, max_iter, zoom::Float64, x_offset::Float64, y_offset::Float64, aspect_ratio::Float64)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= width && j <= height
        x0 = ((i - Float64(width) / 2) / (Float64(width) / 2)) * zoom * aspect_ratio + x_offset
        y0 = ((j - Float64(height) / 2) / (Float64(height) / 2)) * zoom + y_offset
        x = 0.0
        y = 0.0
        iter = 0

        while x * x + y * y <= 4 && iter < max_iter
            xtemp = x * x - y * y + x0
            y = 2 * x * y + y0
            x = xtemp
            iter += 1
        end

        if iter < max_iter
            smooth_iter = iter + 1 - log(log2(sqrt(x * x + y * y)))
            color_hsv = HSV(360 * (smooth_iter / max_iter), 1, 1)
            color_rgb = convert(RGB{N0f8}, color_hsv)
            pixels[j, i] = color_rgb
        else
            pixels[j, i] = RGB{N0f8}(0, 0, 0)
        end
    end
    return nothing
end

# Set parameters
width = 1920
height = 1080
max_iter = 1000
zoom = 1 #1.915736376341104e-16     # Now Float64
x_offset = -0.79271704858622756  # Now Float64
y_offset = -0.16089270323427649  # Now Float64
zoom_rate = 0.01    # Could be Float64 if operated with zoom
frames = 3600
fps = 60
encoder_options = (crf=0, preset="ultrafast")

# Generate Mandelbrot set video (GPU)
println("Generating Mandelbrot set video (GPU)...")
@time imgstack = mandelbrot_gpu_video(width, height, max_iter, zoom, x_offset, y_offset, zoom_rate, frames);

@time VideoIO.save("mandelbrot_gpu_video.mp4", imgstack, framerate=fps, encoder_options=encoder_options, codec_name="libx264rgb");

println("Done!")
