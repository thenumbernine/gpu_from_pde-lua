Here's a quick attempt to take my Lua computer algebra system and combine it with my Lua OpenCL object wrappers.

I've always wanted to automatically generate finite-difference initial-value problem grid solver code.

I have a very simple example of this using the [spring force demo](https://thenumbernine.github.io/symmath/tests/output/spring%20force.html) on my symmath project.

That just runs on the CPU.

I wanted to make something for GPUs.

Right now it handles GMRES convergence on PDE boundary value problems.

I still have to get boundary conditions specified and higher order derivatives.

I plan to incorporate this into ImGuiApp and simulate initial value problems.
