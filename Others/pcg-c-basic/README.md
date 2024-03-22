# PCG Random Number Generation, Minimal C Edition

[PCG-Random website]: http://www.pcg-random.org

This code provides a minimal implementation of one member of the PCG family
of random number generators, which are fast, statistically excellent,
and offer a number of useful features.

Full details can be found at the [PCG-Random website].  This version
of the code provides a single family member and skips some useful features
(such as jump-ahead/jump-back) -- if you want a more full-featured library, 
you may prefer the full version of the C library, or for all features,
the C++ library.

## Documentation and Examples

Visit [PCG-Random website] for information on how to use this library, or look
at the sample code -- hopefully it should be fairly self explanatory.

## Functions
pcg32_srandom_r(rngptr, initstate, initseq)

pcg32_random_r(rngptr)

pcg32_boundedrand_r(rngptr, bound)
