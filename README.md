# CS516 Tour d'Algorithms (Project 2) - Killian Lelong

This project implements various sorting algorithms.
The sorting algorithms included are Bubble Sort (Single Thread), Bitonic Sort (Multi Thread), and the STL `thrust` as a reference implementation.
Compiling this project requires `g++` and `make` and will generate 7 executables in a `bin` directory.

## Executables

The project produces 3 executables:

1. **singlethread** - Bubble Sort (Single Thread)
2. **multithread** - Bitonic Sort (Multi Thread)
3. **thrust** - STL Thrust (Reference Implementation)

## Usage

### SingleThread

Command line format:

```bash
./bin/singlethread [number of random integers] [seed value] [print (0/1)]
```

Example:

```bash
./bin/singlethread 10000 1234 0
```

### MultiThread

Command line format:

```bash
./bin/multithread [number of random integers] [seed value] [print (0/1)]
```

Example:

```bash
./bin/multithread 10000 1234 0
```

## Compile the project

Make sure to have `g++` installed.
(On msys2: `pacman -S mingw-w64-x86_64-gcc`)

```bash
# Create all executables:
$ make all

# Clean and rebuild:
$ make clean

# Build a specific executable (e.g., singlethread):
$ make singlethread
```

## Run Performance Tests

Tests can be run using the `make test` target. You can specify the dataset size using the `TESTSIZE` variable.
A random seed is generated for each test run.

```bash
# Default test (100,000 elements)
$ make test
```

### Custom test size

```bash
# Custom test with 1,000,000 elements
$ make test TESTSIZE=1000000
```
