NAME    =   CUDA_SORTing

CC      =   nvcc

# Output directory for object files
BUILDDIR = obj
# Output directory for binaries
BINDIR   = bin

# Source files are expected in src/
SRC     =   $(wildcard src/*.cu)
OBJ 	= 	$(SRC:src/%.cu=$(BUILDDIR)/%.o)

# The three required executables
TARGETS = thrust singlethread multithread

# Defaults for testing
TESTSIZE ?= 100000
TESTSEED ?= 12345

all: $(TARGETS)

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

$(BINDIR):
	@mkdir -p $(BINDIR)

# NVCC Flags: Arch sm_50 is widely compatible, O2 for optimization
CFLAGS  =  -O2 -arch=sm_50 -I ./include/ --std=c++14

# Compile object files
$(BUILDDIR)/%.o: src/%.cu | $(BUILDDIR)
	@$(CC) $(CFLAGS) -dc $< -o $@
	@echo -e "\e[01m\e[33m    Compiling...\
		\e[00m\e[39m$<\e[032m [OK]\e[00m"

# Link Targets
thrust: $(BUILDDIR)/thrust.o | $(BINDIR)
	@$(CC) $(CFLAGS) -o $(BINDIR)/$@ $<
	@echo -e "\e[32mBuilt thrust executable.\e[0m"

singlethread: $(BUILDDIR)/singlethread.o | $(BINDIR)
	@$(CC) $(CFLAGS) -o $(BINDIR)/$@ $<
	@echo -e "\e[32mBuilt singlethread executable.\e[0m"

multithread: $(BUILDDIR)/multithread.o | $(BINDIR)
	@$(CC) $(CFLAGS) -o $(BINDIR)/$@ $<
	@echo -e "\e[32mBuilt multithread executable.\e[0m"

clean:
	@rm -rf $(BUILDDIR)
	@echo -e "\e[31;1mAll object files have been removed.\e[0m"

fclean: clean
	@rm -rf $(BINDIR)
	@echo -e "\e[38;5;210mBinaries\e[0m\e[38;5;196m have been removed.\e[0m"

re: fclean all
	@echo -e "\e[38;5;42mRebuild Complete.\e[0m"

# Test script adapted for the specific output format of the skeleton code
test: all
	@echo "Running CUDA sorting performance tests..."
	@echo "Dataset size: $(TESTSIZE) elements"
	@SEED=$$(date +%s%N | cut -b1-13); \
	if [ -z "$$SEED" ]; then SEED=$$(date +%s); fi; \
	echo "Random seed: $$SEED"; \
	echo ""; \
	echo "| Algorithm              | Time (seconds) |"; \
	echo "|------------------------|----------------|"; \
	THRUST_TIME=$$(./bin/thrust $(TESTSIZE) $$SEED 0 2>&1 | grep "Total time in seconds:" | awk '{print $$5}'); \
	printf "| %-22s | %-14s |\n" "Thrust (Library)" "$$THRUST_TIME"; \
	SINGLE_TIME=$$(./bin/singlethread $(TESTSIZE) $$SEED 0 2>&1 | grep "Total time in seconds:" | awk '{print $$5}'); \
	printf "| %-22s | %-14s |\n" "Single Thread GPU" "$$SINGLE_TIME"; \
	MULTI_TIME=$$(./bin/multithread $(TESTSIZE) $$SEED 0 2>&1 | grep "Total time in seconds:" | awk '{print $$5}'); \
	printf "| %-22s | %-14s |\n" "Multithread GPU" "$$MULTI_TIME"; \

.PHONY: all clean fclean re test