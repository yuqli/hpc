# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/yuqiongli/Desktop/HPC/lab/lab1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yuqiongli/Desktop/HPC/lab/lab1/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/lab1-c5-c6.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lab1-c5-c6.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lab1-c5-c6.dir/flags.make

CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o: CMakeFiles/lab1-c5-c6.dir/flags.make
CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o: ../lab1-c5-c6.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yuqiongli/Desktop/HPC/lab/lab1/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o   -c /Users/yuqiongli/Desktop/HPC/lab/lab1/lab1-c5-c6.c

CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/yuqiongli/Desktop/HPC/lab/lab1/lab1-c5-c6.c > CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.i

CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/yuqiongli/Desktop/HPC/lab/lab1/lab1-c5-c6.c -o CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.s

CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o.requires:

.PHONY : CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o.requires

CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o.provides: CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o.requires
	$(MAKE) -f CMakeFiles/lab1-c5-c6.dir/build.make CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o.provides.build
.PHONY : CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o.provides

CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o.provides.build: CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o


# Object files for target lab1-c5-c6
lab1__c5__c6_OBJECTS = \
"CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o"

# External object files for target lab1-c5-c6
lab1__c5__c6_EXTERNAL_OBJECTS =

lab1-c5-c6: CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o
lab1-c5-c6: CMakeFiles/lab1-c5-c6.dir/build.make
lab1-c5-c6: CMakeFiles/lab1-c5-c6.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/yuqiongli/Desktop/HPC/lab/lab1/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable lab1-c5-c6"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lab1-c5-c6.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lab1-c5-c6.dir/build: lab1-c5-c6

.PHONY : CMakeFiles/lab1-c5-c6.dir/build

CMakeFiles/lab1-c5-c6.dir/requires: CMakeFiles/lab1-c5-c6.dir/lab1-c5-c6.c.o.requires

.PHONY : CMakeFiles/lab1-c5-c6.dir/requires

CMakeFiles/lab1-c5-c6.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lab1-c5-c6.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lab1-c5-c6.dir/clean

CMakeFiles/lab1-c5-c6.dir/depend:
	cd /Users/yuqiongli/Desktop/HPC/lab/lab1/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yuqiongli/Desktop/HPC/lab/lab1 /Users/yuqiongli/Desktop/HPC/lab/lab1 /Users/yuqiongli/Desktop/HPC/lab/lab1/cmake-build-debug /Users/yuqiongli/Desktop/HPC/lab/lab1/cmake-build-debug /Users/yuqiongli/Desktop/HPC/lab/lab1/cmake-build-debug/CMakeFiles/lab1-c5-c6.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lab1-c5-c6.dir/depend
