# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/siddharth/OpenSfM/opensfm/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/siddharth/OpenSfM/cmake_build

# Include any dependencies generated for this target.
include map/CMakeFiles/pymap.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include map/CMakeFiles/pymap.dir/compiler_depend.make

# Include the progress variables for this target.
include map/CMakeFiles/pymap.dir/progress.make

# Include the compile flags for this target's objects.
include map/CMakeFiles/pymap.dir/flags.make

map/CMakeFiles/pymap.dir/python/pybind.cc.o: map/CMakeFiles/pymap.dir/flags.make
map/CMakeFiles/pymap.dir/python/pybind.cc.o: /home/siddharth/OpenSfM/opensfm/src/map/python/pybind.cc
map/CMakeFiles/pymap.dir/python/pybind.cc.o: map/CMakeFiles/pymap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/siddharth/OpenSfM/cmake_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object map/CMakeFiles/pymap.dir/python/pybind.cc.o"
	cd /home/siddharth/OpenSfM/cmake_build/map && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT map/CMakeFiles/pymap.dir/python/pybind.cc.o -MF CMakeFiles/pymap.dir/python/pybind.cc.o.d -o CMakeFiles/pymap.dir/python/pybind.cc.o -c /home/siddharth/OpenSfM/opensfm/src/map/python/pybind.cc

map/CMakeFiles/pymap.dir/python/pybind.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pymap.dir/python/pybind.cc.i"
	cd /home/siddharth/OpenSfM/cmake_build/map && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/siddharth/OpenSfM/opensfm/src/map/python/pybind.cc > CMakeFiles/pymap.dir/python/pybind.cc.i

map/CMakeFiles/pymap.dir/python/pybind.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pymap.dir/python/pybind.cc.s"
	cd /home/siddharth/OpenSfM/cmake_build/map && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/siddharth/OpenSfM/opensfm/src/map/python/pybind.cc -o CMakeFiles/pymap.dir/python/pybind.cc.s

# Object files for target pymap
pymap_OBJECTS = \
"CMakeFiles/pymap.dir/python/pybind.cc.o"

# External object files for target pymap
pymap_EXTERNAL_OBJECTS =

/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: map/CMakeFiles/pymap.dir/python/pybind.cc.o
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: map/CMakeFiles/pymap.dir/build.make
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: map/libmap.a
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: geometry/libgeometry.a
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: foundation/libfoundation.a
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: bundle/libbundle.a
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: geo/libgeo.a
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: foundation/libfoundation.a
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_gapi.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_highgui.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_ml.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_objdetect.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_photo.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_stitching.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_video.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_calib3d.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_dnn.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_features2d.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_flann.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_videoio.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_imgcodecs.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_imgproc.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/local/lib/libopencv_core.so.4.10.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libglog.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/libceres.so.2.0.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libglog.so.0.4.0
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libunwind.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopenblas.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libspqr.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libtbb.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libcholmod.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libccolamd.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libcamd.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libcolamd.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libcxsparse.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libamd.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopenblas.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libspqr.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libtbb.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libcholmod.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libccolamd.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libcamd.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libcolamd.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libcxsparse.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libamd.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/librt.a
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libmetis.so
/home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so: map/CMakeFiles/pymap.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/siddharth/OpenSfM/cmake_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module /home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so"
	cd /home/siddharth/OpenSfM/cmake_build/map && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pymap.dir/link.txt --verbose=$(VERBOSE)
	cd /home/siddharth/OpenSfM/cmake_build/map && /usr/bin/strip /home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
map/CMakeFiles/pymap.dir/build: /home/siddharth/OpenSfM/opensfm/pymap.cpython-310-x86_64-linux-gnu.so
.PHONY : map/CMakeFiles/pymap.dir/build

map/CMakeFiles/pymap.dir/clean:
	cd /home/siddharth/OpenSfM/cmake_build/map && $(CMAKE_COMMAND) -P CMakeFiles/pymap.dir/cmake_clean.cmake
.PHONY : map/CMakeFiles/pymap.dir/clean

map/CMakeFiles/pymap.dir/depend:
	cd /home/siddharth/OpenSfM/cmake_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/siddharth/OpenSfM/opensfm/src /home/siddharth/OpenSfM/opensfm/src/map /home/siddharth/OpenSfM/cmake_build /home/siddharth/OpenSfM/cmake_build/map /home/siddharth/OpenSfM/cmake_build/map/CMakeFiles/pymap.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : map/CMakeFiles/pymap.dir/depend
