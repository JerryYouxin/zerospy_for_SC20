# Zerospy

There are two main folders in this directory: cctlib-master and cctlib-no_crt. cctlib-master is the default Zerospy tool directory and cctlib-no_crt is a flatten calling context version of Zerospy, which can save memory for those programs with very deep calling contexts. You should always use Zerospy in cctlib-master, except for failures reported for too deep calling contexts.

## Build

The build process of these two directory is same. For cctlib-master, the building script is listed below:

~~~~
$ tar xvzf pin-2.14-71313-gcc4.4.7-linux.tar.gz
$ export PIN_ROOT=`pwd`/pin-2.14-71313-gcc4.4.7-linux 
$ cd cctlib-master
$ ./build.sh
~~~~

For cctlib-no_crt, the building script is listed below:

~~~~
$ tar xvzf pin-3.7-97619-g0d0c92f4f-linux.tar.gz
$ export PIN_ROOT=`pwd`/pin-3.7-97619-g0d0c92f4f-linux
$ cd cctlib-no_crt
$ ./build.sh
~~~~

The compiled Zerospy tool is in cctlib-master/clients/obj-xxx/zerospy_client_fast.so (and cctlib-master/clients/obj-xxx/zerospy_spatial_client_fast.so for data centric analysis)

## Usage

To profile a program with Zerospy, run the program as follows:

~~~~
$ # For cctlib-master (default) version, PIN_ROOT is path to pin-2.14, CCTLIB_ROOT is path to cctlib-master
$ # For cctlib-no_crt (Flatten Calling Context) version, PIN_ROOT is path to pin-3.7, CCTLIB_ROOT is path to cctlib-no_crt
$ $PIN_ROOT/pin.sh -ifeellucky -t $CCTLIB_ROOT/clients/obj-xxx/zerospy_client_fast.so -- $EXE # code centric analysis
$ $PIN_ROOT/pin.sh -ifeellucky -t $CCTLIB_ROOT/clients/obj-xxx/zerospy_spatial_client_fast.so -- $EXE # data centric analysis
~~~~
