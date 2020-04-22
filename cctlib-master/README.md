[![Build Status](https://travis-ci.org/CCTLib/cctlib.svg?branch=master)](https://travis-ci.org/CCTLib/cctlib)

CCTLib is a library to ubiquitously collect calling contexts as well as attribute costs to data objects in an execution of a program.


--------------------------------------------------
	Supported platforms
--------------------------------------------------
1. Linux x86_64

--------------------------------------------------
	Requirements	
--------------------------------------------------

0. Ensure you have g++ 4.8.2 or higher as the default compiler and make sure you compile everything with -std=c++11 flag

1. Download and install the latest Pin framework 2.14 rev 71313 matching your platform from
http://www.pintool.org/downloads.html. In case you don't download Pin, our build script has an option to automatically download from WWW.
Our current release is tested on http://software.intel.com/sites/landingpage/pintool/downloads/pin-2.14-71313-gcc.4.4.7-linux.tar.gz

--------------------------------------------------
	Compiling
--------------------------------------------------

1. Set PIN_ROOT environmental variable to point to the root of the Pin installation. No need to do this step if you want the build.sh script to automatically download Pin from the WWW.

 e.g. 
 export PIN_ROOT=/path/to/pin/root/pin-2.14-67254-gcc.4.4.7-linux/
 
To build, simply type "sh build.sh"
This will configure, make, and check CCTLib. 

This produces libcctlib.a and libcctlib_tree_based.a in the src directory (refer to http://dl.acm.org/citation.cfm?id=2544164 for details).
libcctlib.a is CCTLib with shadow memory-based data-centric attribution.
libcctlib_tree_based.a is CCTLib with balanced binary tree based data-centric attribution.

Please refer to FAQs if you have compilation errors. If you still have issues compiling contact CCTLib Forum (cctlib-forum@lists.wm.edu).

--------------------------------------------------
Documentation of CCTLib's key APIs 
--------------------------------------------------

1. int PinCCTLibInit(IsInterestingInsFptr isInterestingIns, FILE* logFile, CCTLibInstrumentInsCallback userCallback, VOID* userCallbackArg, BOOL doDataCentric = false);
	Description: 
   		CCTLib clients must call this before using CCTLib. 
                Note: For postmortem analysis call PinCCTLibInitForPostmortemAnalysis() instead.
	Arguments:
                isInterestingIns: a client tool callback that should return boolean true/false if a given INS needs to collect context. 
                    Following predefined values are available for client tools: 
                        INTERESTING_INS_ALL => client tool needs the calling context on each instruction. tests/cct_client.cpp is a good example to demonstrate this use case.
                        INTERESTING_INS_MEMORY_ACCESS => client tool needs the calling context on each load/store instruction. tests/cct_client_mem_only.cpp is a good example to demonstrate this use case.
                        INTERESTING_INS_NONE => client tool does not require calling context only to the level of function names and callsites, leaf level instructions are ignored.
		logFile: file pointer where CCTLib will put its output data.
		userCallback: a client callback that CCTLib calls on each INS for which isInterestingIns is true passing it userCallbackArg value.
                userCallbackArg: is a void pointer that CCTLib takes and passes back to the client tool callback provided by userCallback argument.
		doDataCentric: should be set to true if the client wants CCTLib to do data-centric attribution.

2. ContextHandle_t GetContextHandle(THREADID threadId, uint32_t opaqueHandle);
	Description:
		Client tools call this API when they need the calling context handle (ContextHandle_t).
	Arguments:
		threadId: Pin's thread id of the asking thread.
		opaqueHandle: handle passed by CCTLib to the client tool in its userCallback.

3. DataHandle_t GetDataObjectHandle(VOID* address, THREADID threadId);
	Description:
		Client tools call this API when they need handle to the data object (DataHandle_t).
	Arguments:
		address: effective address for which the data object is needed.
		threadId: Pin's thread id of the asking thread.
	Note: Make sure that you have finite stack size. Don't set "ulimit -s unlimited"

4. VOID PrintFullCallingContext(ContextHandle_t ctxtHandle);
	Description:
		Prints the full calling context whose handle is ctxtHandle. Client tools must call PIN_LockClient() before calling this API and release lock via PIN_UnlockClient().
		I have intentionally made client tool to hold lock (PIN_LockClient) instead of CCTLib holding the lock so that it becomes efficient and the granularity of locking is left to the user.
		If the client tool is already holding the lock, it does not make sense for CCTLib to acquire it again (It is not clear from Pin manual if this lock is reentrant), hence this design is justified.
	Typical use:
		PIN_LockClient();

		for (...) {
			PrintFullCallingContext(i);
		}

		PIN_UnlockClient();
		
5. VOID GetFullCallingContext(ContextHandle_t ctxtHandle, vector<Context>& contextVec);
	Description:
		Returns the full calling context whose handle is ctxtHandle. Client tools must call PIN_LockClient() before calling this API and release lock via PIN_UnlockClient().
                I have intentionally made client tool to hold lock (PIN_LockClient) instead of CCTLib holding the lock so that it becomes efficient and the granularity of locking is left to the user.
                If the client tool is already holding the lock, it does not make sense for CCTLib to acquire it again (It is not clear from Pin manual if this lock is reentrant), hence this design is justified.
        Typical use:
                PIN_LockClient();

                for (...) {
                        GetFullCallingContext(...);
                }

                PIN_UnlockClient();


	Arguments:
		ctxtHandle: is the context handle for which the full call path is requested.
		contextVec: is a vector that will be populated with the full call path.

6. int PinCCTLibInitForPostmortemAnalysis(FILE* logFile, string serializedFilesDirectory);
	Description:
		Reads serialized CCT metadata and rebuilds CCTs for postmortem analysis.
	Arguments:
		logFile: file pointer where CCTLib will put its output data.
		serializedFilesDirectory: Path to directory where previously files were serialized.

		Caution: This should never be called with PinCCTLibInit().

7. void SerializeMetadata(string directoryForSerializationFiles = "");
	Description: 
		Serializes all CCTLib data into files for postmortem analysis.

	Arguments:
		directoryForSerializationFiles: directory where serialized files are written.

8. void DottifyAllCCTs()
   Description:
	Dumps all CCTs into DOT files for visualization.

9. bool IsSameSourceLine(ContextHandle_t ctxt1, ContextHandle_t ctxt2) 
   Description:
       Given two contexts handles, returns true if they both map to the same source line (could be different instructions). 
       Client tools must call PIN_LockClient() before calling this API and release lock after via PIN_UnlockClient(). Follow instructions similar to GetFullCallingContext() to decide the granularity of locking. 


--------------------------------------------------
Example uses of CCTLib	
--------------------------------------------------

The "tests" directory contains several simple example uses of CCTLib.

cct_client.cpp is a simple tool that gathers calling context on each instruction.
cct_client_mem_only.cpp is a simple tool that gathers calling context on each memory access.
cct_data_centric_client.cpp is a simple tool that associates each memory access to its associated data object via shadow memory technique.
cct_data_centric_client_tree_based.cpp is a simple tool that associates each memory access to its associated data object via balanced binary tree technique.

The "clients" directory contains advanced example uses of CCTLib.
deadspy_client.cpp is an implementation of DeadSpy that uses CCTLib and serializes the CCT.
cctlib_reader.cpp reads the serialized CCT and build back the CCT for postmortem analysis.


--------------------------------------------------
Example code snippets
--------------------------------------------------

1. Gathering calling context on every instruction by a Pin tool.
   In your Pin tool's main(), after initializing Pin via PIN_Init, call PinCCTLib::PinCCTLibInit(INTERESTING_INS_ALL, gTraceFile, InstrumentInsCallback, 0).
   The arguments to PinCCTLibInit are explained below:
   1.1 INTERESTING_INS_ALL is a predefined callback in CCTLib that tells that the client tool "may" want to gather calling context on any instruction.
      Instead of using INTERESTING_INS_ALL, the client tool may provide its own callback function whose signature must be as follows:      
      typedef BOOL (* IsInterestingInsFptr)(INS ins);      
      INTERESTING_INS_ALL is implemented as follow: BOOL InterestingInsAll(INS ins) { return true;}
      A more sophisticated Pin client tool may inspect INS and decide the return value true/false differently.
      
   1.2 gTraceFile is a file pointer (FILE *) that the client tool must pass to CCTLib so that CCTLib can write its log messages to that file. 
       
   1.3 InstrumentInsCallback is a callback that Pin client tool provides to CCTLib. CCTLib calls this function on each INS for which the first argument "IsInterestingInsFptr" return true.
       The prototype of InstrumentInsCallback is as follow:

       typedef VOID (*CCTLibInstrumentInsCallback)(INS ins, VOID* v, uint32_t slot);

       INS is the instruction on which instrumentation is being performed. 
       v is the same pointer that the client had passed to CCTLib (i.e., 4th argument to  PinCCTLibInit).
       slot is an opaque handle that CCTLib gives to the Pin client which the client MUST faithfully pass as an argument to CCTLib's GetContextHandle() routine called from this INS.

       In the implementation of InstrumentInsCallback callback, the client tool can perform the instrumentation that it would normally perform via functions such as INS_InsertCall() or INS_InsertPredicatedCall().
       For example, a client Pin tool that increments a counter on each instruction and gathers calling context can have its InstrumentInsCallback() implemented as follows:
       
       VOID SimpleCCTQuery(THREADID id, uint32_t slot) {
           // Increment counter
           gCounter++;
           
           // Gather calling context
           GetContextHandle(id, slot);
       }

       VOID InstrumentInsCallback(INS ins, VOID* v, uint32_t slot) {
           INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)SimpleCCTQuery, IARG_THREAD_ID, IARG_UINT32, slot, IARG_END);
       }
       
       
       This completes a simple example of gathering calling context on each instruction. Full code is present in tests/cct_client.cpp


--------------------------------------------------
Some useful control flags / macros
--------------------------------------------------
 
1. MAX_IPNODES is the maximum number of call path handles supported. It is by default set to (1<<32). The virtual address space is eagerly allocated get contiguous memory, but physical memory is consumed iff needed. If your machine has virtual memory limitation, change this value to a suitable value by passing -DMAX_IPNODES=<num> when compiling cctlib.

2. MAX_STRING_POOL_NODES is the maximum number of variable names supported in data-centric analysis. It is by default set to (1<<30). The virtual address space is eagerly allocated to get contiguous memory, but physical memory is consumed iff needed. If your machine has virtual memory limitation, change this value to a suitable value by passing -DMAX_STRING_POOL_NODES=<num> when compiling cctlib.


--------------------------------------------------
FAQs: Frequently asked questions
--------------------------------------------------
Q: How can I compile CCTLib for development?
A: Pass --enable-develop switch to the ./configure step.

Q: How can I cite CCTLib:
A: Please cite this paper: Milind Chabbi, Xu Liu, and John Mellor-Crummey. 2014. Call Paths for Pin Tools. In Proceedings of Annual IEEE/ACM International Symposium on Code Generation and Optimization (CGO '14). ACM, New York, NY, USA, , Pages 76 , 11 pages. DOI=http://dx.doi.org/10.1145/2544137.2544164


--------------------------------------------------
Publications based on CCTLib
--------------------------------------------------
1. Milind Chabbi, Xu Liu, and John Mellor-Crummey. 2014. Call Paths for Pin Tools. In Proceedings of Annual IEEE/ACM International Symposium on Code Generation and Optimization (CGO '14). ACM, New York, NY, USA, , Pages 76 , 11 pages. DOI=http://dx.doi.org/10.1145/2544137.2544164
2. Milind Chabbi, Wim Lavrijsen, Wibe de Jong, Koushik Sen, John Mellor-Crummey, and Costin Iancu. 2015. Barrier elision for production parallel programs. In Proceedings of the 20th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP 2015). ACM, New York, NY, USA, 109-119. DOI=http://dx.doi.org/10.1145/2688500.2688502
3. Milind Chabbi and John Mellor-Crummey. 2012. DeadSpy: a tool to pinpoint program inefficiencies. In Proceedings of the Tenth International Symposium on Code Generation and Optimization (CGO '12). ACM, New York, NY, USA, 124-134. DOI=http://dx.doi.org/10.1145/2259016.2259033
4. Shasha Wen, Xu Liu, Milind Chabbi, "Runtime Value Numbering: A Profiling Technique to Pinpoint Redundant Computations"  The 24th International Conference on Parallel Architectures and Compilation Techniques (PACT'15), Oct 18-21, 2015, San Francisco, California, USA
5. Shasha Wen, Milind Chabbi, and Xu Liu. 2017. REDSPY: Exploring Value Locality in Software. In Proceedings of the Twenty-Second International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '17). ACM, New York, NY, USA, 47-61. DOI: https://doi.org/10.1145/3037697.3037729
6. Pengfei Su, Shasha Wen, Hailong Yang, Milind Chabbi, and Xu Liu. 2019. Redundant Loads: A Software Inefficiency Indicator. In Proceedings of the 41st International Conference on Software Engineering (ICSE '19). IEEE Press, Piscataway, NJ, USA, 982-993. DOI: https://doi.org/10.1109/ICSE.2019.00103



