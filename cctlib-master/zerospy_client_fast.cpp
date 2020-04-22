// @COPYRIGHT@
// Licensed under MIT license.
// See LICENSE.TXT file in the project root for more information.
// ==============================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <malloc.h>
#include <iostream>
#include <unistd.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <string.h>
#include <sys/mman.h>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <list>
#include "pin.H"
#include "cctlib.H"
// #include "shadow_memory.H"
#include <xmmintrin.h>
#include <immintrin.h>
// definition of BYTE_ORDER
#include <endian.h>

extern "C" {
#include "xed-interface.h"
#include "xed-common-hdrs.h"
}

// #include <google/sparse_hash_map>
// #include <google/dense_hash_map>
// using google::sparse_hash_map;  // namespace where class lives by default
// using google::dense_hash_map;

using namespace std;
using namespace PinCCTLib;

// have R, W representative macros
#define READ_ACTION (0)
#define WRITE_ACTION (0xff)

#define ONE_BYTE_READ_ACTION (0)
#define TWO_BYTE_READ_ACTION (0)
#define FOUR_BYTE_READ_ACTION (0)
#define EIGHT_BYTE_READ_ACTION (0)

#define ONE_BYTE_WRITE_ACTION (0xff)
#define TWO_BYTE_WRITE_ACTION (0xffff)
#define FOUR_BYTE_WRITE_ACTION (0xffffffff)
#define EIGHT_BYTE_WRITE_ACTION (0xffffffffffffffff)

#define IS_ACCESS_WITHIN_PAGE_BOUNDARY(accessAddr, accessLen)  (PAGE_OFFSET((accessAddr)) <= (PAGE_OFFSET_MASK - (accessLen)))

/* Other footprint_client settings */
#define MAX_REDUNDANT_CONTEXTS_TO_LOG (1000)
#define THREAD_MAX (1024)

#define ENCODE_ADDRESS_AND_ACCESS_LEN(addr, len) ( (addr) | (((uint64_t)(len)) << 48))
#define DECODE_ADDRESS(addrAndLen) ( (addrAndLen) & ((1L<<48) - 1))
#define DECODE_ACCESS_LEN(addrAndLen) ( (addrAndLen) >> 48)


#define MAX_WRITE_OP_LENGTH (512)
#define MAX_WRITE_OPS_IN_INS (8)
#define MAX_REG_LENGTH (64)

#define MAX_SIMD_LENGTH (64)
#define MAX_SIMD_REGS (32)

#define PAGE_MASK (~0xfff)
#define GET_PAGE_INDEX(x) ((x) & PAGE_MASK)

#define CACHELINE_MASK (~63)
#define GET_CACHELINE_INDEX(x) ((x) & CACHELINE_MASK)

//#define MERGING
// #define NO_APPROXMAP
// #define SKIP_SMALLCASE

#ifdef ENABLE_SAMPLING

#define WINDOW_ENABLE 1000000
#define WINDOW_DISABLE 100000000
#define WINDOW_CLEAN 10
#endif

#define DECODE_DEAD(data) static_cast<uint8_t>(((data)  & 0xffffffffffffffff) >> 32 )
#define DECODE_KILL(data) (static_cast<ContextHandle_t>( (data)  & 0x00000000ffffffff))


#define MAKE_CONTEXT_PAIR(a, b) (((uint64_t)(a) << 32) | ((uint64_t)(b)))

#define delta 0.01

// #define ENABLE_FILTER_BEFORE_SORT

#define MULTI_THREADED

#ifdef NO_CRT
KNOB<BOOL>   KnobFlatProfile(KNOB_MODE_WRITEONCE, "pintool", "fp", "0", "Collect flat profile");
#endif

#define static_assert(x) static_assert(x,#x)

/***********************************************
 ******  shadow memory
 ************************************************/
//ConcurrentShadowMemory<uint8_t, ContextHandle_t> sm;

struct{
    char dummy1[128];
    xed_state_t  xedState;
    char dummy2[128];
} LoadSpyGlobals;

struct RedSpyThreadData{
    uint64_t bytesLoad;
};

// for metric logging
int redload_metric_id = 0;
int redload_approx_metric_id = 0;

//for statistics result
uint64_t grandTotBytesLoad;
uint64_t grandTotBytesRedLoad;
uint64_t grandTotBytesApproxRedLoad;

// 64 byte per cacheline : 8 * int64 
struct __attribute__((aligned(64))) CachelineAlignedInt64_t {
    uint64_t val;
};
CachelineAlignedInt64_t threadBytesLoad[THREAD_MAX] __attribute__((aligned(64)));

//uint64_t localTotBytesLoad[THREAD_MAX] = {0};

// key for accessing TLS storage in the threads. initialized once in main()
static  TLS_KEY client_tls_key;
static RedSpyThreadData* gSingleThreadedTData;

// function to access thread-specific data
inline RedSpyThreadData* ClientGetTLS(const THREADID threadId) {
#ifdef MULTI_THREADED
    RedSpyThreadData* tdata =
    static_cast<RedSpyThreadData*>(PIN_GetThreadData(client_tls_key, threadId));
    return tdata;
#else
    return gSingleThreadedTData;
#endif
}

static INT32 Usage() {
    PIN_ERROR("Pin tool to gather calling context on each load and store.\n" + KNOB_BASE::StringKnobSummary() + "\n");
    return -1;
}

// Main for RedSpy, initialize the tool, register instrumentation functions and call the target program.
static FILE* gTraceFile;
//#define DEBUG_ZEROSPY
#ifdef DEBUG_ZEROSPY
static FILE* gDebugFile[200]={0};
static FILE* getDebugFile(int i) {
    if(!gDebugFile[i]) {
        char filename[20];
        sprintf(filename, "debug.%d.txt",i);
        gDebugFile[i] = fopen(filename,"w");
    }
    return gDebugFile[i];
}
#endif

// Initialized the needed data structures before launching the target program
static void ClientInit(int argc, char* argv[]) {
    // Create output file
    char name[MAX_FILE_PATH] = "zeroLoad.out.";
    char* envPath = getenv("CCTLIB_CLIENT_OUTPUT_FILE");
    
    if(envPath) {
        // assumes max of MAX_FILE_PATH
        strcpy(name, envPath);
    }
    
    gethostname(name + strlen(name), MAX_FILE_PATH - strlen(name));
    pid_t pid = getpid();
    sprintf(name + strlen(name), "%d", pid);
    cerr << "\n Creating log file at:" << name << "\n";
    gTraceFile = fopen(name, "w");
    // print the arguments passed
    fprintf(gTraceFile, "\n");
    
    for(int i = 0 ; i < argc; i++) {
        fprintf(gTraceFile, "%s ", argv[i]);
    }
    
    fprintf(gTraceFile, "\n");

    // Init Xed
    // Init XED for decoding instructions
    xed_state_init(&LoadSpyGlobals.xedState, XED_MACHINE_MODE_LONG_64, (xed_address_width_enum_t) 0, XED_ADDRESS_WIDTH_64b);
}


static const uint64_t READ_ACCESS_STATES [] = {/*0 byte */0, /*1 byte */ ONE_BYTE_READ_ACTION, /*2 byte */ TWO_BYTE_READ_ACTION, /*3 byte */ 0, /*4 byte */ FOUR_BYTE_READ_ACTION, /*5 byte */0, /*6 byte */0, /*7 byte */0, /*8 byte */ EIGHT_BYTE_READ_ACTION};
static const uint64_t WRITE_ACCESS_STATES [] = {/*0 byte */0, /*1 byte */ ONE_BYTE_WRITE_ACTION, /*2 byte */ TWO_BYTE_WRITE_ACTION, /*3 byte */ 0, /*4 byte */ FOUR_BYTE_WRITE_ACTION, /*5 byte */0, /*6 byte */0, /*7 byte */0, /*8 byte */ EIGHT_BYTE_WRITE_ACTION};
static const uint8_t OVERFLOW_CHECK [] = {/*0 byte */0, /*1 byte */ 0, /*2 byte */ 0, /*3 byte */ 1, /*4 byte */ 2, /*5 byte */3, /*6 byte */4, /*7 byte */5, /*8 byte */ 6};

struct RedLogs{
    uint64_t tot;
    uint64_t red;
    uint64_t fred; // full redundancy
    uint64_t redByteMap;
};
struct ApproxRedLogs{
    uint64_t fred; // full redundancy
    uint64_t ftot;
    uint64_t redByteMap;
    uint8_t AccessLen;
    uint8_t size;
};
//static unordered_map<uint64_t, uint64_t> validMap[THREAD_MAX];
static unordered_map<uint64_t, RedLogs> RedMap[THREAD_MAX];
static unordered_map<uint64_t, ApproxRedLogs> ApproxRedMap[THREAD_MAX];
static inline void AddToRedTable(uint64_t key,  uint16_t value, uint64_t byteMap, uint16_t total, THREADID threadId) __attribute__((always_inline,flatten));
static inline void AddToRedTable(uint64_t key,  uint16_t value, uint64_t byteMap, uint16_t total, THREADID threadId) {
    unordered_map<uint64_t, RedLogs>::iterator it = RedMap[threadId].find(key);
    if ( it  == RedMap[threadId].end()) {
        RedLogs log;
        log.red = value;
        log.tot = total;
        log.fred= (value==total);
        log.redByteMap = byteMap;
        //if(total < value) cerr << "ERROR : total " << total << " < value " << value << endl;
        RedMap[threadId][key] = log;
        //printf("Bucket size : %ld\n",RedMap[threadId].bucket_count());
    } else {
        it->second.red += value;
        it->second.tot += total;
        it->second.fred+= (value==total);
        it->second.redByteMap &= byteMap;
        //if(total < value) cerr << "ERROR : total " << total << " < value " << value << endl;
        //assert(it->second.AccessLen == total && "AccessLen not match");
    }
}

static inline void AddToApproximateRedTable(uint64_t key, uint64_t byteMap, uint16_t total, uint16_t zeros, uint16_t nums, uint8_t size, THREADID threadId) __attribute__((always_inline,flatten));
static inline void AddToApproximateRedTable(uint64_t key, uint64_t byteMap, uint16_t total, uint16_t zeros, uint16_t nums, uint8_t size, THREADID threadId) {
    unordered_map<uint64_t, ApproxRedLogs>::iterator it = ApproxRedMap[threadId].find(key);
    if ( it  == ApproxRedMap[threadId].end()) {
        ApproxRedLogs log;
        log.fred= zeros;
        log.ftot= nums;
        log.redByteMap = byteMap;
        log.AccessLen = total;
        log.size = size;
        //if(total < value) cerr << "ERROR : total " << total << " < value " << value << endl;
        ApproxRedMap[threadId][key] = log;
    } else {
        it->second.fred+= zeros;
        it->second.ftot+= nums;
        it->second.redByteMap &= byteMap;
        //if(total < value) cerr << "ERROR : total " << total << " < value " << value << endl;
        //assert(it->second.AccessLen == total && "AccessLen not match");
    }
}


#ifdef ENABLE_SAMPLING

static ADDRINT IfEnableSample(THREADID threadId){
    RedSpyThreadData* const tData = ClientGetTLS(threadId);
    return tData->sampleFlag;
}

#endif

// Certain FP instructions should not be approximated
static inline bool IsOkToApproximate(xed_decoded_inst_t & xedd) {
     xed_category_enum_t cat = xed_decoded_inst_get_category(&xedd);
     xed_iclass_enum_t 	iclass = xed_decoded_inst_get_iclass (&xedd);
     switch(iclass) {
	case XED_ICLASS_FLDENV:
	case XED_ICLASS_FNSTENV:
	case XED_ICLASS_FNSAVE:
	case XED_ICLASS_FLDCW:
	case XED_ICLASS_FNSTCW:
    case XED_ICLASS_FNSTSW:
	case XED_ICLASS_FXRSTOR:
	case XED_ICLASS_FXRSTOR64:
	case XED_ICLASS_FXSAVE:
	case XED_ICLASS_FXSAVE64:
		return false;
	default:
		return true;
     }
}

static inline bool IsFloatInstructionAndOkToApproximate(ADDRINT ip) {
#ifdef DEBUG_ZEROSPY
    FILE* debug = getDebugFile(0);
#endif
    xed_decoded_inst_t  xedd;
    xed_decoded_inst_zero_set_mode(&xedd, &LoadSpyGlobals.xedState);
    
    if(XED_ERROR_NONE == xed_decode(&xedd, (const xed_uint8_t*)(ip), 15)) {
        xed_category_enum_t cat = xed_decoded_inst_get_category(&xedd);
#ifdef DEBUG_ZEROSPY
        char xxx[200] = {0};
        xed_decoded_inst_dump (&xedd, xxx, 200);
        xed_format_context(XED_SYNTAX_ATT, &xedd, xxx, 200,  ip, 0, 0);
        fprintf(debug, " IP = %lx %s\n", ip, xxx);
#endif
        switch (cat) {
            case XED_CATEGORY_AES:
            case XED_CATEGORY_CONVERT:
            case XED_CATEGORY_PCLMULQDQ:
            case XED_CATEGORY_SSE:
            case XED_CATEGORY_AVX2:
            case XED_CATEGORY_AVX:
            case XED_CATEGORY_MMX:
            case XED_CATEGORY_DATAXFER: {
                // Get the mem operand
                
                const xed_inst_t* xi = xed_decoded_inst_inst(&xedd);
                int  noperands = xed_inst_noperands(xi);
                int memOpIdx = -1;
                for( int i =0; i < noperands ; i++) {
                    const xed_operand_t* op = xed_inst_operand(xi,i);
                    xed_operand_enum_t op_name = xed_operand_name(op);
                    if(XED_OPERAND_MEM0 == op_name) {
                        memOpIdx = i;
                        break;
                    }
                }
                if(memOpIdx == -1) {
                    return false;
                }
                
                // TO DO MILIND case XED_OPERAND_MEM1:
                xed_operand_element_type_enum_t eType = xed_decoded_inst_operand_element_type(&xedd,memOpIdx);
                switch (eType) {
                    case XED_OPERAND_ELEMENT_TYPE_FLOAT16:
                    case XED_OPERAND_ELEMENT_TYPE_SINGLE:
                    case XED_OPERAND_ELEMENT_TYPE_DOUBLE:
                    case XED_OPERAND_ELEMENT_TYPE_LONGDOUBLE:
                    case XED_OPERAND_ELEMENT_TYPE_LONGBCD:
                        return IsOkToApproximate(xedd);
                    default:
                        return false;
                }
            }
                break;
            case XED_CATEGORY_X87_ALU:
            case XED_CATEGORY_FCMOV:
                //case XED_CATEGORY_LOGICAL_FP:
                // assumption, the access length must be either 4 or 8 bytes else assert!!!
                //assert(*accessLen == 4 || *accessLen == 8);
                return IsOkToApproximate(xedd);
            case XED_CATEGORY_XSAVE:
            case XED_CATEGORY_AVX2GATHER:
            case XED_CATEGORY_STRINGOP:
            default: return false;
        }
    }else {
        assert(0 && "failed to disassemble instruction");
        //	printf("\n Diassembly failure\n");
        return false;
    }
}

static inline bool IsFloatInstructionOld(ADDRINT ip) {
    xed_decoded_inst_t  xedd;
    xed_decoded_inst_zero_set_mode(&xedd, &LoadSpyGlobals.xedState);
    
    if(XED_ERROR_NONE == xed_decode(&xedd, (const xed_uint8_t*)(ip), 15)) {
        xed_iclass_enum_t iclassType = xed_decoded_inst_get_iclass(&xedd);
        if (iclassType >= XED_ICLASS_F2XM1 && iclassType <=XED_ICLASS_FYL2XP1) {
            return true;
        }
        if (iclassType >= XED_ICLASS_VBROADCASTSD && iclassType <= XED_ICLASS_VDPPS) {
            return true;
        }
        if (iclassType >= XED_ICLASS_VRCPPS && iclassType <= XED_ICLASS_VSQRTSS) {
            return true;
        }
        if (iclassType >= XED_ICLASS_VSUBPD && iclassType <= XED_ICLASS_VXORPS) {
            return true;
        }
        switch (iclassType) {
            case XED_ICLASS_ADDPD:
            case XED_ICLASS_ADDPS:
            case XED_ICLASS_ADDSD:
            case XED_ICLASS_ADDSS:
            case XED_ICLASS_ADDSUBPD:
            case XED_ICLASS_ADDSUBPS:
            case XED_ICLASS_ANDNPD:
            case XED_ICLASS_ANDNPS:
            case XED_ICLASS_ANDPD:
            case XED_ICLASS_ANDPS:
            case XED_ICLASS_BLENDPD:
            case XED_ICLASS_BLENDPS:
            case XED_ICLASS_BLENDVPD:
            case XED_ICLASS_BLENDVPS:
            case XED_ICLASS_CMPPD:
            case XED_ICLASS_CMPPS:
            case XED_ICLASS_CMPSD:
            case XED_ICLASS_CMPSD_XMM:
            case XED_ICLASS_COMISD:
            case XED_ICLASS_COMISS:
            case XED_ICLASS_CVTDQ2PD:
            case XED_ICLASS_CVTDQ2PS:
            case XED_ICLASS_CVTPD2PS:
            case XED_ICLASS_CVTPI2PD:
            case XED_ICLASS_CVTPI2PS:
            case XED_ICLASS_CVTPS2PD:
            case XED_ICLASS_CVTSD2SS:
            case XED_ICLASS_CVTSI2SD:
            case XED_ICLASS_CVTSI2SS:
            case XED_ICLASS_CVTSS2SD:
            case XED_ICLASS_DIVPD:
            case XED_ICLASS_DIVPS:
            case XED_ICLASS_DIVSD:
            case XED_ICLASS_DIVSS:
            case XED_ICLASS_DPPD:
            case XED_ICLASS_DPPS:
            case XED_ICLASS_HADDPD:
            case XED_ICLASS_HADDPS:
            case XED_ICLASS_HSUBPD:
            case XED_ICLASS_HSUBPS:
            case XED_ICLASS_MAXPD:
            case XED_ICLASS_MAXPS:
            case XED_ICLASS_MAXSD:
            case XED_ICLASS_MAXSS:
            case XED_ICLASS_MINPD:
            case XED_ICLASS_MINPS:
            case XED_ICLASS_MINSD:
            case XED_ICLASS_MINSS:
            case XED_ICLASS_MOVAPD:
            case XED_ICLASS_MOVAPS:
            case XED_ICLASS_MOVD:
            case XED_ICLASS_MOVHLPS:
            case XED_ICLASS_MOVHPD:
            case XED_ICLASS_MOVHPS:
            case XED_ICLASS_MOVLHPS:
            case XED_ICLASS_MOVLPD:
            case XED_ICLASS_MOVLPS:
            case XED_ICLASS_MOVMSKPD:
            case XED_ICLASS_MOVMSKPS:
            case XED_ICLASS_MOVNTPD:
            case XED_ICLASS_MOVNTPS:
            case XED_ICLASS_MOVNTSD:
            case XED_ICLASS_MOVNTSS:
            case XED_ICLASS_MOVSD:
            case XED_ICLASS_MOVSD_XMM:
            case XED_ICLASS_MOVSS:
            case XED_ICLASS_MULPD:
            case XED_ICLASS_MULPS:
            case XED_ICLASS_MULSD:
            case XED_ICLASS_MULSS:
            case XED_ICLASS_ORPD:
            case XED_ICLASS_ORPS:
            case XED_ICLASS_ROUNDPD:
            case XED_ICLASS_ROUNDPS:
            case XED_ICLASS_ROUNDSD:
            case XED_ICLASS_ROUNDSS:
            case XED_ICLASS_SHUFPD:
            case XED_ICLASS_SHUFPS:
            case XED_ICLASS_SQRTPD:
            case XED_ICLASS_SQRTPS:
            case XED_ICLASS_SQRTSD:
            case XED_ICLASS_SQRTSS:
            case XED_ICLASS_SUBPD:
            case XED_ICLASS_SUBPS:
            case XED_ICLASS_SUBSD:
            case XED_ICLASS_SUBSS:
            case XED_ICLASS_VADDPD:
            case XED_ICLASS_VADDPS:
            case XED_ICLASS_VADDSD:
            case XED_ICLASS_VADDSS:
            case XED_ICLASS_VADDSUBPD:
            case XED_ICLASS_VADDSUBPS:
            case XED_ICLASS_VANDNPD:
            case XED_ICLASS_VANDNPS:
            case XED_ICLASS_VANDPD:
            case XED_ICLASS_VANDPS:
            case XED_ICLASS_VBLENDPD:
            case XED_ICLASS_VBLENDPS:
            case XED_ICLASS_VBLENDVPD:
            case XED_ICLASS_VBLENDVPS:
            case XED_ICLASS_VBROADCASTSD:
            case XED_ICLASS_VBROADCASTSS:
            case XED_ICLASS_VCMPPD:
            case XED_ICLASS_VCMPPS:
            case XED_ICLASS_VCMPSD:
            case XED_ICLASS_VCMPSS:
            case XED_ICLASS_VCOMISD:
            case XED_ICLASS_VCOMISS:
            case XED_ICLASS_VCVTDQ2PD:
            case XED_ICLASS_VCVTDQ2PS:
            case XED_ICLASS_VCVTPD2PS:
            case XED_ICLASS_VCVTPH2PS:
            case XED_ICLASS_VCVTPS2PD:
            case XED_ICLASS_VCVTSD2SS:
            case XED_ICLASS_VCVTSI2SD:
            case XED_ICLASS_VCVTSI2SS:
            case XED_ICLASS_VCVTSS2SD:
            case XED_ICLASS_VDIVPD:
            case XED_ICLASS_VDIVPS:
            case XED_ICLASS_VDIVSD:
            case XED_ICLASS_VDIVSS:
            case XED_ICLASS_VDPPD:
            case XED_ICLASS_VDPPS:
            case XED_ICLASS_VMASKMOVPD:
            case XED_ICLASS_VMASKMOVPS:
            case XED_ICLASS_VMAXPD:
            case XED_ICLASS_VMAXPS:
            case XED_ICLASS_VMAXSD:
            case XED_ICLASS_VMAXSS:
            case XED_ICLASS_VMINPD:
            case XED_ICLASS_VMINPS:
            case XED_ICLASS_VMINSD:
            case XED_ICLASS_VMINSS:
            case XED_ICLASS_VMOVAPD:
            case XED_ICLASS_VMOVAPS:
            case XED_ICLASS_VMOVD:
            case XED_ICLASS_VMOVHLPS:
            case XED_ICLASS_VMOVHPD:
            case XED_ICLASS_VMOVHPS:
            case XED_ICLASS_VMOVLHPS:
            case XED_ICLASS_VMOVLPD:
            case XED_ICLASS_VMOVLPS:
            case XED_ICLASS_VMOVMSKPD:
            case XED_ICLASS_VMOVMSKPS:
            case XED_ICLASS_VMOVNTPD:
            case XED_ICLASS_VMOVNTPS:
            case XED_ICLASS_VMOVSD:
            case XED_ICLASS_VMOVSS:
            case XED_ICLASS_VMOVUPD:
            case XED_ICLASS_VMOVUPS:
            case XED_ICLASS_VMULPD:
            case XED_ICLASS_VMULPS:
            case XED_ICLASS_VMULSD:
            case XED_ICLASS_VMULSS:
            case XED_ICLASS_VORPD:
            case XED_ICLASS_VORPS:
            case XED_ICLASS_VPABSD:
            case XED_ICLASS_VPADDD:
            case XED_ICLASS_VPCOMD:
            case XED_ICLASS_VPCOMUD:
            case XED_ICLASS_VPERMILPD:
            case XED_ICLASS_VPERMILPS:
            case XED_ICLASS_VPERMPD:
            case XED_ICLASS_VPERMPS:
            case XED_ICLASS_VPGATHERDD:
            case XED_ICLASS_VPGATHERQD:
            case XED_ICLASS_VPHADDBD:
            case XED_ICLASS_VPHADDD:
            case XED_ICLASS_VPHADDUBD:
            case XED_ICLASS_VPHADDUWD:
            case XED_ICLASS_VPHADDWD:
            case XED_ICLASS_VPHSUBD:
            case XED_ICLASS_VPHSUBWD:
            case XED_ICLASS_VPINSRD:
            case XED_ICLASS_VPMACSDD:
            case XED_ICLASS_VPMACSSDD:
            case XED_ICLASS_VPMASKMOVD:
            case XED_ICLASS_VPMAXSD:
            case XED_ICLASS_VPMAXUD:
            case XED_ICLASS_VPMINSD:
            case XED_ICLASS_VPMINUD:
            case XED_ICLASS_VPROTD:
            case XED_ICLASS_VPSUBD:
            case XED_ICLASS_XORPD:
            case XED_ICLASS_XORPS:
                return true;
                
            default: return false;
        }
    } else {
        assert(0 && "failed to disassemble instruction");
        return false;
    }
}

static inline uint16_t FloatOperandSize(ADDRINT ip, uint32_t oper) {
    xed_decoded_inst_t  xedd;
    xed_decoded_inst_zero_set_mode(&xedd, &LoadSpyGlobals.xedState);
    
    if(XED_ERROR_NONE == xed_decode(&xedd, (const xed_uint8_t*)(ip), 15)) {
        xed_operand_element_type_enum_t TypeOperand = xed_decoded_inst_operand_element_type(&xedd,oper);
        if(TypeOperand == XED_OPERAND_ELEMENT_TYPE_SINGLE || TypeOperand == XED_OPERAND_ELEMENT_TYPE_FLOAT16)
            return 4;
        if (TypeOperand == XED_OPERAND_ELEMENT_TYPE_DOUBLE) {
            return 8;
        }
        if (TypeOperand == XED_OPERAND_ELEMENT_TYPE_LONGDOUBLE) {
            return 16;
        }
        assert(0 && "float instruction with unknown operand\n");
        return 0;
    } else {
        assert(0 && "failed to disassemble instruction\n");
        return 0;
    }
}

/*******************************************************************************************/
// single floating point zero byte counter
// 32-bit float: |sign|exp|mantissa| = | 1 | 8 | 23 |
// the redmap of single floating point takes up 5 bits (1 bit sign, 1 bit exp, 3 bit mantissa)
#define SP_MAP_SIZE 5
inline __attribute__((always_inline)) uint64_t count_zero_bytemap_fp(void * addr) {
    register uint32_t xx = *((uint32_t*)addr);
    // reduce by bits until byte level
    // | 0 | x0x0 x0x0 | 0x0 x0x0 x0x0 x0x0 x0x0 x0x0 |
    xx = xx | ((xx>>1)&0xffbfffff);
    // | x | 00xx 00xx | 0xx 00xx 00xx 00xx 00xx 00xx |
    xx = xx | ((xx>>2)&0xffdfffff);
    // | x | 0000 xxxx | 000 xxxx 0000 xxxx 0000 xxxx |
    xx = xx | ((xx>>4)&0xfff7ffff);
    // now xx is byte level reduced, check if it is zero and mask the unused bits
    xx = (~xx) & 0x80810101;
    // narrowing
    xx = xx | (xx>>7) | (xx>>14) | (xx>>20) | (xx>>27);
    xx = xx & 0x1f;
    return xx;
}
inline __attribute__((always_inline)) bool hasRedundancy_fp(void * addr) {
    register uint32_t xx = *((uint32_t*)addr);
    return (xx & 0x007f0000)==0;
}
/*******************************************************************************************/
// double floating point zero byte counter
// 64-bit float: |sign|exp|mantissa| = | 1 | 11 | 52 |
// the redmap of single floating point takes up 10 bits (1 bit sign, 2 bit exp, 7 bit mantissa)
#define DP_MAP_SIZE 10
inline __attribute__((always_inline)) uint64_t count_zero_bytemap_dp(void * addr) {
    register uint64_t xx = (static_cast<uint64_t*>(addr))[0];
    // reduce by bits until byte level
    // | 0 | 0x0 x0x0 x0x0 | x0x0 x0x0_x0x0 x0x0_x0x0 x0x0_x0x0 x0x0_x0x0 x0x0_x0x0 x0x0_x0x0 |
    xx = xx | ((xx>>1)&(~0x4008000000000000LL));
    // | x | 0xx 00xx 00xx | 00xx 00xx_00xx 00xx_00xx 00xx_00xx 00xx_00xx 00xx_00xx 00xx_00xx |
    xx = xx | ((xx>>2)&(~0x200c000000000000LL));
    // | x | xxx 0000 xxxx | xxxx 0000_xxxx 0000_xxxx 0000_xxxx 0000_xxxx 0000_xxxx 0000_xxxx |
    xx = xx | ((xx>>4)&(~0x100f000000000000LL));
    // now xx is byte level reduced, check if it is zero and mask the unused bits
    xx = (~xx) & 0x9011010101010101LL;
    // narrowing
    register uint64_t m = xx & 0x1010101010101LL;
    m = m | (m>>7);
    m = m | (m>>14);
    m = m | (m>>28);
    m = m & 0x7f;
    xx = xx | (xx>>9) | (xx>>7);
    xx = (xx >> 45) & 0x380;
    xx = m | xx;
    return xx;
}
inline __attribute__((always_inline)) bool hasRedundancy_dp(void * addr) {
    register uint64_t xx = *((uint64_t*)addr);
    return (xx & 0x000f000000000000LL)==0;
}
/***************************************************************************************/
/*********************** floating point full redundancy functions **********************/
/***************************************************************************************/

#if __BYTE_ORDER == __BIG_ENDIAN
typedef union {
  float f;
  struct {
    uint32_t sign : 1;
    uint32_t exponent : 8;
    uint32_t mantisa : 23;
  } parts;
  struct {
    uint32_t sign : 1;
    uint32_t value : 31;
  } vars;
} float_cast;

typedef union {
  double f;
  struct {
    uint64_t sign : 1;
    uint64_t exponent : 11;
    uint64_t mantisa : 52;
  } parts;
  struct {
    uint64_t sign : 1;
    uint64_t value : 63;
  } vars;
} double_cast;
#elif __BYTE_ORDER == __LITTLE_ENDIAN
typedef union {
  float f;
  struct {
    uint32_t mantisa : 23;
    uint32_t exponent : 8;
    uint32_t sign : 1;
  } parts;
  struct {
    uint32_t value : 31;
    uint32_t sign : 1;
  } vars;
} float_cast;

typedef union {
  double f;
  struct {
    uint64_t mantisa : 52;
    uint64_t exponent : 11;
    uint64_t sign : 1;
  } parts;
  struct {
    uint64_t value : 63;
    uint64_t sign : 1;
  } vars;
} double_cast;
#else
    #error Unknown Byte Order
#endif

template<int start, int end, int incr>
struct UnrolledConjunctionApprox{
    // if the mantisa is 0, the value of the double/float var must be 0
    static __attribute__((always_inline)) uint64_t BodyZeros(uint8_t* addr){
        if(incr==4)
            return ((*(reinterpret_cast<float_cast*>(&addr[start]))).vars.value==0) + (UnrolledConjunctionApprox<start+incr,end,incr>::BodyZeros(addr));
        else if(incr==8)
            return ((*(reinterpret_cast<double_cast*>(&addr[start]))).vars.value==0) + (UnrolledConjunctionApprox<start+incr,end,incr>::BodyZeros(addr));
        return 0;
    }
    static __attribute__((always_inline)) uint64_t BodyRedMap(uint8_t* addr){
        if(incr==4)
            return count_zero_bytemap_fp((void*)(addr+start)) | (UnrolledConjunctionApprox<start+incr,end,incr>::BodyRedMap(addr)<<SP_MAP_SIZE);
        else if(incr==8)
            return count_zero_bytemap_dp((void*)(addr+start)) | (UnrolledConjunctionApprox<start+incr,end,incr>::BodyRedMap(addr)<<DP_MAP_SIZE);
        else
            assert(0 && "Not Supportted floating size! now only support for FP32 or FP64.");
        return 0;
    }
    static __attribute__((always_inline)) uint64_t BodyHasRedundancy(uint8_t* addr){
        if(incr==4)
            return hasRedundancy_fp((void*)(addr+start)) || (UnrolledConjunctionApprox<start+incr,end,incr>::BodyHasRedundancy(addr));
        else if(incr==8)
            return hasRedundancy_dp((void*)(addr+start)) || (UnrolledConjunctionApprox<start+incr,end,incr>::BodyHasRedundancy(addr));
        else
            assert(0 && "Not Supportted floating size! now only support for FP32 or FP64.");
        return 0;
    }
};

template<int end,  int incr>
struct UnrolledConjunctionApprox<end , end , incr>{
    static __attribute__((always_inline)) uint64_t BodyZeros(uint8_t* addr){
        return 0;
    }
    static __attribute__((always_inline)) uint64_t BodyRedMap(uint8_t* addr){
        return 0;
    }
    static __attribute__((always_inline)) uint64_t BodyHasRedundancy(uint8_t* addr){
        return 0;
    }
};

/****************************************************************************************/
inline __attribute__((always_inline)) uint64_t count_zero_bytemap_int8(uint8_t * addr) {
    register uint8_t xx = *((uint8_t*)addr);
    // reduce by bits until byte level
    xx = xx | (xx>>1) | (xx>>2) | (xx>>3) | (xx>>4) | (xx>>5) | (xx>>6) | (xx>>7);
    // now xx is byte level reduced, check if it is zero and mask the unused bits
    xx = (~xx) & 0x1;
    return xx;
}
inline __attribute__((always_inline)) uint64_t count_zero_bytemap_int16(uint8_t * addr) {
    register uint16_t xx = *((uint16_t*)addr);
    // reduce by bits until byte level
    xx = xx | (xx>>1) | (xx>>2) | (xx>>3) | (xx>>4) | (xx>>5) | (xx>>6) | (xx>>7);
    // now xx is byte level reduced, check if it is zero and mask the unused bits
    xx = (~xx) & 0x101;
    // narrowing
    xx = xx | (xx>>7);
    xx = xx & 0x3;
    return xx;
}
inline __attribute__((always_inline)) uint64_t count_zero_bytemap_int32(uint8_t * addr) {
    register uint32_t xx = *((uint32_t*)addr);
    // reduce by bits until byte level
    xx = xx | (xx>>1) | (xx>>2) | (xx>>3) | (xx>>4) | (xx>>5) | (xx>>6) | (xx>>7);
    // now xx is byte level reduced, check if it is zero and mask the unused bits
    xx = (~xx) & 0x1010101;
    // narrowing
    xx = xx | (xx>>7);
    xx = xx | (xx>>14);
    xx = xx & 0xf;
    return xx;
}
inline __attribute__((always_inline)) uint64_t count_zero_bytemap_int64(uint8_t * addr) {
    register uint64_t xx = *((uint64_t*)addr);
    // reduce by bits until byte level
    xx = xx | (xx>>1) | (xx>>2) | (xx>>3) | (xx>>4) | (xx>>5) | (xx>>6) | (xx>>7);
    // now xx is byte level reduced, check if it is zero and mask the unused bits
    xx = (~xx) & 0x101010101010101LL;
    // narrowing
    xx = xx | (xx>>7);
    xx = xx | (xx>>14);
    xx = xx | (xx>>28);
    xx = xx & 0xff;
    return xx;
}

static const unsigned char BitCountTable4[] __attribute__ ((aligned(64))) = {
    0, 0, 1, 2
};

static const unsigned char BitCountTable8[] __attribute__ ((aligned(64))) = {
    0, 0, 0, 0, 1, 1, 2, 3
};

static const unsigned char BitCountTable16[] __attribute__ ((aligned(64))) = {
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 4
};

static const unsigned char BitCountTable256[] __attribute__ ((aligned(64))) = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8
};

// accessLen & eleSize are size in bits
template<uint32_t AccessLen, uint32_t EleSize>
struct RedMapString {
    static __attribute__((always_inline)) std::string getIntRedMapString(uint64_t redmap) {
        static_assert(AccessLen % EleSize == 0);
        std::string buff = 
            RedMapString<AccessLen-EleSize, EleSize>::getIntRedMapString(redmap>>(AccessLen-EleSize)) + 
            " , " + RedMapString<EleSize, EleSize>::getIntRedMapString(redmap);
        return buff;
    }
};

template<uint32_t AccessLen>
struct RedMapString <AccessLen, AccessLen> {
    static __attribute__((always_inline)) std::string getIntRedMapString(uint64_t redmap) {
        std::string buff = "";
        buff += ((redmap>>(AccessLen-1))&0x1) ? "00 " : "XX ";
        buff += RedMapString<AccessLen-1, AccessLen-1>::getIntRedMapString(redmap>>1);
        return buff;
    }
};

template<>
struct RedMapString <1, 1> {
    static __attribute__((always_inline)) std::string getIntRedMapString(uint64_t redmap) {
        return std::string((redmap&0x1) ? "00" : "XX");
    }
};

template<uint32_t n_exp, uint32_t n_man>
inline __attribute__((always_inline)) std::string __getFpRedMapString(uint64_t redmap) {
    std::string buff = "";
    const uint32_t signPos = n_exp + n_man;
    buff += RedMapString<1,1>::getIntRedMapString(redmap>>signPos) + " | ";
    buff += RedMapString<n_exp,n_exp>::getIntRedMapString(redmap>>n_man) + " | ";
    buff += RedMapString<n_man,n_man>::getIntRedMapString(redmap);
    return buff;
}

template<uint32_t n_exp, uint32_t n_man>
std::string getFpRedMapString(uint64_t redmap, uint64_t accessLen) {
    std::string buff = "";
    uint64_t newAccessLen = accessLen - (n_exp + n_man + 1);
    if(newAccessLen==0) {
        return __getFpRedMapString<n_exp,n_man>(redmap);
    } else {
        return getFpRedMapString<n_exp,n_man>(redmap>>newAccessLen, newAccessLen) + " , " + __getFpRedMapString<n_exp,n_man>(redmap);
    }
    return buff;
}

#define getFpRedMapString_SP(redmap, num) getFpRedMapString<1,3>(redmap, num*5)
#define getFpRedMapString_DP(redmap, num) getFpRedMapString<2,7>(redmap, num*10)

template<int start, int end, int incr>
struct UnrolledConjunction{
    // if the mantisa is 0, the value of the double/float var must be 0
    static __attribute__((always_inline)) uint64_t BodyRedNum(uint64_t rmap){
        static_assert(start < end);
        if(incr==1)
            return ((start==0) ? (rmap&0x1) : ((rmap>>start)&0x1)) + (UnrolledConjunction<start+incr,end,incr>::BodyRedNum(rmap));
        else if(incr==2)
            return ((start==0) ? BitCountTable8[rmap&0x3] : BitCountTable8[(rmap>>start)&0x3]) + (UnrolledConjunction<start+incr,end,incr>::BodyRedNum(rmap));
        else if(incr==4)
            return ((start==0) ? BitCountTable16[rmap&0xf] : BitCountTable16[(rmap>>start)&0xf]) + (UnrolledConjunction<start+incr,end,incr>::BodyRedNum(rmap));
        else if(incr==8)
            return ((start==0) ? BitCountTable256[rmap&0xff] : BitCountTable256[(rmap>>start)&0xff]) + (UnrolledConjunction<start+incr,end,incr>::BodyRedNum(rmap));
        return 0;
    }
    static __attribute__((always_inline)) uint64_t BodyRedMap(uint8_t* addr){
        static_assert(start < end);
        if(incr==1)
            return count_zero_bytemap_int8(addr+start) | (UnrolledConjunction<start+incr,end,incr>::BodyRedMap(addr)<<1);
        else if(incr==2)
            return count_zero_bytemap_int16(addr+start) | (UnrolledConjunction<start+incr,end,incr>::BodyRedMap(addr)<<2);
        else if(incr==4)
            return count_zero_bytemap_int32(addr+start) | (UnrolledConjunction<start+incr,end,incr>::BodyRedMap(addr)<<4);
        else if(incr==8)
            return count_zero_bytemap_int64(addr+start) | (UnrolledConjunction<start+incr,end,incr>::BodyRedMap(addr)<<8);
        else
            assert(0 && "Not Supportted integer size! now only support for INT8, INT16, INT32 or INT64.");
        return 0;
    }
    static __attribute__((always_inline)) bool BodyHasRedundancy(uint8_t* addr){
        if(incr==1)
            return (addr[start]==0) || (UnrolledConjunction<start+incr,end,incr>::BodyHasRedundancy(addr));
        else if(incr==2)
            return (((*((uint16_t*)(&addr[start])))&0xff00)==0) || (UnrolledConjunction<start+incr,end,incr>::BodyRedMap(addr));
        else if(incr==4)
            return (((*((uint32_t*)(&addr[start])))&0xff000000)==0) || (UnrolledConjunction<start+incr,end,incr>::BodyRedMap(addr));
        else if(incr==8)
            return (((*((uint64_t*)(&addr[start])))&0xff00000000000000LL)==0) || (UnrolledConjunction<start+incr,end,incr>::BodyRedMap(addr));
        else
            assert(0 && "Not Supportted integer size! now only support for INT8, INT16, INT32 or INT64.");
        return 0;
    }
};

template<int end,  int incr>
struct UnrolledConjunction<end , end , incr>{
    static __attribute__((always_inline)) uint64_t BodyRedNum(uint64_t rmap){
        return 0;
    }
    static __attribute__((always_inline)) uint64_t BodyRedMap(uint8_t* addr){
        return 0;
    }
    static __attribute__((always_inline)) uint64_t BodyHasRedundancy(uint8_t* addr){
        return 0;
    }
};
/*******************************************************************************************/

uint32_t zeros_g=0;
uint64_t map_g=0;

bool is_pointer_valid(void *p) {
    /* get the page size */
    size_t page_size = sysconf(_SC_PAGESIZE);
    /* find the address of the page that contains p */
    void *base = (void *)((((size_t)p) / page_size) * page_size);
    /* call msync, if it returns non-zero, return false */
    return msync(base, page_size, MS_ASYNC) == 0;
}

template<class T, uint32_t AccessLen, bool isApprox>
struct ZeroSpyAnalysis{
    static __attribute__((always_inline)) VOID CheckNByteValueAfterRead(ADDRINT ip, void* addr, uint32_t opaqueHandle, THREADID threadId){
// #ifdef DEBUG_ZEROSPY
//         fprintf(gDebugFile,"\nINFO : In Check NBytes Value After Read\n");
// #endif
        //RedSpyThreadData* const tData = ClientGetTLS(threadId);
#ifdef DEBUG_ZEROSPY
        FILE* debug = getDebugFile(threadId);
        xed_decoded_inst_t  xedd;
        xed_decoded_inst_zero_set_mode(&xedd, &LoadSpyGlobals.xedState);
        if(XED_ERROR_NONE == xed_decode(&xedd, (const xed_uint8_t*)(ip), 15)) {
            char xxx[200] = {0};
            xed_decoded_inst_dump (&xedd, xxx, 200);
            xed_format_context(XED_SYNTAX_ATT, &xedd, xxx, 200,  ip, 0, 0);
            fprintf(debug, " IP = %lx %s\n", ip, xxx);
        } else {
            fprintf(debug, " Failed to disassemble IP = %lx\n",ip);
        }
        fflush(debug);
#endif
        ContextHandle_t curCtxtHandle = GetContextHandle(threadId, opaqueHandle);
        uint8_t* bytes = static_cast<uint8_t*>(addr);
        if(isApprox) {
            // uint32_t redbyteNum = getRedNum(addr);
            bool hasRedundancy = UnrolledConjunctionApprox<0,AccessLen,sizeof(T)>::BodyHasRedundancy(bytes);
            if(hasRedundancy) {
                uint64_t map = UnrolledConjunctionApprox<0,AccessLen,sizeof(T)>::BodyRedMap(bytes);
                uint32_t zeros = UnrolledConjunctionApprox<0,AccessLen,sizeof(T)>::BodyZeros(bytes);
                //printf("FP Redundancy Detected: %lx %d\n",map, zeros);         
                AddToApproximateRedTable((uint64_t)curCtxtHandle, map, AccessLen, zeros, AccessLen/sizeof(T), sizeof(T), threadId);
            } else {
                AddToApproximateRedTable((uint64_t)curCtxtHandle, 0, AccessLen, 0, AccessLen/sizeof(T), sizeof(T), threadId);
            }
        } else {
            // uint32_t redbyteNum = getRedNum(addr);
            bool hasRedundancy = UnrolledConjunction<0,AccessLen,sizeof(T)>::BodyHasRedundancy(bytes);
            if(hasRedundancy) {
                uint64_t redbyteMap = UnrolledConjunction<0,AccessLen,sizeof(T)>::BodyRedMap(bytes);
                uint32_t redbyteNum = UnrolledConjunction<0,AccessLen,sizeof(T)>::BodyRedNum(redbyteMap);
                AddToRedTable((uint64_t)MAKE_CONTEXT_PAIR(AccessLen, curCtxtHandle), redbyteNum, redbyteMap, AccessLen, threadId);
            } else {
                AddToRedTable((uint64_t)MAKE_CONTEXT_PAIR(AccessLen, curCtxtHandle), 0, 0, AccessLen, threadId);
            }
        }

#ifdef DEBUG_ZEROSPY
        fprintf(debug,"\nINFO : Exit Check NBytes Value After Read\n");
        fflush(debug);
#endif
    }
    static __attribute__((always_inline)) VOID CheckNByteValueAfterVGather(PIN_MULTI_MEM_ACCESS_INFO* multiMemAccessInfo, PIN_REGISTER* ymmMask, uint32_t opaqueHandle, THREADID threadId){
        ContextHandle_t curCtxtHandle = GetContextHandle(threadId, opaqueHandle);
        PIN_MEM_ACCESS_INFO *pinMemAccessInfo = &(multiMemAccessInfo->memop[0]);
        uint32_t num = multiMemAccessInfo->numberOfMemops;
        assert(num*sizeof(T)==AccessLen && "VGather : AccessLen is not match for number of memops");
        if(isApprox) {
            uint32_t zeros=0;
            uint64_t map=0;
            //for(UINT32 k=0;k<multiMemAccessInfo->numberOfMemops;++k) { 
            for(uint32_t k=0;k<AccessLen/sizeof(T);++k) {
                //printf("%d\n",k); fflush(stdout);
                if(sizeof(T)==4) assert((!!(ymmMask->dword[k] & (0x1<<31)))==pinMemAccessInfo->maskOn);
                else             assert((!!(ymmMask->qword[k] & (0x1LL<<63)))==pinMemAccessInfo->maskOn);
                if(pinMemAccessInfo->maskOn) { // memop without masked is not accessed
                    //printf("%d %lx\n", pinMemAccessInfo->bytesAccessed, pinMemAccessInfo->memoryAddress); fflush(stdout);
                    assert(sizeof(T)==pinMemAccessInfo->bytesAccessed && "VGather : Size not matched for accessed bytes");
                    uint8_t* bytes = reinterpret_cast<uint8_t*>(pinMemAccessInfo->memoryAddress);
                    // TODO: memoryAddress may be invalid (SIGSEGV when access *((T*)memoryAddress) ) even if maskOn is set to true, which violates the description of pin tool IARG_MULTI_MEMORYACCESS_EA. Need further checking
                    // uint32_t redbyteNum = getRedNum((void*)bytes);
                    bool hasRedundancy = (bytes[sizeof(T)-1]==0);
                    if(hasRedundancy) {
                        if(sizeof(T)==4) { map = (map<<SP_MAP_SIZE) |  UnrolledConjunctionApprox<0,sizeof(T),sizeof(T)>::BodyRedMap(bytes); }
                        else if(sizeof(T)==8) { map = (map<<DP_MAP_SIZE) |  UnrolledConjunctionApprox<0,sizeof(T),sizeof(T)>::BodyRedMap(bytes); }
                        zeros = zeros + UnrolledConjunctionApprox<0,sizeof(T),sizeof(T)>::BodyZeros(bytes);
                    }
                } else {
                    if(sizeof(T)==4) { map = (map<<SP_MAP_SIZE);}
                    else if(sizeof(T)==8) { map = (map<<DP_MAP_SIZE); }
                }
                ++pinMemAccessInfo;
                //printf("Next\n"); fflush(stdout);
            }
            map_g = map;
            zeros_g = zeros;
            //printf("Adding\n"); fflush(stdout);
            //AddToApproximateRedTable((uint64_t)curCtxtHandle, map, AccessLen, zeros, AccessLen/sizeof(T), sizeof(T), threadId);
        } else {
            assert(0 && "VGather should be a floating point operation!");
        }

#ifdef DEBUG_ZEROSPY
        fprintf(debug,"\nINFO : Exit Check NBytes Value After Read\n");
        fflush(debug);
#endif
    }
};

static inline VOID CheckAfterLargeRead(void* addr, UINT32 accessLen, uint32_t opaqueHandle, THREADID threadId){
// #ifdef DEBUG_ZEROSPY
//     fprintf(gDebugFile,"\nINFO : In Check After Large Read\n");
//     if(accessLen > 32) {
//         fprintf(gDebugFile,"ERROR : AccessLen too large : %d\n",accessLen);
//         assert(0 && (accessLen <= 32));
//     }
// #endif
    ContextHandle_t curCtxtHandle = GetContextHandle(threadId, opaqueHandle);
    uint8_t* bytes = static_cast<uint8_t*>(addr);
    // quick check whether the most significant byte of the read memory is redundant zero or not
    bool hasRedundancy = (bytes[accessLen-1]==0);
    if(hasRedundancy) {
        // calculate redmap by binary reduction with bitwise operation
        register uint64_t redbyteMap = 0;
        int restLen = accessLen;
        int index = 0;
        while(restLen>=8) {
            redbyteMap = (redbyteMap<<8) | count_zero_bytemap_int64(bytes+index);
            restLen -= 8;
            index += 8;
        }
        while(restLen>=4) {
            redbyteMap = (redbyteMap<<4) | count_zero_bytemap_int32(bytes+index);
            restLen -= 4;
            index += 4;
        }
        while(restLen>=2) {
            redbyteMap = (redbyteMap<<2) | count_zero_bytemap_int16(bytes+index);
            restLen -= 2;
            index += 2;
        }
        while(restLen>=1) {
            redbyteMap = (redbyteMap<<1) | count_zero_bytemap_int8(bytes+index);
            restLen -= 1;
            index += 1;
        }
        // now redmap is calculated, count for redundancy
        bool counting = true;
        register uint64_t redbytesNum = 0;
        restLen = accessLen;
        while(counting && restLen>=8) {
            restLen -= 8;
            register uint8_t t = BitCountTable256[(redbyteMap>>restLen)&0xff];
            redbytesNum += t;
            if(t!=8) {
                counting = false;
                break;
            }
        }
        while(counting && restLen>=4) {
            restLen -= 4;
            register uint8_t t = BitCountTable16[(redbyteMap>>restLen)&0xf];
            redbytesNum += t;
            if(t!=4) {
                counting = false;
                break;
            }
        }
        while(counting && restLen>=2) {
            restLen -= 2;
            register uint8_t t = BitCountTable4[(redbyteMap>>restLen)&0x3];
            redbytesNum += t;
            if(t!=8) {
                counting = false;
                break;
            }
        }
        // dont check here as this loop must execute only once
        while(counting && restLen>=1) {
            restLen -= 1;
            register uint8_t t = (redbyteMap>>restLen)&0x1;
            redbytesNum += t;
        }
        // report in RedTable
        AddToRedTable((uint64_t)MAKE_CONTEXT_PAIR(accessLen, curCtxtHandle), redbytesNum, redbyteMap, accessLen, threadId);
    }
    else {
        AddToRedTable((uint64_t)MAKE_CONTEXT_PAIR(accessLen, curCtxtHandle), 0, 0, accessLen, threadId);
    }
// #ifdef DEBUG_ZEROSPY
//     fprintf(gDebugFile,"\nINFO : Exit Check After Large Read\n");
// #endif
}

#ifdef ENABLE_SAMPLING

#define HANDLE_CASE(T, ACCESS_LEN, IS_APPROX) \
INS_InsertIfPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)IfEnableSample, IARG_THREAD_ID,IARG_END);\
INS_InsertThenPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) ZeroSpyAnalysis<T, (ACCESS_LEN), (IS_APPROX)>::CheckNByteValueAfterRead, IARG_MEMORYOP_EA, memOp, IARG_UINT32, opaqueHandle, IARG_THREAD_ID, IARG_INST_PTR,IARG_END)

#define HANDLE_LARGE() \
INS_InsertIfPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)IfEnableSample, IARG_THREAD_ID,IARG_END);\
INS_InsertThenPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) CheckAfterLargeRead, IARG_MEMORYOP_EA, memOp, IARG_MEMORYREAD_SIZE, IARG_UINT32, opaqueHandle, IARG_THREAD_ID, IARG_END)
#else

#define HANDLE_CASE(T, ACCESS_LEN, IS_APPROX) \
INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) ZeroSpyAnalysis<T, (ACCESS_LEN), (IS_APPROX)>::CheckNByteValueAfterRead, IARG_ADDRINT, INS_Address(ins), IARG_MEMORYOP_EA, memOp, IARG_UINT32, opaqueHandle, IARG_THREAD_ID, IARG_INST_PTR,IARG_END)

// #define HANDLE_CASE_LOWER(T, LOWERT, ACCESS_LEN) 
// INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) ZeroSpyApproxAnalysis<T, LOWERT, (ACCESS_LEN)>::CheckNByteValueAfterRead, IARG_MEMORYOP_EA, memOp, IARG_UINT32, opaqueHandle, IARG_THREAD_ID, IARG_INST_PTR,IARG_END)

#define HANDLE_LARGE() \
INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) CheckAfterLargeRead, IARG_MEMORYOP_EA, memOp, IARG_MEMORYREAD_SIZE, IARG_UINT32, opaqueHandle, IARG_THREAD_ID, IARG_END)
#endif

#define HANDLE_VGATHER(T, ACCESS_LEN, IS_APPROX, ins, ymmMask) \
INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) ZeroSpyAnalysis<T, (ACCESS_LEN), (IS_APPROX)>::CheckNByteValueAfterVGather, IARG_MULTI_MEMORYACCESS_EA, IARG_REG_REFERENCE, ymmMask, IARG_UINT32, opaqueHandle, IARG_THREAD_ID, IARG_END)

static int GetNumReadOperandsInIns(INS ins, UINT32 & whichOp){
    int numReadOps = 0;
    UINT32 memOperands = INS_MemoryOperandCount(ins);
    for(UINT32 memOp = 0; memOp < memOperands; memOp++) {
        if (INS_MemoryOperandIsRead(ins, memOp)) {
            numReadOps++;
            whichOp = memOp;
        }
    }
    return numReadOps;
}


struct LoadSpyInstrument{
    static __attribute__((always_inline)) void InstrumentReadValueBeforeAndAfterLoading(INS ins, UINT32 memOp, uint32_t opaqueHandle){
// #ifdef DEBUG_ZEROSPY
//         fprintf(gDebugFile,"\nINFO : In InstrumentReadValueBeforeAndAfterLoading\n");
// #endif
        UINT32 refSize = INS_MemoryOperandSize(ins, memOp);
        
        if (IsFloatInstructionAndOkToApproximate(INS_Address(ins))) {
            unsigned int operSize = FloatOperandSize(INS_Address(ins),INS_MemoryOperandIndexToOperandIndex(ins,memOp));
            switch(refSize) {
                case 1:
                case 2: assert(0 && "memory read floating data with unexptected small size");
                case 4: HANDLE_CASE(float, 4, true); break;
                case 8: HANDLE_CASE(double, 8, true); break;
                // case 8: HANDLE_CASE_LOWER(double, float, 8); break;
                case 10: HANDLE_CASE(uint8_t, 10, true); break;
                case 16: {
                    switch (operSize) {
                        case 4: HANDLE_CASE(float, 16, true); break;
                        case 8: HANDLE_CASE(double, 16, true); break;
                        // case 8: HANDLE_CASE_LOWER(double, float, 16); break;
                        default: assert(0 && "handle large mem read with unexpected operand size\n"); break;
                    }
                }break;
                case 32: {
                    switch (operSize) {
                        case 4: HANDLE_CASE(float, 32, true); break;
                        case 8: HANDLE_CASE(double, 32, true); break;
                        // case 8: HANDLE_CASE_LOWER(double, float, 32); break;
                        default: assert(0 && "handle large mem read with unexpected operand size\n"); break;
                    }
                }break;
                default: assert(0 && "unexpected large memory read\n"); break;
            }
        }else{
            switch(refSize) {
#ifdef SKIP_SMALLCASE
                // do nothing when access is small
                case 1: break;
                case 2: break;
#else
                case 1: HANDLE_CASE(uint8_t, 1, false); break;
                case 2: HANDLE_CASE(uint16_t, 2, false); break;
#endif
                case 4: HANDLE_CASE(uint32_t, 4, false); break;
                case 8: HANDLE_CASE(uint64_t, 8, false); break;
                    
                default: {
                    HANDLE_LARGE();
                }
            }
        }
// #ifdef DEBUG_ZEROSPY
//         fprintf(gDebugFile,"\nINFO : Exit InstrumentReadValueBeforeAndAfterLoading\n");
// #endif
    }
    static __attribute__((always_inline)) void InstrumentReadValueBeforeVGather(INS ins, REG ymmMask, uint32_t opaqueHandle){
        UINT32 memOperands = INS_MemoryOperandCount(ins);
        UINT32 operSize = INS_MemoryOperandSize(ins, 0); // VGather's second operand is the memory operand
        UINT32 refSize = operSize*memOperands;
        // unsigned int operSize = FloatOperandSize(INS_Address(ins),INS_MemoryOperandIndexToOperandIndex(ins,0));
        switch(refSize) {
            case 1:
            case 2: 
            case 4: 
            case 8: 
            case 10: assert(0 && "memory read floating data with unexptected small size");
            case 16: {
                switch (operSize) {
                    case 4: HANDLE_VGATHER(float, 16, true, ins, ymmMask); break;
                    case 8: HANDLE_VGATHER(double, 16, true, ins, ymmMask); break;
                    default: assert(0 && "handle large mem read with unexpected operand size\n"); break;
                }
            }break;
            case 32: {
                switch (operSize) {
                    case 4: HANDLE_VGATHER(float, 32, true, ins, ymmMask); break;
                    case 8: HANDLE_VGATHER(double, 32, true, ins, ymmMask); break;
                    default: assert(0 && "handle large mem read with unexpected operand size\n"); break;
                }
            }break;
            default: assert(0 && "unexpected large memory read\n"); break;
        }
    }
};

/*********************  instrument analysis  ************************/

static inline bool INS_IsIgnorable(INS ins){
    if( INS_IsFarJump(ins) || INS_IsDirectFarJump(ins)
#if (PIN_PRODUCT_VERSION_MAJOR >= 3) && (PIN_PRODUCT_VERSION_MINOR >= 7)
       // INS_IsMaskedJump has disappeared in 3,7
#else
       || INS_IsMaskedJump(ins)
#endif
    )
        return true;
    else if(INS_IsRet(ins) || INS_IsIRet(ins))
        return true;
    else if(INS_IsCall(ins) || INS_IsSyscall(ins))
        return true;
    else if(INS_IsBranch(ins) || INS_IsRDTSC(ins) || INS_IsNop(ins))
        return true;
    else if(INS_IsPrefetch(ins)) // Prefetch instructions might access addresses which are invalid.
        return true;
    // XSAVEC and XRSTOR are problematic since its access length is variable. 
    // Execution of XSAVEC is similar to that of XSAVE. XSAVEC differs from XSAVE in that it uses compaction and that it may use the init optimization.
    // It fails with "Cannot use IARG_MEMORYWRITE_SIZE on non-standard memory access of instruction at 0xfoo: xsavec ptr [rsp]" error.
    // A correct solution should use INS_hasKnownMemorySize() which is not available in Pin 2.14.
    if(INS_Mnemonic(ins) == "XSAVEC")
        return true;
    if(INS_Mnemonic(ins) == "XSAVE")
        return true;
    if(INS_Mnemonic(ins) == "XRSTOR")
         return true;
    return false;
}

// static inline bool INS_IsVGather(INS ins){
//     // For vector gather instruction, the address may be given invalid regardless of its mask
//     switch(INS_Opcode(ins)) {
//         case XED_ICLASS_VGATHERDPD:
//         case XED_ICLASS_VGATHERDPS:
//         case XED_ICLASS_VGATHERQPD:
//         case XED_ICLASS_VGATHERQPS:
//         case XED_ICLASS_VPGATHERDD:
//         case XED_ICLASS_VPGATHERDQ:
//         case XED_ICLASS_VPGATHERQD:
//         case XED_ICLASS_VPGATHERQQ:
//         return true;
//     }
//     return false;
// }

static VOID InstrumentInsCallback(INS ins, VOID* v, const uint32_t opaqueHandle) {
// #ifdef DEBUG_ZEROSPY
//     fprintf(gDebugFile,"\nINFO : In InstrumentInsCallback\n");
// #endif
    if (!INS_HasFallThrough(ins)) return;
    if (INS_IsIgnorable(ins))return;
    if (INS_IsBranchOrCall(ins) || INS_IsRet(ins)) return;
    if (INS_IsVgather(ins)) {
        // Do not handle Vgather as Vgather will always fail as memoryAccess given by Pin_Multi_memoryAccess_EA may be invalid (segfault when read even if maskon is set to true)
        // Furthermore, the example tool given by pin also fails (segfault) at this point, so just ignore vgather.
#if 0
        REG ymmMask = INS_OperandReg(ins, 2);
        if (!REG_is_ymm(ymmMask))
        {
            ymmMask = (REG)(ymmMask - REG_XMM0 + REG_YMM0);
        }
        // TODO: position the root cause of abnormal pin args (memoryAccess).
        // HANDLE_GATHER(ymmMask); // VGather need special care
        LoadSpyInstrument::InstrumentReadValueBeforeVGather(ins, ymmMask, opaqueHandle);
#endif
        return ;
    }
    
    //Instrument memory reads to find redundancy
    // Special case, if we have only one read operand
    UINT32 whichOp = 0;
    if(GetNumReadOperandsInIns(ins, whichOp) == 1){
        // Read the value at location before and after the instruction
        LoadSpyInstrument::InstrumentReadValueBeforeAndAfterLoading(ins, whichOp, opaqueHandle);
    }else{
        UINT32 memOperands = INS_MemoryOperandCount(ins);
        for(UINT32 memOp = 0; memOp < memOperands; memOp++) {
            
            if(!INS_MemoryOperandIsRead(ins, memOp))
                continue;
            LoadSpyInstrument::InstrumentReadValueBeforeAndAfterLoading(ins, memOp, opaqueHandle);
        }
    }
// #ifdef DEBUG_ZEROSPY
//     fprintf(gDebugFile,"\nINFO : Exit InstrumentInsCallback\n");
// #endif
}

/**********************************************************************************/

#ifdef ENABLE_SAMPLING
#error Sampling should not be enabled as it has not been tested yet!
inline VOID UpdateAndCheck(uint32_t count, uint32_t bytes, THREADID threadId) {
    
    RedSpyThreadData* const tData = ClientGetTLS(threadId);
    
    if(tData->sampleFlag){
        tData->numIns += count;
        if(tData->numIns > WINDOW_ENABLE){
            tData->sampleFlag = false;
            tData->numIns = 0;
        }
    }else{
        tData->numIns += count;
        if(tData->numIns > WINDOW_DISABLE){
            tData->sampleFlag = true;
            tData->numIns = 0;
        }
    }
    if (tData->sampleFlag) {
        tData->bytesLoad += bytes;
    }
}

inline VOID Update(uint32_t count, uint32_t bytes, THREADID threadId){
    RedSpyThreadData* const tData = ClientGetTLS(threadId);
    tData->numIns += count;
    if (tData->sampleFlag) {
        tData->bytesLoad += bytes;
    }
}

//instrument the trace, count the number of ins in the trace, decide to instrument or not
static void InstrumentTrace(TRACE trace, void* f) {
    bool check = false;
    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl))
    {
        uint32_t totInsInBbl = BBL_NumIns(bbl);
        uint32_t totBytes = 0;
        for(INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins)) {
            
            if (!INS_HasFallThrough(ins)) continue;
            if (INS_IsIgnorable(ins)) continue;
            if (INS_IsBranchOrCall(ins) || INS_IsRet(ins)) continue;
            
            if(INS_IsMemoryRead(ins)) {
                totBytes += INS_MemoryReadSize(ins);
            }
        }
        
        if (BBL_InsTail(bbl) == BBL_InsHead(bbl)) {
            BBL_InsertCall(bbl,IPOINT_BEFORE,(AFUNPTR)UpdateAndCheck,IARG_UINT32, totInsInBbl, IARG_UINT32,totBytes, IARG_THREAD_ID, IARG_CALL_ORDER, CALL_ORDER_FIRST, IARG_END);
        }else if(INS_IsIndirectBranchOrCall(BBL_InsTail(bbl))){
            BBL_InsertCall(bbl,IPOINT_BEFORE,(AFUNPTR)UpdateAndCheck,IARG_UINT32, totInsInBbl, IARG_UINT32,totBytes, IARG_THREAD_ID,IARG_CALL_ORDER, CALL_ORDER_FIRST, IARG_END);
        }else{
            if (check) {
                BBL_InsertCall(bbl,IPOINT_BEFORE,(AFUNPTR)UpdateAndCheck,IARG_UINT32, totInsInBbl, IARG_UINT32, totBytes, IARG_THREAD_ID,IARG_CALL_ORDER, CALL_ORDER_FIRST, IARG_END);
                check = false;
            } else {
                BBL_InsertCall(bbl,IPOINT_BEFORE,(AFUNPTR)Update,IARG_UINT32, totInsInBbl, IARG_UINT32, totBytes, IARG_THREAD_ID, IARG_CALL_ORDER, CALL_ORDER_FIRST, IARG_END);
                check = true;
            }
        }
    }
}

#else

inline VOID Update(uint32_t bytes, THREADID threadId){
// #ifdef DEBUG_ZEROSPY
//     fprintf(gDebugFile,"\nINFO : In Update\n");
// #endif
    RedSpyThreadData* const tData = ClientGetTLS(threadId);
    tData->bytesLoad += bytes;
// #ifdef DEBUG_ZEROSPY
//     fprintf(gDebugFile,"\nINFO : Exit Update\n");
// #endif
}

//instrument the trace, count the number of ins in the trace, decide to instrument or not
static void InstrumentTrace(TRACE trace, void* f) {
// #ifdef DEBUG_ZEROSPY
//     fprintf(gDebugFile,"\nINFO : In InstrumentTrace\n");
// #endif
    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl))
    {
        uint32_t totBytes = 0;
        for(INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins)) {
            
            if (!INS_HasFallThrough(ins)) continue;
            if (INS_IsIgnorable(ins)) continue;
            if (INS_IsBranchOrCall(ins) || INS_IsRet(ins)) continue;
            
            if(INS_IsMemoryRead(ins)) {
                totBytes += INS_MemoryReadSize(ins);
            }
        }
        if(totBytes) {
            BBL_InsertCall(bbl,IPOINT_BEFORE,(AFUNPTR)Update, IARG_UINT32, totBytes, IARG_THREAD_ID, IARG_END);
        }
    }
// #ifdef DEBUG_ZEROSPY
//     fprintf(gDebugFile,"\nINFO : Exit InstrumentTrace\n");
// #endif
}

#endif

struct RedundacyData {
    ContextHandle_t cntxt;
    uint64_t frequency;
    uint64_t all0freq;
    uint64_t ltot;
    uint64_t byteMap;
    uint8_t accessLen;
};

struct ApproxRedundacyData {
    ContextHandle_t cntxt;
    uint64_t all0freq;
    uint64_t ftot;
    uint64_t byteMap;
    uint8_t accessLen;
    uint8_t size;
};

static inline bool RedundacyCompare(const struct RedundacyData &first, const struct RedundacyData &second) {
    return first.frequency > second.frequency ? true : false;
}
static inline bool ApproxRedundacyCompare(const struct ApproxRedundacyData &first, const struct ApproxRedundacyData &second) {
    return first.all0freq > second.all0freq ? true : false;
}
//#define SKIP_SMALLACCESS
#ifdef SKIP_SMALLACCESS
#define LOGGING_THRESHOLD 100
#endif

#include <omp.h>

static void PrintRedundancyPairs(THREADID threadId) {
    printf("%d %lx\n",zeros_g, map_g);
    vector<RedundacyData> tmpList;
    vector<RedundacyData>::iterator tmpIt;
    
    uint64_t grandTotalRedundantBytes = 0;
    uint64_t grandTotalRedundantIns = 0;
    tmpList.reserve(RedMap[threadId].size());
    printf("Dumping INTEGER Redundancy Info... Total num : %ld\n",RedMap[threadId].size());
    fflush(stdout);
    fprintf(gTraceFile, "\n--------------- Dumping INTEGER Redundancy Info ----------------\n");
    fprintf(gTraceFile, "\n*************** Dump Data from Thread %d ****************\n", threadId);
    uint64_t count = 0; uint64_t rep = -1;
#ifdef MERGING
    fprintf(gTraceFile, "\n*************** Merging Redundancy Info, The Caller Prefix Printed is useless ****************\n");
    for (unordered_map<uint64_t, RedLogs>::iterator it = RedMap[threadId].begin(); it != RedMap[threadId].end(); ++it) {
        ++count;
        if(100 * count / RedMap[threadId].size()!=rep) {
            rep = 100 * count / RedMap[threadId].size();
            printf("%ld%%  Finish, current list size = %ld\n",rep, tmpList.size());
            fflush(stdout);
        }
        grandTotalRedundantBytes += (*it).second.red;
#ifdef SKIP_SMALLACCESS
        if((*it).second.tot>LOGGING_THRESHOLD*((*it).second.AccessLen)) continue;
#endif
        ContextHandle_t cur = static_cast<ContextHandle_t>((*it).first);

        if(cur!=0) {
            for(tmpIt = tmpList.begin();tmpIt != tmpList.end(); ++tmpIt){
                if((*tmpIt).cntxt == 0){
                    continue;
                }
                if (!HaveSameCallerPrefix(cur,(*tmpIt).cntxt)) {
                    continue;
                }
                bool ct1 = IsSameSourceLine(cur,(*tmpIt).cntxt);
                if(ct1){
                    (*tmpIt).frequency += (*it).second.red;
                    (*tmpIt).all0freq += (*it).second.fred;
                    (*tmpIt).ltot += (*it).second.tot;
                    (*tmpIt).byteMap &= (*it).second.redByteMap;
                    grandTotalRedundantIns += 1;
                    break;
                }
            }
        }
        if(tmpIt == tmpList.end()){
            RedundacyData tmp = { DECODE_KILL((*it).first), (*it).second.red,(*it).second.fred,(*it).second.tot,(*it).second.redByteMap,DECODE_DEAD((*it).first)};
            tmpList.push_back(tmp);
        }
    }
#else
    for (unordered_map<uint64_t, RedLogs>::iterator it = RedMap[threadId].begin(); it != RedMap[threadId].end(); ++it) {
        ++count;
        if(100 * count / RedMap[threadId].size()!=rep) {
            rep = 100 * count / RedMap[threadId].size();
            printf("%ld%%  Finish\n",rep);
            fflush(stdout);
        }
        RedundacyData tmp = { DECODE_KILL((*it).first), (*it).second.red,(*it).second.fred,(*it).second.tot,(*it).second.redByteMap,DECODE_DEAD((*it).first)};
        tmpList.push_back(tmp);
        grandTotalRedundantBytes += tmp.frequency;
    }
#endif
    
    __sync_fetch_and_add(&grandTotBytesRedLoad,grandTotalRedundantBytes);
    printf("Extracted Raw data, now sorting...\n");
    fflush(stdout);
    
    fprintf(gTraceFile, "\n Total redundant bytes = %f %%\n", grandTotalRedundantBytes * 100.0 / threadBytesLoad[threadId].val);//ClientGetTLS(threadId)->bytesLoad);
    fprintf(gTraceFile, "\n INFO : Total redundant bytes = %f %% (%ld / %ld) \n", grandTotalRedundantBytes * 100.0 / threadBytesLoad[threadId].val, grandTotalRedundantBytes, threadBytesLoad[threadId].val);
    
#ifdef ENABLE_FILTER_BEFORE_SORT
#define FILTER_THESHOLD 1000
    fprintf(gTraceFile, "\n Filter out small redundancies according to the predefined threshold: %.2lf %%\n", 100.0/(double)FILTER_THESHOLD);
    // ApproxRedMap[threadId].clear();
    vector<RedundacyData> tmpList2;
    tmpList2 = std::move(tmpList);
    tmpList.clear(); // make sure it is empty
    tmpList.reserve(tmpList2.size());
    for(tmpIt = tmpList2.begin();tmpIt != tmpList2.end(); ++tmpIt) {
        if(tmpIt->frequency * FILTER_THESHOLD > tmpIt->ltot) {
            tmpList.push_back(*tmpIt);
        }
    }
    fprintf(gTraceFile, " Remained Redundancies: %ld (%.2lf %%)\n", tmpList.size(), (double)tmpList.size()/(double)tmpList2.size());
    tmpList2.clear();
#undef FILTER_THRESHOLD
#endif

    sort(tmpList.begin(), tmpList.end(), RedundacyCompare);
    printf("Sorted, Now generating reports...\n");
    fflush(stdout);
    int cntxtNum = 0;
    for (vector<RedundacyData>::iterator listIt = tmpList.begin(); listIt != tmpList.end(); ++listIt) {
        if (cntxtNum < MAX_REDUNDANT_CONTEXTS_TO_LOG) {
            fprintf(gTraceFile, "\n\n======= (%f) %% of total Redundant, with local redundant %f %% (%ld Bytes / %ld Bytes) ======\n", 
                (*listIt).frequency * 100.0 / grandTotalRedundantBytes,
                (*listIt).frequency * 100.0 / (*listIt).ltot,
                (*listIt).frequency,(*listIt).ltot);
            fprintf(gTraceFile, "\n\n======= with All Zero Redundant %f %% (%ld / %ld) ======\n", 
                (*listIt).all0freq * (*listIt).accessLen * 100.0 / (*listIt).ltot,
                (*listIt).all0freq,(*listIt).ltot/(*listIt).accessLen);
            fprintf(gTraceFile, "\n======= Redundant byte map : [0] ");
            for(uint32_t i=0;i<(*listIt).accessLen;++i) {
                if((*listIt).byteMap & (1<<i)) {
                    fprintf(gTraceFile, "00 ");
                }
                else {
                    fprintf(gTraceFile, "XX ");
                }
            }
            fprintf(gTraceFile, " [AccessLen=%d] =======\n", (*listIt).accessLen);
            fprintf(gTraceFile, "\n---------------------Redundant load with---------------------------\n");
            PrintFullCallingContext((*listIt).cntxt);
        }
        else {
            break;
        }
        cntxtNum++;
    }
    fprintf(gTraceFile, "\n------------ Dumping INTEGER Redundancy Info Finish -------------\n");
    printf("INTEGER Report dumped\n");
    fflush(stdout);
}

static void PrintApproximationRedundancyPairs(THREADID threadId) {
    vector<ApproxRedundacyData> tmpList;
    vector<ApproxRedundacyData>::iterator tmpIt;
    tmpList.reserve(ApproxRedMap[threadId].size());
    
    uint64_t grandTotalRedundantBytes = 0;
    uint64_t grandTotalRedundantIns = 0;
    fprintf(gTraceFile, "\n--------------- Dumping Approximation Redundancy Info ----------------\n");
    fprintf(gTraceFile, "\n*************** Dump Data(delta=%.2f%%) from Thread %d ****************\n", delta*100,threadId);
#ifdef MERGING
    fprintf(gTraceFile, "\n*************** Merging Approximation Redundancy Info, The Caller Prefix Printed is useless ****************\n");
    for (unordered_map<uint64_t, ApproxRedLogs>::iterator it = ApproxRedMap[threadId].begin(); it != ApproxRedMap[threadId].end(); ++it) {
        grandTotalRedundantBytes += (*it).second.fred * (*it).second.AccessLen;
#ifdef SKIP_SMALLACCESS
        // only merging logs access more than LOGGING_THRESHOLD times
        if((*it).second.ftot>LOGGING_THRESHOLD) continue;
#endif
        ContextHandle_t cur = static_cast<ContextHandle_t>((*it).first);
        
        for(tmpIt = tmpList.begin();tmpIt != tmpList.end(); ++tmpIt){
            if(cur == 0 || ((*tmpIt).cntxt) == 0){
                continue;
            }
            if (!HaveSameCallerPrefix(cur,(*tmpIt).cntxt)) {
                continue;
            }
            bool ct1 = IsSameSourceLine(cur,(*tmpIt).cntxt);
            if(ct1){
                (*tmpIt).all0freq += (*it).second.fred;
                (*tmpIt).ftot += (*it).second.ftot;
                (*tmpIt).byteMap &= (*it).second.redByteMap;
                grandTotalRedundantIns += 1;
                break;
            }
        }
        if(tmpIt == tmpList.end()){
            ApproxRedundacyData tmp = { static_cast<ContextHandle_t>((*it).first), (*it).second.fred,(*it).second.ftot, (*it).second.redByteMap,(*it).second.AccessLen, (*it).second.size};
            tmpList.push_back(tmp);
        }
    }
#else
    for (unordered_map<uint64_t, ApproxRedLogs>::iterator it = ApproxRedMap[threadId].begin(); it != ApproxRedMap[threadId].end(); ++it) {
        ApproxRedundacyData tmp = { static_cast<ContextHandle_t>((*it).first), (*it).second.fred,(*it).second.ftot, (*it).second.redByteMap,(*it).second.AccessLen, (*it).second.size};
        tmpList.push_back(tmp);
        grandTotalRedundantBytes += (*it).second.fred * (*it).second.AccessLen;
    }
#endif
    
    __sync_fetch_and_add(&grandTotBytesApproxRedLoad,grandTotalRedundantBytes);
    
    // fprintf(gTraceFile, "\n Total redundant bytes = %f %%\n", grandTotalRedundantBytes * 100.0 / ClientGetTLS(threadId)->bytesLoad);
    // fprintf(gTraceFile, "\n INFO : Total redundant bytes = %f %% (%ld / %ld) \n", grandTotalRedundantBytes * 100.0 / ClientGetTLS(threadId)->bytesLoad, grandTotalRedundantBytes, ClientGetTLS(threadId)->bytesLoad);
    fprintf(gTraceFile, "\n Total redundant bytes = %f %%\n", grandTotalRedundantBytes * 100.0 / threadBytesLoad[threadId].val);
    fprintf(gTraceFile, "\n INFO : Total redundant bytes = %f %% (%ld / %ld) \n", grandTotalRedundantBytes * 100.0 / threadBytesLoad[threadId].val, grandTotalRedundantBytes, threadBytesLoad[threadId].val);
    
#ifdef ENABLE_FILTER_BEFORE_SORT
#define FILTER_THESHOLD 1000
    fprintf(gTraceFile, "\n Filter out small redundancies according to the predefined threshold: %.2lf %%\n", 100.0/(double)FILTER_THESHOLD);
    // ApproxRedMap[threadId].clear();
    vector<ApproxRedundacyData> tmpList2;
    tmpList2 = std::move(tmpList);
    tmpList.clear(); // make sure it is empty
    tmpList.reserve(tmpList2.size());
    for(tmpIt = tmpList2.begin();tmpIt != tmpList2.end(); ++tmpIt) {
        if(tmpIt->all0freq * FILTER_THESHOLD > tmpIt->ftot) {
            tmpList.push_back(*tmpIt);
        }
    }
    fprintf(gTraceFile, " Remained Redundancies: %ld (%.2lf %%)\n", tmpList.size(), (double)tmpList.size()/(double)tmpList2.size());
    tmpList2.clear();
#undef FILTER_THRESHOLD
#endif

    sort(tmpList.begin(), tmpList.end(), ApproxRedundacyCompare);
    int cntxtNum = 0;
    for (vector<ApproxRedundacyData>::iterator listIt = tmpList.begin(); listIt != tmpList.end(); ++listIt) {
        if (cntxtNum < MAX_REDUNDANT_CONTEXTS_TO_LOG) {
            fprintf(gTraceFile, "\n======= (%f) %% of total Redundant, with local redundant %f %% (%ld Zeros / %ld Reads) ======\n",
                (*listIt).all0freq * 100.0 / grandTotalRedundantBytes,
                (*listIt).all0freq * 100.0 / (*listIt).ftot,
                (*listIt).all0freq,(*listIt).ftot); 
            fprintf(gTraceFile, "\n======= Redundant byte map : [mantiss | exponent | sign] ========\n");
            if((*listIt).size==4) {
                fprintf(gTraceFile, "%s", getFpRedMapString_SP((*listIt).byteMap, (*listIt).accessLen/4).c_str());
            } else {
                fprintf(gTraceFile, "%s", getFpRedMapString_DP((*listIt).byteMap, (*listIt).accessLen/8).c_str());
            }
            fprintf(gTraceFile, "\n===== [AccessLen=%d, typesize=%d] =======\n", (*listIt).accessLen, (*listIt).size);
            fprintf(gTraceFile, "\n---------------------Redundant load with---------------------------\n");
            //PrintFullCallingContext((*listIt).kill);
            PrintFullCallingContext((*listIt).cntxt);
        }
        else {
            break;
        }
        cntxtNum++;
    }
    fprintf(gTraceFile, "\n------------ Dumping Approximation Redundancy Info Finish -------------\n");
}

static void updateThreadByteLoad(THREADID threadid, uint64_t val) {
    threadBytesLoad[threadid].val = val;
}
static uint64_t getThreadByteLoad(THREADID threadId) {
    register uint64_t x = threadBytesLoad[threadId].val;
    for (unordered_map<uint64_t, RedLogs>::iterator it = RedMap[threadId].begin(); it != RedMap[threadId].end(); ++it) {
        x += (*it).second.tot;
    }
    for (unordered_map<uint64_t, ApproxRedLogs>::iterator it = ApproxRedMap[threadId].begin(); it != ApproxRedMap[threadId].end(); ++it) {
        x += (*it).second.ftot * (*it).second.AccessLen;
    }
    return x;
}

// On each Unload of a loaded image, the accummulated redundancy information is dumped
static VOID ImageUnload(IMG img, VOID* v) {
    printf("\nImage %s Unloading...\n", IMG_Name(img).c_str());
    fflush(stdout);
    fprintf(gTraceFile, "\n TODO .. Multi-threading is not well supported.");
    THREADID  threadid =  PIN_ThreadId();
    fprintf(gTraceFile, "\nUnloading %s", IMG_Name(img).c_str());
    if (RedMap[threadid].empty() && ApproxRedMap[threadid].empty()) return;
    updateThreadByteLoad(threadid, getThreadByteLoad(threadid));
    printf("Now locking client for updating...\n");
    fflush(stdout);
    // Update gTotalInstCount first
    PIN_LockClient();
    printf("Client locked, now generate report\n");
    fflush(stdout);
    PrintRedundancyPairs(threadid);
    printf("Generate Floating point report\n");
    fflush(stdout);
    PrintApproximationRedundancyPairs(threadid);
    printf("all generated\n");
    fflush(stdout);
    PIN_UnlockClient();
    printf("Unlocked\n");
    fflush(stdout);
    // clear redmap now
    RedMap[threadid].clear();
    ApproxRedMap[threadid].clear();
    printf("...Finish\n");
    fflush(stdout);
}

static VOID ThreadFiniFunc(THREADID threadid, const CONTEXT *ctxt, INT32 code, VOID *v) {
    // __sync_fetch_and_add(&grandTotBytesLoad, ClientGetTLS(threadid)->bytesLoad);
    __sync_fetch_and_add(&grandTotBytesLoad, getThreadByteLoad(threadid));
}

static VOID FiniFunc(INT32 code, VOID *v) {
    // do whatever you want to the full CCT with footpirnt
    uint64_t redReadTmp = 0;
    uint64_t approxRedReadTmp = 0;
    for(int i = 0; i < THREAD_MAX; ++i){
        unordered_map<uint64_t, RedLogs>::iterator it;
        if(!RedMap[i].empty()){
            for (it = RedMap[i].begin(); it != RedMap[i].end(); ++it) {
                redReadTmp += (*it).second.red;
            }
        }
        unordered_map<uint64_t, ApproxRedLogs>::iterator ait;
        if(!ApproxRedMap[i].empty()){
            for (ait = ApproxRedMap[i].begin(); ait != ApproxRedMap[i].end(); ++ait) {
                approxRedReadTmp += (*ait).second.fred;
            }
        }
    }
    grandTotBytesRedLoad += redReadTmp;
    grandTotBytesApproxRedLoad += approxRedReadTmp;
    
    fprintf(gTraceFile, "\n#Redundant Read:");
    fprintf(gTraceFile, "\nTotalBytesLoad: %lu \n",grandTotBytesLoad);
    fprintf(gTraceFile, "\nRedundantBytesLoad: %lu %.2f\n",grandTotBytesRedLoad, grandTotBytesRedLoad * 100.0/grandTotBytesLoad);
    fprintf(gTraceFile, "\nApproxRedundantBytesLoad: %lu %.2f\n",grandTotBytesApproxRedLoad, grandTotBytesApproxRedLoad * 100.0/grandTotBytesLoad);
}

static void InitThreadData(RedSpyThreadData* tdata){
    tdata->bytesLoad = 0;
/*    for (int i = 0; i < THREAD_MAX; ++i) {
        RedMap[i].set_empty_key(0);
        ApproxRedMap[i].set_empty_key(0);
    }
*/
}

static VOID ThreadStart(THREADID threadid, CONTEXT* ctxt, INT32 flags, VOID* v) {
    RedSpyThreadData* tdata = (RedSpyThreadData*)memalign(32,sizeof(RedSpyThreadData));
    InitThreadData(tdata);
    RedMap[threadid].rehash(10000000);
    ApproxRedMap[threadid].rehash(10000000);
    //    __sync_fetch_and_add(&gClientNumThreads, 1);
#ifdef MULTI_THREADED
    PIN_SetThreadData(client_tls_key, tdata, threadid);
#else
    gSingleThreadedTData = tdata;
#endif
}

// user-defined function for metric computation
// hpcviewer can only show the numbers for the metric
uint64_t computeMetricVal(void *metric)
{
    if (!metric) return 0;
    return (uint64_t)metric;
}

int main(int argc, char* argv[]) {
    // Initialize PIN
    if(PIN_Init(argc, argv))
        return Usage();
    
    // Initialize Symbols, we need them to report functions and lines
    PIN_InitSymbols();
    
    // Init Client
    ClientInit(argc, argv);
    // Intialize CCTLib
#ifdef NO_CRT
    // use flat profile to avoid out of memory caused by too deep cct
    PinCCTLibInit(INTERESTING_INS_ALL, gTraceFile, InstrumentInsCallback, 0, 0, KnobFlatProfile);
#else
    PinCCTLibInit(INTERESTING_INS_ALL, gTraceFile, InstrumentInsCallback, 0);
#endif
    // Obtain  a key for TLS storage.
    client_tls_key = PIN_CreateThreadDataKey(0 /*TODO have a destructir*/);
    // Register ThreadStart to be called when a thread starts.
    PIN_AddThreadStartFunction(ThreadStart, 0);
    
    
    // fini function for post-mortem analysis
    PIN_AddThreadFiniFunction(ThreadFiniFunc, 0);
    PIN_AddFiniFunction(FiniFunc, 0);
    
    // TRACE_AddInstrumentFunction(InstrumentTrace, 0);
    
    // Register ImageUnload to be called when an image is unloaded
    IMG_AddUnloadFunction(ImageUnload, 0);
    
    // Launch program now
    PIN_StartProgram();
    return 0;
}

