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
// #ifdef NDEBUG
// #undef NDEBUG
// #endif
#include <assert.h>
#include <string.h>
#include <sys/mman.h>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <list>
#include "pin.H"

#define MULTI_THREADED

//#define SKIP_SMALL_VARS
#ifdef SKIP_SMALL_VARS
#define SMALL_VAR_THRESHOLD 16
#endif

//enable Data-centric
#define USE_TREE_BASED_FOR_DATA_CENTRIC
#define USE_TREE_WITH_ADDR
//#define USE_SHADOW_FOR_DATA_CENTRIC
//#define USE_ADDR_RANGE
#include "cctlib.H"

#define OBJTYPE2STRING(t) ((t==DYNAMIC_OBJECT)?"DYNAMIC":((t==STATIC_OBJECT)?"STATIC":((t==STACK_OBJECT)?"STACK":"UNKNOWN")))
#define SYMNAME2STRING(t,s) ((t==DYNAMIC_OBJECT)?"DYNAMIC":((t==STATIC_OBJECT)?GetStringFromStringPool(s):((t==STACK_OBJECT)?"STACK":"UNKNOWN")))

#include <xmmintrin.h>
#include <immintrin.h>

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
//#define MAX_REDUNDANT_CONTEXTS_TO_LOG (1000)
#define MAX_OBJS_TO_LOG 100
#define MAX_REDUNDANT_CONTEXTS_PER_OBJ_TO_LOG 10
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


#ifdef ENABLE_SAMPLING

#define WINDOW_ENABLE 1000000
#define WINDOW_DISABLE 100000000
#define WINDOW_CLEAN 10
#endif

#define MAKE_OBJID(a, b) (((uint64_t)(a)<<32) | (b))
#define DECODE_TYPE(a) (((uint64_t)(a)&(0xffffffffffffffff))>>32)
#define DECODE_NAME(b) ((uint64_t)(b)&(0x00000000ffffffff))

#define MAKE_CNTXT(a, b, c) (((uint64_t)(a)<<32) | ((uint64_t)(b)<<16) | (uint64_t)(c))
#define DECODE_CNTXT(a) (static_cast<ContextHandle_t>((((a)&(0xffffffffffffffff))>>32)))
#define DECODE_ACCLN(b) (((uint64_t)(b)&(0x00000000ffff0000))>>16)
#define DECODE_TYPSZ(c)  ((uint64_t)(c)&(0x000000000000ffff))

#define delta 0.01

#define CACHE_LINE_SIZE (64)
#ifndef PAGE_SIZE
#define PAGE_SIZE (4*1024)
#endif

#ifdef NO_CRT
KNOB<BOOL>   KnobFlatProfile(KNOB_MODE_WRITEONCE, "pintool", "fp", "0", "Collect flat profile");
#endif

#define static_assert(x) static_assert(x,#x)

/************************************************/
/****************** Bit Vector ******************/
struct bitvec_t { 
    union {
        uint64_t stat; // static 64 bits small cases
        uint64_t* dyn; // dynamic allocate memory for bitvec larger than 64
    } data;
    size_t size;
    size_t capacity;
};
typedef bitvec_t* bitref_t;
inline void bitvec_alloc(bitref_t bitref, size_t size) {
    bitref->size = size;
    if(size>64) {
        bitref->capacity = (size+63)/64;
        assert(bitref->capacity > 0);
        // Only Dynamic Malloc for large cases (>64 Bytes)
        // TODO: USE memaligned malloc
        bitref->data.dyn = (uint64_t*)malloc(sizeof(uint64_t)*(size+63)/64);
        assert(bitref->data.dyn!=NULL);
        // TODO: may be slow, use avx
        memset(bitref->data.dyn, -1, sizeof(uint64_t)*(size+63)/64);
    } else {
        bitref->capacity = 1;
        bitref->data.stat = -1; // 0xffffffffffffffffLL;
    }
}

inline void bitvec_free(bitref_t bitref) {
    if(bitref->size>64) {
        free(bitref->data.dyn);
    }
}

inline void bitvec_and(bitref_t bitref, uint64_t val, size_t offset, size_t size) {
    if(bitref->size>64) {
        // if(offset+size>bitref->size) {
        //     printf("%ld %ld %ld %ld %p %lx\n", bitref->size, bitref->capacity, offset, size, bitref->data.dyn, val); fflush(stdout);
        // }
        // assert(offset+size<=bitref->size);
        size_t bytePos = offset / 64;
        size_t bitPos = offset % 64;
        size_t rest = 64-bitPos;
        assert(bytePos<bitref->capacity);
        if(rest<size) {
            assert(bytePos+1<bitref->capacity);
            register uint64_t mask = (0x1LL << (size-rest)) - 1;
            mask = ~mask;
            bitref->data.dyn[bytePos+1] &= ((val>>rest)|mask);
            size = rest;
        }
        register uint64_t mask = (0x1LL << size) - 1;
        mask = mask << bitPos;
        mask = ~mask;
        bitref->data.dyn[bytePos] &= ((val<<bitPos)|mask);
    } else {
        assert(offset<64);
        register uint64_t mask = (0x1LL << size) - 1;
        mask = mask << offset;
        mask = ~mask;
        bitref->data.stat &= ((val<<offset)|mask);
    }
}

inline bool bitvec_at(bitref_t bitref, size_t pos) {
    if(bitref->size>64) {
        size_t bytePos = pos / 64;
        size_t bitPos = pos % 64;
        assert(bytePos<bitref->capacity);
        if(bitPos!=0) {
            return (bitref->data.dyn[bytePos] & (0x1LL << bitPos))!=0 ? true : false;
        } else {
            return (bitref->data.dyn[bytePos] & (0x1LL))!=0 ? true : false;
        }
    } else {
        return (bitref->data.stat & (0x1LL << pos))!=0 ? true : false;
    }
}
/************************************************/

/***********************************************
 ******  shadow memory
 ************************************************/
//ConcurrentShadowMemory<uint8_t, DataHandle_t> sm;

struct{
    char dummy1[128];
    xed_state_t  xedState;
    char dummy2[128];
} LoadSpyGlobals;

////////////////////////////////////////////////

struct RedSpyThreadData{
    
    uint64_t bytesLoad;
    
    long long numIns;
    bool sampleFlag;
};

// for metric logging
int redload_metric_id = 0;
int redload_approx_metric_id = 0;

//for statistics result
uint64_t grandTotBytesLoad;
uint64_t grandTotBytesRedLoad;
uint64_t grandTotBytesApproxRedLoad;

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

// Initialized the needed data structures before launching the target program
static void ClientInit(int argc, char* argv[]) {
    // Create output file
    char name[MAX_FILE_PATH] = "zeroLoad.dataCentric.out.";
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

// #define DATA_STATE_NOT_VISIT 0
// #define DATA_STATE_ONLY_ZERO 1
// #define DATA_STATE_NOT_ZERO  2

struct RedLogs{
    uint64_t red;  // how many byte zero
    bitvec_t redmap; // bitmap logging if a byte is redundant
    bitvec_t accmap; // bitmap logging if a byte is accessed
};

static unordered_map<uint64_t, unordered_map<uint64_t, RedLogs> > RedMap[THREAD_MAX];
static unordered_map<uint64_t, unordered_map<uint64_t, RedLogs> > ApproxRedMap[THREAD_MAX];

static inline void AddToRedTable(uint64_t addr, DataHandle_t data, uint16_t value, uint16_t total, uint32_t redmap, THREADID threadId) __attribute__((always_inline,flatten));
static inline void AddToRedTable(uint64_t addr, DataHandle_t data, uint16_t value, uint16_t total, uint32_t redmap, THREADID threadId) {
    assert(addr<=data.end_addr);
    size_t offset = addr-data.beg_addr;
    uint64_t key = MAKE_OBJID(data.objectType,data.symName);
    unordered_map<uint64_t, unordered_map<uint64_t, RedLogs> >::iterator it2 = RedMap[threadId].find(key);
    unordered_map<uint64_t, RedLogs>::iterator it;
    size_t size = data.end_addr - data.beg_addr;
    if ( it2  == RedMap[threadId].end() || (it = it2->second.find(size)) == it2->second.end()) {
        RedLogs log;
        log.red = value;
        bitvec_alloc(&log.redmap, size);
        bitvec_and(&log.redmap, redmap, offset, total);
        bitvec_alloc(&log.accmap, size);
        bitvec_and(&log.accmap, 0, offset, total);
        RedMap[threadId][key][size] = log;
    } else {
        assert(it->second.redmap.size==it->second.accmap.size);
        assert(size == it->second.redmap.size);
        it->second.red += value;
        bitvec_and(&(it->second.redmap), redmap, offset, total);
        bitvec_and(&(it->second.accmap), 0, offset, total);
    }
}

static inline void AddToApproximateRedTable(uint64_t addr, DataHandle_t data, uint16_t value, uint16_t total, uint64_t redmap, uint32_t typesz, THREADID threadId) __attribute__((always_inline,flatten));
static inline void AddToApproximateRedTable(uint64_t addr, DataHandle_t data, uint16_t value, uint16_t total, uint64_t redmap, uint32_t typesz, THREADID threadId) {
    assert(addr<=data.end_addr);
    assert((addr-data.beg_addr)%typesz==0);
    size_t offset = (addr-data.beg_addr)/typesz;
    uint64_t key = MAKE_OBJID(data.objectType,data.symName);
    unordered_map<uint64_t, unordered_map<uint64_t, RedLogs> >::iterator it2 = ApproxRedMap[threadId].find(key);
    unordered_map<uint64_t, RedLogs>::iterator it;
    // the data size may not aligned with typesz, so use upper bound as the bitvec size
    // Note: not aligned case : struct/class with floating and int.
    size_t size = (data.end_addr - data.beg_addr+(typesz-1))/typesz;
    if(value > total) {
        cerr << "** Warning AddToApproximateTable : value " << value << ", total " << total << " **" << endl;
        assert(0 && "** BUG #0 Detected. Existing **");
    }
    if ( it2  == ApproxRedMap[threadId].end() || (it = it2->second.find(size)) == it2->second.end()) {
        RedLogs log;
        log.red = value;
        bitvec_alloc(&log.redmap, size);
        bitvec_and(&log.redmap, redmap, offset, total);
        bitvec_alloc(&log.accmap, size);
        bitvec_and(&log.accmap, 0, offset, total);
        ApproxRedMap[threadId][key][size] = log;
    } else {
        assert(it->second.redmap.size==it->second.accmap.size);
        assert(size == it->second.redmap.size);
        it->second.red += value;
        bitvec_and(&(it->second.redmap), redmap, offset, total);
        bitvec_and(&(it->second.accmap), 0, offset, total);
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
    xed_decoded_inst_t  xedd;
    xed_decoded_inst_zero_set_mode(&xedd, &LoadSpyGlobals.xedState);
    
    if(XED_ERROR_NONE == xed_decode(&xedd, (const xed_uint8_t*)(ip), 15)) {
        xed_category_enum_t cat = xed_decoded_inst_get_category(&xedd);
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
    #error Known Byte Order
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
            return ((*(reinterpret_cast<float_cast*>(&addr[start]))).vars.value==0) | (UnrolledConjunctionApprox<start+incr,end,incr>::BodyRedMap(addr)<<1);
        else if(incr==8)
            return ((*(reinterpret_cast<double_cast*>(&addr[start]))).vars.value==0) | (UnrolledConjunctionApprox<start+incr,end,incr>::BodyRedMap(addr)<<1);
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

template<class T, uint32_t AccessLen, bool isApprox>
struct ZeroSpyAnalysis{
    static __attribute__((always_inline)) VOID CheckNByteValueAfterRead(void* addr, uint32_t opaqueHandle, THREADID threadId){
#ifdef DEBUG_ZEROSPY_SPATIAL
        printf("\n In CheckNByteValueAfterRead Begin : %p %d %d\n", addr,opaqueHandle, threadId);
#endif
        DataHandle_t curDataHandle = GetDataObjectHandle(addr, threadId);
        if(curDataHandle.objectType!=DYNAMIC_OBJECT && curDataHandle.objectType!=STATIC_OBJECT) {
            return;
        }
#ifdef SKIP_SMALL_VARS
        // if it is a small var, skip logging
        if(curDataHandle.end_addr-curDataHandle.beg_addr<=SMALL_VAR_THRESHOLD) return;
#endif
        uint8_t* bytes = static_cast<uint8_t*>(addr);
        if(isApprox) {
            assert(((uint64_t)addr-curDataHandle.beg_addr)%sizeof(T)==0);
            uint32_t zeros = UnrolledConjunctionApprox<0,AccessLen,sizeof(T)>::BodyZeros(bytes);
            if(zeros) {
                uint64_t map = UnrolledConjunctionApprox<0,AccessLen,sizeof(T)>::BodyRedMap(bytes);
                AddToApproximateRedTable((uint64_t)addr,curDataHandle,zeros,AccessLen/sizeof(T),map,sizeof(T),threadId);
            } else {
                AddToApproximateRedTable((uint64_t)addr,curDataHandle,0,AccessLen/sizeof(T),0,sizeof(T),threadId);
            }
        } else {
            // uint32_t redbyteNum = getRedNum(addr);
            bool hasRedundancy = UnrolledConjunction<0,AccessLen,sizeof(T)>::BodyHasRedundancy(bytes);
            if(hasRedundancy) {
                uint64_t redbyteMap = UnrolledConjunction<0,AccessLen,sizeof(T)>::BodyRedMap(bytes);
                uint32_t redbyteNum = UnrolledConjunction<0,AccessLen,sizeof(T)>::BodyRedNum(redbyteMap);
                AddToRedTable((uint64_t)addr,curDataHandle,redbyteNum,AccessLen,redbyteMap,threadId);
            } else {
                AddToRedTable((uint64_t)addr,curDataHandle,0,AccessLen,0,threadId);
            }
        }
#ifdef DEBUG_ZEROSPY_SPATIAL
        printf("\n In CheckNByteValueAfterRead Finish \n");
#endif
    }
    static __attribute__((always_inline)) VOID CheckNByteValueAfterVGather(ADDRINT ip, PIN_MULTI_MEM_ACCESS_INFO* multiMemAccessInfo, uint32_t opaqueHandle, THREADID threadId){
        uint32_t num = multiMemAccessInfo->numberOfMemops;
        assert(num*sizeof(T)==AccessLen && "VGather : AccessLen is not match for number of memops");
        if(isApprox) {
            uint32_t zeros=0;
            uint64_t map=0;
            for(UINT32 k=0;k<multiMemAccessInfo->numberOfMemops;++k) { 
                if(!multiMemAccessInfo->memop[k].maskOn) continue; // memop without masked is not accessed
                assert(sizeof(T)==multiMemAccessInfo->memop[k].bytesAccessed && "VGather : Size not matched for accessed bytes");
                PIN_MEM_ACCESS_INFO memop = multiMemAccessInfo->memop[k];
                T* bytes = reinterpret_cast<T*>(memop.memoryAddress);
                DataHandle_t curDataHandle = GetDataObjectHandle((void*)bytes, threadId);
                if(curDataHandle.objectType!=DYNAMIC_OBJECT && curDataHandle.objectType!=STATIC_OBJECT) {
                    return;
                }
#ifdef SKIP_SMALL_VARS
                // if it is a small var, skip logging
                if(curDataHandle.end_addr-curDataHandle.beg_addr<=SMALL_VAR_THRESHOLD) return;
#endif
                uint64_t val = (bytes[0] == 0) ? 1 : 0;
                assert((memop.memoryAddress-curDataHandle.beg_addr)%sizeof(T)==0);
                AddToApproximateRedTable(memop.memoryAddress,curDataHandle,val,1,val,sizeof(T),threadId);
            }
        } else {
            assert(0 && "VGather should be a floating point operation!");
        }
    }
};


static inline VOID CheckAfterLargeRead(void* addr, UINT32 accessLen, uint32_t opaqueHandle, THREADID threadId){
#ifdef DEBUG_ZEROSPY_SPATIAL
    printf("\n In CheckAfterLargeRead Begin : %p %d %d %d\n", addr,accessLen, opaqueHandle, threadId);    
#endif
    DataHandle_t curDataHandle = GetDataObjectHandle(addr, threadId);
    if(curDataHandle.objectType!=DYNAMIC_OBJECT && curDataHandle.objectType!=STATIC_OBJECT) {
        return;
    }
#ifdef SKIP_SMALL_VARS
    // if it is a small var, skip logging
    if(curDataHandle.end_addr-curDataHandle.beg_addr<=SMALL_VAR_THRESHOLD) return;
#endif
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
        AddToRedTable((uint64_t)addr,curDataHandle,redbytesNum,accessLen,redbyteMap,threadId);
    }
    else {
        AddToRedTable((uint64_t)addr,curDataHandle,0,accessLen,0,threadId);
    }
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
INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) ZeroSpyAnalysis<T, (ACCESS_LEN), (IS_APPROX)>::CheckNByteValueAfterRead, IARG_MEMORYOP_EA, memOp, IARG_UINT32, opaqueHandle, IARG_THREAD_ID, IARG_INST_PTR,IARG_END)

#define HANDLE_LARGE() \
INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) CheckAfterLargeRead, IARG_MEMORYOP_EA, memOp, IARG_MEMORYREAD_SIZE, IARG_UINT32, opaqueHandle, IARG_THREAD_ID, IARG_END)
#endif

#define HANDLE_VGATHER(T, ACCESS_LEN, IS_APPROX, ins) \
INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) ZeroSpyAnalysis<T, (ACCESS_LEN), (IS_APPROX)>::CheckNByteValueAfterVGather, IARG_ADDRINT, INS_Address(ins), IARG_MULTI_MEMORYACCESS_EA, IARG_UINT32, opaqueHandle, IARG_THREAD_ID, IARG_END)

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
        UINT32 refSize = INS_MemoryOperandSize(ins, memOp);
        
        if (IsFloatInstructionAndOkToApproximate(INS_Address(ins))) {
            unsigned int operSize = FloatOperandSize(INS_Address(ins),INS_MemoryOperandIndexToOperandIndex(ins,memOp));
            switch(refSize) {
                case 1:
                case 2: assert(0 && "memory read floating data with unexptected small size");
                case 4: HANDLE_CASE(float, 4, true); break;
                case 8: HANDLE_CASE(double, 8, true); break;
                case 10: HANDLE_CASE(uint8_t, 10, true); break;
                case 16: {
                    switch (operSize) {
                        case 4: HANDLE_CASE(float, 16, true); break;
                        case 8: HANDLE_CASE(double, 16, true); break;
                        default: assert(0 && "handle large mem read with unexpected operand size\n"); break;
                    }
                }break;
                case 32: {
                    switch (operSize) {
                        case 4: HANDLE_CASE(float, 32, true); break;
                        case 8: HANDLE_CASE(double, 32, true); break;
                        default: assert(0 && "handle large mem read with unexpected operand size\n"); break;
                    }
                }break;
                default: assert(0 && "unexpected large memory read\n"); break;
            }
        }else{
            switch(refSize) {
                case 1: HANDLE_CASE(uint8_t, 1, false); break;
                case 2: HANDLE_CASE(uint16_t, 2, false); break;
                case 4: HANDLE_CASE(uint32_t, 4, false); break;
                case 8: HANDLE_CASE(uint64_t, 8, false); break;
                    
                default: {
                    HANDLE_LARGE();
                }
            }
        }
    }
    static __attribute__((always_inline)) void InstrumentReadValueBeforeVGather(INS ins, uint32_t opaqueHandle){
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
                    case 4: HANDLE_VGATHER(float, 16, true, ins); break;
                    case 8: HANDLE_VGATHER(double, 16, true, ins); break;
                    default: assert(0 && "handle large mem read with unexpected operand size\n"); break;
                }
            }break;
            case 32: {
                switch (operSize) {
                    case 4: HANDLE_VGATHER(float, 32, true, ins); break;
                    case 8: HANDLE_VGATHER(double, 32, true, ins); break;
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
    if(INS_Mnemonic(ins) == "XSAVEC")
        return true;
    if(INS_Mnemonic(ins) == "XSAVE")
        return true;
    if(INS_Mnemonic(ins) == "XRSTOR")
         return true;
    switch(INS_Opcode(ins)) {
        case XED_ICLASS_VGATHERDPD:
        case XED_ICLASS_VGATHERDPS:
        case XED_ICLASS_VGATHERQPD:
        case XED_ICLASS_VGATHERQPS:
        case XED_ICLASS_VPGATHERDD:
        case XED_ICLASS_VPGATHERDQ:
        case XED_ICLASS_VPGATHERQD:
        case XED_ICLASS_VPGATHERQQ:
        return true;
    }
    return false;
}

static inline bool INS_IsVGather(INS ins){
    // For vector gather instruction, the address may be given invalid regardless of its mask
    switch(INS_Opcode(ins)) {
        case XED_ICLASS_VGATHERDPD:
        case XED_ICLASS_VGATHERDPS:
        case XED_ICLASS_VGATHERQPD:
        case XED_ICLASS_VGATHERQPS:
        case XED_ICLASS_VPGATHERDD:
        case XED_ICLASS_VPGATHERDQ:
        case XED_ICLASS_VPGATHERQD:
        case XED_ICLASS_VPGATHERQQ:
        return true;
    }
    return false;
}

static VOID InstrumentInsCallback(INS ins, VOID* v, const uint32_t opaqueHandle) {
    if (!INS_HasFallThrough(ins)) return;
    if (INS_IsIgnorable(ins))return;
    if (INS_IsBranchOrCall(ins) || INS_IsRet(ins)) return;
    if (INS_IsVGather(ins)) {
        // Do not handle Vgather as Vgather will always fail as memoryAccess given by Pin_Multi_memoryAccess_EA may be invalid (segfault when read even if maskon is set to true)
        // Furthermore, the example tool given by pin also fails (segfault) at this point, so just ignore vgather.
#if 0
        // VGather need special care
        LoadSpyInstrument::InstrumentReadValueBeforeVGather(ins, opaqueHandle);
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
}

/**********************************************************************************/

#ifdef ENABLE_SAMPLING
#error Shold not use ENABLE_SAMPLING!!!
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
    //printf("\nUpdate Begin\n");
    RedSpyThreadData* const tData = ClientGetTLS(threadId);
    tData->bytesLoad += bytes;
    //printf("\nUpdate Finish\n");
}

//instrument the trace, count the number of ins in the trace, decide to instrument or not
static void InstrumentTrace(TRACE trace, void* f) {
    //printf("\nInstrumentTrace Begin\n");
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
        if(totBytes>0) {
            BBL_InsertCall(bbl,IPOINT_BEFORE,(AFUNPTR)Update, IARG_UINT32, totBytes, IARG_THREAD_ID, IARG_END);
        }
    }
    //printf("\nInstrumentTrace Finish\n");
}

#endif

#ifdef PRINT_MEM_INFO
struct AlignedRedMemoryElement {
    DataHandle_t data;
    size_t beg_index;
    size_t end_index;
};

template<size_t ALIGNMENT>
struct AlignedRedMemory {
    uint64_t index;
    uint64_t frequency;
    uint64_t total;
    list<AlignedRedMemoryElement> info; 
};

struct DRedData {
    DataHandle_t data;
    uint8_t* byteState;
};

static inline bool DRedundacyCompare(const struct DRedData &first, const struct DRedData &second) {
    return first.data.beg_addr < second.data.beg_addr ? true : false;
}

#define LEVEL_1_RED_THRESHOLD 0.90
#define LEVEL_2_RED_THRESHOLD 0.70
#define LEVEL_3_RED_THRESHOLD 0.50

#define ALIGN(x,a) ((x)&(~(a-1)))

template<size_t ALIGNMENT>
static void MemLogByMemIndex(AlignedRedMemory<ALIGNMENT> &memLog, uint64_t &flast, vector<DRedData>::iterator &first, vector<DRedData>::iterator fend, uint64_t curMemIndex) {
    uint64_t lastMemIndex = ALIGN((*first).data.beg_addr+flast,ALIGNMENT);
    while(first!=fend && lastMemIndex==curMemIndex) {
        // first byte must be in this alignment memory
        switch((*first).byteState[flast]) {
            case DATA_STATE_ONLY_ZERO:
                ++memLog.frequency;
            case DATA_STATE_NOT_ZERO:
                ++memLog.total;
                break;
            default:
                break;
        }
        AlignedRedMemoryElement infoElement = {(*first).data, flast, 0};
        uint64_t firstDataSize = (*first).data.end_addr - (*first).data.beg_addr;
        ++flast;
        for(;flast<firstDataSize && lastMemIndex==curMemIndex;++flast) {
            lastMemIndex = ALIGN((*first).data.beg_addr+flast,ALIGNMENT);
            if(lastMemIndex!=curMemIndex) break;
            switch((*first).byteState[flast]) {
                case DATA_STATE_ONLY_ZERO:
                    ++memLog.frequency;
                case DATA_STATE_NOT_ZERO:
                    ++memLog.total;
                    break;
                default:
                    break;
            }
        }
        infoElement.end_index = flast;
        memLog.info.push_back(infoElement);
        if(flast==firstDataSize) {
            ++first; // next
            flast = 0;
        }
        lastMemIndex = ALIGN((*first).data.beg_addr+flast,ALIGNMENT);
    }
}

template<size_t ALIGNMENT>
static void printMemLog(AlignedRedMemory<ALIGNMENT> memLogs) {
#ifdef PRINT_ALL_PAGE_INFO

#endif
}

template<size_t ALIGNMENT>
static void PrintMemoryRedundancy(const char* rname, THREADID threadId) {
    vector<DRedData> tmpList;
    
    uint64_t grandTotalReadBytes = 0;
    uint64_t grandTotalRedundantBytes = 0;
    uint64_t grandTotalRedundant_level1 = 0;
    uint64_t grandTotalRedundant_level2 = 0;
    uint64_t grandTotalRedundant_level3 = 0;
    uint64_t grandTotal = 0;
    float maxrate = 0;
    float minrate = 100;
    fprintf(gTraceFile, "\n--------------- Dumping %s Redundancy Info : ALIGNMENT %ld Bytes ----------------\n",rname, ALIGNMENT);
    fprintf(gTraceFile, "\n*************** Dump Data from Thread %d ****************\n", threadId);
    
    for (unordered_map<uint64_t, RedLogs>::iterator it = RedMap[threadId].begin(); it != RedMap[threadId].end(); ++it) {
        DRedData tmp = { (*it).second.data, (*it).second.byteState};
        tmpList.push_back(tmp);
    }

    for (unordered_map<uint64_t, RedLogs>::iterator it = ApproxRedMap[threadId].begin(); it != ApproxRedMap[threadId].end(); ++it) {
        DRedData tmp = { (*it).second.data, (*it).second.byteState};
        tmpList.push_back(tmp);
    }

    sort(tmpList.begin(),tmpList.end(),DRedundacyCompare);

    uint64_t flast=0;
    vector<DRedData>::iterator listIt = tmpList.begin();
    while(listIt!=tmpList.end()) {
        uint64_t curMemIndex = ALIGN((*listIt).data.beg_addr+flast,ALIGNMENT);
        AlignedRedMemory<ALIGNMENT> memLogs;
        memLogs.index = curMemIndex;
        memLogs.frequency = 0;
        memLogs.total = 0;
        memLogs.info.clear();
        MemLogByMemIndex<ALIGNMENT>(memLogs,flast,listIt,tmpList.end(),curMemIndex);
        printMemLog<ALIGNMENT>(memLogs);

        grandTotal++;
        grandTotalRedundantBytes += memLogs.frequency;
        grandTotalReadBytes += memLogs.total;
        if((float)memLogs.frequency/(float)memLogs.total > LEVEL_1_RED_THRESHOLD) {
            grandTotalRedundant_level1++;
        }
        if((float)memLogs.frequency/(float)memLogs.total > LEVEL_2_RED_THRESHOLD) {
            grandTotalRedundant_level2++;
        }
        if((float)memLogs.frequency/(float)memLogs.total > LEVEL_3_RED_THRESHOLD) {
            grandTotalRedundant_level3++;
        }
        if(maxrate < (float)memLogs.frequency/(float)memLogs.total) {
            maxrate = (float)memLogs.frequency/(float)memLogs.total;
        }
        if(minrate > (float)memLogs.frequency/(float)memLogs.total) {
            minrate = (float)memLogs.frequency/(float)memLogs.total;
        }
    }
    
    //__sync_fetch_and_add(&grandTotBytesRedLoad,grandTotalRedundantBytes);
    
    fprintf(gTraceFile, "\n Total redundant bytes = %f %% (%ld Bytes / %ld Bytes), rate range from [%f, %f] %%\n", grandTotalRedundantBytes * 100.0 / grandTotalReadBytes, grandTotalRedundantBytes, grandTotalReadBytes, minrate*100, maxrate*100);
    
    fprintf(gTraceFile, "\n Total redundant bytes (local redundant rate > %f %%) = %f %% (%ld %ss / %ld %ss)\n", LEVEL_1_RED_THRESHOLD * 100.0, grandTotalRedundant_level1 * 100.0 / grandTotal, grandTotalRedundant_level1, rname, grandTotal, rname);
    fprintf(gTraceFile, "\n Total redundant bytes (local redundant rate > %f %%) = %f %% (%ld %ss / %ld %ss)\n", LEVEL_2_RED_THRESHOLD * 100.0, grandTotalRedundant_level2 * 100.0 / grandTotal, grandTotalRedundant_level2, rname, grandTotal, rname);
    fprintf(gTraceFile, "\n Total redundant bytes (local redundant rate > %f %%) = %f %% (%ld %ss / %ld %ss)\n", LEVEL_3_RED_THRESHOLD * 100.0, grandTotalRedundant_level3 * 100.0 / grandTotal, grandTotalRedundant_level3, rname, grandTotal, rname);

    fprintf(gTraceFile, "\n------------ Dumping %s Redundancy Info Finish -------------\n",rname);
}
#endif

// redundant data for a object
struct ObjRedundancy {
    uint64_t objID;
    uint64_t bytes;
};

static inline bool ObjRedundancyCompare(const struct ObjRedundancy &first, const struct ObjRedundancy &second) {
    return first.bytes > second.bytes ? true : false;
}

static inline void PrintSize(uint64_t size, const char* unit="B") {
    if(size >= (1<<20)) {
        fprintf(gTraceFile, "%lf M%s",(double)size/(double)(1<<20),unit);
    } else if(size >= (1<<10)) {
        fprintf(gTraceFile, "%lf K%s",(double)size/(double)(1<<10),unit);
    } else {
        fprintf(gTraceFile, "%ld %s",size,unit);
    }
}

#define MAX_REDMAP_PRINT_SIZE 128
// only print top 5 redundancy with full redmap to file
#define MAX_PRINT_FULL 5

static void PrintRedundancyPairs(THREADID threadId) {
    vector<ObjRedundancy> tmpList;
    
    uint64_t grandTotalRedundantBytes = 0;
    fprintf(gTraceFile, "\n--------------- Dumping Data Redundancy Info ----------------\n");
    fprintf(gTraceFile, "\n*************** Dump Data from Thread %d ****************\n", threadId);

    int count=0;
    int rep=-1;
    int total = RedMap[threadId].size();
    tmpList.reserve(total);
    for(unordered_map<uint64_t, unordered_map<uint64_t, RedLogs> >::iterator it = RedMap[threadId].begin(); it != RedMap[threadId].end(); ++it) {
        ++count;
        if(100 * count / total!=rep) {
            rep = 100 * count / total;
            printf("Stage 1 : %d%%  Finish\n",rep);
            fflush(stdout);
        }
        ObjRedundancy tmp = {(*it).first, 0};
        for(unordered_map<uint64_t, RedLogs>::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
            tmp.bytes += it2->second.red;
        }
        if(tmp.bytes==0) continue;
        grandTotalRedundantBytes += tmp.bytes;
        tmpList.push_back(tmp); 
    }

    __sync_fetch_and_add(&grandTotBytesRedLoad,grandTotalRedundantBytes);
    fprintf(gTraceFile, "\n Total redundant bytes = %f %%\n", grandTotalRedundantBytes * 100.0 / ClientGetTLS(threadId)->bytesLoad);
#ifndef ZEROSPY_NO_DETAIL
    sort(tmpList.begin(), tmpList.end(), ObjRedundancyCompare);

    int objNum = 0;
    rep = -1;
    total = tmpList.size()<MAX_OBJS_TO_LOG?tmpList.size():MAX_OBJS_TO_LOG;
    for(vector<ObjRedundancy>::iterator listIt = tmpList.begin(); listIt != tmpList.end(); ++listIt) {
        if(objNum++ >= MAX_OBJS_TO_LOG) break;
        if(100 * objNum / total!=rep) {
            rep = 100 * objNum / total;
            printf("Stage 2 : %d%%  Finish\n",rep);
            fflush(stdout);
        }
        if((uint8_t)DECODE_TYPE((*listIt).objID) == DYNAMIC_OBJECT) {
            fprintf(gTraceFile, "\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Dynamic Object: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
            PrintFullCallingContext(DECODE_NAME((*listIt).objID)); // segfault might happen if the shadow memory based data centric is used
        } else  
            fprintf(gTraceFile, "\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Static Object: %s ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", GetStringFromStringPool((uint32_t)DECODE_NAME((*listIt).objID)));

        fprintf(gTraceFile, "\n\n==========================================\n");
        fprintf(gTraceFile, "Redundancy Ratio = %f %% (%ld Bytes)\n", (*listIt).bytes * 100.0 / grandTotalRedundantBytes, (*listIt).bytes);

        for(unordered_map<uint64_t, RedLogs>::iterator it2 = RedMap[threadId][(*listIt).objID].begin(); it2 != RedMap[threadId][(*listIt).objID].end(); ++it2) {
            uint64_t dfreq = 0;
            uint64_t dread = 0;
            uint64_t dsize = it2->first;
            bitref_t accmap = &(it2->second.accmap);
            bitref_t redmap = &(it2->second.redmap);

            assert(accmap->size==dsize);
            assert(accmap->size==redmap->size);

            for(size_t i=0;i<accmap->size;++i) {
                if(!bitvec_at(accmap, i)) {
                    ++dread;
                    if(bitvec_at(redmap, i)) ++dfreq;
                }
            }
                
            fprintf(gTraceFile, "\n\n======= DATA SIZE : ");
            PrintSize(dsize);
            fprintf(gTraceFile, "( Not Accessed Data %f %% (%ld Bytes), Redundant Data %f %% (%ld Bytes) )", 
                    (dsize-dread) * 100.0 / dsize, dsize-dread, 
                    dfreq * 100.0 / dsize, dfreq);

            fprintf(gTraceFile, "\n======= Redundant byte map : [0] ");
            uint32_t num=0;
            for(size_t i=0;i<accmap->size;++i) {
                if(!bitvec_at(accmap, i)) {
                    if(bitvec_at(redmap, i)) {
                        fprintf(gTraceFile, "00 ");
                    } else {
                        fprintf(gTraceFile, "XX ");
                    }
                } else {
                    fprintf(gTraceFile, "?? ");
                }
                ++num;
                if(num>MAX_REDMAP_PRINT_SIZE) {
                    fprintf(gTraceFile, "... ");
                    break;
                }
            }
        }
#if 0
        if(objNum<=MAX_PRINT_FULL) {
            char fn[50] = {};
            sprintf(fn,"%lx.redmap",(*listIt).objID);
            FILE* fp = fopen(fn,"w");
            if((uint8_t)DECODE_TYPE((*listIt).objID) == DYNAMIC_OBJECT) {
                fprintf(fp, "\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Dynamic Object: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
                PrintFullCallingContext(DECODE_NAME((*listIt).objID)); // segfault might happen if the shadow memory based data centric is used
            } else  
                fprintf(fp, "\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Static Object: %s ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n", GetStringFromStringPool((uint32_t)DECODE_NAME((*listIt).objID)));
            for(size_t i=0;i<accmap->size;++i) {
                if(!bitvec_at(accmap, i)) {
                    if(bitvec_at(redmap, i)) {
                        fprintf(fp, "00 ");
                    } else {
                        fprintf(fp, "XX ");
                    }
                } else {
                    fprintf(fp, "?? ");
                }
            }
        }
#endif
    }
#endif
    fprintf(gTraceFile, "\n------------ Dumping Redundancy Info Finish -------------\n");
}

static void PrintApproximationRedundancyPairs(THREADID threadId) {
    vector<ObjRedundancy> tmpList;
    
    uint64_t grandTotalRedundantBytes = 0;
    fprintf(gTraceFile, "\n--------------- Dumping Data Approximation Redundancy Info ----------------\n");
    fprintf(gTraceFile, "\n*************** Dump Data(delta=%.2f%%) from Thread %d ****************\n", delta*100,threadId);

    int count=0;
    int rep=-1;
    int total = ApproxRedMap[threadId].size();
    tmpList.reserve(total);
    for(unordered_map<uint64_t, unordered_map<uint64_t, RedLogs> >::iterator it = ApproxRedMap[threadId].begin(); it != ApproxRedMap[threadId].end(); ++it) {
        ++count;
        if(100 * count / total!=rep) {
            rep = 100 * count / total;
            printf("Stage 1 : %d%%  Finish\n",rep);
            fflush(stdout);
        }
        ObjRedundancy tmp = {(*it).first, 0};
        for(unordered_map<uint64_t, RedLogs>::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
            tmp.bytes += it2->second.red;
        }
        if(tmp.bytes==0) continue;
        grandTotalRedundantBytes += tmp.bytes;
        tmpList.push_back(tmp); 
    }

    __sync_fetch_and_add(&grandTotBytesApproxRedLoad,grandTotalRedundantBytes);

    fprintf(gTraceFile, "\n Total redundant bytes = %f %%\n", grandTotalRedundantBytes * 100.0 / ClientGetTLS(threadId)->bytesLoad);
#ifndef ZEROSPY_NO_DETAIL
    sort(tmpList.begin(), tmpList.end(), ObjRedundancyCompare);

    int objNum = 0;
    vector<uint8_t> state;
    for(vector<ObjRedundancy>::iterator listIt = tmpList.begin(); listIt != tmpList.end(); ++listIt) {
        if(objNum++ >= MAX_OBJS_TO_LOG) break;
        if(100 * objNum / total!=rep) {
            rep = 100 * objNum / total;
            printf("Stage 2 : %d%%  Finish\n",rep);
            fflush(stdout);
        }
        fflush(gTraceFile);
        if((uint8_t)DECODE_TYPE((*listIt).objID) == DYNAMIC_OBJECT) {
            fprintf(gTraceFile, "\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Dynamic Object: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
            PrintFullCallingContext(DECODE_NAME((*listIt).objID)); // segfault might happen if the shadow memory based data centric is used
        } else  
            fprintf(gTraceFile, "\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Static Object: %s ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", GetStringFromStringPool((uint32_t)DECODE_NAME((*listIt).objID)));
        fflush(gTraceFile);
        fprintf(gTraceFile, "\n\n==========================================\n");
        fprintf(gTraceFile, "Redundancy Ratio = %f %% (%ld Bytes)\n", (*listIt).bytes * 100.0 / grandTotalRedundantBytes, (*listIt).bytes);

        for(unordered_map<uint64_t, RedLogs>::iterator it2 = ApproxRedMap[threadId][(*listIt).objID].begin(); it2 != ApproxRedMap[threadId][(*listIt).objID].end(); ++it2) {
            uint64_t dfreq = 0;
            uint64_t dread = 0;
            uint64_t dsize = it2->first;
            bitref_t accmap = &(it2->second.accmap);
            bitref_t redmap = &(it2->second.redmap);

            assert(accmap->size==dsize);
            assert(accmap->size==redmap->size);

            for(size_t i=0;i<accmap->size;++i) {
                if(!bitvec_at(accmap, i)) {
                    ++dread;
                    if(bitvec_at(redmap, i)) ++dfreq;
                }
            }
                
            fprintf(gTraceFile, "\n\n======= DATA SIZE : ");
            PrintSize(dsize, " Elements");
            fprintf(gTraceFile, "( Not Accessed Data %f %% (%ld Reads), Redundant Data %f %% (%ld Reads) )", 
                    (dsize-dread) * 100.0 / dsize, dsize-dread, 
                    dfreq * 100.0 / dsize, dfreq);

            fprintf(gTraceFile, "\n======= Redundant byte map : [0] ");
            uint32_t num=0;
            for(size_t i=0;i<accmap->size;++i) {
                if(!bitvec_at(accmap, i)) {
                    if(bitvec_at(redmap, i)) {
                        fprintf(gTraceFile, "00 ");
                    } else {
                        fprintf(gTraceFile, "XX ");
                    }
                } else {
                    fprintf(gTraceFile, "?? ");
                }
                ++num;
                if(num>MAX_REDMAP_PRINT_SIZE) {
                    fprintf(gTraceFile, "... ");
                    break;
                }
            }
        }
#if 0
        if(objNum<=MAX_PRINT_FULL) {
            char fn[50] = {};
            sprintf(fn,"%lx.redmap",(*listIt).objID);
            FILE* fp = fopen(fn,"w");
            if((uint8_t)DECODE_TYPE((*listIt).objID) == DYNAMIC_OBJECT) {
                fprintf(fp, "\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Dynamic Object: %lx^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",(*listIt).objID);
            } else  
                fprintf(fp, "\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Static Object: %s, %lx ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n", GetStringFromStringPool((uint32_t)DECODE_NAME((*listIt).objID)),(*listIt).objID);
            for(size_t i=0;i<accmap->size;++i) {
                if(!bitvec_at(accmap, i)) {
                    if(bitvec_at(redmap, i)) {
                        fprintf(fp, "00 ");
                    } else {
                        fprintf(fp, "XX ");
                    }
                } else {
                    fprintf(fp, "?? ");
                }
            }
        }
#endif
    }
#endif
    fprintf(gTraceFile, "\n------------ Dumping Approx Redundancy Info Finish -------------\n");
}
// On each Unload of a loaded image, the accummulated redundancy information is dumped
static VOID ImageUnload(IMG img, VOID* v) {
    printf("==== PIN CLIENT ZEROSPY : Unloading %s, now collecting analysis data ===\n",IMG_Name(img).c_str());
    fprintf(gTraceFile, "\n TODO .. Multi-threading is not well supported.");
    THREADID  threadid =  PIN_ThreadId();
    fprintf(gTraceFile, "\nUnloading %s", IMG_Name(img).c_str());
    if (RedMap[threadid].empty() && ApproxRedMap[threadid].empty()) return;
    // Update gTotalInstCount first
    PIN_LockClient();
    printf("==== PIN CLIENT ZEROSPY : Print Redundancy info ... ===\n");
    PrintRedundancyPairs(threadid);
    printf("==== PIN CLIENT ZEROSPY : Print Approximation Redundancy info ... ===\n");
    PrintApproximationRedundancyPairs(threadid);
#ifdef PRINT_MEM_INFO
    printf("==== PIN CLIENT ZEROSPY : Print Cacheline Redundancy info ... ===\n");
    PrintMemoryRedundancy<CACHE_LINE_SIZE>("Cacheline", threadid);
    printf("==== PIN CLIENT ZEROSPY : Print Page Redundancy info ... ===\n");
    PrintMemoryRedundancy<PAGE_SIZE>("Page", threadid);
#endif
    PIN_UnlockClient();
    // clear redmap now
    RedMap[threadid].clear();
    ApproxRedMap[threadid].clear();
}

static VOID ThreadFiniFunc(THREADID threadid, const CONTEXT *ctxt, INT32 code, VOID *v) {
    __sync_fetch_and_add(&grandTotBytesLoad, ClientGetTLS(threadid)->bytesLoad);
}

static VOID FiniFunc(INT32 code, VOID *v) {
    // do whatever you want to the full CCT with footpirnt
    uint64_t redReadTmp = 0;
    uint64_t approxRedReadTmp = 0;
    for(int i = 0; i < THREAD_MAX; ++i) {
        if(!RedMap[i].empty()) {
            for(unordered_map<uint64_t, unordered_map<uint64_t, RedLogs> >::iterator it = RedMap[i].begin(); it != RedMap[i].end(); ++it) {
                for(unordered_map<uint64_t, RedLogs>::iterator it2=it->second.begin(); it2 != it->second.end();++it2) {
                    redReadTmp += (*it2).second.red;
                }
            }
        }
        if(!ApproxRedMap[i].empty()) {
            for(unordered_map<uint64_t, unordered_map<uint64_t, RedLogs> >::iterator it = ApproxRedMap[i].begin(); it != ApproxRedMap[i].end(); ++it) {
                for(unordered_map<uint64_t, RedLogs>::iterator it2=it->second.begin(); it2 != it->second.end();++it2) {
                    approxRedReadTmp += (*it2).second.red;
                }
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
    tdata->sampleFlag = true;
    tdata->numIns = 0;
    fprintf(gTraceFile, "\nInit Thread Data Finish\n");
/*    for (int i = 0; i < THREAD_MAX; ++i) {
        RedMap[i].set_empty_key(0);
        ApproxRedMap[i].set_empty_key(0);
    }
*/
}

static VOID ThreadStart(THREADID threadid, CONTEXT* ctxt, INT32 flags, VOID* v) {
    RedSpyThreadData* tdata = (RedSpyThreadData*)memalign(32,sizeof(RedSpyThreadData));
    InitThreadData(tdata);
    //    __sync_fetch_and_add(&gClientNumThreads, 1);
#ifdef MULTI_THREADED
    PIN_SetThreadData(client_tls_key, tdata, threadid);
#else
    gSingleThreadedTData = tdata;
#endif
    fprintf(gTraceFile, "\nInit ThreadStart Finish\n");
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
    PinCCTLibInit(INTERESTING_INS_ALL, gTraceFile, InstrumentInsCallback, 0, true, KnobFlatProfile);
#else
    PinCCTLibInit(INTERESTING_INS_ALL, gTraceFile, InstrumentInsCallback, 0,/*Do data centric work*/true);
#endif
    // Obtain  a key for TLS storage.
    client_tls_key = PIN_CreateThreadDataKey(0 /*TODO have a destructir*/);
    // Register ThreadStart to be called when a thread starts.
    PIN_AddThreadStartFunction(ThreadStart, 0);
    
    // fini function for post-mortem analysis
    PIN_AddThreadFiniFunction(ThreadFiniFunc, 0);
    PIN_AddFiniFunction(FiniFunc, 0);
    
    TRACE_AddInstrumentFunction(InstrumentTrace, 0);
    
    // Register ImageUnload to be called when an image is unloaded
    IMG_AddUnloadFunction(ImageUnload, 0);
    printf("==== PIN CLIENT : Launch program now ===\n");
    // Launch program now
    PIN_StartProgram();
    return 0;
}

