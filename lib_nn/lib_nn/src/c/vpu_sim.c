
#include "vpu_sim.h"

#include <stdio.h>

static xs3_vpu vpu;



/**
 * Saturate to the relevent bounds.
 */
static int64_t saturate(
    const int64_t input,
    const unsigned bits)
{
    const int64_t max_val = (((int64_t)1)<<(bits-1))-1;
    const int64_t min_val = -max_val;
    
    return (input >= max_val)?  max_val : (input <= min_val)? min_val : input;
}

/**
 * Get the accumulator for the VPU's current mode
 */
static int64_t get_accumulator(
    unsigned index)
{
    if(vpu.mode == MODE_S8 || vpu.mode == MODE_S16){
        union {
            int16_t s16[2];
            int32_t s32;
        } acc;
        acc.s16[1] = vpu.vD.s16[index];
        acc.s16[0] = vpu.vR.s16[index];
        
        return acc.s32;
    } else {
        assert(0); // TODO
    }

}

/** 
 * Set the accumulator for the VPU's current mode
 */
static void set_accumulator(
    unsigned index,
    int64_t acc)
{
    if(vpu.mode == MODE_S8 || vpu.mode == MODE_S16){
        
        unsigned mask = (1<<VPU_INT8_ACC_VR_BITS)-1;
        vpu.vR.s16[index] = acc & mask;
        mask = mask << VPU_INT8_ACC_VR_BITS;
        vpu.vD.s16[index] = ((acc & mask) >> VPU_INT8_ACC_VR_BITS);

    } else {
        assert(0); // TODO
    }
}

/**
 * Rotate the accumulators following a VLMACCR
 */
static void rotate_accumulators()
{
    if(vpu.mode == MODE_S8 || vpu.mode == MODE_S16){
        data16_t tmpD = vpu.vD.u16[VPU_INT8_ACC_PERIOD-1];
        data16_t tmpR = vpu.vR.u16[VPU_INT8_ACC_PERIOD-1];
        for(int i = VPU_INT8_ACC_PERIOD-1; i > 0; i--){
            vpu.vD.u16[i] = vpu.vD.u16[i-1];
            vpu.vR.u16[i] = vpu.vR.u16[i-1];
        }
        vpu.vD.u16[0] = tmpD;
        vpu.vR.u16[0] = tmpR;
    } else if(vpu.mode == MODE_S32) {
        uint32_t tmpD = vpu.vD.u32[VPU_INT32_ACC_PERIOD-1];
        uint32_t tmpR = vpu.vR.u32[VPU_INT32_ACC_PERIOD-1];
        for(int i = VPU_INT32_ACC_PERIOD-1; i > 0; i--){
            vpu.vD.u32[i] = vpu.vD.u32[i-1];
            vpu.vR.u32[i] = vpu.vR.u32[i-1];
        }
        vpu.vD.u32[0] = tmpD;
        vpu.vR.u32[0] = tmpR;
    } else {
        assert(0); //How'd this happen?
    }
}




void VSETC(const vector_mode mode){
    vpu.mode = mode;
}

void VCLRDR(){
    memset(&vpu.vR.u8[0], 0 ,XS3_VPU_VREG_WIDTH_BYTES);
    memset(&vpu.vD.u8[0], 0 ,XS3_VPU_VREG_WIDTH_BYTES);
}

void VLDR(const void* addr){
    memcpy(&vpu.vR.u8[0], addr, XS3_VPU_VREG_WIDTH_BYTES);
}

void VLDD(const void* addr){
    memcpy(&vpu.vD.u8[0], addr, XS3_VPU_VREG_WIDTH_BYTES);
}

void VLDC(const void* addr){
    memcpy(&vpu.vC.u8[0], addr, XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTR(void* addr){
    memcpy(addr, &vpu.vR.u8[0], XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTD(void* addr){
    memcpy(addr, &vpu.vD.u8[0], XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTC(void* addr){
    memcpy(addr, &vpu.vC.u8[0], XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTRPV(void* addr, unsigned mask){
    int8_t* addr8 = (int8_t*) addr;

    for(int i = 0; i < 32; i++){
        if(mask & (1 << i)){
            addr8[i] = vpu.vR.s8[i];
        }
    }
}

void VLMACC(
    const void* addr)
{
    if(vpu.mode == MODE_S8){
        const int8_t* addr8 = (const int8_t*) addr;

        for(int i = 0; i < VPU_INT8_VLMACC_ELMS; i++){
            int64_t acc = get_accumulator(i);
            acc = acc + (((int32_t)vpu.vC.s8[i]) * addr8[i]);

            set_accumulator(i, saturate(acc, 32));
        }
    } else if(vpu.mode == MODE_S16){
        const int16_t* addr16 = (const int16_t*) addr;

        for(int i = 0; i < VPU_INT16_VLMACC_ELMS; i++){
            int64_t acc = get_accumulator(i);
            acc = acc + (((int32_t)vpu.vC.s16[i]) * addr16[i]);

            set_accumulator(i, saturate(acc, 32));
        }
    } else if(vpu.mode == MODE_S32){
        const int32_t* addr32 = (const int32_t*) addr;

        for(int i = 0; i < VPU_INT32_VLMACC_ELMS; i++){
            int64_t acc = get_accumulator(i);
            acc = acc + (((int64_t)vpu.vC.s32[i]) * addr32[i]);

            set_accumulator(i, saturate(acc, 40));
        }
    } else { 
        assert(0); //How'd this happen?
    }
}

void VLMACCR(
    const void* addr)
{
    if(vpu.mode == MODE_S8){
        const int8_t* addr8 = (const int8_t*) addr;
        int64_t acc = get_accumulator(VPU_INT8_ACC_PERIOD-1);

        for(int i = 0; i < VPU_INT8_EPV; i++)
            acc = acc + (((int32_t)vpu.vC.s8[i]) * addr8[i]);

        acc = saturate(acc, 32);
        rotate_accumulators();
        set_accumulator(0, acc);
    } else if(vpu.mode == MODE_S16){
        const int16_t* addr16 = (const int16_t*) addr;
        int64_t acc = get_accumulator(VPU_INT16_ACC_PERIOD-1);

        for(int i = 0; i < VPU_INT16_EPV; i++)
            acc = acc + (((int32_t)vpu.vC.s16[i]) * addr16[i]);

        acc = saturate(acc, 32);
        rotate_accumulators();
        set_accumulator(0, acc);
    } else if(vpu.mode == MODE_S32){
        const int32_t* addr32 = (const int32_t*) addr;
        int32_t acc = get_accumulator(VPU_INT32_ACC_PERIOD-1);

        for(int i = 0; i < VPU_INT32_EPV; i++)
            acc = acc + (((int32_t)vpu.vC.s32[i]) * addr32[i]);

        acc = saturate(acc, 40);
        rotate_accumulators();
        set_accumulator(0, acc);
    } else { 
        assert(0); //How'd this happen?
    }
}

void VLSAT(
    const void* addr)
{
    if(vpu.mode == MODE_S8){
        const uint16_t* addr16 = (const uint16_t*) addr;

        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            int32_t acc = get_accumulator(i);
            acc = acc + (1 << (addr16[i]-1));   //Round
            acc = acc >> addr16[i];             //Shift
            int8_t val = saturate(acc, 8);      //Saturate

            vpu.vR.s8[i] = val;
        }
        memset(&vpu.vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);
        memset(&vpu.vR.u8[VPU_INT8_ACC_PERIOD], 0, VPU_INT8_ACC_PERIOD);
    } else if(vpu.mode == MODE_S16){
        const uint16_t* addr16 = (const uint16_t*) addr;

        for(int i = 0; i < VPU_INT16_ACC_PERIOD; i++){
            int32_t acc = get_accumulator(i);
            acc = acc + (1 << (addr16[i]-1));   //Round
            acc = acc >> addr16[i];             //Shift
            int16_t val = saturate(acc, 16);    //Saturate

            vpu.vR.s16[i] = val;
        }
        memset(&vpu.vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);

    } else if(vpu.mode == MODE_S32){
        const uint32_t* addr32 = (const uint32_t*) addr;

        for(int i = 0; i < VPU_INT32_ACC_PERIOD; i++){
            int64_t acc = get_accumulator(i);
            acc = acc + (1 << (addr32[i]-1));   //Round
            acc = acc >> addr32[i];             //Shift
            int32_t val = saturate(acc, 32);    //Saturate

            vpu.vR.s32[i] = val;
        }
        memset(&vpu.vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);
    } else { 
        assert(0); //How'd this happen?
    }
}


void print_reg(
    const char* which, 
    const vpu_vector_t* vec, 
    const unsigned hex,
    const char* extra, 
    const unsigned line)
{
    printf("%s = [ ", which);
    if(vpu.mode == MODE_S8){
        for(int i = 0; i < VPU_INT8_EPV; i++){
            int8_t v = vec->s8[i];
            if(hex) printf("0x%X, ",(unsigned)v);
            else    printf("%d, ", v);
        }
    } else if(vpu.mode == MODE_S16) {
        for(int i = 0; i < VPU_INT16_EPV; i++){
            int16_t v = vec->s16[i];
            if(hex) printf("0x%X, ",(unsigned)v);
            else    printf("%d, ", v);
        }
    } else if(vpu.mode == MODE_S32) {
        for(int i = 0; i < VPU_INT32_EPV; i++){
            int32_t v = vec->s32[i];
            if(hex) printf("0x%X, ",(unsigned)v);
            else    printf("%ld, ", v);
        }
    } else {
        assert(0); //How'd this happen?
    }
    printf(" ]");

    if(extra != NULL)
        printf(" %s", extra);
    if(line != 0)
        printf("%u", line);
    printf("\n");
}

void print_vR(const unsigned hex, const char* extra, const unsigned line){    print_reg("vR", &vpu.vR, hex, extra, line);   }
void print_vD(const unsigned hex, const char* extra, const unsigned line){    print_reg("vD", &vpu.vD, hex, extra, line);   }
void print_vC(const unsigned hex, const char* extra, const unsigned line){    print_reg("vC", &vpu.vC, hex, extra, line);   }

void print_accumulators(const unsigned hex)
{
    printf("vD:vR = [ ");
    if(vpu.mode == MODE_S8 || vpu.mode == MODE_S16){
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            int32_t acc = get_accumulator(i);
            if(hex) printf("0x%X, ", (unsigned)acc);
            else    printf("%ld, ", acc);
        }
    } else if(vpu.mode == MODE_S32) {
        for(int i = 0; i < VPU_INT32_ACC_PERIOD; i++){
            int64_t acc = get_accumulator(i);
            if(hex) printf("0x%llX, ", (uint64_t)acc);
            else    printf("%lld, ", acc);
        }
    } else {
        assert(0); //How'd this happen?
    }
    printf(" ]\n");
}