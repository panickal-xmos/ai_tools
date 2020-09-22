
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "tst_common.h"
#include "nn_operator.h"
#include "nn_types.h"
#include "xs3_vpu.h"
#include "nn_bin_utils.h"

#include "unity.h"

void larq_ref_bconv2d(const nn_image_params_t* x, const nn_image_params_t* y,
                      const nn_window_params_t* k,
                      const int32_t* packed_input_data,
                      const int32_t* packed_filter_data,
                      int32_t* packed_output_data, const long* thresholds);

/*
X_ref and K_ref must be initialised before running this.

This function test whole images, i.e. it wont work on a sub image.
*/
int run_config(bnn_b32_t* Y_p, bnn_b32_t* Y_ref_p, bnn_b256_t* X_ref,
               bnn_b256_t* K_p, bnn_b256_t* K_ref_p, int32_t* thresholds_ref,
               int32_t* thresholds_p, unsigned x_height, unsigned x_width,
               unsigned k_height, unsigned k_width, unsigned chans_in,
               unsigned chans_out, unsigned h_stride, unsigned v_stride) {
  assert(Y_p != Y_ref_p);
  assert(K_p != K_ref_p);
  assert(thresholds_p != thresholds_ref);

  unsigned y_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, v_stride);
  unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, h_stride);

  unsigned X_bytes = (x_height * x_width * chans_in) / 8;
  unsigned K_bytes = (k_width * k_height * chans_in * chans_out) / 8;
  unsigned Y_bytes = (y_width * y_height * chans_out) / 8;

  // This makes for a nice threshold for a random input
  for (unsigned i = 0; i < chans_out; i++)
    thresholds_ref[i] = i + ((chans_in * k_height * k_width - chans_out) / 2);

  nn_image_params_t x;
  x.height = x_height;
  x.width = x_width;
  x.channels = chans_in;
  nn_image_params_t y;
  y.height = y_height;
  y.width = y_width;
  y.channels = chans_out;
  nn_window_params_t k;
  k.shape.height = k_height;
  k.shape.width = k_width;
  k.stride.horizontal = h_stride;
  k.stride.vertical = v_stride;
  k.dilation.horizontal = 1;
  k.dilation.vertical = 1;

  larq_ref_bconv2d(&x, &y, &k, (int32_t*)X_ref, (int32_t*)K_ref_p,
                   (int32_t*)Y_ref_p, (const long *)thresholds_ref);

#if defined(__XS3A__)

  bnn_reorder_threshold_tensor(thresholds_p, thresholds_ref, chans_out,
                               k_width * k_height * chans_in);

  bnn_reorder_kernel_tensor(K_p, K_ref_p, k_height, k_width, chans_in,
                            chans_out);

  bnn_conv2d_bin_out((bnn_b32_t*)Y_p, (const bnn_b256_t*)X_ref,
                      (const bnn_b256_t*)K_p, thresholds_p, &x, &y, &k,
    0, 0, y_width, y_height,
    0, 0, 
    0, 0, k_width, k_height);

#else
  bnn_conv2d_bin_out((bnn_b32_t*)Y_p, (const bnn_b256_t*)X_ref,
                      (const bnn_b256_t*)K_ref_p, thresholds_ref, &x, &y, &k,
    0, 0, y_width, y_height,
    0, 0, 
    0, 0, k_width, k_height);
#endif

  unsigned chan_b32_out = (chans_out + 32 - 1) / 32;
  bnn_b32_t(*Y)[y_width][chan_b32_out] =
      (bnn_b32_t(*)[y_width][chan_b32_out])Y_p;

  bnn_b32_t(*Y_ref)[y_width][chan_b32_out] =
      (bnn_b32_t(*)[y_width][chan_b32_out])Y_ref_p;

  int all_equal = 1;
  for (unsigned h = 0; h < y_height; h++) {
    for (unsigned w = 0; w < y_width; w++) {
      for (unsigned c = 0; c < chan_b32_out; c++) {
        int e = (Y_ref[h][w][c] == Y[h][w][c]);
        all_equal &= e;
      }
    }
  }

  return 1 - all_equal;
}

void test_bnn_conv2d_bin_out_pseudo_directed() {
#define X_V_DILATION 1
#define X_H_DILATION 1

#define X_HEIGHT 5
#define X_WIDTH 5
#define K_HEIGHT 3
#define K_WIDTH 3
#define CHANS_IN 256
#define CHANS_OUT 32
#define H_STRIDE 1
#define V_STRIDE 1

#define Y_HEIGHT \
  CONV2D_OUTPUT_LENGTH(X_HEIGHT, K_HEIGHT, X_V_DILATION, V_STRIDE)
#define Y_WIDTH CONV2D_OUTPUT_LENGTH(X_WIDTH, K_WIDTH, X_H_DILATION, H_STRIDE)

#define CHAN_WORDS_IN \
  ((CHANS_IN + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS)
#define CHAN_WORDS_OUT ((CHANS_OUT + 32 - 1) / 32)

  bnn_b256_t WORD_ALIGNED K_ref[CHANS_OUT][K_HEIGHT][K_WIDTH][CHAN_WORDS_IN];
  bnn_b256_t WORD_ALIGNED
      K[CHANS_OUT / 16][K_HEIGHT][K_WIDTH][CHAN_WORDS_IN][16];

  bnn_b256_t WORD_ALIGNED X_ref[X_HEIGHT][X_WIDTH][CHAN_WORDS_IN];
  bnn_b32_t WORD_ALIGNED Y_ref[Y_HEIGHT][Y_WIDTH][CHAN_WORDS_OUT];
  bnn_b32_t WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHAN_WORDS_OUT];

  int32_t WORD_ALIGNED thresholds_ref[CHANS_OUT];
  int32_t WORD_ALIGNED thresholds[CHANS_OUT];

  srand(69);
  pseudo_rand_bytes((char*)X_ref, sizeof(X_ref));
  pseudo_rand_bytes((char*)K_ref, sizeof(K_ref));

  memset(Y, 0, sizeof(Y));
  memset(Y_ref, 0, sizeof(Y_ref));

  int failure =
      run_config((bnn_b32_t*)Y, (bnn_b32_t*)Y_ref, (bnn_b256_t*)X_ref,
                 (bnn_b256_t*)K, (bnn_b256_t*)K_ref, (int32_t*)thresholds_ref,
                 (int32_t*)thresholds, X_HEIGHT, X_WIDTH, K_HEIGHT, K_WIDTH,
                 CHANS_IN, CHANS_OUT, H_STRIDE, V_STRIDE);

  if (failure) {
    printf("it was wrong\n");
    exit(1);
  }
  TEST_ASSERT_FALSE(failure);

#undef H_STRIDE
#undef V_STRIDE
#undef H_OFFSET
#undef V_OFFSET
#undef K_HEIGHT
#undef K_WIDTH
#undef CHANS_IN
#undef CHANS_OUT
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
}

void test_bnn_conv2d_bin_out_pseudo_random() {
#define MIN_H_STRIDE 1
#define MIN_V_STRIDE 1
#define MAX_H_STRIDE 4
#define MAX_V_STRIDE 4

#define MIN_K_HEIGHT 1
#define MIN_K_WIDTH 1
#define MAX_K_HEIGHT 5
#define MAX_K_WIDTH 5

#define MIN_CHANS_IN 256
#define MAX_CHANS_IN 512

#define MIN_CHANS_OUT 32
#define MAX_CHANS_OUT 64

#define MIN_X_HEIGHT MIN_K_HEIGHT
#define MIN_X_WIDTH MIN_K_WIDTH
#define MAX_X_HEIGHT 7
#define MAX_X_WIDTH 7

#define MAX_CHAN_WORDS_IN \
  ((MAX_CHANS_IN + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS)
#define MAX_CHAN_WORDS_OUT ((MAX_CHANS_OUT + 32 - 1) / 32)

#define MAX_Y_HEIGHT (((MAX_X_HEIGHT - MIN_K_HEIGHT + 1) / MIN_V_STRIDE))
#define MAX_Y_WIDTH (((MAX_X_WIDTH - MIN_K_WIDTH + 1) / MIN_H_STRIDE))

  bnn_b256_t WORD_ALIGNED
      K_ref[MAX_CHANS_OUT][MAX_K_HEIGHT][MAX_K_WIDTH][MAX_CHAN_WORDS_IN];
  bnn_b256_t WORD_ALIGNED
      K[MAX_CHANS_OUT][MAX_K_HEIGHT][MAX_K_WIDTH][MAX_CHAN_WORDS_IN];

  bnn_b256_t WORD_ALIGNED X_ref[MAX_X_HEIGHT][MAX_X_WIDTH][MAX_CHAN_WORDS_IN];
  bnn_b32_t WORD_ALIGNED Y_ref[MAX_Y_HEIGHT][MAX_Y_WIDTH][MAX_CHAN_WORDS_OUT];
  bnn_b32_t WORD_ALIGNED Y[MAX_Y_HEIGHT][MAX_Y_WIDTH][MAX_CHAN_WORDS_OUT];

  int32_t WORD_ALIGNED thresholds_ref[MAX_CHANS_OUT];
  int32_t WORD_ALIGNED thresholds[MAX_CHANS_OUT];

  assert(((int)K & 0x3) == 0);
  assert(((int)K_ref & 0x3) == 0);
  assert(((int)X_ref & 0x3) == 0);
  assert(((int)Y & 0x3) == 0);
  assert(((int)Y_ref & 0x3) == 0);

  assert(((int)thresholds_ref & 0x3) == 0);
  assert(((int)thresholds & 0x3) == 0);

  pseudo_rand_bytes((char*)X_ref, sizeof(X_ref));
  pseudo_rand_bytes((char*)K_ref, sizeof(K_ref));

  srand(69);
  for (unsigned h_stride = MIN_H_STRIDE; h_stride <= MAX_H_STRIDE; ++h_stride) {
    for (unsigned v_stride = MIN_V_STRIDE; v_stride <= MAX_V_STRIDE;
         ++v_stride) {
      for (unsigned k_height = MIN_K_HEIGHT; k_height <= MAX_K_HEIGHT;
           ++k_height) {
        for (unsigned k_width = MIN_K_WIDTH; k_width <= MAX_K_WIDTH;
             ++k_width) {
          for (unsigned x_height = k_height; x_height <= MAX_X_HEIGHT;
               ++x_height) {
            for (unsigned x_width = k_width; x_width <= MAX_X_WIDTH;
                 ++x_width) {
              for (unsigned chans_in = MIN_CHANS_IN; chans_in <= MAX_CHANS_IN;
                   chans_in += 256) {
                for (unsigned chans_out = MIN_CHANS_OUT;
                     chans_out <= MAX_CHANS_OUT; chans_out += 32) {
                      //  printf("x_height:%u, x_width:%u, k_height:%u, k_width:%u, chans_in:%u, chans_out:%u, h_stride:%u, v_stride:%u\n", x_height,
                      // x_width, k_height, k_width, chans_in, chans_out, h_stride,
                      // v_stride);
                  int r = run_config(
                      (bnn_b32_t*)Y, (bnn_b32_t*)Y_ref, (bnn_b256_t*)X_ref,
                      (bnn_b256_t*)K, (bnn_b256_t*)K_ref,
                      (int32_t*)thresholds_ref, (int32_t*)thresholds, x_height,
                      x_width, k_height, k_width, chans_in, chans_out, h_stride,
                      v_stride);
                  TEST_ASSERT_FALSE(r);
                }
              }
            }
          }
        }
      }
    }
  }
}
void bnn_conv2d_bin_out_valid(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, 
    const bnn_b256_t* K_p, 
    const int32_t* thresholds_p,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height
);

int run_sub_image(bnn_b32_t* Y_p, const bnn_b32_t* Y_ref_p, const bnn_b256_t* X_ref,
              bnn_b256_t* K_p, const bnn_b256_t* K_ref_p, const int32_t* thresholds_ref,
              int32_t* thresholds_p, 
              const nn_image_params_t* x,
              const nn_image_params_t* y,
              const nn_window_params_t* k,
              unsigned y_loc_x, unsigned y_loc_y, 
              unsigned y_sub_width, unsigned y_sub_height){

#if defined(__XS3A__)

  bnn_reorder_threshold_tensor(thresholds_p, thresholds_ref, y->channels,
                               k->shape.width * k->shape.height * x->channels);

  bnn_reorder_kernel_tensor(K_p, K_ref_p, k->shape.height , k->shape.width, x->channels,
                            y->channels);

  bnn_conv2d_bin_out_valid((bnn_b32_t*)Y_p, (const bnn_b256_t*)X_ref,
                      (const bnn_b256_t*)K_p, thresholds_p, x, y, k,
                       y_loc_x, y_loc_y, y_sub_width, y_sub_height);
#else
  bnn_conv2d_bin_out_valid((bnn_b32_t*)Y_p, (const bnn_b256_t*)X_ref,
                      (const bnn_b256_t*)K_ref_p, thresholds_ref, x, y, k,
                      y_loc_x, y_loc_y, y_sub_width, y_sub_height);
#endif

  unsigned chan_b32_out = (y->channels + 32 - 1) / 32;
  bnn_b32_t(*Y)[y->width][chan_b32_out] =
      (bnn_b32_t(*)[y->width][chan_b32_out])Y_p;

  bnn_b32_t(*Y_ref)[y->width][chan_b32_out] =
      (bnn_b32_t(*)[y->width][chan_b32_out])Y_ref_p;

  int all_equal = 1;
  for (unsigned h = y_loc_y; h < y_loc_y + y_sub_height; h++) {
    for (unsigned w = y_loc_x; w < y_loc_x + y_sub_width; w++) {
      for (unsigned c = 0; c < chan_b32_out; c++) {
        int e = (Y_ref[h][w][c] == Y[h][w][c]);
        all_equal &= e;
      }
    }
  }
  return 1 - all_equal;
}

#undef CHAN_WORDS_IN
void test_bnn_conv2d_bin_out_sub_image(){

  #define FULL_X_HEIGHT 7
  #define FULL_X_WIDTH 8
  #define FULL_K_HEIGHT 3
  #define FULL_K_WIDTH 5
  #define CHANS_IN 256
  #define CHANS_OUT 32
  #define X_V_DILATION 1
  #define V_STRIDE 1
  #define X_H_DILATION 1
  #define H_STRIDE 1

  #define CHAN_WORDS_IN ((CHANS_IN + 32 - 1) / 32)
  #define CHAN_WORDS_OUT ((CHANS_OUT + 32 - 1) / 32)
  #define FULL_Y_HEIGHT \
    CONV2D_OUTPUT_LENGTH(FULL_X_HEIGHT, FULL_K_HEIGHT, X_V_DILATION, V_STRIDE)
  #define FULL_Y_WIDTH CONV2D_OUTPUT_LENGTH(FULL_X_WIDTH, FULL_K_WIDTH, X_H_DILATION, H_STRIDE)

  bnn_b256_t WORD_ALIGNED
      K_ref[CHANS_OUT][FULL_K_HEIGHT][FULL_K_WIDTH][CHAN_WORDS_IN];
  bnn_b256_t WORD_ALIGNED
      K[CHANS_OUT][FULL_K_HEIGHT][FULL_K_WIDTH][CHAN_WORDS_IN];

  bnn_b256_t WORD_ALIGNED X_ref[FULL_X_HEIGHT][FULL_X_WIDTH][CHAN_WORDS_IN];
  bnn_b32_t WORD_ALIGNED Y_ref[FULL_Y_HEIGHT][FULL_Y_WIDTH][CHAN_WORDS_OUT];
  bnn_b32_t WORD_ALIGNED Y[FULL_Y_HEIGHT][FULL_Y_WIDTH][CHAN_WORDS_OUT];

  int32_t WORD_ALIGNED thresholds_ref[CHANS_OUT];
  int32_t WORD_ALIGNED thresholds[CHANS_OUT];

  assert(((int)K & 0x3) == 0);
  assert(((int)K_ref & 0x3) == 0);
  assert(((int)X_ref & 0x3) == 0);
  assert(((int)Y & 0x3) == 0);
  assert(((int)Y_ref & 0x3) == 0);

  assert(((int)thresholds_ref & 0x3) == 0);
  assert(((int)thresholds & 0x3) == 0);

  pseudo_rand_bytes((char*)X_ref, sizeof(X_ref));
  pseudo_rand_bytes((char*)K_ref, sizeof(K_ref));

  srand(42);

  for (unsigned h_stride=1; h_stride < 5; h_stride++){

    for (unsigned v_stride=1; v_stride < 5; v_stride++){
        
      nn_image_params_t x;
      x.height = FULL_X_HEIGHT;
      x.width = FULL_X_WIDTH;
      x.channels = CHANS_IN;
      nn_image_params_t y;
      y.height = CONV2D_OUTPUT_LENGTH(FULL_X_HEIGHT, FULL_K_HEIGHT, X_V_DILATION, v_stride);
      y.width = CONV2D_OUTPUT_LENGTH(FULL_X_WIDTH, FULL_K_WIDTH, X_H_DILATION, h_stride);
      y.channels = CHANS_OUT;
      nn_window_params_t k;
      k.shape.height = FULL_K_HEIGHT;
      k.shape.width = FULL_K_WIDTH;
      k.stride.horizontal = h_stride;
      k.stride.vertical = v_stride;
      k.dilation.horizontal = X_H_DILATION;
      k.dilation.vertical = X_V_DILATION;

      for (unsigned i = 0; i < y.channels; i++)
        thresholds_ref[i] = ((x.channels * k.shape.height * k.shape.width ) / 2);
        // thresholds_ref[i] = i + ((x.channels * k.shape.height * k.shape.width - y.channels) / 2);

      //Calculate the entire reference image

      larq_ref_bconv2d(&x, &y, &k, (int32_t*)X_ref, (int32_t*)K_ref,
                      (int32_t*)Y_ref, (const long *)thresholds_ref);

      for (unsigned y_loc_x = 0; y_loc_x<y.width;++y_loc_x){
        for (unsigned y_loc_y = 0; y_loc_y<y.height;++y_loc_y){
          for (unsigned y_sub_width = 1; y_sub_width<y.width-y_loc_x;++y_sub_width){
            for (unsigned y_sub_height = 1; y_sub_height<y.height-y_loc_y;++y_sub_height){
                // printf("%u %u %u %u %u %u\n", y_loc_x, y_loc_y, y_sub_width, y_sub_height, h_stride, v_stride);

                memset(Y, 0xaa, sizeof(Y));
                memset(thresholds, 0, sizeof(thresholds));
                int r = run_sub_image((bnn_b32_t*)Y, (const bnn_b32_t*)Y_ref, 
                (const bnn_b256_t*) X_ref, (bnn_b256_t*) K, 
                (const bnn_b256_t*) K_ref, (const int32_t*)thresholds_ref, 
                (int32_t*)thresholds, &x, &y, &k,
                  y_loc_x, y_loc_y, y_sub_width, y_sub_height
                );
                TEST_ASSERT_FALSE(r);
              }
            }
          } 
        }
    }
  }
}

void test_bnn_conv2d() {
  UNITY_SET_FILE();
  RUN_TEST(test_bnn_conv2d_bin_out_pseudo_directed);
  RUN_TEST(test_bnn_conv2d_bin_out_pseudo_random);
  RUN_TEST(test_bnn_conv2d_bin_out_sub_image);
}