// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATOR_DISPATCHER_H_
#define XCORE_OPERATOR_DISPATCHER_H_

#include "lib_ops/api/par.h"
#include "lib_ops/api/allocator.h"
#include "lib_ops/api/lib_ops.h"

#ifdef XCORE

extern "C" {
#include "lib_ops/src/xs1.h"  // FIXME: remove someday
//    this must appear BEFORE including xcore/thread.h
#include <xcore/thread.h>
}

#define ATTRIBUTE_THREAD_FUNCTION __attribute__((fptrgroup("thread_function")))
#define STRINGIFY(NAME) #NAME
#define GET_STACKWORDS(DEST, NAME) \
  asm("ldc %[__dest], " STRINGIFY(NAME) ".nstackwords" : [ __dest ] "=r"(DEST))

#else // not XCORE
#include <vector>
#include <thread>

#define ATTRIBUTE_THREAD_FUNCTION
#define GET_STACKWORDS(DEST, NAME) DEST=0

typedef void (*thread_function_t)(void*);
typedef std::vector<std::thread> threadgroup_t;
#endif

namespace xcore {

typedef struct Task {
  ATTRIBUTE_THREAD_FUNCTION thread_function_t function;
  void* argument;
  size_t stack_words;
  void* stack;
} Task;

typedef struct TaskArray {
  int size;
  Task* data;
} TaskArray;

class Dispatcher {
 public:
  Dispatcher(void* buffer, size_t size, int num_cores, bool use_current_core = true);
  ~Dispatcher();

  void* AllocateStackBuffer(int32_t num_threads, size_t stack_words);
  void* AllocatePersistentBuffer(size_t size);
  XCoreStatus Reset();
  XCoreStatus Add(thread_function_t function, void* argument,
                  size_t stack_words);
  void Start();
  void Wait();

 private:
  int num_threads_;
  bool use_current_thread_;
  size_t stack_size_;
  void* stack_ptr_;
  threadgroup_t group_;
  TaskArray tasks_;
  LinearAllocator allocator_;
};

// static, shared Dispatcher object
Dispatcher* GetDispatcher();
XCoreStatus InitializeXCore(Dispatcher* dispatcher);

}  // namespace xcore

#endif  // XCORE_OPERATOR_DISPATCHER_H_