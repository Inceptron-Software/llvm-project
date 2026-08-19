// RUN: mlir-opt -gpu-async-region %s | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL: func @if_without_else_threads_token(
  // CHECK: %[[SEED:.*]] = gpu.wait async
  // CHECK: %[[NEXT:.*]] = scf.if {{.*}} -> (!gpu.async.token) {
  // CHECK: %[[DEALLOC:.*]] = gpu.dealloc async [%[[SEED]]] %{{.*}}
  // CHECK: scf.yield %[[DEALLOC]] : !gpu.async.token
  // CHECK: } else {
  // CHECK: scf.yield %[[SEED]] : !gpu.async.token
  // CHECK: }
  // CHECK: gpu.wait [%[[NEXT]]]
  func.func @if_without_else_threads_token(%condition : i1, %buffer : memref<7xf32>) {
    %seed = gpu.wait async
    scf.if %condition {
      gpu.dealloc %buffer : memref<7xf32>
    }
    gpu.wait
    return
  }
}
