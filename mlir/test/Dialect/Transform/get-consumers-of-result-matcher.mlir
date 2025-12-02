// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics

// Minimal check that a matcher using transform.get_consumers_of_result can be
// called from collect_matching and foreach_match. This requires that the matcher
// implements MatchOpInterface.
module attributes { transform.with_named_sequence } {
  func.func @test() -> i32 {
    %c0 = arith.constant 0 : i32
    %add = arith.addi %c0, %c0 : i32
    // expected-remark @below {{from collect_matching}}
    // expected-remark @below {{from foreach_match}}
    return %add : i32
  }

  transform.named_sequence @matcher(
      %op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %op ["arith.addi"] : !transform.any_op
    %consumers = transform.get_consumers_of_result %op[0]
        : (!transform.any_op) -> !transform.any_op
    transform.yield %consumers : !transform.any_op
  }

  transform.named_sequence @action(
      %consumer: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %consumer, "from foreach_match"
        : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.sequence %root : !transform.any_op failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
      %collected = transform.collect_matching @matcher in %arg0
          : (!transform.any_op) -> !transform.any_op
      transform.foreach %collected : !transform.any_op {
      ^bb1(%c: !transform.any_op):
        transform.debug.emit_remark_at %c, "from collect_matching"
            : !transform.any_op
        transform.yield
      }

      transform.foreach_match in %arg0
          @matcher -> @action
          : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    transform.yield
  }
}
