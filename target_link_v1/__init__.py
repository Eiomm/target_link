"""V1: target-link fine-grained trajectory representation learning.

Strict implementation of `representationV1.md`:
    - target link only (no upstream/downstream buffer, no zone/approach tokens)
    - long links split into configurable sub-links (default 200 m)
    - 10 m spatial bins along driving direction -> local speed profile
    - lightweight Transformer encoder + masked mean pooling
    - mean aggregation over vehicles -> r_link
    - downstream sees [mean_speed ; r_link]
"""
