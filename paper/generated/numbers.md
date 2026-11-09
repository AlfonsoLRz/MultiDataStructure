# Numbers used in the manuscript

| macro | value | source |
|---|---|---|
| `\nCells` | 73 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nScenes` | 17 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nSearchedCells` | 67 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nTransferCells` | 6 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nCellsGround` | 31 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nCellsAerial` | 25 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nCellsForest` | 16 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nestedSingleAll` | 1.011 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nestedPickedAll` | 0 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\pickOracleAll` | 1.003 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nestedSingleLo` | 0.97 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nestedSingleHi` | 1.07 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nestedWinsAll` | 1 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nestedTiesAll` | 60 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nestedLossesAll` | 12 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nestedSingleGround` | 1.020 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nestedSingleAerial` | 1.005 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nestedSingleForest` | 1.002 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nestedSingleOneB` | 1.004 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nestedSingleHundredM` | 1.012 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\leverOctreeDefaultTuned` | 1.66 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\leverKdDefaultTuned` | 1.53 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\leverOctreeDefaultPrimitive` | 1.36 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\leverKdDefaultPrimitive` | 1.26 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\leverPrimitiveTuned` | 1.21 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition |
| `\nWorkloads` | 7 | workload_sweep_makinon2 |
| `\workloadNestedSingleMax` | 1.022 | workload_sweep_makinon2 |
| `\workloadNestedSingleMin` | 1.002 | workload_sweep_makinon2 |
| `\gpuNestedSingleTotal` | 1.010 | gpu_normals_makinon2/confirm, fastest round per schema |
| `\gpuNestedSingleKernel` | 1.013 | gpu_normals_makinon2/confirm |
| `\oursOverCudaKdTree` | 1.34 | gpu_normals_makinon2/confirm: our best schema / cudaKDTree, end to end |
| `\nGpuRows` | 12 | gpu_normals_makinon2/confirm |
| `\nodeCostNestedSingleZero` | 1.008 | node_cost*_makinon2/d0 |
| `\nodeCostNestedPickedZero` | 0 | node_cost*_makinon2/d0 |
| `\nodeCostNestedSingleHundredNs` | 1.000 | node_cost*_makinon2/d100 |
| `\nodeCostNestedPickedHundredNs` | 1 | node_cost*_makinon2/d100 |
| `\nodeCostNestedSingleOneUs` | 1.024 | node_cost*_makinon2/d1000 |
| `\nodeCostNestedPickedOneUs` | 2 | node_cost*_makinon2/d1000 |
| `\nodeCostNestedSingleTenUs` | 1.046 | node_cost*_makinon2/d10000 |
| `\nodeCostNestedPickedTenUs` | 1 | node_cost*_makinon2/d10000 |
| `\ruleRegret` | 1.043 | rule_fit, instrumented column |
| `\ruleRegretTimed` | 1.076 | rule_fit, timed-passes column |
| `\ruleRegretP` | 1.10 | rule_fit, p90 instrumented |
| `\ruleCellsOver` | 2 | rule_fit: cells with regret > 1.15, instrumented |
| `\globalSchemaRegret` | 1.067 | rule_fit |
| `\groupSchemaRegret` | 1.078 | rule_fit |
| `\octreeDefaultRegret` | 1.746 | rule_fit |
| `\octreeDefaultRegretTimed` | 1.950 | rule_fit |
| `\kdDefaultRegret` | 1.630 | rule_fit |
| `\floorRegret` | 1.011 | rule_fit |
| `\ruleCapturePct` | 89 | rule_fit: share of log(default -> oracle) closed by the rule, the lower of the two columns |
| `\nRuleCells` | 68 | rule_fit: searched cells with the 55 tuned singles on the opt half |
| `\nLibraryCells` | 15 | external_makinon2 (+ external_1B_makinon2) |
| `\oursOctreeDefaultOverPick` | 1.73 | external_report medians |
| `\oursKdDefaultOverPick` | 1.42 | external_report medians |
| `\ruleOverPick` | 1.01 | external_report medians |
| `\nanoflannDefaultAll` | 1.02 | external_makinon2 |
| `\nanoflannDefaultRadius` | 1.27 | external_makinon2 |
| `\nanoflannDefaultKnn` | 0.72 | external_makinon2 |
| `\nanoflannBestAll` | 0.95 | external_makinon2 |
| `\nanoflannBestKnn` | 0.68 | external_makinon2 |
| `\nanoflannBestRadius` | 1.18 | external_makinon2 |
| `\pclKdAll` | 2.59 | external_makinon2 |
| `\pclOctreeAll` | 5.99 | external_makinon2 |
| `\openThreeDAll` | 1.60 | external_makinon2 |
| `\unibnRadius` | 1.65 | external_makinon2 |
| `\boostAll` | 2.46 | external_makinon2 |
| `\libraryLeverMin` | 1.04 | external_makinon2: each library's default / its best setting |
| `\libraryLeverMax` | 1.07 | external_makinon2 |
| `\hugeSinglesHundredM` | 0.915 | cache_accounting_makinon2 |
| `\hugeBigPools` | 0.902 | cache_accounting_makinon2 |
| `\hugeSmallPools` | 0.978 | cache_accounting_makinon2 |
| `\hugeSinglesTwentyFiveM` | 0.974 | cache_accounting_makinon2 |
| `\hugeMin` | 0.709 | cache_accounting_makinon2 |
| `\prefetchSinglesHundredM` | 0.941 | cache_accounting_makinon2 |
| `\prefetchHGrid` | 1.092 | cache_accounting_makinon2 |
| `\prefetchMin` | 0.717 | cache_accounting_makinon2 |
| `\bothSinglesHundredM` | 0.928 | cache_accounting_makinon2 |
| `\hugeNestedHundredM` | 0.945 | cache_accounting_makinon2 |
| `\visitCostTopNs` | 35 | cache_accounting_makinon2: DomFountain 100M kd48l128sm, depths 0-7, instrumented pass |
| `\visitCostMiddleNs` | 172 | cache_accounting_makinon2: DomFountain 100M kd48l128sm, middle depths |
| `\visitCostLeafNs` | 352 | cache_accounting_makinon2: DomFountain 100M kd48l128sm, deepest four levels |
| `\topLevelsShareDomFountain` | 4 | cache_accounting_makinon2: share of the instrumented query time in depths 0-7 |
| `\savedVisitCostNs` | 39 | cache_accounting_makinon2: (instrumented ns of the pick - of the nested finalist) / visits saved, DomFountain and Lille 100M |
| `\domFountainPoolMb` | 75 | cache_accounting_makinon2: kd48l128sm nodes x 32 B |
| `\oocInRamAdopted` | 0 | ooc.py: stage B adoptions with the hierarchy in RAM |
| `\oocInRamCells` | 27 | ooc.py |
| `\oocPagedAdopted` | 4 | ooc.py: stage B adoptions with the hierarchy paged |
| `\oocPagedCells` | 25 | ooc.py |
| `\pagedOverPinnedMinNoCache` | 3.3 | ooc.py: best schema with the hierarchy paged / pinned, no cache, four device models |
| `\pagedOverPinnedMaxNoCache` | 6.1 | ooc.py |
| `\pagedOverPinnedMinCache` | 1.2 | ooc.py: best schema with the hierarchy paged / pinned, 1% LRU, four device models |
| `\pagedOverPinnedMaxCache` | 1.9 | ooc.py |
| `\nestedBuildRatio` | 0.941 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition: build time of the best nested finalist / best single (held-out structure run) |
| `\nestedNodesRatio` | 0.915 | e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition: nodes of the best nested finalist / best single |
| `\oracleRepeatCvPct` | 7.7 | rule_fit: median repeat CV of the oracle_opt rows |
| `\exampleOracleName` | kd-tree (sliding midpoint), leaf 128 | rule_fit: DomFountain 25M oracle_opt |
| `\exampleOctreeVisits` | 37.9 | e9 opt half, DomFountain 25M, instrumented |
| `\exampleOctreePoints` | 215 | e9 opt half, DomFountain 25M |
| `\exampleOctreeUs` | 3.31 | e9 opt half, DomFountain 25M |
| `\exampleOctreeNodes` | 2.70 | e9 opt half, DomFountain 25M, millions of nodes |
| `\exampleOracleVisits` | 39.3 | e9 opt half, DomFountain 25M, instrumented |
| `\exampleOraclePoints` | 391 | e9 opt half, DomFountain 25M |
| `\exampleOracleUs` | 2.51 | e9 opt half, DomFountain 25M |
| `\exampleOracleNodes` | 0.59 | e9 opt half, DomFountain 25M, millions of nodes |
| `\aloneOverInterleavedKdBase` | 0.846 | cache_accounting_makinon2/interleave: kd48l128sm alone / interleaved, base arm |
| `\aloneOverInterleavedOtBase` | 0.949 | cache_accounting_makinon2/interleave: ot24l128tc alone / interleaved, base arm |
| `\aloneOverInterleavedKdHuge` | 0.973 | cache_accounting_makinon2/interleave: kd48l128sm alone / interleaved, huge arm |
| `\aloneOverInterleavedOtHuge` | 0.936 | cache_accounting_makinon2/interleave: ot24l128tc alone / interleaved, huge arm |
| `\hugeAloneQt` | 0.811 | cache_accounting_makinon2/interleave: SanAndreas 100M qt24l32 alone, huge / base |
| `\hugeAloneKd` | 0.890 | cache_accounting_makinon2/interleave: SanAndreas 100M kd48l128sm alone, huge / base |
| `\nullPairMin` | 0.95 | cache_accounting_makinon2: SanAndreas 100M qt9_ot24l32 / qt24l32 (the same tree) across arms |
| `\nullPairMax` | 1.17 | cache_accounting_makinon2: the same |
| `\isprsOverLeafMatchedMin` | 0.90 | ooc.py: ISPRS nested schemas / leaf-matched depth-10 singles, SATA model, geomean per regime |
| `\isprsOverLeafMatchedMax` | 0.95 | ooc.py |
| `\isprsOverBestSingleMin` | 1.9 | ooc.py: ISPRS nested schemas / the regime's best single, SATA model |
| `\isprsOverBestSingleMax` | 3.3 | ooc.py |
| `\hierarchyMbMin` | 1.1 | ooc.py: node pool of the best single with leaves on the device (0:memory), over the four device models, MB |
| `\hierarchyMbMax` | 20 | ooc.py |
| `\nAnchorCells` | 3 | external_1B_makinon2: three 1B clouds, sequential protocol, 1000 queries |
| `\anchorNanoflannDefaultAll` | 1.01 | external_1B_makinon2: three 1B clouds, sequential protocol, 1000 queries: nanoflann leaf 10 / the selected schema |
| `\anchorNanoflannDefaultMin` | 0.91 | external_1B_makinon2: three 1B clouds, sequential protocol, 1000 queries |
| `\anchorNanoflannDefaultMax` | 1.10 | external_1B_makinon2: three 1B clouds, sequential protocol, 1000 queries |
| `\anchorNanoflannBestAll` | 0.98 | external_1B_makinon2: three 1B clouds, sequential protocol, 1000 queries |
| `\anchorNanoflannDefaultKnn` | 0.82 | external_1B_makinon2: three 1B clouds, sequential protocol, 1000 queries |
| `\anchorNanoflannDefaultRadius` | 1.19 | external_1B_makinon2: three 1B clouds, sequential protocol, 1000 queries |
| `\anchorOursOctreeDefault` | 1.74 | external_1B_makinon2: three 1B clouds, sequential protocol, 1000 queries |
| `\anchorOursKdDefault` | 1.63 | external_1B_makinon2: three 1B clouds, sequential protocol, 1000 queries |
| `\anchorRuleOverPick` | 1.02 | external_1B_makinon2: three 1B clouds, sequential protocol, 1000 queries |
| `\nPanelCells` | 9 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds |
| `\panelOctreeDefaultBase` | 1.723 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds, base arm: octree default / pick |
| `\panelOctreeDefaultHuge` | 1.755 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds, huge arm: octree default / pick |
| `\panelKdDefaultBase` | 1.464 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds, base arm: kd default / pick |
| `\panelKdDefaultHuge` | 1.511 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds, huge arm: kd default / pick |
| `\panelRuleBase` | 1.009 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds, base arm: rule / pick |
| `\panelRuleHuge` | 1.041 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds, huge arm: rule / pick |
| `\panelPickOracleBase` | 1.002 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds, base arm: pick / best single |
| `\panelPickOracleHuge` | 1.000 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds, huge arm: pick / best single |
| `\panelNestedSingleBase` | 1.021 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds, base arm: nested / best single |
| `\panelNestedSingleHuge` | 1.032 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds, huge arm: nested / best single |
| `\panelStageBBase` | 2.425 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds, base arm: stage B winner / pick |
| `\panelStageBHuge` | 2.514 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds, huge arm: stage B winner / pick |
| `\panelNestedSingleBaseMax` | 1.05 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds |
| `\panelHugeOverBaseSingles` | 0.958 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds: huge / base over the single schemas |
| `\panelHugeOverBaseSinglesMin` | 0.92 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds |
| `\panelHugeOverBaseSinglesMax` | 1.01 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds |
| `\panelHugeOverBaseNested` | 0.957 | leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds |
| `\leafKdOverThirtyTwoBase` | 1.076 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds: kd48sm leaf 32 / best leaf, base |
| `\leafKdOverHundredTwentyEightBase` | 1.013 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds: kd48sm leaf 128 / best leaf, base |
| `\leafKdOverThirtyTwoHuge` | 1.040 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds: kd48sm leaf 32 / best leaf, huge |
| `\leafKdOverHundredTwentyEightHuge` | 1.036 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds: kd48sm leaf 128 / best leaf, huge |
| `\leafQtOverThirtyTwoBase` | 1.315 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds: qt24tc leaf 32 / best leaf, base |
| `\leafQtOverHundredTwentyEightBase` | 1.037 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds: qt24tc leaf 128 / best leaf, base |
| `\leafQtOverThirtyTwoHuge` | 1.297 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds: qt24tc leaf 32 / best leaf, huge |
| `\leafQtOverHundredTwentyEightHuge` | 1.037 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds: qt24tc leaf 128 / best leaf, huge |
| `\leafWorstMin` | 1.40 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds: worst leaf / best leaf |
| `\leafWorstMax` | 2.50 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds |
| `\leafPairs` | 18 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds: (cell, primitive) pairs with both page modes |
| `\leafShiftPairs` | 4 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds: pairs whose best leaf is smaller on huge pages |
| `\leafKdShiftPairs` | 4 | leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds: kd-tree pairs whose best leaf is smaller on huge pages |
