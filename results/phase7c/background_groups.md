## Background groups (distance < 1.25 grey levels, over all 85 training.zip + 24 val.zip scenes)

| group | scenes | split role |
|---|---|---|
| 1 | 000001 000009 000010 | train (contains val.zip scenes: 7b train / old val) |
| 2 | 000002 | train |
| 3 | 000003 000004 000005 000006 000007 000011 000012 | train (contains val.zip scenes: 7b train / old val) |
| 4 | 000008 | train |
| 5 | 000021 000022 000023 000024 000025 000026 000027 000028 000029 000030 | TEST group: test 000029, 000030; training.zip members EXCLUDED |
| 6 | 000034 000035 000036 000037 000038 000039 ... 000125 (36 scenes) | train (contains val.zip scenes: 7b train / old val) |
| 7 | 000074 000075 000076 000077 000078 000079 ... 000088 (15 scenes) | train (contains val.zip scenes: 7b train / old val) |
| 8 | 000099 000100 000101 | train |
| 9 | 000104 000105 | train |
| 10 | 000106 | train |
| 11 | 000109 000110 000111 | train (contains val.zip scenes: 7b train / old val) |
| 12 | 000114 | train |
| 13 | 000115 | train |
| 14 | 000126 000127 | train |
| 15 | 000128 000129 000130 000131 000132 000133 000134 000135 000136 000137 | TEST group: test 000136, 000137; training.zip members EXCLUDED |
| 16 | 000143 | train |
| 17 | 000144 | train |
| 18 | 000145 | train |
| 19 | 000146 000147 000148 | train |
| 20 | 000150 000151 000152 000153 | VALIDATION |
| 21 | 000155 | train |
| 22 | 000156 000157 | train |

## Candidate validation groups (2-6 training.zip scenes, no val.zip scene in the group)

| scenes | frames | range (m) | nearest scene outside the group (distance) | min distance to a test scene |
|---|---|---|---|---|
| 000099 000100 000101 | 278 | 199-310 | T000143 (9.1) | 15.8 |
| 000104 000105 | 200 | 202-251 | T000106 (2.9) | 22.4 |
| 000126 000127 | 146 | 474-500 | T000074 (6.7) | 24.2 |
| 000146 000147 000148 | 253 | 416-500 | V000010 (11.2) | 22.3 |
| 000150 000151 000152 000153 **(chosen)** | 331 | 371-494 | V000009 (9.2) | 26.3 |
| 000156 000157 | 111 | 277-499 | T000155 (3.8) | 4.2 |

Only 000150-000153 is a complete group of 4-6 scenes that has no other member; it is clearly separate from every other scene (the others in the table have 2-3 scenes). It adds 331 frames with ranges up to 494 m (the test scenes reach 433 m).

Excluded from training (test groups): 16 scenes: 000021 000022 000023 000024 000025 000026 000027 000028 000128 000129 000130 000131 000132 000133 000134 000135.
Scenes in the TEST groups that were not in the first list (000021-28, 000128-130, 000132): 000131 000133 000134 000135 (distance to test scene 000136/000137 1.0-1.2, closer than 000130 to 000137).
Training: 65 training.zip scenes + 16 Phase 7b training scenes + 4 old validation scenes.
