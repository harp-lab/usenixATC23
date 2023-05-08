Benchmark for CA-HepTh
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| CA-HepTh | 51,971 | 74,619,885 | 18 | 3,456 x 512 | 2.7330 |


Initialization: 0.0015, Read: 0.0118
Hashtable rate: 814,209,619 keys/s, time: 0.0001
Join: 0.5877
Deduplication: 1.8560 (sort: 1.6606, unique: 0.1954)
Memory clear: 0.1673
Union: 0.1086 (merge: 0.0541)
Total: 2.7330

Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 512 | 11.4015 |


Initialization: 0.0059, Read: 0.0505
Hashtable rate: 7,104,205,160 keys/s, time: 0.0000
Join: 1.7366
Deduplication: 3.3479 (sort: 2.0922, unique: 1.2557)
Memory clear: 1.0063
Union: 5.2542 (merge: 0.7313)
Total: 11.4015

Benchmark for ego-Facebook
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| ego-Facebook | 88,234 | 2,508,102 | 17 | 3,456 x 512 | 0.5368 |


Initialization: 0.0069, Read: 0.0179
Hashtable rate: 112,210,171 keys/s, time: 0.0008
Join: 0.1389
Deduplication: 0.2715 (sort: 0.2213, unique: 0.0503)
Memory clear: 0.0390
Union: 0.0619 (merge: 0.0231)
Total: 0.5368

Benchmark for wiki-Vote
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| wiki-Vote | 103,689 | 11,947,132 | 10 | 3,456 x 512 | 1.1121 |


Initialization: 0.0048, Read: 0.0231
Hashtable rate: 131,013,421 keys/s, time: 0.0008
Join: 0.2541
Deduplication: 0.7433 (sort: 0.6787, unique: 0.0647)
Memory clear: 0.0509
Union: 0.0351 (merge: 0.0153)
Total: 1.1121

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 512 | 0.6860 |


Initialization: 0.0040, Read: 0.0063
Hashtable rate: 697,007,046 keys/s, time: 0.0000
Join: 0.1693
Deduplication: 0.3553 (sort: 0.2739, unique: 0.0814)
Memory clear: 0.0787
Union: 0.0725 (merge: 0.0337)
Total: 0.6860

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 512 | 1.8779 |


Initialization: 0.0041, Read: 0.0084
Hashtable rate: 719,899,198 keys/s, time: 0.0001
Join: 0.4322
Deduplication: 1.1298 (sort: 0.9703, unique: 0.1594)
Memory clear: 0.1496
Union: 0.1537 (merge: 0.0675)
Total: 1.8779

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 512 | 0.4571 |


Initialization: 0.0046, Read: 0.0053
Hashtable rate: 1,358,359,423 keys/s, time: 0.0000
Join: 0.0185
Deduplication: 0.0232 (sort: 0.0077, unique: 0.0155)
Memory clear: 0.1392
Union: 0.2662 (merge: 0.0063)
Total: 0.4571

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 512 | 0.1657 |


Initialization: 0.0031, Read: 0.0053
Hashtable rate: 1,437,153,864 keys/s, time: 0.0000
Join: 0.0078
Deduplication: 0.0314 (sort: 0.0246, unique: 0.0068)
Memory clear: 0.0455
Union: 0.0726 (merge: 0.0021)
Total: 0.1657

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 512 | 0.0975 |


Initialization: 0.0023, Read: 0.0021
Hashtable rate: 463,774,803 keys/s, time: 0.0000
Join: 0.0074
Deduplication: 0.0098 (sort: 0.0050, unique: 0.0048)
Memory clear: 0.0331
Union: 0.0428 (merge: 0.0020)
Total: 0.0975

Benchmark for luxembourg_osm
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| luxembourg_osm | 119,666 | 5,022,084 | 426 | 3,456 x 512 | 1.2934 |


Initialization: 0.0018, Read: 0.0240
Hashtable rate: 6,431,926,901 keys/s, time: 0.0000
Join: 0.0437
Deduplication: 0.1015 (sort: 0.0322, unique: 0.0694)
Memory clear: 0.3523
Union: 0.7701 (merge: 0.0371)
Total: 1.2934

Benchmark for fe_sphere
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| fe_sphere | 49,152 | 78,557,912 | 188 | 3,456 x 512 | 13.2685 |


Initialization: 0.0016, Read: 0.0101
Hashtable rate: 2,575,155,865 keys/s, time: 0.0000
Join: 2.9783
Deduplication: 7.6862 (sort: 6.4548, unique: 1.2314)
Memory clear: 0.8564
Union: 1.7359 (merge: 0.5120)
Total: 13.2685

Benchmark for fe_body
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| fe_body | 163,734 | 156,120,489 | 188 | 3,456 x 512 | 47.7355 |


Initialization: 0.0084, Read: 0.0337
Hashtable rate: 5,097,412,907 keys/s, time: 0.0000
Join: 9.8148
Deduplication: 34.1434 (sort: 31.4644, unique: 2.6790)
Memory clear: 1.6269
Union: 2.1083 (merge: 0.9378)
Total: 47.7355

Benchmark for cti
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cti | 48,232 | 6,859,653 | 53 | 3,456 x 512 | 0.4146 |


Initialization: 0.0087, Read: 0.0168
Hashtable rate: 2,551,147,783 keys/s, time: 0.0000
Join: 0.0968
Deduplication: 0.0935 (sort: 0.0624, unique: 0.0311)
Memory clear: 0.0808
Union: 0.1180 (merge: 0.0192)
Total: 0.4146

Benchmark for fe_ocean
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| fe_ocean | 409,593 | 1,669,750,513 | 247 | 3,456 x 512 | 140.4886 |


Initialization: 0.0136, Read: 0.0819
Hashtable rate: 8,947,571,924 keys/s, time: 0.0000
Join: 1.7637
Deduplication: 8.3061 (sort: 1.4520, unique: 6.8541)
Memory clear: 23.3293
Union: 106.9938 (merge: 5.1659)
Total: 140.4886

Benchmark for wing
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| wing | 121,544 | 329,438 | 11 | 3,456 x 512 | 0.0703 |


Initialization: 0.0242, Read: 0.0247
Hashtable rate: 4,850,506,824 keys/s, time: 0.0000
Join: 0.0012
Deduplication: 0.0030 (sort: 0.0022, unique: 0.0009)
Memory clear: 0.0088
Union: 0.0083 (merge: 0.0004)
Total: 0.0703

Benchmark for loc-Brightkite
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| loc-Brightkite | 214,078 | 138,269,412 | 24 | 3,456 x 512 | 15.8365 |


Initialization: 0.0023, Read: 0.0419
Hashtable rate: 1,856,253,468 keys/s, time: 0.0001
Join: 3.2291
Deduplication: 11.9419 (sort: 11.3281, unique: 0.6137)
Memory clear: 0.4026
Union: 0.2187 (merge: 0.0992)
Total: 15.8365

Benchmark for delaunay_n16
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| delaunay_n16 | 196,575 | 6,137,959 | 101 | 3,456 x 512 | 1.1289 |


Initialization: 0.0126, Read: 0.0389
Hashtable rate: 5,796,278,822 keys/s, time: 0.0000
Join: 0.2728
Deduplication: 0.3800 (sort: 0.1636, unique: 0.2165)
Memory clear: 0.1878
Union: 0.2367 (merge: 0.0862)
Total: 1.1289

Benchmark for usroads
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| usroads | 165,435 | 871,365,688 | 606 | 3,456 x 512 | 363.0640 |


Initialization: 0.0023, Read: 0.0332
Hashtable rate: 7,495,242,841 keys/s, time: 0.0000
Join: 46.5787
Deduplication: 115.8058 (sort: 98.1751, unique: 17.6307)
Memory clear: 27.3912
Union: 173.2528 (merge: 8.9575)
Total: 363.0640
