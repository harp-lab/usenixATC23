Benchmark for random 1000000
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| random 1000000 | 1,000,000 | 30,511,908 | 3,456 x 512 | 0.118800 |

Initialization: 0.003516, Read: 0.000452
Hashtable rate: 905,114,949 keys/s, time: 0.001105
Join: 0.086287
Memory clear: 0.027440
Total: 0.118800

Benchmark for random 2000000
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| random 2000000 | 2,000,000 | 122,065,482 | 3,456 x 512 | 0.443422 |

Initialization: 0.005974, Read: 0.000872
Hashtable rate: 461,967,203 keys/s, time: 0.004329
Join: 0.325866
Memory clear: 0.106380
Total: 0.443422

Benchmark for random 3000000
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| random 3000000 | 3,000,000 | 274,633,933 | 3,456 x 512 | 0.986914 |

Initialization: 0.008218, Read: 0.001308
Hashtable rate: 307,958,636 keys/s, time: 0.009742
Join: 0.729421
Memory clear: 0.238226
Total: 0.986914

Benchmark for random 4000000
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| random 4000000 | 4,000,000 | 488,266,157 | 3,456 x 512 | 1.727381 |

Initialization: 0.010703, Read: 0.001733
Hashtable rate: 231,690,875 keys/s, time: 0.017264
Join: 1.283241
Memory clear: 0.414439
Total: 1.727381

Benchmark for random 5000000
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| random 5000000 | 5,000,000 | 762,962,523 | 3,456 x 512 | 2.640382 |

Initialization: 0.013112, Read: 0.002058
Hashtable rate: 236,218,801 keys/s, time: 0.021167
Join: 1.961875
Memory clear: 0.642171
Total: 2.640382

Benchmark for string 1000000
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| string 1000000 | 1,000,000 | 999,999 | 3,456 x 512 | 0.014831 |

Initialization: 0.005044, Read: 0.000463
Hashtable rate: 4,295,532,426 keys/s, time: 0.000233
Join: 0.006254
Memory clear: 0.002837
Total: 0.014831

Benchmark for string 2000000
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| string 2000000 | 2,000,000 | 1,999,999 | 3,456 x 512 | 0.022604 |

Initialization: 0.006602, Read: 0.000876
Hashtable rate: 4,132,777,725 keys/s, time: 0.000484
Join: 0.009639
Memory clear: 0.005003
Total: 0.022604

Benchmark for string 3000000
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| string 3000000 | 3,000,000 | 2,999,999 | 3,456 x 512 | 0.031009 |

Initialization: 0.009622, Read: 0.001289
Hashtable rate: 4,008,980,336 keys/s, time: 0.000748
Join: 0.012381
Memory clear: 0.006969
Total: 0.031009

Benchmark for string 4000000
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| string 4000000 | 4,000,000 | 3,999,999 | 3,456 x 512 | 0.038893 |

Initialization: 0.012064, Read: 0.001720
Hashtable rate: 3,901,494,725 keys/s, time: 0.001025
Join: 0.015481
Memory clear: 0.008604
Total: 0.038893

Benchmark for string 5000000
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| string 5000000 | 5,000,000 | 4,999,999 | 3,456 x 512 | 0.049386 |


Benchmark for CA-HepTh
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| CA-HepTh | 51,971 | 651,469 | 3,456 x 512 | 0.019241 |

Initialization: 0.001112, Read: 0.014293
Hashtable rate: 787,248,541 keys/s, time: 0.000066
Join: 0.002542
Memory clear: 0.001228
Total: 0.019241

Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 273,550 | 3,456 x 512 | 0.050398 |

Initialization: 0.001475, Read: 0.046141
Hashtable rate: 6,172,525,478 keys/s, time: 0.000036
Join: 0.001503
Memory clear: 0.001242
Total: 0.050398

Benchmark for ego-Facebook
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| ego-Facebook | 88,234 | 2,690,019 | 3,456 x 512 | 0.031749 |

Initialization: 0.001456, Read: 0.016832
Hashtable rate: 84,308,591 keys/s, time: 0.001047
Join: 0.009200
Memory clear: 0.003214
Total: 0.031749

Benchmark for wiki-Vote
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| wiki-Vote | 103,689 | 4,542,805 | 3,456 x 512 | 0.040988 |

Initialization: 0.001513, Read: 0.020214
Hashtable rate: 104,478,022 keys/s, time: 0.000992
Join: 0.013883
Memory clear: 0.004386
Total: 0.040988

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 108,864 | 3,456 x 512 | 0.008062 |

Initialization: 0.001233, Read: 0.005538
Hashtable rate: 519,761,010 keys/s, time: 0.000050
Join: 0.000628
Memory clear: 0.000614
Total: 0.008062

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 180,230 | 3,456 x 512 | 0.010563 |

Initialization: 0.001115, Read: 0.008123
Hashtable rate: 693,954,770 keys/s, time: 0.000058
Join: 0.000675
Memory clear: 0.000593
Total: 0.010563

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 19,836 | 3,456 x 512 | 0.006451 |

Initialization: 0.001136, Read: 0.004514
Hashtable rate: 1,089,881,402 keys/s, time: 0.000020
Join: 0.000156
Memory clear: 0.000625
Total: 0.006451

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 24,274 | 3,456 x 512 | 0.008955 |

Initialization: 0.001082, Read: 0.007070
Hashtable rate: 1,118,534,491 keys/s, time: 0.000021
Join: 0.000162
Memory clear: 0.000621
Total: 0.008955

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 7,445 | 3,456 x 512 | 0.002515 |

Initialization: 0.000729, Read: 0.001334
Hashtable rate: 339,265,062 keys/s, time: 0.000021
Join: 0.000151
Memory clear: 0.000280
Total: 0.002515

Benchmark for luxembourg_osm
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| luxembourg_osm | 119,666 | 114,532 | 3,456 x 512 | 0.026753 |

Initialization: 0.001132, Read: 0.024419
Hashtable rate: 4,769,850,424 keys/s, time: 0.000025
Join: 0.000575
Memory clear: 0.000602
Total: 0.026753

Benchmark for fe_sphere
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| fe_sphere | 49,152 | 146,350 | 3,456 x 512 | 0.011760 |

Initialization: 0.001112, Read: 0.009787
Hashtable rate: 1,898,640,364 keys/s, time: 0.000026
Join: 0.000223
Memory clear: 0.000612
Total: 0.011760

Benchmark for fe_body
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| fe_body | 163,734 | 609,957 | 3,456 x 512 | 0.037617 |

Initialization: 0.001493, Read: 0.032568
Hashtable rate: 4,016,238,050 keys/s, time: 0.000041
Join: 0.002352
Memory clear: 0.001163
Total: 0.037617

Benchmark for cti
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| cti | 48,232 | 130,492 | 3,456 x 512 | 0.011542 |

Initialization: 0.001113, Read: 0.009569
Hashtable rate: 1,970,261,485 keys/s, time: 0.000024
Join: 0.000215
Memory clear: 0.000620
Total: 0.011542

Benchmark for fe_ocean
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| fe_ocean | 409,593 | 1,175,076 | 3,456 x 512 | 0.091541 |

Initialization: 0.001828, Read: 0.083341
Hashtable rate: 7,028,984,494 keys/s, time: 0.000058
Join: 0.004178
Memory clear: 0.002134
Total: 0.091541

Benchmark for wing
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| wing | 121,544 | 116,371 | 3,456 x 512 | 0.034421 |

Initialization: 0.001120, Read: 0.032061
Hashtable rate: 3,712,854,416 keys/s, time: 0.000033
Join: 0.000602
Memory clear: 0.000605
Total: 0.034421

Benchmark for loc-Brightkite
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| loc-Brightkite | 214,078 | 3,368,451 | 3,456 x 512 | 0.099064 |

Initialization: 0.001538, Read: 0.083631
Hashtable rate: 1,575,214,818 keys/s, time: 0.000136
Join: 0.009898
Memory clear: 0.003862
Total: 0.099064

Benchmark for delaunay_n16
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| delaunay_n16 | 196,575 | 393,028 | 3,456 x 512 | 0.058860 |

Initialization: 0.001534, Read: 0.054110
Hashtable rate: 4,503,642,734 keys/s, time: 0.000044
Join: 0.001836
Memory clear: 0.001336
Total: 0.058860

Benchmark for usroads
----------------------------------------------------------

| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- |
| usroads | 165,435 | 206,898 | 3,456 x 512 | 0.037773 |

Initialization: 0.001466, Read: 0.033987
Hashtable rate: 5,706,229,503 keys/s, time: 0.000029
Join: 0.001409
Memory clear: 0.000883
Total: 0.037773
