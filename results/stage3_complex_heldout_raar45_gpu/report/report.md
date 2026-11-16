# Post hoc: RAAR with shrinkwrap at equal wall time (45 starts), complex objects, n = 100

Reproduction of seeds 0-4: same selected start on 100/100 images; 500/500 starts within 0.1 dB.

| K | mean | median | failures | exact | images with an exact start |
|--:|--:|--:|--:|--:|--:|
| 1 | 22.76 | 13.89 | 65 | 17 | 17 |
| 2 | 25.35 | 14.75 | 51 | 19 | 19 |
| 5 | 31.47 | 18.40 | 29 | 25 | 25 |
| 10 | 36.62 | 33.19 | 17 | 33 | 33 |
| 15 | 38.56 | 37.36 | 13 | 36 | 36 |
| 20 | 39.68 | 41.36 | 10 | 38 | 39 |
| 30 | 41.57 | 53.36 | 9 | 41 | 42 |
| 45 | 43.91 | 55.79 | 6 | 47 | 48 |

FE-softmax: mean 17.67, failures 21. RAAR best45: mean 43.91, failures 6, exact 47.
Difference FE-softmax minus RAAR best45: mean -26.24 dB [-29.86, -22.63], median -37.88 dB; FE-softmax better on 9, RAAR on 91.
On the 6 RAAR failures FE-softmax was better on 6.

| t | RAAR used | mean | failures | exact |
|--:|--:|--:|--:|--:|
| 0.001 | 56 | 40.67 | 13 | 47 |
| 0.003 | 58 | 41.35 | 12 | 47 |
| 0.01 | 65 | 42.51 | 11 | 47 |
| 0.02 | 76 | 43.63 | 7 | 47 |
| 0.03 | 89 | 43.94 | 4 | 47 |
