# Post hoc: RAAR with shrinkwrap at equal wall time (45 starts), complex objects, n = 100

Reproduction of seeds 0-4: same selected start on 51/100 images; 109/500 starts within 0.1 dB.

| K | mean | median | failures | exact | images with an exact start |
|--:|--:|--:|--:|--:|--:|
| 1 | 20.60 | 13.73 | 67 | 13 | 13 |
| 2 | 23.83 | 14.71 | 52 | 18 | 18 |
| 5 | 28.90 | 17.47 | 35 | 21 | 21 |
| 10 | 33.95 | 24.11 | 25 | 29 | 29 |
| 15 | 36.95 | 33.19 | 16 | 33 | 33 |
| 20 | 37.43 | 34.10 | 11 | 33 | 34 |
| 30 | 41.34 | 51.07 | 11 | 42 | 43 |
| 45 | 43.36 | 54.61 | 5 | 46 | 47 |

FE-softmax: mean 17.67, failures 21. RAAR best45: mean 43.36, failures 5, exact 46.
Difference FE-softmax minus RAAR best45: mean -25.68 dB [-29.28, -22.09], median -37.27 dB; FE-softmax better on 10, RAAR on 90.
On the 5 RAAR failures FE-softmax was better on 4.

| t | RAAR used | mean | failures | exact |
|--:|--:|--:|--:|--:|
| 0.001 | 54 | 39.74 | 14 | 46 |
| 0.003 | 57 | 40.71 | 13 | 46 |
| 0.01 | 63 | 41.69 | 12 | 46 |
| 0.02 | 76 | 43.05 | 9 | 46 |
| 0.03 | 93 | 43.33 | 4 | 46 |
