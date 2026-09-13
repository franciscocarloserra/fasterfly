# event_fused vs csr

| rate_hz | variant | steps/s | spikes | %/step |
|---|---|---|---|---|
| 100.0 | csr | 1608 | 59338 | 0.072 |
| 100.0 | event_fused | 3380 | 59338 | 0.072 |
| 100.0 | event_fused_batched | 2341 | 59338 | 0.072 |
| 1000.0 | csr | 1676 | 807820 | 0.978 |
| 1000.0 | event_fused | 3391 | 807820 | 0.978 |

## validation @ rate_hz=100.0
- scatter all_ones: max|diff|=1.801e-03 allclose=True
- scatter top1000_outdeg: max|diff|=1.144e-05 allclose=True
- scatter rand_1pct: max|diff|=4.768e-07 allclose=True
- scatter rand_50pct: max|diff|=5.951e-04 allclose=True
- raster diff: 0 events, 0 on DNp01/02/04/11

## validation @ rate_hz=1000.0
- scatter all_ones: max|diff|=1.816e-03 allclose=True
- scatter top1000_outdeg: max|diff|=9.537e-06 allclose=True
- scatter rand_1pct: max|diff|=4.768e-07 allclose=True
- scatter rand_50pct: max|diff|=6.180e-04 allclose=True
- raster diff: 0 events, 0 on DNp01/02/04/11
