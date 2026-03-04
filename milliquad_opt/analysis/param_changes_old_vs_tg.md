# Parameter Changes: Old vs Terrain-Gated Runs

Bold = >2× change or >50% shift.

## Flat → Flat TG

| Parameter | Old | New | Change |
|---|---|---|---|
| friction (sliding) | 0.368 | 0.493 | <span style="color:green">+34%</span> |
| friction (torsional) | 1.40e-4 | 6.11e-4 | <span style="color:green">**+4.4×**</span> |
| friction (rolling) | 4.30e-6 | 1.11e-6 | <span style="color:red">**-74%**</span> |
| solref (timeconst) | 1.36e-3 | 1.32e-3 | <span style="color:red">-3%</span> |
| solref (dampratio) | 6.08 | 2.74 | <span style="color:red">**-55%**</span> |
| solimp (dmin) | 0.231 | 0.756 | <span style="color:green">**+3.3×**</span> |
| solimp (dmax) | 0.944 | 0.927 | <span style="color:red">-2%</span> |
| solimp (width) | 2.90e-5 | 2.26e-5 | <span style="color:red">-22%</span> |
| solimp (midpoint) | 0.534 | 0.697 | <span style="color:green">+31%</span> |
| solimp (power) | 5.08 | 4.93 | <span style="color:red">-3%</span> |
| moment_fudge | 0.881 | 0.522 | <span style="color:red">**-41%**</span> |
| field_fudge | 1.04 | 1.49 | <span style="color:green">+43%</span> |
| damping | 3.40e-10 | 3.60e-10 | <span style="color:green">+6%</span> |
| noslip_iterations | 0 | 1 | <span style="color:green">0→1</span> |
| noslip_tolerance | 1.00e-6 | 7.57e-6 | <span style="color:green">**+7.6×**</span> |
| o_margin | 1.70e-4 | 3.03e-4 | <span style="color:green">**+78%**</span> |

## Step → Step 065

| Parameter | Old | New | Change |
|---|---|---|---|
| friction (sliding) | 0.358 | 0.734 | <span style="color:green">**+2.0×**</span> |
| friction (torsional) | 1.70e-3 | 2.72e-4 | <span style="color:red">**-84%**</span> |
| friction (rolling) | 2.70e-5 | 2.09e-6 | <span style="color:red">**-92%**</span> |
| solref (timeconst) | 7.28e-3 | 2.42e-3 | <span style="color:red">**-67%**</span> |
| solref (dampratio) | 2.38 | 1.53 | <span style="color:red">-36%</span> |
| solimp (dmin) | 0.432 | 0.312 | <span style="color:red">-28%</span> |
| solimp (dmax) | 0.836 | 0.850 | <span style="color:green">+2%</span> |
| solimp (width) | 2.00e-4 | 1.53e-5 | <span style="color:red">**-92%**</span> |
| solimp (midpoint) | 0.366 | 0.925 | <span style="color:green">**+2.5×**</span> |
| solimp (power) | 4.14 | 4.85 | <span style="color:green">+17%</span> |
| moment_fudge | 0.797 | 0.907 | <span style="color:green">+14%</span> |
| field_fudge | 0.802 | 0.802 | 0% |
| damping | 9.30e-10 | 2.98e-10 | <span style="color:red">**-68%**</span> |
| noslip_iterations | 30 | 30 | 0% |
| noslip_tolerance | 1.50e-5 | 1.99e-5 | <span style="color:green">+33%</span> |
| o_margin | 2.70e-4 | 5.74e-4 | <span style="color:green">**+2.1×**</span> |

## Rough → Rough TG

| Parameter | Old | New | Change |
|---|---|---|---|
| friction (sliding) | 0.504 | 0.640 | <span style="color:green">+27%</span> |
| friction (torsional) | 1.10e-4 | 2.35e-5 | <span style="color:red">**-79%**</span> |
| friction (rolling) | 1.90e-6 | 1.48e-6 | <span style="color:red">-22%</span> |
| solref (timeconst) | 4.30e-4 | 8.33e-4 | <span style="color:green">**+94%**</span> |
| solref (dampratio) | 3.03 | 3.67 | <span style="color:green">+21%</span> |
| solimp (dmin) | 0.321 | 0.282 | <span style="color:red">-12%</span> |
| solimp (dmax) | 0.521 | 0.508 | <span style="color:red">-2%</span> |
| solimp (width) | 3.60e-5 | 1.58e-5 | <span style="color:red">**-56%**</span> |
| solimp (midpoint) | 0.787 | 0.814 | <span style="color:green">+3%</span> |
| solimp (power) | 5.50 | 5.43 | <span style="color:red">-1%</span> |
| moment_fudge | 1.14 | 0.853 | <span style="color:red">**-25%**</span> |
| field_fudge | 1.03 | 1.50 | <span style="color:green">**+46%**</span> |
| damping | 7.30e-10 | 7.65e-10 | <span style="color:green">+5%</span> |
| noslip_iterations | 0 | 0 | 0% |
| noslip_tolerance | 1.60e-6 | 1.01e-6 | <span style="color:red">-37%</span> |
| o_margin | 2.90e-6 | 3.00e-5 | <span style="color:green">**+10×**</span> |

## Summary

- **Step had the most drastic changes** (6 bold entries) — torsional/rolling/width all dropped ~90%, midpoint jumped 2.5×
- **Flat** reshuffled moment/field fudge and dmin heavily, but dmax/timeconst/power stayed put
- **Rough was most stable** — mostly <30% shifts, except torsional (-79%), margin (+10×), and the moment↔field fudge rebalance
- **field_fudge × moment_fudge product**: Flat 0.92→0.78, Step 0.64→0.73, Rough 1.17→1.28 — effective torque scaling shifted modestly despite individual params moving a lot
