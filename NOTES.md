Significant motions at:
01:05, 02:14, 06:54, 07:22, ...

trigger/fire goes to 1 at 08:15.65, no data prior.
Then it stays at 1 for ~the rest of the dataset.
Trigger seems to have an effect from ~08:15.65 to 08:28.71

## Latency comparison between actuators

- I won't filter the data. The noise levels look low already and filtering would just introduce latency.

- Calculate latency between signals with cross-correlation (numpy.correlate)

- Throw out data around trigger/fire since that's not a normal motion.

## Motion magnitude vs latency

- Break each signal pair into, say, 0.5 second chunks

- Compute latency for each chunk

- Run a Fourier transform, get the primary signal frequency for each chunk

- Plot magnitude vs latency for each axis and all motions.

- Try a line of best fit, compute R^2, see if it looks good. If not, try a polynomial.

- For each motion chunk and each axis, calculate and save overshoot and settling time.

## Deflection when triggered

- Crop the data around trigger/fire going to 1

- Estimate magnitude of deflection

## Bonus

Make a nice plot of inaccuracy at 100 yards relative to drone

## Include in writeup

Harmonic vs planetary vs ...

Preload with FF torque prior to the shot

Write up methods
