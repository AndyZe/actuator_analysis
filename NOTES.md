Recording starts at 2025-11-05 18:00:48.0, ends at ~ 2025-11-05 20:51:40

Significant motions at:
18:01:05, 18:02:14, 18:06:54, 18:07:22, ...

trigger/fire goes to 1 at 18:08:15.65, no data prior.
Then it stays at 1 for ~the rest of the dataset.
Trigger seems to have an effect from ~18:08:15.65 to 18:08:28.71

## Latency comparison between actuators

- [x] Calculate latency between signals with cross-correlation (numpy.correlate)

- [x] Throw out data around trigger/fire since that's not a normal motion.

## Motion magnitude vs latency

- [x] Break each signal pair into, say, 1 minute chunks

- [x] Compute latency for each chunk

- [x] Run an FFT, get the primary signal frequency energy for each chunk

- [x] Plot frequency vs latency for each axis and all motions.

- [x] Try a line of best fit, compute R^2, see if it looks good. If not, try a polynomial.

- [ ] For each motion chunk and each axis, calculate and save overshoot and settling time.

## Deflection when triggered

- [ ] Crop the data around trigger/fire going to 1

- [ ] Estimate magnitude of deflection

## Bonus

Make a nice plot of inaccuracy at 100 yards relative to drone

## Include in writeup

Harmonic vs planetary vs ...

I didn't filter the data. The noise levels look low already and filtering would just introduce latency.

It looks like the signals aren't perfectly synchronized. During firing, the /pitch/target plot jumps prior to trigger/fire going high.

Preload with FF torque prior to the shot

Write up methods

Mention subtracting a window around firing for the actuator latency comparison

Velocity vs position control

Since your PID controllers don't track low-frequency commands well: add a bit of integral gain. Compensate for gravity, inertia, Coriolis, and centripetal forces with a dynamic model. Or, you may not care about low-frequency tracking.

Use actuator brakes to stabilize during firing.
