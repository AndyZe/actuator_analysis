### What is the "latency of best fit" for each actuator across the entire dataset?

For this calculation I used numpy.correlate. It looks at the entire dataset minus a small window around firing (excluded because firing is not normal motion.)

For pitch, the "latency of best fit" was **0.2408 seconds**. For yaw, the "latency of best fit" was **0.2145 seconds**. Very similar.

The script used in this calculation was scripts/bulk_latency_calculation.py.

As a sanity check, here are graphs comparing the two signals for the first 100s.

![Pitch current and target except firing (first 100 s)](./graphics/pitch_current_and_target_except_firing_first_100s.png)

![Yaw current and target except firing (first 100 s)](./graphics/yaw_current_and_target_except_firing_first_100s.png)


I can tell from the pitch graph already that **pitch does not track slow position changes well**.

### How does latency vary with movement magnitude?

I used the frequency of the signal as a proxy for "magnitude", because I think the question is really asking about frequency response (slow motions vs fast motions). To grok this, I separated the data into 60-second segments (again, excluding a small window around firing because it's not normal motion). Then I calculated the spectral centroid of target position for each segment and the "latency of best fit." For latency of best fit, again, I used numpy.centroid. Thus I could plot latency vs signal frequency and check if it's linear.

This was implemented in scripts/latency_vs_signal_frequency.py.

The raw plot of latency vs signal centroid frequency is pretty noisy and has a lot of outliers, so it didn't tell me much initially. Here it is:

![Latency vs signal centroid frequency](./graphics/latency_vs_signal_frequency.png)

Attempting to make better sense of the data, I removed outliers beyond 1.5 standard deviations. From eye-balling the graph, the latency is actually worse at lower signal frequencies. Your PID controllers do not track well at low frequencies, as already mentioned before. They seem to be tuned more for high-frequency response, i.e. tuned for better disturbance rejection I suppose.

![Latency vs signal centroid frequency (outliers removed)](./graphics/latency_vs_signal_frequency_without_outliers.png)

I added lines of best fit just because your questions asked if the trend is linear, but it's clear from a quick glance that it's not. Latency is much worse for low-frequency data. Latency is low and relatively constant for high-frequency tracking. The R^2 value is very low, less than 0.1.
