### What is the "latency of best fit" for each actuator across the entire dataset?

For this calculation I used numpy.correlate. It looks at the entire dataset minus a small window around firing (because firing is not normal motion.)


For pitch, the "latency of best fit" was **X seconds**. For yaw, the "latency of best fit" was **X seconds**. Very similar.

The script used in this calculation was scripts/bulk_latency_calculation.py.


