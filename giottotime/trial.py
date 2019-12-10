from giottotime.feature_creation import CalendarFeature
import matplotlib.pyplot as plot
import pandas.util.testing as testing


if __name__ == "__main__":
    testing.N, testing.K = 200, 1
    X = testing.makeTimeDataFrame(freq="MS")
    cs = CalendarFeature(
        "america",
        "Brazil",
        kernel=list(range(1, 11)) + list(range(-10, 0)),
        output_name="fsfs",
    )
    # cs = CalendarFeature("america", "Brazil", kernel=None, output_name="fsfs")
    events = cs.transform()
    print(X)
    print(events)
    events.plot()
    plot.show()
