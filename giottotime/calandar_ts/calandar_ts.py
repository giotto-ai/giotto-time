import importlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

year = datetime.now().year


def holiday_ts(region="america", country="Brazil", start_date='01/01/2018',
               end_date='01/01/2020', kernel="none"):
    mod = importlib.import_module(f".{region}", f"workalendar")
    country_mod = getattr(mod, country)()

    index = pd.date_range(start=start_date, end=end_date, freq='D')

    years = index.year.unique()

    events = pd.DataFrame()

    for year in years:
        events = events.append(country_mod.holidays(year))

    events = events.rename(columns={0: "date", 1: "events"})
    events['date'] = pd.to_datetime(events['date'])
    events['status'] = 1
    events = events.set_index('date')

    events = events.reindex(index)

    events['events'] = events['events'].fillna("none")
    events['status'] = events['status'].fillna(0).astype(int)

    events = events.sort_index()

    klen = len(kernel)

    if kernel == "none":
        return events
    else:
        def ip(w):
            if sum(w) != 0:
                return np.dot(w, kernel) / sum(w)
            else:
                return 0

        events['status'] = events['status'].rolling(klen, center=True).apply(
            ip)

        return events


###########################################################################

if __name__ == "__main__":
    # kernel = list( range(1,11) ) + list( range(-10,1) )

    kernel = signal.parzen(20)  # list( range(1, 11) )

    e = holiday_ts(region="asia", country="Qatar", kernel=kernel)
    print(e)

    e.plot()
    plt.show()
