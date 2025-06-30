import pandas as pd
import numpy as np

def year_fraction_exact(start: pd.Timestamp, end: pd.Timestamp):
    if start > end:
        raise ValueError("start date must be <= end date")
    if start.year == end.year:
        return (end - start).days / pd.Timestamp(f"{start.year}-12-31").dayofyear
    else:
        start_year_end = pd.Timestamp(f"{start.year}-12-31")
        days_in_start_year = pd.Timestamp(f"{start.year}-12-31").dayofyear
        start_frac = (start_year_end - start).days / days_in_start_year

        end_year_start = pd.Timestamp(f"{end.year}-01-01")
        days_in_end_year = pd.Timestamp(f"{end.year}-12-31").dayofyear
        end_frac = (end - end_year_start).days / days_in_end_year

        years_between = end.year - start.year - 1
        return start_frac + years_between + end_frac

def discounted_cf_exact(cf: pd.Series, rfr: pd.Series, valuation_date=None):
    if valuation_date is None:
        valuation_date = cf.index.min()
    else:
        valuation_date = pd.to_datetime(valuation_date)
    year_fractions = [
        year_fraction_exact(valuation_date, dt)
        for dt in cf.index
    ]
    rfr_interp = np.interp(year_fractions, rfr.index.astype(float), rfr.values)
    discounted = cf.values * np.exp(-rfr_interp * year_fractions)
    return pd.Series(discounted, index=cf.index)

# Data prep
cf = pd.Series(
    [1000, 1200, 1100],
    index=pd.to_datetime(['2026-02-15', '2027-04-15', '2028-07-20'])
)
rfr = pd.Series(
    [0.025, 0.03, 0.033, 0.035],
    index=[1, 2, 3, 4]
)

# Run function
discounted = discounted_cf_exact(cf, rfr, pd.Timestamp('2025-12-31'))
print(discounted)
print("Total PV:", discounted.sum())
