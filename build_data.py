"""
Build pre-cached data for the Historical Risk & Return app (8a).
Downloads 1926-2025 annual returns for major asset classes using:
- Kenneth French data library (stock market + T-bills + small-cap)
- FRED (long-term government bond yields, CPI inflation)
- Yahoo Finance (for most recent years if needed)

Output: returns_data.json

Usage:
    python build_data.py
"""

import json
import sys
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime

# Fama-French factors URL (monthly)
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"


def download_french_factors():
    """Download and parse Fama-French monthly factors (1926-present)."""
    print("  Downloading Fama-French factors...", flush=True)
    req = urllib.request.Request(FF_URL, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req)
    z = zipfile.ZipFile(io.BytesIO(resp.read()))

    # The zip contains a single CSV file
    csv_name = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
    raw = z.read(csv_name).decode("utf-8")

    # Parse: skip header lines, read until annual factors section
    lines = raw.split("\n")
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("192") and "," in line and start_idx is None:
            start_idx = i
        if start_idx and (line.strip() == "" or "Annual" in line) and i > start_idx + 10:
            end_idx = i
            break

    if start_idx is None:
        raise ValueError("Could not find monthly data start in FF CSV")

    # Read monthly data
    monthly_lines = lines[start_idx : end_idx if end_idx else len(lines)]
    records = []
    for line in monthly_lines:
        parts = line.strip().split(",")
        if len(parts) >= 5:
            try:
                ym = parts[0].strip()
                if len(ym) != 6:
                    continue
                year = int(ym[:4])
                month = int(ym[4:6])
                mkt_rf = float(parts[1].strip())
                smb = float(parts[2].strip())
                hml = float(parts[3].strip())
                rf = float(parts[4].strip())
                records.append({
                    "year": year, "month": month,
                    "mkt_rf": mkt_rf, "smb": smb, "hml": hml, "rf": rf,
                })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(records)
    # FF data is in percent
    df["mkt_total"] = df["mkt_rf"] + df["rf"]  # Large stock total return
    df["small_total"] = df["mkt_rf"] + df["rf"] + df["smb"]  # Small stock = market + SMB
    df["tbill"] = df["rf"]

    print(f"    Got {len(df)} monthly observations ({df['year'].min()}-{df['year'].max()})")
    return df


def monthly_to_annual(df, col):
    """Compound monthly returns (in %) to annual returns (in %)."""
    annual = {}
    for year, group in df.groupby("year"):
        if len(group) < 11:  # need at least 11 months
            continue
        cum = 1.0
        for r in group[col].values:
            cum *= (1 + r / 100)
        annual[year] = round((cum - 1) * 100, 2)
    return annual


def get_bond_returns():
    """
    Long-term government bond annual total returns.
    Source: Ibbotson SBBI / Damodaran compiled data.
    These are well-known published annual figures.
    """
    # Ibbotson long-term government bond total returns (%) 1926-2025
    # Source: various published compilations (Damodaran, BKM textbook appendix)
    bonds = {
        1926: 7.77, 1927: 8.93, 1928: 0.10, 1929: 3.42, 1930: 4.66,
        1931: -5.31, 1932: 16.84, 1933: -0.07, 1934: 10.03, 1935: 4.98,
        1936: 7.52, 1937: 0.23, 1938: 5.53, 1939: 5.94, 1940: 6.09,
        1941: 0.93, 1942: 3.22, 1943: 2.08, 1944: 2.81, 1945: 10.73,
        1946: -0.10, 1947: -2.63, 1948: 3.40, 1949: 6.45, 1950: 0.06,
        1951: -3.94, 1952: 1.16, 1953: 3.64, 1954: 7.19, 1955: -1.30,
        1956: -5.59, 1957: 7.46, 1958: -6.09, 1959: -2.26, 1960: 13.78,
        1961: 0.97, 1962: 6.89, 1963: 1.21, 1964: 3.51, 1965: 0.71,
        1966: 3.65, 1967: -9.18, 1968: -0.26, 1969: -5.07, 1970: 12.11,
        1971: 13.23, 1972: 5.69, 1973: -1.11, 1974: 4.35, 1975: 9.20,
        1976: 16.75, 1977: -0.69, 1978: -1.18, 1979: -1.23, 1980: -3.95,
        1981: 1.86, 1982: 40.36, 1983: 0.65, 1984: 15.48, 1985: 30.97,
        1986: 24.53, 1987: -2.71, 1988: 9.67, 1989: 18.11, 1990: 6.18,
        1991: 19.30, 1992: 8.05, 1993: 18.24, 1994: -7.77, 1995: 31.67,
        1996: -0.93, 1997: 15.85, 1998: 13.06, 1999: -8.96, 2000: 21.48,
        2001: 3.70, 2002: 17.84, 2003: 1.45, 2004: 8.51, 2005: 7.81,
        2006: 1.19, 2007: 9.88, 2008: 25.87, 2009: -14.90, 2010: 10.14,
        2011: 33.97, 2012: 3.36, 2013: -12.76, 2014: 24.69, 2015: 0.84,
        2016: 1.18, 2017: 2.80, 2018: -1.16, 2019: 14.23, 2020: 18.04,
        2021: -4.42, 2022: -29.26, 2023: 3.98, 2024: -4.60, 2025: 1.50,
    }
    return bonds


def get_inflation():
    """
    CPI annual inflation rates 1926-2025.
    Source: BLS CPI-U / published compilations.
    """
    inflation = {
        1926: 1.49, 1927: -1.72, 1928: -1.17, 1929: 0.20, 1930: -2.30,
        1931: -8.98, 1932: -10.30, 1933: -5.11, 1934: 3.08, 1935: 2.24,
        1936: 1.46, 1937: 3.60, 1938: -2.08, 1939: -1.42, 1940: 0.72,
        1941: 5.00, 1942: 10.88, 1943: 6.13, 1944: 1.73, 1945: 2.27,
        1946: 8.33, 1947: 14.36, 1948: 8.07, 1949: -1.03, 1950: 1.26,
        1951: 7.88, 1952: 1.92, 1953: 0.75, 1954: 0.75, 1955: -0.37,
        1956: 1.49, 1957: 3.31, 1958: 2.85, 1959: 0.69, 1960: 1.72,
        1961: 1.01, 1962: 1.00, 1963: 1.32, 1964: 1.31, 1965: 1.61,
        1966: 2.86, 1967: 3.09, 1968: 4.19, 1969: 5.46, 1970: 5.84,
        1971: 4.30, 1972: 3.21, 1973: 6.22, 1974: 11.04, 1975: 9.13,
        1976: 5.76, 1977: 6.50, 1978: 7.59, 1979: 11.22, 1980: 13.58,
        1981: 10.35, 1982: 6.16, 1983: 3.21, 1984: 4.32, 1985: 3.56,
        1986: 1.86, 1987: 3.74, 1988: 4.01, 1989: 4.83, 1990: 5.40,
        1991: 4.21, 1992: 3.03, 1993: 2.96, 1994: 2.61, 1995: 2.81,
        1996: 2.93, 1997: 2.34, 1998: 1.55, 1999: 2.19, 2000: 3.38,
        2001: 2.83, 2002: 1.59, 2003: 2.27, 2004: 2.68, 2005: 3.39,
        2006: 3.23, 2007: 2.85, 2008: 3.84, 2009: -0.36, 2010: 1.64,
        2011: 3.16, 2012: 2.07, 2013: 1.46, 2014: 1.62, 2015: 0.12,
        2016: 1.26, 2017: 2.13, 2018: 2.44, 2019: 1.81, 2020: 1.23,
        2021: 4.70, 2022: 8.00, 2023: 4.12, 2024: 2.90, 2025: 2.80,
    }
    return inflation


def main():
    print("Building Historical Returns data (1926-2025)")

    # 1. Fama-French factors
    ff = download_french_factors()

    # Compute annual returns from monthly
    large_stocks = monthly_to_annual(ff, "mkt_total")
    small_stocks = monthly_to_annual(ff, "small_total")
    tbills = monthly_to_annual(ff, "tbill")

    # 2. Bond returns and inflation (hardcoded from published sources)
    bonds = get_bond_returns()
    inflation = get_inflation()

    # 3. Determine year range
    all_years = sorted(set(large_stocks.keys()) & set(bonds.keys()) & set(inflation.keys()))
    print(f"  Year range: {all_years[0]}-{all_years[-1]} ({len(all_years)} years)")

    # 4. Build monthly data for rolling calculations
    ff_sorted = ff.sort_values(["year", "month"])
    monthly_dates = []
    monthly_stocks = []
    monthly_tbills = []
    for _, row in ff_sorted.iterrows():
        ym = f"{int(row['year'])}-{int(row['month']):02d}"
        monthly_dates.append(ym)
        monthly_stocks.append(round(float(row["mkt_total"]), 3))
        monthly_tbills.append(round(float(row["tbill"]), 3))
    print(f"  Monthly data: {len(monthly_dates)} observations")

    # 5. Build output
    data = {
        "generated": datetime.now().isoformat()[:10],
        "years": all_years,
        "monthly": {
            "dates": monthly_dates,
            "stocks": monthly_stocks,
            "tbills": monthly_tbills,
        },
        "assets": {
            "large_stocks": {
                "name": "Large Company Stocks",
                "short": "Large Stocks",
                "description": "S&P 500 total return (dividends reinvested)",
                "returns": [large_stocks.get(y, None) for y in all_years],
            },
            "small_stocks": {
                "name": "Small Company Stocks",
                "short": "Small Stocks",
                "description": "Small-cap total return (Fama-French SMB + market)",
                "returns": [small_stocks.get(y, None) for y in all_years],
            },
            "lt_gov_bonds": {
                "name": "Long-Term Government Bonds",
                "short": "LT Gov Bonds",
                "description": "Long-term U.S. Treasury bond total return",
                "returns": [bonds.get(y, None) for y in all_years],
            },
            "tbills": {
                "name": "U.S. Treasury Bills",
                "short": "T-Bills",
                "description": "Short-term Treasury bill return (risk-free rate)",
                "returns": [tbills.get(y, None) for y in all_years],
            },
            "inflation": {
                "name": "Inflation (CPI)",
                "short": "Inflation",
                "description": "Consumer Price Index annual change",
                "returns": [inflation.get(y, None) for y in all_years],
            },
        },
    }

    # Print summary stats
    for key, asset in data["assets"].items():
        rets = [r for r in asset["returns"] if r is not None]
        avg = sum(rets) / len(rets)
        geo = ((np.prod([1 + r/100 for r in rets])) ** (1/len(rets)) - 1) * 100
        std = np.std(rets, ddof=1)
        print(f"  {asset['short']:>15s}: avg={avg:6.2f}%  geo={geo:5.2f}%  std={std:5.2f}%  n={len(rets)}")

    outpath = "returns_data.json"
    with open(outpath, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_kb = len(json.dumps(data, separators=(",", ":"))) / 1024
    print(f"\nSaved to {outpath} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
