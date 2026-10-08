"""
Synthetic data generator for the Forecasting & S&OP Planning project.

Everything here is FICTIONAL. It simulates "Meridian Systems", a made-up hardware company with six
product lines, and builds the kinds of inputs a demand-planning / S&OP team actually works with:

  * sales history (orders = unconstrained demand, shipments = supply-constrained)
  * product master + lifecycle (launches, end-of-life, last-time-buy)
  * promotions, marketing plan, competitor signals, macro indicators
  * supply-chain and operations signals (lead time, allocation, supplier OTD, yield, capacity)
  * a leading indicator (sales pipeline / web traffic) and a legacy "sales-submitted" forecast

The data-generating process is documented in docs/data_dictionary.md. The effect sizes are modelling
ASSUMPTIONS chosen to be plausible; they are not measurements from any real company.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_MONTHS = 72                      # Jan-2020 .. Dec-2025 (observed)
N_EXTRA = 3                        # extra simulated months so forward-looking signals can be built
START = "2020-01-01"
TIERS = ["Entry", "Mainstream", "Premium"]
TIER_W = np.array([0.50, 0.35, 0.15])

# ----------------------------------------------------------------------------------------------
# Line definitions (fictional)
# ----------------------------------------------------------------------------------------------
LINES = {
    "Client PCs": dict(
        code="PC", total0=18000, asp=(650, 1000, 1800), cogs=0.80, lead=10, consumer=True,
        season=[0.85, 0.82, 0.90, 0.88, 0.90, 0.95, 1.00, 1.15, 1.15, 1.00, 1.15, 1.25],
        vol=0.10, deal_p=0.00, promo_p=0.20, promo_qe=0.0, promo_depth=(5, 15), promo_el=2.5,
        shortage_sens=0.8, price_el=1.0, macro_el=0.6, mkt_pct=0.04, li_sd=0.12),
    "Accessories & Peripherals": dict(
        code="AC", total0=30000, asp=(80, 140, 260), cogs=0.70, lead=6, consumer=True,
        season=[0.88, 0.85, 0.92, 0.92, 0.95, 0.95, 1.00, 1.10, 1.05, 1.00, 1.20, 1.18],
        vol=0.09, deal_p=0.00, promo_p=0.30, promo_qe=0.0, promo_depth=(5, 20), promo_el=2.5,
        shortage_sens=0.4, price_el=1.0, macro_el=0.6, mkt_pct=0.04, li_sd=0.12),
    "Network Switches": dict(
        code="NW", total0=1800, asp=(1500, 4500, 12000), cogs=0.62, lead=14, consumer=False,
        season=[0.88, 0.90, 1.10, 0.92, 0.95, 1.08, 0.95, 0.97, 1.08, 1.00, 1.02, 1.15],
        vol=0.12, deal_p=0.02, promo_p=0.08, promo_qe=0.25, promo_depth=(3, 10), promo_el=1.2,
        shortage_sens=1.0, price_el=0.5, macro_el=0.8, mkt_pct=0.06, li_sd=0.10),
    "Servers": dict(
        code="SV", total0=900, asp=(6000, 15000, 45000), cogs=0.72, lead=16, consumer=False,
        season=[0.80, 0.85, 1.15, 0.85, 0.90, 1.15, 0.90, 0.92, 1.20, 0.95, 1.00, 1.30],
        vol=0.14, deal_p=0.05, promo_p=0.06, promo_qe=0.20, promo_depth=(3, 10), promo_el=1.2,
        shortage_sens=1.0, price_el=0.5, macro_el=0.8, mkt_pct=0.06, li_sd=0.10),
    "Storage Arrays": dict(
        code="ST", total0=260, asp=(40000, 110000, 300000), cogs=0.55, lead=14, consumer=False,
        season=[0.78, 0.82, 1.20, 0.80, 0.88, 1.18, 0.85, 0.90, 1.18, 0.92, 1.00, 1.35],
        vol=0.16, deal_p=0.06, promo_p=0.05, promo_qe=0.20, promo_depth=(3, 10), promo_el=1.2,
        shortage_sens=0.6, price_el=0.5, macro_el=0.8, mkt_pct=0.06, li_sd=0.10),
    "Edge Sensors & Modules": dict(
        code="SN", total0=1200, asp=(400, 1200, 3500), cogs=0.65, lead=20, consumer=False,
        season=[0.90, 0.95, 1.05, 1.00, 1.05, 1.00, 0.95, 0.95, 1.05, 1.05, 1.00, 1.05],
        vol=0.15, deal_p=0.05, promo_p=0.03, promo_qe=0.0, promo_depth=(3, 8), promo_el=1.0,
        shortage_sens=0.5, price_el=0.5, macro_el=0.8, mkt_pct=0.05, li_sd=0.15),
}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def gauss(t, c, s):
    return np.exp(-0.5 * ((t - c) / s) ** 2)


def sig(t, c, w):
    return 1.0 / (1.0 + np.exp(-(t - c) / w))


def month_label(t: int) -> pd.Timestamp:
    return pd.Timestamp(START) + pd.DateOffset(months=int(t))


def line_structure(code: str, t: np.ndarray) -> dict:
    """Underlying (pre-seasonality) demand path + the known event index for a line."""
    event_known = np.zeros_like(t, dtype=float)
    if code == "PC":
        wfh = 0.35 * sig(t, 4, 2) * (1 - sig(t, 26, 4))
        hang = -0.20 * gauss(t, 38, 6)
        refresh = 0.30 * gauss(t, 67, 4)               # OS end-of-support refresh (announced, so "known")
        level = np.exp(-0.02 * t / 12) * (1 + wfh + hang + refresh)
        event_known = refresh / 0.30
    elif code == "AC":
        wfh = 0.35 * sig(t, 4, 2) * (1 - sig(t, 26, 4))
        hang = -0.20 * gauss(t, 38, 6)
        refresh = 0.30 * gauss(t, 67, 4)
        level = np.exp(0.03 * t / 12) * (1 + 0.6 * (wfh + hang + refresh))
        event_known = refresh / 0.30 * 0.6
    elif code == "NW":
        level = np.exp(0.03 * t / 12) * (1 + 0.70 * sig(t, 46, 8)) * (1 - 0.08 * gauss(t, 4, 3))
    elif code == "SV":
        level = np.exp(0.05 * t / 12) * (1 + 1.10 * sig(t, 48, 8)) * (1 - 0.10 * gauss(t, 4, 3))
    elif code == "ST":
        level = np.exp(0.04 * t / 12) * (1 + 0.55 * sig(t, 50, 9)) * (1 - 0.08 * gauss(t, 4, 3))
    else:  # SN: step-ups when new fleets / cities come online (customer-driven)
        steps = np.ones_like(t, dtype=float)
        for c, a in [(18, 0.15), (34, 0.25), (52, 0.35), (63, 0.30)]:
            steps *= 1 + a * sig(t, c, 0.8)
        level = np.exp(0.10 * t / 12) * steps
    return dict(level=level, event_known=event_known)


# ----------------------------------------------------------------------------------------------
# main generator
# ----------------------------------------------------------------------------------------------
def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    T = N_MONTHS + N_EXTRA
    t = np.arange(T, dtype=float)
    cal_month = np.array([month_label(i).month for i in range(T)])
    dates = [month_label(i) for i in range(T)]

    # ---- global drivers -------------------------------------------------------------------
    shortage = np.interp(t, [0, 14, 20, 28, 34, T], [0, 0, 1, 1, 0, 0])        # component shortage
    digestion = gauss(t, 37, 2.5)                                              # post-shortage inventory digestion

    def ar_series(level_fn, rho, sd):
        x = np.zeros(T)
        x[0] = level_fn(0)
        for i in range(1, T):
            x[i] = rho * x[i - 1] + (1 - rho) * level_fn(i) + rng.normal(0, sd)
        return x

    macro_it = ar_series(lambda i: 100 + 0.15 * i - 6 * gauss(i, 4, 2), 0.92, 1.0)
    macro_cons = ar_series(lambda i: 100 + 0.05 * i - 6 * gauss(i, 4, 2) - 8 * gauss(i, 32, 5), 0.92, 1.0)
    commodity = ar_series(lambda i: 100 + 45 * shortage[i] + 35 * sig(i, 64, 2), 0.85, 2.0)

    # ---- product master ---------------------------------------------------------------------
    sku_rows, sku_params = [], {}
    for line, d in LINES.items():
        eol1 = rng.integers(56, 71, size=3)                       # gen-1 end-of-life month index (some inside 2025)
        launch2 = np.minimum(rng.integers(38, 58, size=3), eol1 - 4)
        for k, tier in enumerate(TIERS):
            for gen in (1, 2):
                sku = f"{d['code']}-{tier[0]}{gen}"
                launch = -24 if gen == 1 else int(launch2[k])
                eol = int(eol1[k]) if gen == 1 else 999
                sku_rows.append(dict(
                    sku=sku, product_line=line, tier=tier, generation=gen,
                    launch_idx=launch, eol_idx=eol,
                    launch_date=month_label(launch).date(),
                    eol_date=(month_label(eol).date() if eol < 999 else pd.NaT),
                    asp_usd=d["asp"][k] * (1.0 if gen == 1 else 1.08),
                    unit_cost_usd=d["asp"][k] * (1.0 if gen == 1 else 1.08) * d["cogs"],
                    base_lead_time_weeks=d["lead"], moq=int(max(1, round(d["total0"] * TIER_W[k] / 40))),
                    supplier=f"Supplier-{d['code']}-{1 + (k % 2)}", cm_site=["CM-Penang", "CM-Shenzhen", "CM-Austin"][(k + gen) % 3],
                ))
                sku_params[sku] = dict(line=line, k=k, gen=gen, launch=launch, eol=eol)
    sku_master = pd.DataFrame(sku_rows)

    # ---- per-line external series -----------------------------------------------------------------
    comp_rows, mkt_rows, sup_rows, ops_rows, ev_rows = [], [], [], [], []
    line_ext = {}
    for line, d in LINES.items():
        code = d["code"]
        # competitor price war + launches
        n_launch = 3
        launch_t = np.sort(rng.choice(np.arange(10, N_MONTHS - 3), size=n_launch, replace=False))
        comp_launch = np.zeros(T)
        comp_effect = np.ones(T)
        for lt in launch_t:
            comp_launch[lt] = 1
            after = t >= lt
            comp_effect[after] *= 1 - 0.07 * np.exp(-(t[after] - lt) / 2.5)
            ev_rows.append(dict(date=month_label(int(lt)).date(), event="Competitor product launch", product_line=line, known_in_advance=False))
        comp_px = 100 * np.exp(np.cumsum(rng.normal(-0.0015, 0.010, T)))
        for wt in rng.choice(np.arange(8, N_MONTHS - 2), size=2, replace=False):
            comp_px[wt:] *= 0.95                                   # price-war step-downs
            ev_rows.append(dict(date=month_label(int(wt)).date(), event="Competitor price cut", product_line=line, known_in_advance=False))
        our_px = 100 * np.exp(np.cumsum(rng.normal(-0.0020, 0.005, T)))
        rel_px = our_px / comp_px
        price_effect = np.clip(rel_px, 0.7, 1.4) ** (-d["price_el"])
        # competitor share (analyst data, quarterly, 1-quarter lag)
        share_true = 30 + 1.5 * np.convolve(comp_launch, np.exp(-np.arange(8) / 3.0))[:T] + 8 * (rel_px - 1) + rng.normal(0, 0.4, T)
        qstart = (np.arange(T) // 3) * 3
        share_obs = np.array([share_true[max(0, q - 3)] for q in qstart]) + rng.normal(0, 0.3, T)

        macro = macro_cons if d["consumer"] else macro_it
        macro_effect = (macro / 100.0) ** d["macro_el"]

        # known events
        st = line_structure(code, t)
        ext = dict(comp_launch=comp_launch, comp_effect=comp_effect, comp_px=comp_px, our_px=our_px, rel_px=rel_px,
                   price_effect=price_effect, share_obs=share_obs, macro=macro, macro_effect=macro_effect,
                   level=st["level"], event_known=st["event_known"])
        # supply side
        sens = d["shortage_sens"]
        alloc = np.clip(1 - 0.30 * shortage * sens + rng.normal(0, 0.01, T), 0.5, 1.0)
        lead = d["lead"] * (1 + 0.9 * shortage * sens) + rng.normal(0, 0.4, T)
        otd = np.clip(93 - 18 * shortage * sens + rng.normal(0, 1.2, T), 55, 99)
        ext.update(alloc=alloc, lead=lead, otd=otd)
        line_ext[line] = ext
        for i in range(N_MONTHS):
            comp_rows.append(dict(date=dates[i].date(), product_line=line, competitor_price_index=round(comp_px[i], 2),
                                  our_price_index=round(our_px[i], 2), relative_price=round(rel_px[i], 4),
                                  competitor_launch_flag=int(comp_launch[i]), competitor_share_pct_observed=round(share_obs[i], 2)))
    for ev, tt in [("Component shortage begins", 16), ("Component shortage eases", 34)]:
        ev_rows.append(dict(date=month_label(tt).date(), event=ev, product_line="All", known_in_advance=False))
    ev_rows.append(dict(date=month_label(67).date(), event="OS end-of-support refresh peak (PCs, Accessories)", product_line="Client PCs / Accessories", known_in_advance=True))

    # ---- SKU-level demand -----------------------------------------------------------------------
    sku_ids = sku_master["sku"].tolist()
    mu = {s: np.zeros(T) for s in sku_ids}               # structural mean (no noise)
    mu_noinfl = {s: np.zeros(T) for s in sku_ids}
    orders = {s: np.zeros(T) for s in sku_ids}
    promo_flag = {s: np.zeros(T, dtype=int) for s in sku_ids}
    promo_depth = {s: np.zeros(T) for s in sku_ids}
    mkt_mult_line = {}
    for line, d in LINES.items():
        ext = line_ext[line]
        code = d["code"]
        seas = np.array(d["season"]) / np.mean(d["season"])
        season_t = seas[cal_month - 1]
        # marketing campaign multiplier: trade-show months + launch campaigns
        mm = np.ones(T)
        mm[np.isin(cal_month, [3, 10])] *= 1.15
        for s in [x for x in sku_ids if sku_params[x]["line"] == line and sku_params[x]["gen"] == 2]:
            l = sku_params[s]["launch"]
            for dm, m in [(-1, 1.5), (0, 1.8), (1, 1.5)]:
                if 0 <= l + dm < T:
                    mm[l + dm] *= m
        mm *= rng.lognormal(0, 0.05, T)
        mkt_mult_line[line] = mm
        mkt_uplift = 1 + 0.10 * (np.r_[1, mm[:-1]] - 1) + 0.04 * (mm - 1)
        infl = 1 + 0.12 * shortage * d["shortage_sens"]
        dig = 1 - 0.10 * digestion * d["shortage_sens"]
        ext["mkt_uplift"], ext["infl"], ext["dig"], ext["season"] = mkt_uplift, infl, dig, season_t

        for k, tier in enumerate(TIERS):
            g1 = f"{code}-{tier[0]}1"
            g2 = f"{code}-{tier[0]}2"
            p1, p2 = sku_params[g1], sku_params[g2]
            lc1 = np.where(t < p1["eol"], 1.0, np.exp(-(t - p1["eol"]) / 3.5))
            r2 = np.where(t >= p2["launch"], 1 - np.exp(-(t - p2["launch"] + 1) / 5.0), 0.0)
            share2 = 1.6 * r2 / (lc1 + 1.6 * r2 + 1e-12)
            share2 = np.where(r2 == 0, 0.0, share2)
            share = {g1: 1 - share2, g2: share2}
            for s, p in [(g1, p1), (g2, p2)]:
                # promotions (planned): seasonal pattern + end-of-life clearance
                pp = np.full(T, d["promo_p"])
                if d["promo_qe"]:
                    pp = np.where(np.isin(cal_month, [3, 6, 9, 12]), d["promo_qe"], d["promo_p"])
                flag = (rng.random(T) < pp).astype(int)
                depth = np.where(flag == 1, rng.uniform(*d["promo_depth"], T), 0.0) / 100.0
                if p["eol"] < 999:
                    for dm in (-1, 0):
                        if 0 <= p["eol"] + dm < T:
                            flag[p["eol"] + dm] = 1
                            depth[p["eol"] + dm] = rng.uniform(0.15, 0.25)
                promo_flag[s], promo_depth[s] = flag, depth
                promo_up = 1 + d["promo_el"] * depth
                promo_dip = np.where(np.r_[0, flag[:-1]] == 1, 0.95, 1.0)
                npi = np.ones(T)
                ltb = np.ones(T)
                if p["gen"] == 2:
                    npi = np.where(t >= p["launch"], 1 + 0.35 * np.exp(-(t - p["launch"]) / 2.0), 1.0)
                if p["eol"] < 999:
                    ltb = np.where((t >= p["eol"] - 2) & (t <= p["eol"] - 1), 1.30, 1.0)
                base = (d["total0"] * TIER_W[k] * ext["level"] * season_t * share[s] * npi * ltb * promo_up * promo_dip
                        * ext["comp_effect"] * ext["price_effect"] * ext["macro_effect"] * mkt_uplift)
                mu_noinfl[s] = base
                mu[s] = base * infl * dig
                # noisy realisation (orders): lognormal multiplicative noise -> Poisson, plus occasional large deals
                noise = rng.lognormal(-0.5 * d["vol"] ** 2, d["vol"], T)
                lam = mu[s] * noise
                y = rng.poisson(np.clip(lam, 0, None)).astype(float)
                if d["deal_p"] > 0:
                    deal = rng.random(T) < d["deal_p"]
                    y = y + deal * np.round(lam * rng.uniform(1.0, 2.5, T))
                orders[s] = np.where(t < p["launch"], 0, y)
                mu[s] = np.where(t < p["launch"], 0, mu[s])

    # ---- supply-constrained shipments / backlog --------------------------------------------------
    ship, backlog = {}, {}
    for s in sku_ids:
        ext = line_ext[sku_params[s]["line"]]
        cap = ext["alloc"] * 1.20 * mu[s]
        sh, bl, b = np.zeros(T), np.zeros(T), 0.0
        for i in range(T):
            sh[i] = min(b + orders[s][i], cap[i])
            b = b + orders[s][i] - sh[i]
            bl[i] = b
        ship[s], backlog[s] = np.round(sh), np.round(bl)

    # ---- leading indicator (line level) + legacy sales forecast ------------------------------------
    leading = {}
    for line, d in LINES.items():
        struct = sum(mu[s] for s in sku_ids if sku_params[s]["line"] == line)
        li = np.zeros(T - N_EXTRA)
        for i in range(T - N_EXTRA):
            ref = struct[max(0, i - 11): i + 1].mean() + 1e-9
            li[i] = 100 * struct[i + 2] / ref * rng.lognormal(0, d["li_sd"])
        leading[line] = li
    # Legacy sales-submitted forecast (made 3 months ahead). Modelled as: +8% structural optimism, a persistent
    # LINE-level error shared by all SKUs in the line (AR(1), sd 0.15 -> planners are optimistic/pessimistic together),
    # plus independent SKU-level noise (sd 0.27). Without the shared component, errors would diversify away unrealistically.
    line_err = {}
    for line in LINES:
        e = np.zeros(T)
        for i in range(1, T):
            e[i] = 0.6 * e[i - 1] + rng.normal(0, 0.15 * np.sqrt(1 - 0.6 ** 2))
        line_err[line] = e
    sales_fc = {}
    for s in sku_ids:
        sd_tot2 = 0.27 ** 2 + 0.15 ** 2
        sales_fc[s] = np.round(mu[s] * np.exp(np.log(1.08) - 0.5 * sd_tot2 + line_err[sku_params[s]["line"]] + rng.normal(0, 0.27, T)))

    # ---- marketing / ops tables -------------------------------------------------------------------
    for line, d in LINES.items():
        ext = line_ext[line]
        struct = sum(mu_noinfl[s] for s in sku_ids if sku_params[s]["line"] == line)
        avg_asp = np.mean(d["asp"])
        spend_actual = struct * avg_asp * d["mkt_pct"] / 1000.0 / ext["season"] * mkt_mult_line[line]
        spend_plan = spend_actual * rng.lognormal(0, 0.06, T)
        for i in range(N_MONTHS):
            mkt_rows.append(dict(date=dates[i].date(), product_line=line, marketing_spend_actual_usd_k=round(spend_actual[i], 1),
                                 marketing_spend_plan_usd_k=round(spend_plan[i], 1), campaign_multiplier=round(mkt_mult_line[line][i], 3)))
        npi_int = np.zeros(T)
        for s in [x for x in sku_ids if sku_params[x]["line"] == line and sku_params[x]["gen"] == 2]:
            l = sku_params[s]["launch"]
            npi_int += np.where((t >= l) & (t < l + 4), 1.0, 0.0)
        yld = np.clip(96 - 5 * shortage * d["shortage_sens"] - 1.5 * npi_int + rng.normal(0, 0.5, T), 80, 99)
        cap_units = sum(ext["alloc"] * 1.20 * mu[s] for s in sku_ids if sku_params[s]["line"] == line)
        bl_line = sum(backlog[s] for s in sku_ids if sku_params[s]["line"] == line)
        ext["yield"] = yld
        for i in range(N_MONTHS):
            sup_rows.append(dict(date=dates[i].date(), product_line=line, supplier_lead_time_weeks=round(ext["lead"][i], 1),
                                 component_allocation_pct=round(100 * ext["alloc"][i], 1), supplier_otd_pct=round(ext["otd"][i], 1)))
            ops_rows.append(dict(date=dates[i].date(), product_line=line, cm_capacity_units=int(cap_units[i]),
                                 factory_yield_pct=round(yld[i], 1), line_backlog_units=int(bl_line[i])))
    macro_df = pd.DataFrame(dict(date=[d.date() for d in dates[:N_MONTHS]], it_capex_index=np.round(macro_it[:N_MONTHS], 2),
                                 consumer_demand_index=np.round(macro_cons[:N_MONTHS], 2),
                                 commodity_price_index=np.round(commodity[:N_MONTHS], 2)))

    # ---- assemble tables ------------------------------------------------------------------------------
    sales_rows, promo_rows = [], []
    for s in sku_ids:
        for i in range(N_MONTHS):
            sales_rows.append(dict(date=dates[i].date(), sku=s, orders=int(orders[s][i]), shipments=int(ship[s][i]),
                                   backlog_units=int(backlog[s][i]), sales_forecast_3m_ahead=int(sales_fc[s][i])))
            promo_rows.append(dict(date=dates[i].date(), sku=s, promo_flag=int(promo_flag[s][i]), promo_depth_pct=round(100 * promo_depth[s][i], 1)))
    sales = pd.DataFrame(sales_rows)
    promo = pd.DataFrame(promo_rows)
    comp = pd.DataFrame(comp_rows)
    mkt = pd.DataFrame(mkt_rows)
    sup = pd.DataFrame(sup_rows)
    ops = pd.DataFrame(ops_rows)
    events = pd.DataFrame(ev_rows)
    extra_ev = []
    for s in sku_ids:
        p = sku_params[s]
        if p["gen"] == 2:
            extra_ev.append(dict(date=month_label(p["launch"]).date(), event=f"NPI launch {s}", product_line=p["line"], known_in_advance=True))
        if p["eol"] < 999:
            extra_ev.append(dict(date=month_label(p["eol"]).date(), event=f"End-of-life start {s}", product_line=p["line"], known_in_advance=True))
    events = pd.concat([events, pd.DataFrame(extra_ev)]).sort_values("date").reset_index(drop=True)
    li_rows = []
    for line in LINES:
        for i in range(N_MONTHS):
            li_rows.append(dict(date=dates[i].date(), product_line=line, leading_indicator_index=round(leading[line][i], 2),
                                refresh_event_index=round(line_ext[line]["event_known"][i], 3)))
    leading_df = pd.DataFrame(li_rows)

    # merged modelling panel (one row per SKU-month)
    panel = (sales.merge(sku_master[["sku", "product_line", "tier", "generation", "asp_usd", "unit_cost_usd"]], on="sku")
             .merge(promo, on=["date", "sku"]))
    for df in (comp, mkt, sup, ops, leading_df):
        panel = panel.merge(df, on=["date", "product_line"])
    panel = panel.merge(macro_df, on="date")
    return dict(sales_history=sales, sku_master=sku_master, promo_calendar=promo, marketing=mkt, competitor=comp,
                supply_chain=sup, operations=ops, macro=macro_df, leading_indicators=leading_df, events=events, panel=panel)


def main(out_dir: str | Path = "data"):
    out = Path(out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "processed").mkdir(parents=True, exist_ok=True)
    tables = generate()
    for name, df in tables.items():
        folder = "processed" if name == "panel" else "raw"
        df.to_csv(out / folder / f"{name}.csv", index=False)
    print({k: v.shape for k, v in tables.items()})


if __name__ == "__main__":
    main()
