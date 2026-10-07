"""CostSentinel: synthetic proof of concept and reproducible evaluation.

PLAIN ENGLISH: this file (1) makes up two years of cost data for a pretend factory
(about 1,200 jobs a year, roughly 1 in 4 going more than 10% over budget), (2) lets an
AI model learn from year 1 what normal spending looks like for each type of job, and
(3) replays year 2 day by day to see how early, and how accurately, the model warns
compared with the monthly cost report and two simple rules. It also makes the data
messier on purpose to see which data problems break the idea. No real data is used.


Run ``python run_all.py`` to regenerate the four scenarios, ablations, dashboard
data and dashboard. Requirements and exact metric definitions are in README.txt.
Only synthetic data is used. The model score is NOT a calibrated probability:
its training target combines final observed overrun >10% with current cost excess
above the learned curve >3% of budget. Real-world benefit remains untested.
"""
import json, os
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_env] = "1"
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WD_PER_MONTH = 21          # working days per month
CLOSE_LAG = 5              # assumed working-day lag after the month-end posting cutoff
OVERRUN_LABEL = 0.10       # >10% over budget at completion = "overrun job"
MIN_PRECISION = 0.50       # validation constraint; held-out pass/fail is reported separately
TRAIN_CUTOFF = 252         # only closed jobs and postings available before test-year start
ALERT_COOLDOWN = 5         # common working-day cooldown after a dismissed alert
CALIBRATION_ROWS = []
ELEMENTS = ["mat", "lab", "sup"]

FAMILIES = {
    # share of jobs, duration range (working days), budget range (EUR), cost mix (mat, lab, sup), planned curves (beta a,b)
    "Machining":   dict(share=0.40, dur=(10, 30), bud=(15e3, 60e3),  mix=(0.45, 0.45, 0.10),
                        curve={"mat": (1.0, 4.0), "lab": (1.6, 1.6), "sup": (1.5, 1.5)}),
    "Fabrication": dict(share=0.35, dur=(20, 45), bud=(30e3, 120e3), mix=(0.55, 0.35, 0.10),
                        curve={"mat": (1.4, 3.0), "lab": (2.0, 1.8), "sup": (1.5, 1.5)}),
    "Assembly":    dict(share=0.25, dur=(25, 60), bud=(50e3, 200e3), mix=(0.65, 0.28, 0.07),
                        curve={"mat": (1.2, 2.2), "lab": (2.6, 1.6), "sup": (1.5, 1.5)}),
}
DRIVERS = ["Scrap / rework", "Material price spike", "Labour inefficiency", "Engineering change"]


def planned_daily(total, a, b, D):
    """Planned spend per day from a beta-shaped cumulative curve."""
    x = np.arange(D + 1) / D
    cdf = beta.cdf(x, a, b)
    return total * np.diff(cdf)


EXTRA = []   # (job, day, injected overrun EUR) — ground truth of WHEN overrun cost is incurred


def simulate_jobs(n_jobs, year_offset, rng):
    """Returns (jobs dataframe, postings dataframe of TRUE costs)."""
    fams = list(FAMILIES)
    probs = [FAMILIES[f]["share"] for f in fams]
    jobs, posts = [], []
    for j in range(n_jobs):
        jid = f"J{year_offset}{j:04d}"
        fam = rng.choice(fams, p=probs)
        F = FAMILIES[fam]
        D = int(rng.integers(*F["dur"]))
        B = float(np.exp(rng.uniform(np.log(F["bud"][0]), np.log(F["bud"][1]))))
        start = int(year_offset * 252 + rng.integers(0, 252 - 10))
        bud = {e: B * m for e, m in zip(ELEMENTS, F["mix"])}
        eff = {e: rng.normal(1.0, 0.03) for e in ELEMENTS}          # normal job-to-job noise
        timing_shift = rng.random() < 0.25                           # materials bought early (legit, not an overrun)

        daily = {}
        for e in ELEMENTS:
            a, b = F["curve"][e]
            if e == "mat":
                if timing_shift:
                    a, b = max(0.6, a * 0.6), b * 1.6
                k = int(rng.integers(3, 7))                           # materials issued in lumps
                days = np.clip((beta.rvs(a, b, size=k, random_state=rng) * D).astype(int), 0, D - 1)
                arr = np.zeros(D)
                for d in days:
                    arr[d] += bud[e] / k * eff[e] * rng.lognormal(0, 0.10)
            else:
                arr = planned_daily(bud[e], a, b, D) * eff[e] * rng.lognormal(0, 0.25, D)
            daily[e] = arr

        # ---- inject overrun driver --------------------------------------------------
        driver, s_frac, mag = None, np.nan, 0.0
        base = {e: daily[e].copy() for e in ELEMENTS}
        if rng.random() < 0.28:
            driver = DRIVERS[int(rng.integers(0, 4))]
            s_frac = rng.uniform(0.05, 0.85)
            s = int(s_frac * D)
            mag = rng.uniform(0.10, 0.40)
            extra = mag * B
            rem = D - s
            if driver == "Scrap / rework":
                daily["lab"][s:] += extra * 0.5 / rem
                daily["mat"][s:] += extra * 0.5 / rem
            elif driver == "Material price spike":
                pm = planned_daily(bud["mat"], *F["curve"]["mat"], D)[s:]
                extra = min(extra, pm.sum() * 0.6)                    # can only inflate material still to be bought
                if pm.sum() > 0:
                    daily["mat"][s:] += extra * pm / pm.sum()
            elif driver == "Labour inefficiency":
                pl = planned_daily(bud["lab"], *F["curve"]["lab"], D)[s:]
                daily["lab"][s:] += extra * pl / pl.sum()
            else:  # Engineering change: lump of material + extra labour over next 30% of job
                daily["mat"][s] += extra * 0.6
                w = max(1, int(0.3 * D))
                daily["lab"][s:s + w] += extra * 0.4 / len(daily["lab"][s:s + w])

        true_final = sum(daily[e].sum() for e in ELEMENTS)
        extra_daily = sum(daily[e] - base[e] for e in ELEMENTS)
        for d in np.nonzero(extra_daily > 0)[0]:
            EXTRA.append((jid, d, extra_daily[d]))
        jobs.append(dict(job=jid, family=fam, D=D, budget=B, start=start, driver=driver,
                         start_frac=s_frac, final_true=true_final,
                         overrun_true=true_final / B - 1, timing_shift=timing_shift,
                         **{f"bud_{e}": bud[e] for e in ELEMENTS}))
        for e in ELEMENTS:
            nz = np.nonzero(daily[e] > 0)[0]
            for d in nz:
                posts.append((jid, d, start + d, e, daily[e][d]))
    jobs = pd.DataFrame(jobs)
    jobs["is_overrun"] = jobs["overrun_true"] > OVERRUN_LABEL
    posts = pd.DataFrame(posts, columns=["job", "day", "abs_day", "elem", "amt"])
    return jobs, posts


def degrade(posts, jobs, p_late, p_mis, p_batch, rng):
    """Apply data-quality problems to postings. Returns OBSERVED postings."""
    p = posts.copy()
    p["post_abs"] = p["abs_day"].astype(int)
    # (1) late postings: delay 2–10 working days
    late = rng.random(len(p)) < p_late
    p.loc[late, "post_abs"] += rng.integers(2, 11, late.sum())
    # (2) weekly batch timesheets for a share of jobs (labour keyed every 5th day)
    batch_jobs = set(jobs.loc[rng.random(len(jobs)) < p_batch, "job"])
    m = p["job"].isin(batch_jobs) & (p["elem"] == "lab")
    p.loc[m, "post_abs"] = ((p.loc[m, "post_abs"] // 5) + 1) * 5
    # (3) mis-coding: posting lands on another job active that day
    mis = np.nonzero(rng.random(len(p)) < p_mis)[0]
    if len(mis):
        js = jobs[["job", "start", "D"]].to_numpy()
        starts = jobs["start"].to_numpy(); ends = (jobs["start"] + jobs["D"]).to_numpy()
        for i in mis:
            t = p.at[i, "abs_day"]
            active = np.nonzero((starts <= t) & (ends > t))[0]
            if len(active) > 1:
                p.at[i, "job"] = js[rng.choice(active)][0]
    # convert to job-relative posting day
    p = p.merge(jobs[["job", "start", "D"]], on="job", how="left")
    p["pday"] = p["post_abs"] - p["start"]
    p = p[p["pday"] >= 0]
    return p[["job", "pday", "post_abs", "elem", "amt"]]


def learn_curves(jobs, obs):
    """Median cumulative cost curve per family & element on a 20-bin grid of % elapsed (from history)."""
    grid = np.linspace(0, 1, 21)
    curves = {}
    J = jobs.set_index("job")
    for fam in FAMILIES:
        ids = J.index[J["family"] == fam]
        o = obs[obs["job"].isin(ids)]
        for e in ELEMENTS + ["tot"]:
            mats = []
            for jid, g in (o if e == "tot" else o[o["elem"] == e]).groupby("job"):
                D = J.at[jid, "D"]
                fr = np.clip((g["pday"].to_numpy() + 1) / D, 0, 1.5)
                amt = g["amt"].to_numpy()
                denom = J.at[jid, "budget"] if e == "tot" else J.at[jid, f"bud_{e}"]
                mats.append([amt[fr <= x].sum() / denom for x in grid])
            curves[(fam, e)] = np.median(np.array(mats), axis=0) if mats else grid
    return grid, curves


def snapshots(jobs, obs, grid, curves, step=1):
    """Daily feature snapshots per job using only postings visible by that day."""
    rows = []
    J = jobs.set_index("job")
    byjob = {k: g for k, g in obs.groupby("job")}
    for jid, jr in J.iterrows():
        D = jr["D"]; fam = jr["family"]; B = jr["budget"]
        g = byjob.get(jid)
        if g is None:
            continue
        pd_ = g["pday"].to_numpy(); el = g["elem"].to_numpy(); am = g["amt"].to_numpy()
        for t in range(1, D + 1, step):
            vis = pd_ < t
            frac = t / D
            cum = {e: am[vis & (el == e)].sum() for e in ELEMENTS}
            tot = sum(cum.values())
            def exp(e, f):
                return np.interp(f, grid, curves[(fam, e)])
            r = dict(job=jid, t=t, frac=frac, abs_day=jr["start"] + t,
                     cum_tot=tot, cum_ratio=tot / B, lin_ratio=tot / (B * frac))
            r["dev_tot"] = tot / max(B * exp("tot", frac), 1e-6 * B)
            for e in ELEMENTS:
                be = jr[f"bud_{e}"]
                r[f"dev_{e}"] = cum[e] / max(be * exp(e, frac), 0.02 * be)
                r[f"ratio_{e}"] = cum[e] / be
            r["excess_tot"] = (tot - B * exp("tot", frac)) / B
            for e in ("mat", "lab"):
                be = jr[f"bud_{e}"]
                r[f"excess_{e}"] = (cum[e] - be * exp(e, frac)) / B
            for w in (3, 5, 10):
                lo = max(0, (t - w) / D)
                exp_rec = B * (exp("tot", frac) - exp("tot", lo))
                r[f"burn{w}"] = am[vis & (pd_ >= t - w)].sum() / max(exp_rec, 0.01 * B)
                bl = jr["bud_lab"] * (exp("lab", frac) - exp("lab", lo))
                r[f"lburn{w}"] = am[vis & (pd_ >= t - w) & (el == "lab")].sum() / max(bl, 0.01 * jr["bud_lab"])
            r["fam_" + fam] = 1
            r["logB"] = np.log(B); r["D"] = D
            rows.append(r)
    S = pd.DataFrame(rows).fillna(0)
    for f in FAMILIES:
        if "fam_" + f not in S:
            S["fam_" + f] = 0
    return S


FEATS = ["frac", "cum_ratio", "dev_tot", "dev_mat", "dev_lab", "dev_sup", "ratio_mat", "ratio_lab",
         "ratio_sup", "excess_tot", "excess_mat", "excess_lab", "burn3", "burn5", "burn10",
         "lburn3", "lburn5", "lburn10", "logB", "D"] + ["fam_" + f for f in FAMILIES]


def flag_days(S, flag_col, sev_col, consecutive=1):
    """Flag only after consecutive calendar-working-day snapshots, not merely rows."""
    out = {}
    for jid, g in S.sort_values(["job", "t"]).groupby("job"):
        run, previous, days = 0, None, []
        for t, flag, severity in zip(g["t"], g[flag_col], g[sev_col]):
            run = (run + 1 if previous is not None and t == previous + 1 else 1) if flag else 0
            if run >= consecutive:
                days.append((int(t), float(severity)))
            previous = t
        out[jid] = days
    return out


def alert_events(flags, jobs_true, cooldown=ALERT_COOLDOWN):
    """Synthetic manager: knows true cause onset/final class; 5-day common cooldown.

    A true-overrun job's first emitted alert at/after its simulated cause is a valid
    catch and stops further alerts. Other alerts are dismissed. Real managers do
    not have this oracle information; this is an explicit evaluation convention.
    """
    J = jobs_true.set_index("job")
    histories = {}
    for jid, days in flags.items():
        if jid not in J.index:
            continue
        D, ov, sf = J.at[jid, "D"], J.at[jid, "is_overrun"], J.at[jid, "start_frac"]
        cause = int(sf * D) if ov and not np.isnan(sf) else (0 if ov else 10**6)
        next_allowed, events = -10**6, []
        for day, score in days:
            if day < next_allowed or day >= D + 3 * WD_PER_MONTH:
                continue
            valid = bool(ov and day >= cause)
            events.append(dict(day=int(day), score=float(score), status="valid" if valid else "dismissed"))
            if valid:
                break
            next_allowed = day + cooldown
        histories[jid] = events
    return histories


def manage_alerts(flags, jobs_true):
    histories = alert_events(flags, jobs_true)
    valid = {j: e["day"] for j, es in histories.items() for e in es if e["status"] == "valid"}
    return valid, sum(map(len, histories.values()))


def month_end_flags(jobs, obs):
    """Fixed status quo: +10% phased-budget variance at month-end; report 5 days later.

    No accruals or invoices arriving during the close are included. This deliberately
    simple monthly-report comparator is not a universal description of ERP practice.
    """
    out = {}
    for jr in jobs.itertuples():
        g = obs[obs["job"] == jr.job]
        out[jr.job] = []
        me = ((jr.start // WD_PER_MONTH) + 1) * WD_PER_MONTH
        while me <= jr.start + jr.D + 2 * WD_PER_MONTH:
            rel = me - jr.start
            frac = min(rel / jr.D, 1.0)
            planned = sum(getattr(jr, f"bud_{e}") * beta.cdf(frac, *FAMILIES[jr.family]["curve"][e]) for e in ELEMENTS)
            actual = g.loc[g["post_abs"] < me, "amt"].sum()
            if planned > 0 and actual > 1.10 * planned and frac > 0.05:
                out[jr.job].append((int(rel + CLOSE_LAG), float(actual / planned)))
            me += WD_PER_MONTH
    return out


def evaluate(jobs_true, flags, label):
    J = jobs_true.set_index("job")
    pos = J.index[J["is_overrun"]]
    valid, n_alerts = manage_alerts(flags, jobs_true)
    open_valid = [j for j in valid if valid[j] < J.at[j, "D"]]
    X = pd.DataFrame(EXTRA, columns=["job", "day", "amt"])
    X = X[X["job"].isin(pos)]
    den = float(X["amt"].sum())
    caught = X["job"].map(lambda j: valid.get(j, 10**6))
    remaining = float(X.loc[X["day"] >= caught, "amt"].sum())
    remaining_2day = float(X.loc[X["day"] >= caught + 2, "amt"].sum())
    precision = len(valid) / n_alerts if n_alerts else 0.0
    lags = [(valid[j] - int(J.at[j, "start_frac"] * J.at[j, "D"])) / J.at[j, "D"]
            for j in valid if not np.isnan(J.at[j, "start_frac"])]
    return dict(arm=label, recall_open=len(open_valid) / len(pos) if len(pos) else 0.0,
                recall_any=len(valid) / len(pos) if len(pos) else 0.0,
                precision=precision, alerts=n_alerts, alerts_per_week=n_alerts / (252 / 5),
                false_alerts_per_week=(n_alerts - len(valid)) / (252 / 5),
                median_pct_elapsed=float(np.median([valid[j] / J.at[j, "D"] for j in valid])) if valid else None,
                median_lag_after_cause=float(np.median(lags)) if lags else None,
                share_after_close=float(np.mean([valid[j] >= J.at[j, "D"] for j in valid])) if valid else None,
                addressable_share=remaining / den if den else 0.0,
                addressable_share_2day=remaining_2day / den if den else 0.0,
                remaining_driver_eur=remaining, remaining_driver_eur_2day=remaining_2day,
                injected_driver_eur=den, n_overrun_jobs=len(pos), n_valid_detections=len(valid),
                n_open_detections=len(open_valid), n_false_alerts=n_alerts - len(valid),
                precision_guardrail_pass=bool(precision >= MIN_PRECISION),
                review_cost_eur=n_alerts * 0.5 * 60)


def add_plan_features(S, jobs):
    """Known planner budget, separate from curves learned on the training fold."""
    J = jobs.set_index("job")
    S = S.copy()
    S = S.join(J[["family", "bud_mat", "bud_lab", "bud_sup"]], on="job")
    planned = np.zeros(len(S))
    for family, settings in FAMILIES.items():
        mask = S["family"].eq(family).to_numpy()
        fractions = S.loc[mask, "frac"].to_numpy()
        for e in ELEMENTS:
            planned[mask] += S.loc[mask, f"bud_{e}"].to_numpy() * beta.cdf(fractions, *settings["curve"][e])
    S["plan_cum"] = planned
    S["plan_ratio"] = S["cum_tot"] / S["plan_cum"].clip(lower=1)
    return S


def make_model():
    # Fixed settings, no internal row-level validation split between related snapshots.
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=24,
                                         l2_regularization=1.0, random_state=0, early_stopping=False)


def labels_for(jobs, observed):
    J = jobs.set_index("job")
    recorded = observed.groupby("job")["amt"].sum().reindex(J.index).fillna(0)
    return recorded / J["budget"] - 1 > OVERRUN_LABEL


def arm_flags(S, key, threshold):
    score = {"B1": "lin_ratio", "B2": "plan_ratio", "C": "p"}[key]
    flag = S[score] > threshold
    if key != "C":
        flag &= S["frac"] >= 0.10
    return flag_days(S.assign(_flag=flag), "_flag", score, consecutive=2)


ARM_NAMES = {"A": "A  Month-end variance report", "B1": "B1 Rule vs linear budget",
             "B2": "B2 Rule vs time-phased plan", "C": "C  CostSentinel ML"}


def calibrate(S, jobs, arm, scenario):
    candidates = np.round(np.arange(.20, .951, .05), 2) if arm == "C" else np.round(np.arange(1.05, 2.501, .05), 2)
    rows = []
    for threshold in candidates:
        row = evaluate(jobs, arm_flags(S, arm, threshold), ARM_NAMES[arm])
        row.update(scenario=scenario, threshold=float(threshold), calibration_jobs=len(jobs))
        rows.append(row)
    eligible = [r for r in rows if r["precision_guardrail_pass"] and r["alerts"] > 0]
    if eligible:
        chosen = max(eligible, key=lambda r: (r["addressable_share"], r["precision"], -r["threshold"]))
    else:
        # Diagnostic score only. Infeasibility is explicit in every exported arm result.
        chosen = max(rows, key=lambda r: (r["precision"], r["addressable_share"], -r["threshold"]))
    for r in rows:
        r["selected"] = r is chosen
    CALIBRATION_ROWS.extend(rows)
    return dict(threshold=chosen["threshold"], pass_guardrail=bool(eligible),
                validation_precision=chosen["precision"], validation_addressable_share=chosen["addressable_share"],
                status="validation_pass" if eligible else "validation_infeasible_diagnostic_only",
                candidate_min=float(candidates[0]), candidate_max=float(candidates[-1]), candidate_step=.05)


def run_scenario(name, p_late, p_mis, p_batch, seed=7, keep=False):
    EXTRA.clear()
    rng = np.random.default_rng(seed)
    cohort_tr, posts_tr = simulate_jobs(1200, 0, rng)
    jobs_te, posts_te = simulate_jobs(1200, 1, rng)
    # Separate degradation prevents either cohort donating postings to the other.
    obs_tr_all = degrade(posts_tr, cohort_tr, p_late, p_mis, p_batch, np.random.default_rng(seed + 1))
    obs_te = degrade(posts_te, jobs_te, p_late, p_mis, p_batch, np.random.default_rng(seed + 2))
    jobs_tr = cohort_tr[(cohort_tr["start"] + cohort_tr["D"]) <= TRAIN_CUTOFF].copy()
    obs_tr = obs_tr_all[(obs_tr_all["post_abs"] < TRAIN_CUTOFF) & obs_tr_all["job"].isin(jobs_tr["job"])].copy()
    vjobs = set(jobs_tr["job"].sample(frac=.25, random_state=1))
    jobs_fit = jobs_tr[~jobs_tr["job"].isin(vjobs)]
    jobs_va = jobs_tr[jobs_tr["job"].isin(vjobs)]
    obs_fit = obs_tr[obs_tr["job"].isin(jobs_fit["job"])]
    grid_fit, curves_fit = learn_curves(jobs_fit, obs_fit)
    train = snapshots(jobs_fit, obs_fit, grid_fit, curves_fit)
    val = snapshots(jobs_va, obs_tr[obs_tr["job"].isin(vjobs)], grid_fit, curves_fit)
    y_fit = labels_for(jobs_fit, obs_fit)
    train["y"] = (train["job"].map(y_fit).astype(bool) & (train["excess_tot"] > .03)).astype(int)
    model = make_model()
    model.fit(train[FEATS], train["y"])
    val["p"] = model.predict_proba(val[FEATS])[:, 1]
    val = add_plan_features(val, jobs_va)
    calibration = {arm: calibrate(val, jobs_va, arm, name) for arm in ("B1", "B2", "C")}

    # Final refit uses all eligible previous-year closed jobs, never test observations.
    grid, curves = learn_curves(jobs_tr, obs_tr)
    train_all = snapshots(jobs_tr, obs_tr, grid, curves)
    y_all = labels_for(jobs_tr, obs_tr)
    train_all["y"] = (train_all["job"].map(y_all).astype(bool) & (train_all["excess_tot"] > .03)).astype(int)
    model = make_model()
    model.fit(train_all[FEATS], train_all["y"])
    test = snapshots(jobs_te, obs_te, grid, curves)
    test["p"] = model.predict_proba(test[FEATS])[:, 1]
    test = add_plan_features(test, jobs_te)
    flags = {"A": month_end_flags(jobs_te, obs_te)}
    flags.update({arm: arm_flags(test, arm, calibration[arm]["threshold"]) for arm in ("B1", "B2", "C")})
    results = []
    tau = calibration["C"]["threshold"]
    for arm in ("A", "B1", "B2", "C"):
        r = evaluate(jobs_te, flags[arm], ARM_NAMES[arm])
        c = calibration.get(arm)
        r.update(scenario=name, tau=tau, threshold=c["threshold"] if c else 1.10,
                 calibration_pass=c["pass_guardrail"] if c else None,
                 calibration_precision=c["validation_precision"] if c else None,
                 calibration_status=c["status"] if c else "fixed_status_quo",
                 simulation_guardrails_pass=bool(c and c["pass_guardrail"] and r["precision_guardrail_pass"]),
                 deployment_approved=False)
        results.append(r)
    extra = None
    if keep:
        extra = dict(jobs_te=jobs_te, S_te=test, tau=tau, model=model, calibration=calibration,
                     F_ml=flags["C"], F_me=flags["A"], F_lin=flags["B1"], F_plan=flags["B2"],
                     ml_valid=manage_alerts(flags["C"], jobs_te)[0], me_valid=manage_alerts(flags["A"], jobs_te)[0],
                     ml_history=alert_events(flags["C"], jobs_te), me_history=alert_events(flags["A"], jobs_te),
                     EXTRA=list(EXTRA), n_train_jobs=len(jobs_tr), n_fit_jobs=len(jobs_fit),
                     n_validation_jobs=len(jobs_va), n_train_postings=len(obs_tr),
                     n_excluded_open_training_jobs=len(cohort_tr) - len(jobs_tr), results=results)
    return results, extra


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(filename, value):
    with open(os.path.join(OUT, filename), "w", encoding="utf-8") as f:
        json.dump(json_clean(value), f, indent=2, allow_nan=False, ensure_ascii=False)


METRIC_DEFINITIONS = {
    "recall_open": "Validly detected overrun jobs before planned closure / all jobs with true final cost >110% of budget.",
    "precision": "Valid catches / all emitted alerts, including repeated dismissed alerts; oracle-manager simulation convention.",
    "share_after_close": "Validly detected overrun jobs caught at/after closure / all validly detected overrun jobs, not all alerts.",
    "addressable_share": "Injected driver cost on true-overrun jobs scheduled at/after the first valid alert / all injected driver cost on those jobs; zero response delay, not realized savings.",
    "addressable_share_2day": "Same driver-cost metric, requiring cost to occur at least two working days after the valid alert.",
    "alerts_per_week": "Emitted alerts for the 1,200-job test cohort / 50.4 working weeks; normalized annual-cohort load.",
    "model_score": "Classifier score for observed final overrun >10% AND current learned-curve excess >3% of budget; not calibrated final-overrun probability.",
    "false_alarm_anatomy": "Distinct non-overrun jobs receiving at least one emitted ML alert, not false-alert events; categories may overlap."
}


METHODOLOGY = {
    "data": "One fixed-seed synthetic plant; no real-ERP, customer or user validation.",
    "seed": 7, "training_cutoff_absolute_working_day": TRAIN_CUTOFF,
    "data_quality_rates": "Scenario percentages are nominal probabilities. Mis-coding attempts can select the same active job; realized wrong-job shares are therefore slightly lower.",
    "training": "Only closed previous-year jobs and postings available before test start. Cohorts are degraded separately. Learned curves fit inside the training fold before calibration, then refit on all eligible history.",
    "calibration": "Job-level 25% validation split. All daily arms use daily snapshots and two consecutive daily threshold exceedances. Rules and ML choose thresholds on validation only: maximize remaining driver cost subject to precision >=50%. If infeasible, the highest-precision candidate is exported as diagnostic only.",
    "realert": "All arms use a five-working-day cooldown after a dismissed alert. First valid catch stops further alerts.",
    "manager": "Oracle simulation: manager knows whether final true cost exceeds budget by >10% and whether injected cause is active. This does not represent observed human accuracy or speed.",
    "month_end": "Fixed 10% variance vs phased plan; only postings visible before month-end; report five working days later. No accruals or arrivals during close.",
    "deployment": "A prototype, not approved for production. Validation and held-out precision checks are reported separately; simulation pass is not field validation.",
    "limitations": "One stationary synthetic data-generating process and seed; no uncertainty interval, measured intervention effect, calibrated probability, or evidence of willingness to pay. Cost still ahead may be committed or unavoidable. Zero-delay and two-day-delay opportunities are both shown.",
    "response_delay_working_days": [0, 2], "minimum_precision": MIN_PRECISION,
    "review_minutes_per_alert": 30, "review_cost_eur_per_hour": 60,
    "model_target": METRIC_DEFINITIONS["model_score"]
}


def export_comparator_value(results):
    """Same-effectiveness value comparison; rule maintenance costs remain unknown."""
    baseline = next(r for r in results if r["arm"].startswith("A"))
    rows = []
    for r in results:
        for delay, field in [(0, "remaining_driver_eur"), (2, "remaining_driver_eur_2day")]:
            is_base = r["arm"].startswith("A")
            increment = r[field] - baseline[field]
            subscription = 2490 * 12 if r["arm"].startswith("C") else 0
            review = 0 if is_base else r["review_cost_eur"]
            rows.append(dict(arm=r["arm"], scenario="Typical", response_delay_working_days=delay,
                             effectiveness_assumption=.30, remaining_driver_eur=r[field],
                             opportunity_difference_vs_monthend_eur=increment,
                             conditional_benefit_vs_monthend_eur=increment * .30,
                             additional_review_cost_eur=review, subscription_assumed_eur=subscription,
                             conditional_net_before_rule_maintenance_eur=increment * .30 - review - subscription,
                             precision=r["precision"], simulation_guardrails_pass=r["simulation_guardrails_pass"],
                             cost_scope="Rule setup/maintenance costs unestimated; no rule subscription assumed. Monthly report costs are common baseline costs.",
                             limitation="Standalone-arm opportunity difference, not measured savings or a simulation of combined workflows. Same 30% effectiveness is unvalidated for every arm."))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "comparator_value.csv"), index=False)
    write_json("comparator_value.json", rows)
    return rows


def export_typical(keep):
    EXTRA[:] = keep["EXTRA"]
    J = keep["jobs_te"].set_index("job")
    pos, valid = J[J["is_overrun"]], keep["ml_valid"]
    rows = []
    for driver in DRIVERS:
        ids = pos.index[pos["driver"] == driver]
        caught = [j for j in ids if j in valid and valid[j] < J.at[j, "D"]]
        rows.append(dict(driver=driver, n=len(ids), recall=len(caught) / len(ids) if len(ids) else 0,
                         median_start_frac=float(pos.loc[ids, "start_frac"].median())))
    for title, mask in [("Cause starts in last 30% of job", pos["start_frac"] > .7),
                        ("Cause starts in first 40% of job", pos["start_frac"] <= .4)]:
        ids = pos.index[mask]
        rows.append(dict(driver=title, n=len(ids), recall=sum(j in valid and valid[j] < J.at[j, "D"] for j in ids) / max(len(ids), 1)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "typical_by_driver.csv"), index=False)
    fpjobs = [j for j, events in keep["ml_history"].items() if events and not J.at[j, "is_overrun"]]
    S, threshold_rows = keep["S_te"], []
    for tau in np.round(np.arange(.20, .951, .05), 2):
        r = evaluate(keep["jobs_te"], arm_flags(S, "C", tau), f"tau={tau:.2f}")
        r.update(tau=float(tau), selected_on_validation=bool(tau == keep["tau"]),
                 use="test-set diagnostic curve only; not used to select threshold")
        threshold_rows.append(r)
    pd.DataFrame(threshold_rows).to_csv(os.path.join(OUT, "typical_threshold_curve.csv"), index=False)
    injected = float(pd.DataFrame(EXTRA, columns=["job", "day", "amt"]).query("job in @pos.index")["amt"].sum())
    ml = next(r for r in keep["results"] if r["arm"].startswith("C"))
    me = next(r for r in keep["results"] if r["arm"].startswith("A"))
    delta = ml["remaining_driver_eur"] - me["remaining_driver_eur"]
    delta2 = ml["remaining_driver_eur_2day"] - me["remaining_driver_eur_2day"]
    summary = dict(n_test_jobs=len(J), n_overrun=int(J["is_overrun"].sum()), overrun_rate=float(J["is_overrun"].mean()),
                   n_train_jobs=keep["n_train_jobs"], n_fit_jobs=keep["n_fit_jobs"], n_validation_jobs=keep["n_validation_jobs"],
                   n_excluded_open_training_jobs=keep["n_excluded_open_training_jobs"],
                   total_budget_test=float(J["budget"].sum()), total_actual_cost_test=float(J["final_true"].sum()),
                   total_overrun_eur=float((J["final_true"] - J["budget"])[J["is_overrun"]].sum()),
                   injected_overrun_eur=injected, ml_fp_jobs=len(fpjobs),
                   ml_fp_share_timing_shift=float(np.mean([J.at[j, "timing_shift"] for j in fpjobs])) if fpjobs else 0,
                   base_rate_timing_shift=float(J["timing_shift"].mean()),
                   ml_fp_share_near_miss_3_10pct=float(np.mean([.03 < J.at[j, "overrun_true"] <= .10 for j in fpjobs])) if fpjobs else 0,
                   tau=keep["tau"], calibration=keep["calibration"], ml_metrics=ml, month_end_metrics=me,
                   incremental_remaining_driver_eur=delta, incremental_remaining_driver_eur_2day=delta2,
                   conditional_avoided_eur_15_30_45=[delta*x for x in (.15, .30, .45)],
                   conditional_avoided_eur_2day_15_30_45=[delta2*x for x in (.15, .30, .45)],
                   control_annual_subscription_eur=2490*12, ml_review_cost_eur=ml["review_cost_eur"],
                   conditional_net_benefit_15_30_45=[delta*x - 2490*12 - ml["review_cost_eur"] for x in (.15, .30, .45)],
                   benefit_cost_ratio_15_30_45=[delta*x / (2490*12 + ml["review_cost_eur"]) for x in (.15, .30, .45)],
                   ml_to_monthend_open_recall=ml["recall_open"] / me["recall_open"] if me["recall_open"] else None,
                   average_active_jobs=float(J["D"].sum()/252), deployment_approved=False,
                   note="All value figures are conditional scenarios, not measured savings or deployment approval.")
    summary["comparator_value"] = export_comparator_value(keep["results"])
    write_json("typical_summary.json", summary)
    samples = []
    for jid in list(pos.index[:8]) + list(J.index[~J["is_overrun"]][:5]):
        jr = J.loc[jid]
        g = S[S["job"] == jid].sort_values("t")
        mh, ah = keep["ml_history"].get(jid, []), keep["me_history"].get(jid, [])
        samples.append(dict(job=jid, family=jr["family"], D=int(jr["D"]), budget=float(jr["budget"]),
                            driver=jr["driver"], overrun_true=float(jr["overrun_true"]),
                            cause_day=None if np.isnan(jr["start_frac"]) else int(jr["start_frac"] * jr["D"]),
                            alert_day=mh[0]["day"] if mh else None, me_alert_day=ah[0]["day"] if ah else None,
                            ml_valid_day=keep["ml_valid"].get(jid), me_valid_day=keep["me_valid"].get(jid),
                            ml_alert_history=mh, me_alert_history=ah,
                            series=[dict(t=int(r.t), cum=float(r.cum_tot), p=float(r.p), plan=float(r.plan_cum),
                                         mat=float(r.dev_mat), lab=float(r.dev_lab), burn=float(r.burn5), lburn=float(r.lburn5))
                                    for r in g.itertuples()]))
    write_json("dashboard_sample_jobs.json", samples)
    return summary, samples


def run_ablations():
    rows = []
    for name, late, mis, batch in [("Late postings only (35%)", .35, 0, 0),
                                  ("Weekly batch timesheets only (60% of jobs)", 0, 0, .60),
                                  ("Mis-coded postings only (8%)", 0, .08, 0)]:
        results, _ = run_scenario(name, late, mis, batch)
        rows.extend(r for r in results if r["arm"].startswith(("A", "C")))
        print(name, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ablation_poor_level.csv"), index=False)
    return rows


def main():
    CALIBRATION_ROWS.clear()
    all_results, typical, summary, samples = [], None, None, None
    with threadpool_limits(limits=1):
        # Typical first provides early, immutable feedback; no outcome-based tuning follows.
        for name, late, mis, batch in [("Typical", .15, .03, .30), ("Clean", 0, 0, 0),
                                       ("Poor", .35, .08, .60), ("Very poor", .60, .15, .90)]:
            results, extra = run_scenario(name, late, mis, batch, keep=name == "Typical")
            all_results.extend(results)
            if extra:
                typical = extra
                summary, samples = export_typical(typical)
                write_json("typical_results.json", results)
            print(name, "done", flush=True)
            print(pd.DataFrame(results)[["arm", "recall_open", "precision", "alerts_per_week", "addressable_share", "threshold", "calibration_pass"]].to_string(index=False), flush=True)
        scenario_order = {s: i for i, s in enumerate(("Clean", "Typical", "Poor", "Very poor"))}
        all_results.sort(key=lambda r: (scenario_order[r["scenario"]], r["arm"]))
        pd.DataFrame(all_results).to_csv(os.path.join(OUT, "results_by_scenario.csv"), index=False)
        ablation = run_ablations()
    pd.DataFrame(CALIBRATION_ROWS).to_csv(os.path.join(OUT, "validation_threshold_curves.csv"), index=False)
    ml = summary["ml_metrics"]
    guardrail = dict(minimum_precision=MIN_PRECISION,
                     calibration_pass=summary["calibration"]["C"]["pass_guardrail"],
                     validation_precision=summary["calibration"]["C"]["validation_precision"],
                     test_precision=ml["precision"], test_pass=ml["precision_guardrail_pass"],
                     simulation_guardrails_pass=ml["simulation_guardrails_pass"], deployment_approved=False,
                     status="Simulation checks passed; field validation still required" if ml["simulation_guardrails_pass"] else "Precision guardrail not met; diagnostic prototype only")
    write_json("dashboard_data.json", dict(jobs=samples, tau=summary["tau"], results=all_results,
                                          ablation=ablation, summary=summary, methodology=METHODOLOGY,
                                          metric_definitions=METRIC_DEFINITIONS, guardrail=guardrail,
                                          sample_selection="First eight overrun and first five non-overrun jobs by generated ID; 13 examples, not the full test cohort."))
    write_json("metric_definitions.json", dict(definitions=METRIC_DEFINITIONS, methodology=METHODOLOGY))
    print("All result files regenerated; no threshold selected using test outcomes.", flush=True)


if __name__ == "__main__":
    main()
