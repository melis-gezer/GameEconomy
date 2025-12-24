import numpy as np
import pandas as pd

def generate_events(n_users=5000, days=14, seed=42, sink_multiplier=1.0, price_multiplier=1.0):
    rng = np.random.default_rng(seed)

    # 1) users
    user_ids = np.array([f"u{i:06d}" for i in range(n_users)])
    install_date = pd.Timestamp("2025-01-01")
    install_times = install_date + pd.to_timedelta(rng.integers(0, 24*60*60, size=n_users), unit="s")

    # segments: free / occasional / whale
    segments = rng.choice(["free", "occasional", "whale"], size=n_users, p=[0.85, 0.13, 0.02])
    ab_group = rng.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])

    # segment params
    base_session_lambda = {"free": 1.1, "occasional": 1.4, "whale": 1.7}
    p_iap = {"free": 0.001, "occasional": 0.03, "whale": 0.18}
    p_ad = {"free": 0.35, "occasional": 0.25, "whale": 0.10}

    events = []
    balance = np.zeros(n_users, dtype=int)

    # retention curve (simple): p(active on day d) = exp(-k*d) scaled by segment
    k = 0.22
    seg_scale = {"free": 1.0, "occasional": 1.1, "whale": 1.15}

    for d in range(days):
        day_start = install_date + pd.Timedelta(days=d)

        # user active?
        p_active = np.exp(-k * d) * np.vectorize(seg_scale.get)(segments)
        is_active = rng.random(n_users) < np.clip(p_active, 0, 1)

        active_idx = np.where(is_active)[0]
        if len(active_idx) == 0:
            continue

        # sessions per active user
        sess_lam = np.array([base_session_lambda[s] for s in segments[active_idx]])
        n_sess = rng.poisson(sess_lam) + 1  # at least 1 session if active

        for i, ns in zip(active_idx, n_sess):
            for s in range(ns):
                session_id = f"{user_ids[i]}_d{d}_s{s}"
                # random time within day
                t0 = day_start + pd.to_timedelta(rng.integers(0, 24*60*60), unit="s")

                # session_start
                events.append({
                    "event_time": t0,
                    "user_id": user_ids[i],
                    "install_time": install_times[i],
                    "event_name": "session_start",
                    "session_id": session_id,
                    "level": None,
                    "currency_delta": 0,
                    "currency_balance": int(balance[i]),
                    "iap_usd": 0.0,
                    "ad_usd": 0.0,
                    "segment": segments[i],
                    "ab_group": ab_group[i],
                    "cohort_day": d,
                })

                # levels
                n_levels = int(rng.integers(1, 4))
                for lv in range(n_levels):
                    earn = int(rng.integers(40, 90))
                    spend = int(rng.integers(20, 70) * sink_multiplier)  # sink knob burada

                    balance[i] += earn
                    events.append({
                        "event_time": t0 + pd.Timedelta(seconds=10 + 20*lv),
                        "user_id": user_ids[i],
                        "install_time": install_times[i],
                        "event_name": "currency_earn",
                        "session_id": session_id,
                        "level": lv + 1,
                        "currency_delta": earn,
                        "currency_balance": int(balance[i]),
                        "iap_usd": 0.0,
                        "ad_usd": 0.0,
                        "segment": segments[i],
                        "ab_group": ab_group[i],
                        "cohort_day": d,
                    })

                    balance[i] = max(0, balance[i] - spend)
                    events.append({
                        "event_time": t0 + pd.Timedelta(seconds=15 + 20*lv),
                        "user_id": user_ids[i],
                        "install_time": install_times[i],
                        "event_name": "currency_spend",
                        "session_id": session_id,
                        "level": lv + 1,
                        "currency_delta": -spend,
                        "currency_balance": int(balance[i]),
                        "iap_usd": 0.0,
                        "ad_usd": 0.0,
                        "segment": segments[i],
                        "ab_group": ab_group[i],
                        "cohort_day": d,
                    })

                    events.append({
                        "event_time": t0 + pd.Timedelta(seconds=18 + 20*lv),
                        "user_id": user_ids[i],
                        "install_time": install_times[i],
                        "event_name": "level_complete",
                        "session_id": session_id,
                        "level": lv + 1,
                        "currency_delta": 0,
                        "currency_balance": int(balance[i]),
                        "iap_usd": 0.0,
                        "ad_usd": 0.0,
                        "segment": segments[i],
                        "ab_group": ab_group[i],
                        "cohort_day": d,
                    })

                # ad impression
                if rng.random() < p_ad[segments[i]]:
                    ad_usd = float(rng.uniform(0.002, 0.02))  # simplistic per-impression revenue
                    events.append({
                        "event_time": t0 + pd.Timedelta(seconds=120),
                        "user_id": user_ids[i],
                        "install_time": install_times[i],
                        "event_name": "ad_impression",
                        "session_id": session_id,
                        "level": None,
                        "currency_delta": 0,
                        "currency_balance": int(balance[i]),
                        "iap_usd": 0.0,
                        "ad_usd": ad_usd,
                        "segment": segments[i],
                        "ab_group": ab_group[i],
                        "cohort_day": d,
                    })

                # iap purchase
                if rng.random() < p_iap[segments[i]]:
                    base_price = rng.choice([0.99, 2.99, 4.99, 9.99])
                    if ab_group[i] == "treatment":
                        base_price *= price_multiplier  # pricing test knob burada
                    events.append({
                        "event_time": t0 + pd.Timedelta(seconds=180),
                        "user_id": user_ids[i],
                        "install_time": install_times[i],
                        "event_name": "iap_purchase",
                        "session_id": session_id,
                        "level": None,
                        "currency_delta": 0,
                        "currency_balance": int(balance[i]),
                        "iap_usd": float(base_price),
                        "ad_usd": 0.0,
                        "segment": segments[i],
                        "ab_group": ab_group[i],
                        "cohort_day": d,
                    })

    df = pd.DataFrame(events).sort_values("event_time").reset_index(drop=True)
    return df

df = generate_events(n_users=10000, days=14, sink_multiplier=1.2, price_multiplier=0.9)
print(df.head())
print("rows:", len(df))

# save
df.to_parquet("events.parquet", index=False)
