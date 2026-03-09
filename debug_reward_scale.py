import argparse


def compute_reward(gmv, orr, gmv_weight, orr_weight, gmv_scale):
    return gmv_weight * (gmv * gmv_scale) + orr_weight * orr


def main():
    parser = argparse.ArgumentParser(description="Debug reward scale for GMV vs ORR modes.")
    parser.add_argument("--gmv-high", type=float, default=1500.0)
    parser.add_argument("--gmv-low", type=float, default=200.0)
    parser.add_argument("--orr-high", type=float, default=0.9)
    parser.add_argument("--orr-low", type=float, default=0.3)
    parser.add_argument("--gmv-scale", type=float, default=1.0)
    parser.add_argument("--rush-gmv", type=float, default=0.05)
    parser.add_argument("--rush-orr", type=float, default=0.1)
    parser.add_argument("--normal-gmv", type=float, default=0.001)
    parser.add_argument("--normal-orr", type=float, default=2.0)
    args = parser.parse_args()

    cases = [
        ("High GMV / Low ORR", args.gmv_high, args.orr_low),
        ("Low GMV / High ORR", args.gmv_low, args.orr_high),
    ]

    modes = [
        ("Rush (GMV-focused)", args.rush_gmv, args.rush_orr),
        ("Normal (ORR-focused)", args.normal_gmv, args.normal_orr),
    ]

    print("GMV scale:", args.gmv_scale)
    for mode_name, gmv_w, orr_w in modes:
        print(f"\n[{mode_name}] weights: gmv={gmv_w}, orr={orr_w}")
        rewards = []
        for label, gmv, orr in cases:
            r = compute_reward(gmv, orr, gmv_w, orr_w, args.gmv_scale)
            rewards.append(r)
            print(f"  {label}: GMV={gmv}, ORR={orr} -> reward={r:.4f}")
        if rewards[1] != 0:
            ratio = rewards[0] / rewards[1]
        else:
            ratio = float("inf")
        print(f"  Reward ratio (HighGMV/LowORR vs LowGMV/HighORR): {ratio:.3f}")


if __name__ == "__main__":
    main()
