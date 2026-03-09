import argparse
import csv
import os
import sys
from datetime import datetime, timedelta

import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from tools.load_data import load_envs_custom, load_envs_NYU143
from algo.MAPPO import PPO

try:
    from llm_instructor import GlobalInstructionGenerator
except Exception:
    GlobalInstructionGenerator = None


def _find_latest_best(log_root: str) -> str:
    latest = None
    latest_mtime = -1
    for root, _, files in os.walk(log_root):
        if "Best.pkl" in files:
            path = os.path.join(root, "Best.pkl")
            mtime = os.path.getmtime(path)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest = path
    if latest is None:
        raise FileNotFoundError(f"No Best.pkl found under {log_root}")
    return latest


def _load_args():
    run_dir = os.path.join(os.path.dirname(__file__), "run")
    sys.path.append(run_dir)
    import run_MAPPO  # type: ignore

    argv_backup = sys.argv[:]
    sys.argv = [argv_backup[0]]
    try:
        return run_MAPPO.get_parameter()
    finally:
        sys.argv = argv_backup


def _load_checkpoint(agent: PPO, ckpt_path: str, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device)

    def _filter_state(src, dst):
        filtered = {}
        skipped = []
        for k, v in src.items():
            if k in dst and v.shape == dst[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        return filtered, skipped

    actor_state = agent.actor.state_dict()
    critic_state = agent.critic.state_dict()

    actor_filtered, actor_skipped = _filter_state(state["actor net"], actor_state)
    critic_filtered, critic_skipped = _filter_state(state["critic net"], critic_state)

    agent.actor.load_state_dict(actor_filtered, strict=False)
    agent.critic.load_state_dict(critic_filtered, strict=False)

    if actor_skipped or critic_skipped:
        print(
            f"Warning: skipped {len(actor_skipped)} actor keys and {len(critic_skipped)} critic keys due to shape mismatch."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to Best.pkl (leave empty to auto-detect latest).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join("data", "NYC2015Jan_h3_Surge300.pkl"),
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="surge_eval_log.csv",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Fixed instruction text for LLM embedding (zero-shot transfer).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM instruction embedding (paper-standard CoopRide).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2015-01-26",
        help="Start date for real_time column, format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--drivers",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--max-order-num",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax (deterministic) action selection instead of sampling.",
    )
    parser.add_argument(
        "--debug-steps",
        type=int,
        default=0,
        help="Print available orders and served orders for the first N steps.",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=144,
    )
    parser.add_argument(
        "--high-value-ratio",
        type=float,
        default=0.0,
        help="Fraction of real orders to prioritize as high-value (0 disables quota).",
    )
    parser.add_argument(
        "--min-orr",
        type=float,
        default=0.0,
        help="Disable high-value bias when ORR falls below this threshold (0 disables).",
    )
    args_cli = parser.parse_args()
    os.chdir(os.path.join(os.path.dirname(__file__), "run"))

    args = _load_args()
    args.custom_data_path = args_cli.data_path
    args.driver_num = args_cli.drivers
    args.TIME_LEN = args_cli.time_steps
    args.feature_normal = 3
    args.log_feature = False

    os.environ["COOPRIDE_MAX_ORDER_NUM"] = str(args_cli.max_order_num)
    os.environ["COOPRIDE_DISPATCH_LIMIT"] = str(args_cli.max_order_num)
    if args_cli.high_value_ratio > 0:
        os.environ["COOPRIDE_HIGH_VALUE_RATIO"] = str(args_cli.high_value_ratio)
    if args_cli.min_orr > 0:
        os.environ["COOPRIDE_MIN_ORR"] = str(args_cli.min_orr)

    data_basename = os.path.basename(args.custom_data_path) if args.custom_data_path else ""
    if data_basename.lower() == "nyu_grid143.pkl":
        env, _, _, _, args.grid_num = load_envs_NYU143(driver_num=args.driver_num)
    else:
        env, _, _, _, args.grid_num = load_envs_custom(
            args.custom_data_path, driver_num=args.driver_num
        )
    if hasattr(env, "fleet_help"):
        env.fleet_help = True
    if hasattr(env, "n_intervals"):
        args.TIME_LEN = min(args.TIME_LEN, int(env.n_intervals))

    instruction_text = args_cli.instruction.strip() if args_cli.instruction else ""
    if not args_cli.no_llm:
        if GlobalInstructionGenerator is None:
            if instruction_text:
                raise RuntimeError("llm_instructor.py not available for instruction embedding.")
        else:
            generator = GlobalInstructionGenerator(
                enabled=True,
                override_text=instruction_text or None,
            )
            global_vec = generator.get_instruction_vector(env, 0)
            env.set_global_instruction_vector(global_vec)
            env.state_space = None
            norm = float(torch.norm(torch.tensor(global_vec, dtype=torch.float32)))
            print(f"DEBUG: Goal Embedding Norm: {norm:.6f}")

    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")

    agent = PPO(env, args, device)
    agent.move_device(device)
    agent.actor.eval()
    agent.critic.eval()
    print(f"DEBUG: agent.max_order_num={agent.max_order_num}, grid_num={agent.agent_num}")

    ckpt_path = args_cli.checkpoint
    if not ckpt_path:
        log_root = os.path.join(os.path.dirname(__file__), "logs")
        ckpt_path = _find_latest_best(log_root)
    _load_checkpoint(agent, ckpt_path, device)

    start_dt = datetime.strptime(args_cli.start_date, "%Y-%m-%d")
    step_minutes = args.dispatch_interval

    states_node, _, order_states, order_idx, order_feature, _ = env.reset(mode="PPO2")
    state = agent.process_state(states_node, 0)
    state_rnn_actor = torch.zeros((1, agent.agent_num, agent.hidden_dim), dtype=torch.float)
    state_rnn_critic = torch.zeros((1, agent.agent_num, agent.hidden_dim), dtype=torch.float)
    order, mask_order = agent.process_order(order_states, 0)
    order = agent.remove_order_grid(order)
    mask_order = agent.mask_fake(order, mask_order)

    total_demand_sum = 0
    total_served_sum = 0
    total_gmv = 0.0

    with open(args_cli.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "real_time",
                "total_demand",
                "new_orders",
                "orders_served",
                "gap",
                "orr",
                "orr_pending",
                "gmv",
                "avg_driver_income",
                "reposition_count",
            ]
        )

        total_reposition_sum = 0
        for step in range(args.TIME_LEN):
            total_new = 0
            pending_real = 0
            for node in env.nodes:
                if node is not None:
                    total_new += int(getattr(node, "new_order", 0))
                    pending_real += int(getattr(node, "real_order_num", 0))

            total_demand = pending_real

            prev_finished = env.record_finish_order

            action, local_value, global_value, logp, mask_agent, mask_order_multi, mask_action, mask_entropy, next_state_rnn_actor, next_state_rnn_critic, action_ids, selected_ids = agent.action(
                state,
                order,
                state_rnn_actor,
                state_rnn_critic,
                mask_order,
                order_idx,
                device,
                sample=not args_cli.deterministic,
                random_action=False,
                FM_mode=args.FM_mode,
            )

            orders = env.get_orders_by_id(action_ids)
            reposition_count = 0
            for node_orders in orders:
                for order in node_orders:
                    service_type = getattr(order, "service_type", getattr(order, "_service_type", -1))
                    try:
                        if int(service_type) > 0:
                            reposition_count += 1
                    except (TypeError, ValueError):
                        continue
            next_states_node, next_order_states, next_order_idx, next_order_feature = env.step(
                orders, generate_order=1, mode="PPO2"
            )

            orders_served = env.record_finish_order - prev_finished
            gmv = float(env.gmv)
            gap = total_demand - orders_served
            orr_pending = float(orders_served) / max(1, total_demand)
            orr = float(getattr(env, "order_response_rate", -1.0))
            if orr < 0:
                orr = orr_pending
            avg_driver_income = gmv / orders_served if orders_served > 0 else 0.0
            real_time = (start_dt + timedelta(minutes=step_minutes * step)).strftime("%Y-%m-%d %H:%M")

            if args_cli.debug_steps > 0 and step < args_cli.debug_steps:
                pending_real = 0
                for node in env.nodes:
                    if node is not None:
                        pending_real += int(getattr(node, "real_order_num", 0))
                print(
                    f"[DEBUG step {step}] available_real={pending_real}, served={orders_served}, "
                    f"total_new={total_demand}, orr_step={orr:.4f}"
                )

            writer.writerow(
                [
                    step,
                    real_time,
                    total_demand,
                    total_new,
                    orders_served,
                    gap,
                    orr,
                    orr_pending,
                    gmv,
                    avg_driver_income,
                    reposition_count,
                ]
            )

            total_demand_sum += total_demand
            total_served_sum += orders_served
            total_gmv += gmv
            total_reposition_sum += reposition_count

            state = agent.process_state(next_states_node, step + 1)
            order, mask_order = agent.process_order(next_order_states, step + 1)
            order = agent.remove_order_grid(order)
            mask_order = agent.mask_fake(order, mask_order)
            state_rnn_actor = next_state_rnn_actor
            state_rnn_critic = next_state_rnn_critic
            order_idx = next_order_idx

    mean_orr = float(total_served_sum) / max(1, total_demand_sum)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Total GMV: {total_gmv:.4f}")
    print(f"Mean ORR: {mean_orr:.6f}")
    print(f"Total Repositions: {total_reposition_sum}")
    print(f"CSV saved: {args_cli.out_csv}")


if __name__ == "__main__":
    main()
