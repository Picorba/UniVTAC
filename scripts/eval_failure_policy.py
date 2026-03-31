"""
eval_failure_policy.py

Like eval_policy.py, but for failure analysis: records the domain randomization
parameters actually drawn each episode (stored by the task in self.domain_rand_params)
alongside the success/failure outcome, then generates:
  - 1-D success-rate curves vs each parameter (binned)
  - 2-D scatter plots for all pairs of parameters
"""

from shutil import ExecError
import sys

sys.path.append(".")
sys.path.append(f"./policy")

import time
import json
import yaml
import argparse
import traceback
from pathlib import Path
from typing import Literal

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval Policy – Failure Analysis")
parser.add_argument("task_name",     type=str, help="Task name")
parser.add_argument("task_config",   type=str, help="Task config")
parser.add_argument("deploy_config", type=str, help="Deploy file name")
parser.add_argument("--expert_check", action='store_true')
parser.add_argument("--start_seed",  type=int, default=-1)
parser.add_argument("--max_seed",    type=int, default=-1)
parser.add_argument("--total_num",   type=int, default=100)
parser.add_argument("--print_only",  action='store_true')
parser.add_argument("--plot_only",   action='store_true',
                    help="Skip eval, regenerate plots from existing failure_records.json")
parser.add_argument("--n_bins",      type=int, default=10,
                    help="Number of bins for success-rate curves")
parser.add_argument("--density",     type=int, default=-1,
                    help="Density override for lift_oscar (sets LIFT_OSCAR_DENSITY env var). -1 means use default.")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.livestream = 0
args_cli.headless = False
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg
    from policy._base_policy import BasePolicy

log_path = Path('./log')


def log(msg):
    global log_path, args_cli
    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    if not args_cli.print_only:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
    print(msg)


def eval_failure_policy(
    task: 'BaseTask', policy: 'BasePolicy', expert_check,
    start_seed, max_seed, test_total_num, instructions,
    instruciton_type: Literal['seen', 'unseen'] = 'seen'
) -> list[dict]:
    """
    Evaluate the policy and return per-episode records:
        { success: bool, domain_rand_params: dict, eval_cost: float }

    domain_rand_params is whatever the task stores in self.domain_rand_params
    after reset (set in _reset_actors / pre_move for task-specific randomisation).
    """
    records = []
    test_num, succ_num, seed = 0, 0, start_seed

    seed_path = task.save_root.parent / 'seeds.json'
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    if seed_path.exists():
        with open(seed_path, 'r') as f:
            seed_status = json.load(f)
    else:
        seed_status = {}

    while test_num < test_total_num and (max_seed == -1 or seed <= max_seed):
        if not seed_status.get(str(seed), True):
            seed += 1
            continue

        if expert_check and str(seed) not in seed_status:
            test_start = time.perf_counter()
            task.mode = 'eval_test'
            try:
                task.reset(seed=seed)
                task.play_once()
                if not task.check_success() or not task.plan_success:
                    raise ExecError(f'seed {seed} Expert check failed, '
                                    f'check {task.check_success()}, plan {task.plan_success}.')
                seed_status[seed] = True
                with open(seed_path, 'w') as f:
                    json.dump(seed_status, f)
                test_cost = time.perf_counter() - test_start
                task.clean_cache(result='test_success')
                log(f'Expert check succ, seed {seed}, cost {test_cost:.2f}s')
            except Exception as e:
                test_cost = time.perf_counter() - test_start
                log(f'Expert check failed, seed {seed}, cost {test_cost:.2f}s, '
                    f'with exception {e}')
                task.clean_cache(result='test_fail')
                seed_status[seed] = False
                with open(seed_path, 'w') as f:
                    json.dump(seed_status, f)
                seed += 1
                continue

        test_num += 1
        succ = False
        eval_start = time.perf_counter()
        task.mode = 'eval'

        try:
            task.reset(seed=seed, instructions=instructions[instruciton_type])
            task.mean_steps = task.cfg.step_lim

            # Read domain randomisation values drawn this episode.
            # The task populates self.domain_rand_params in _reset_actors / pre_move.
            rand_params = dict(getattr(task, 'domain_rand_params', {}))

            policy.reset()
            while task.take_action_cnt < task.cfg.step_lim:
                observation = task._get_observations()
                policy.eval(task, observation)
                if task.eval_success:
                    succ = True
                    break
                if task.check_early_stop():
                    break

        except Exception as e:
            log(f"[{test_num:<3d}] Seed {seed} occurred exception: "
                f"{e}\n{traceback.format_exc()}")
            task.clean_cache(result='error')
            test_num -= 1

        else:
            eval_cost = time.perf_counter() - eval_start
            if succ:
                succ_num += 1
            succ_status = 'success' if succ else 'failed'
            task.clean_cache(result=succ_status)
            log(f"[{test_num:<3d}] Seed {seed} {succ_status} after {eval_cost:.2f} s.\n"
                f"steps: {task.step_count:<5d}, actions: {task.take_action_cnt:<5d}.\n"
                f"Instruction: {task.instruction}\n"
                f"Total {succ_num}/{test_num}({succ_num/test_num*100:.2f}%) success.")

            records.append({
                'success': succ,
                'domain_rand_params': rand_params,
                'eval_cost': eval_cost,
            })

        finally:
            seed += 1

    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plots(records: list[dict], output_dir: Path, n_bins: int = 10):
    """
    For each domain randomisation parameter:
      - 1-D success-rate curve (binned along that axis)
    For each pair of parameters:
      - 2-D scatter plot coloured by success / failure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    if not records:
        print("No records to plot.")
        return

    param_names = list(records[0]['domain_rand_params'].keys())
    if not param_names:
        print("No domain_rand_params found in records – nothing to plot.")
        return

    successes = np.array([r['success'] for r in records], dtype=bool)
    data = {
        p: np.array([r['domain_rand_params'][p] for r in records])
        for p in param_names
    }

    overall_rate = successes.mean()

    # ------------------------------------------------------------------ #
    # 1-D success-rate curves                                             #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, len(param_names),
                              figsize=(5 * len(param_names), 4))
    if len(param_names) == 1:
        axes = [axes]

    for ax, param in zip(axes, param_names):
        vals = data[param]
        eff_bins = min(n_bins, max(2, len(records) // 3))
        edges = np.linspace(vals.min(), vals.max(), eff_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        rates, counts = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (vals >= lo) & (vals <= hi)
            if mask.sum() > 0:
                rates.append(float(successes[mask].mean()))
                counts.append(int(mask.sum()))
            else:
                rates.append(float('nan'))
                counts.append(0)

        rates  = np.array(rates)
        valid  = ~np.isnan(rates)

        ax.plot(centers[valid], rates[valid],
                marker='o', color='steelblue', linewidth=2, markersize=6)
        ax.fill_between(centers[valid], rates[valid], alpha=0.15, color='steelblue')
        ax.axhline(overall_rate, color='gray', linestyle='--', alpha=0.6,
                   label=f'overall {overall_rate*100:.1f}%')

        for xc, r, n in zip(centers[valid], rates[valid], np.array(counts)[valid]):
            ax.annotate(f'n={n}', (xc, r),
                        textcoords='offset points', xytext=(0, 7),
                        ha='center', fontsize=7, color='dimgray')

        ax.set_ylim(-0.05, 1.10)
        ax.set_xlabel(param)
        ax.set_ylabel('Success rate')
        ax.set_title(f'success rate vs {param}')
        ax.legend(fontsize=8)

    fig.suptitle('Success Rate vs Domain Randomisation Parameters', fontsize=13)
    fig.tight_layout()
    fname = output_dir / 'success_rate_curves.png'
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"Saved: {fname}")

    # ------------------------------------------------------------------ #
    # 2-D scatter plots for each pair of parameters                       #
    # ------------------------------------------------------------------ #
    pairs = [(param_names[i], param_names[j])
             for i in range(len(param_names))
             for j in range(i + 1, len(param_names))]

    for p1, p2 in pairs:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(data[p1][~successes], data[p2][~successes],
                   c='red',   alpha=0.7, label='Failure', s=55,
                   marker='x', linewidths=1.5)
        ax.scatter(data[p1][successes],  data[p2][successes],
                   c='green', alpha=0.6, label='Success', s=40, marker='o')
        ax.set_xlabel(p1)
        ax.set_ylabel(p2)
        ax.set_title(f'{p1} vs {p2}  —  Success / Failure\n'
                     f'({successes.sum()}/{len(successes)} success, '
                     f'{overall_rate*100:.1f}%)')
        ax.legend()
        fig.tight_layout()
        fname = output_dir / f'scatter_{p1}_vs_{p2}.png'
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"Saved: {fname}")

    print(f"\nOverall: {successes.sum()}/{len(records)} "
          f"({overall_rate*100:.1f}%) success")
    print(f"Plots saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Config helpers (same as eval_policy.py)
# ---------------------------------------------------------------------------

def get_config(file, default_root: Path, type: Literal['yaml', 'json']):
    if type == 'yaml':
        if file.endswith('.yml') or file.endswith('.yaml'):
            file = Path(file)
        else:
            file = default_root / f'{file}.yml'
        with open(file, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        if file.endswith('.json'):
            file = Path(file)
        else:
            file = default_root / f'{file}.json'
        with open(file, 'r') as f:
            config = json.load(f)
        return config, file


task_module, policy_module = None, None


def main():
    global args_cli, task_module, policy_module, log_path

    task_file_name = args_cli.task_name
    task_config, task_config_file = get_config(
        args_cli.task_config,
        default_root=Path(__file__).parent.parent / 'task_config',
        type='yaml'
    )
    deploy_config, deploy_config_file = get_config(
        args_cli.deploy_config,
        default_root=Path(__file__).parent.parent / 'policy',
        type='yaml'
    )
    policy_name = deploy_config['policy_name']
    deploy_config['task_name'] = task_file_name
    deploy_config['task_config'] = task_config_file.stem

    deploy_config['instuction_file'] = deploy_config.get('instuction_file', task_file_name)
    if deploy_config['instuction_file'] is not None:
        instructions, _ = get_config(
            deploy_config['instuction_file'],
            default_root=Path(__file__).parent.parent / 'instructions',
            type='json'
        )
    else:
        instructions = {'seen': ['Empty'], 'unseen': ['Empty']}

    if args_cli.density != -1:
        import os
        os.environ['LIFT_OSCAR_DENSITY'] = str(args_cli.density)

    task_module   = importlib.import_module(f"envs.{task_file_name}")
    policy_module = importlib.import_module(f"policy.{policy_name}")

    curr_time = time.strftime(r'%Y-%m-%d_%H:%M:%S')

    env_cfg: 'BaseTaskCfg' = task_module.TaskCfg()
    save_dir = Path('failure_analysis') / policy_name / task_file_name / deploy_config_file.stem
    if args_cli.density != -1:
        save_dir = save_dir / f'density_{args_cli.density}'
    env_cfg.save_dir = save_dir / curr_time
    env_cfg.decimation      = task_config.get("decimation",      env_cfg.decimation)
    env_cfg.obs_data_type   = task_config.get("observations",    {})
    env_cfg.save_frequency  = task_config.get("save_frequency",  env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.random_texture  = task_config.get("random_texture",  False)

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    seed = deploy_config.get("seed", 0)

    init_start = time.perf_counter()
    policy: 'BasePolicy' = policy_module.Policy(deploy_config)
    policy_init_cost = time.perf_counter() - init_start

    init_start = time.perf_counter()
    task: 'BaseTask' = task_module.Task(env_cfg, mode='eval')
    task_init_cost = time.perf_counter() - init_start

    import os
    if os.environ.get('TRAIN_CONFIG'):
        deploy_config['train_config'] = os.environ['TRAIN_CONFIG']

    log_path = task.save_root / 'log.log'
    log(f"Task Name:   {task_file_name}")
    log(f"Task Config: {task_config_file.absolute()}")
    log(f"Eval Config: {json.dumps(deploy_config, ensure_ascii=False, indent=4)}\n{'-'*20}\n")
    log(f"Task init finish in {task_init_cost:.2f} seconds.")
    log(f"Policy init finish in {policy_init_cost:.2f} seconds.")

    records_path = task.save_root / 'failure_records.json'

    if args_cli.plot_only:
        if not records_path.exists():
            raise FileNotFoundError(
                f"--plot_only requested but {records_path} does not exist. "
                "Run a full eval first."
            )
        with open(records_path, 'r') as f:
            records = json.load(f)
        log(f"Loaded {len(records)} records from {records_path}")
    else:
        records = eval_failure_policy(
            task=task, policy=policy,
            expert_check=args_cli.expert_check,
            start_seed=(1000000 * (1 + seed)
                        if args_cli.start_seed == -1 else args_cli.start_seed),
            max_seed=args_cli.max_seed,
            test_total_num=args_cli.total_num,
            instructions=instructions,
            instruciton_type=deploy_config.get("instruction_type", "seen")
        )

        with open(records_path, 'w') as f:
            json.dump(records, f, indent=2)
        log(f"Records saved to {records_path}")

        n_succ = sum(r['success'] for r in records)
        log(f"Final Result: {n_succ}/{len(records)}"
            f"({n_succ/max(len(records), 1)*100:.2f}%) success.")

    task.close()
    policy.close()
    simulation_app.close()

    # Generate plots after the sim closes – matplotlib does not need it
    generate_plots(records, task.save_root / 'plots', n_bins=args_cli.n_bins)


if __name__ == "__main__":
    main()
