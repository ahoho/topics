import argparse
import copy
import operator
import random
import shutil
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# from git import Repo
from sklearn.model_selection import ParameterGrid

import yaml

try:
    import git
    _git_available = False
except ImportError:
    warnings.warn("`GitPython` not available, so no git log will be saved")
    _git_available = False

def load_yaml(path):
    with open(path, "r") as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)


def save_yaml(yml, path):
    with open(path, "w") as outfile:
        return yaml.dump(yml, outfile)


def load_text(path):
    with open(path, "r") as infile:
        return infile.read()


def save_text(text, path):
    with open(path, "w") as outfile:
        return outfile.write(text)


def create_git_log(path=None, include_untracked=False):
    """
    For the current respository, save the current state.
    If the repo is clean, record the commit hash
    """
    try:
        repo = git.Repo(path=path, search_parent_directories=False)
    except git.exc.IvalidGitRepositoryError:
        return ""

    is_dirty = ""
    if repo.is_dirty() or (include_untracked and repo.untracked_files):
        warnings.warn(f"git repo at {repo.working_dir} is dirty. Please commit.")
        is_dirty = " (dirty)"

    commit = repo.head.commit # head
    log = (
        f"{path}:\n"
        f"\tOn branch {repo.active_branch}{is_dirty}\n"
        f"\tcommit {commit.hexsha}\n"
        f"\tDate: {commit.committed_datetime:%Y-%m-%d %H:%M:%S %z}\n\n"
        f"\t\t{commit.message}"
    )

    return log


def save_code(code_locations, base_output_path):
    """
    Store git logs of code being used in sweep as well as the code itself
    """
    # create a git log
    if _git_available:
        code_paths = set()
        git_logs = ""
        for path in code_locations:
            path = Path(path)
            path = path if path.is_dir() else path.parent
            if path not in code_paths:
                git_logs += f"{create_git_log(path)}\n"
                code_paths.update([path])
        save_text(git_logs, Path(base_output_path, f"git.log"))

    # copy the code
    code_dir = Path(base_output_path, f"stored_code")
    code_dir.mkdir(exist_ok=True)
    pcounts = defaultdict(int)
    for path in code_locations:
        path = Path(path)
        name = path.name

        # make sure we don't overwrite anybody
        pcounts[name] += 1
        newpath = str(code_dir / name)
        if pcounts[name] > 1:
            newpath += f"-{pcounts[name]}"

        # copy directory or file
        if path.is_dir():
            shutil.copytree(path, newpath, dirs_exist_ok=True)
        else:
            shutil.copy(path, newpath)


def apply_constraints(params, constraints):
    """
    Ensure that `params` meet `constraints`
    `constraints` is a list of (parameter, constraint function, constraint parameter) tuples
    """
    if constraints is None:
        return True
    
    operator_map = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne,
    }

    for param_name, operator_name, constraint_name in constraints:
        try:
            op = operator_map[operator_name]
        except KeyError:
            raise KeyError(
                "Constraints must be one of "
                "<, <=, >, >=, ==, !="
            )
        if not op(params[param_name], params[constraint_name]):
            return False
    return True


def hyper_to_configs(hyper_conf, random_runs=None, seed=42):
    """
    Create the set of configurations from the grid
    """
    configs = []
    hyper_settings = hyper_conf.pop("hyper", None)
    constraints = hyper_conf.pop("constraints", None)
    name_template = hyper_conf["templates"].pop("run_name", None)

    if hyper_settings is not None:
        # Convert list of hyperparams to sweep
        grid = ParameterGrid(hyper_settings)
        if random_runs:
            random.seed(seed)
            random_runs = None if random_runs == -1 else random_runs
            grid = sorted(grid, key=lambda k: random.random())[:random_runs]
        
        for params in grid:
            if constraints is None or apply_constraints(params, constraints):
                # Deep copy to avoid overwriting
                conf = copy.deepcopy(hyper_conf)
                # Fill in value of each configuation
                conf["params"].update(params)
                if name_template:
                    # remap to friendly names
                    renamed_params = {
                        k: hyper_settings[k][v]
                        for k, v in params.items()
                        if isinstance(hyper_settings[k], dict)
                    }
                    name_params = {**conf['params'], **renamed_params}
                    conf["conf_name"] = name_template.format(**name_params)
                else:
                    # Possible that microsecond (%f) isn't sufficiently granular to
                    # ensure uniqueness, but I doubt it
                    conf["conf_name"] = f"{datetime.now():%Y%m%d_%H%M%S%f}"
                configs.append(conf)
        # Make sure no repeated names
        assert(len(set(conf["conf_name"] for conf in configs)) == len(configs))
        return configs
    else:
        return [hyper_conf]


def hyper(
    hyper_settings_yml_path: str,
    base_output_path: str,
    base_config_yml_path: str = None,
    random_runs: int = -1,
    skip_if_file_exists: str = None,
    dry_run: bool = False,
    seed: int = None,
    ):
    """
    Generate batch file and directories for the runs
    """
    random.seed(seed)

    # 1) Generate all the configuration files and directories
    hyper_conf = load_yaml(hyper_settings_yml_path)
    
    # hyper_conf_path is a yml file defining the hyper parameter sweep
    configs = hyper_to_configs(hyper_conf, random_runs, seed=seed)

    # base_config_yml_path is the template for the configuration
    base_config = load_yaml(base_config_yml_path) if base_config_yml_path else {}
    job_name = hyper_conf["job_name"]
    run_template = hyper_conf["templates"]["command"]
    code_locations = hyper_conf.get("code_locations", [])
    commands = []

    n_runs = 0
    for c in configs:
        # This defines the path like models/{base_output_path}/{conf_name}
        conf_name = c["conf_name"]

        # Create the output directory and the config file
        output_dir = Path(base_output_path, conf_name).absolute()
        conf_path = Path(output_dir, "config.yml")
        if skip_if_file_exists and Path(output_dir, skip_if_file_exists).exists():
            continue

        n_runs += 1
        if dry_run:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        filled_conf = {**base_config, **c["params"]}
        save_yaml(filled_conf, conf_path)

        run_command = run_template.format(config_path=conf_path, output_dir=output_dir)
        commands.append(run_command)

    print(f"Found {n_runs} configurations.")
    if dry_run:
        return

    # add slurm-specific items
    base_log_dir = Path(base_output_path, "_run-logs")
    base_log_dir.mkdir(exist_ok=True)
    slurm_template = hyper_conf["templates"].pop("slurm_header", None)
    if slurm_template:
        slurm_log_dir = base_log_dir / "slurm-logs"
        slurm_log_dir.mkdir(exist_ok=True)
        slurm_header = slurm_template.format(n_jobs=len(commands)-1, job_name=job_name, log_dir=slurm_log_dir)
        commands = [slurm_header] + [
            f"test ${{SLURM_ARRAY_TASK_ID}} -eq {run_id} && {run_command}"
            for run_id, run_command in enumerate(commands)
        ]

    # save runs here
    save_text("\n".join(commands), f"{job_name}-runs.sh")

    # for reproducability, save the code and re-save the runs
    current_run = f"{datetime.now():%Y%m%d_%H%M%S}-{job_name}"
    current_log_dir = Path(base_log_dir, current_run)
    current_log_dir.mkdir()
    if code_locations:
        save_code(code_locations, current_log_dir)
    shutil.copy(hyper_settings_yml_path, current_log_dir)
    save_text(
        "\n".join(commands),
        current_log_dir / f"{job_name}-runs.sh"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hyper_settings_yml_path")
    parser.add_argument("base_output_path")
    parser.add_argument(
        "--base_config_yml_path",
        default=None,
        help="Standard params can be placed here if desired"
    )
    parser.add_argument(
        "--random_runs",
        type=int,
        default=None,
        help="Randomize the grid and use the first `random_runs` runs. Use all with -1"
    )
    parser.add_argument(
        "--skip_if_file_exists",
        type=str,
        default=None,
        help="If this file exists in an output model directory, do not create a run"
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=11235)
    
    args = parser.parse_args()
    hyper(
        hyper_settings_yml_path=args.hyper_settings_yml_path,
        base_output_path=args.base_output_path,
        base_config_yml_path=args.base_config_yml_path,
        random_runs=args.random_runs,
        skip_if_file_exists=args.skip_if_file_exists,
        dry_run=args.dry_run,
        seed=args.seed,
    )