import os
from datetime import datetime
from pathlib import Path

import git


def get_new_model_version(model_dir: str) -> str:
    """
    A model will have multiple runs. Each run will have a different version.
    """
    versions = []
    for version_dir in os.listdir(model_dir):
        try:
            versions.append(int(version_dir))
        except:
            print(f'Invalid subdirectory:{model_dir}/{version_dir}. Only integer versions are allowed')
            exit()
    if len(versions) == 0:
        return '0'
    return f'{max(versions) + 1}'


def get_model_name(config):
    std = config['psf_list']
    return f"{config['name']}_N{len(config['psf_list'])}_{min(std)}-{max(std)}"


def get_month():
    return datetime.now().strftime("%y%m")


def get_workdir(config, root_dir, use_max_version):
    rel_path = get_month()
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    rel_path = os.path.join(rel_path, get_model_name(config))
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    if use_max_version:
        # Used for debugging.
        version = int(get_new_model_version(cur_workdir))
        if version > 0:
            version = f'{version - 1}'

        rel_path = os.path.join(rel_path, str(version))
    else:
        rel_path = os.path.join(rel_path, get_new_model_version(cur_workdir))

    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)
    return cur_workdir


def add_git_info(config):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    repo = git.Repo(dir_path, search_parent_directories=True)
    config['git_changedFiles'] = [item.a_path for item in repo.index.diff(None)]
    config['git_branch'] = repo.active_branch.name
    config['git_untracked_files'] = repo.untracked_files
    config['git_latest_commit'] = repo.head.object.hexsha
