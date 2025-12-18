from datetime import datetime
from pathlib import Path
import pandas as pd
import mesa
import matplotlib.pylab as plt
import yaml
import copy

from model import PublishingModel, DEFAULT_JOURNAL_SPECS

from visualization_utils import *

def no_economics_exp(gamma, beta_p, journal_specs, folder):
    params = {
        "n_groups": 1000,
        "n_journals": 100,
        "enable_economics": True,
        "prestige_decay": 0.001,
        "prestige_social_multiplier": 0,
        "journal_reputation_decay": 0.01,
        "journal_setup": [journal_specs],
        "researcher_preferences": [(1.0, 0.0, 0.0)],
        "g0": 1000,
        "gamma": gamma,
        "beta_p": beta_p,
        "budget_slope_lambda": 100,
    }

    max_steps = 400

    now = datetime.now()
    dir_name = Path(f'experiments/{now.strftime("%Y-%m-%d %H-%M-%S")} {folder}')
    Path.mkdir(dir_name, exist_ok=True, parents=True)

    # save params as .yaml file
    yaml_path = dir_name / 'parameters.yaml'
    with open(yaml_path, 'w') as f:
        # Dump the full params. ensure journal_specs is serializable (dicts/lists)
        yaml.dump(params, f, default_flow_style=False)

    result = mesa.batch_run(
        PublishingModel,
        number_processes=None,
        iterations=100,
        data_collection_period=1,
        parameters=params,
        max_steps=max_steps
    )

    df = pd.DataFrame(result)

    # Prestige & reputation
    plot_dynamics(df)
    plt.savefig(dir_name / 'fig_1.png', dpi=300)
    plt.close()

    # Dynamics of Papers
    plot_category_dynamics(df)
    plt.savefig(dir_name / 'fig_2.png', dpi=300)
    plt.close()

    # Distributions Start vs End
    plot_start_end_distribution_reputation(df, max_steps=max_steps)
    plt.savefig(dir_name / 'fig_3.png', dpi=300)
    plt.close()

    # Distributions Start vs End
    plot_start_end_distribution_prestige(df, max_steps=max_steps)
    plt.savefig(dir_name / 'fig_4.png', dpi=300)
    plt.close()

    # Gini Coefficients
    plot_gini(df)
    plt.savefig(dir_name / 'fig_5.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    # order: predatory, commercial, society
    original_specs = copy.deepcopy(DEFAULT_JOURNAL_SPECS)

    # scenario 1
    scenario_1 = copy.deepcopy(original_specs)
    selectivity_threshold_thetas = [-10, 3.0, 1.0]
    for i, j in enumerate(scenario_1):
        j["params"]["selectivity_threshold_theta"] = selectivity_threshold_thetas[i]
        j["params"]["screening_noise_tau"] = 0.5
        j["params"]["initial_reputation"] = 1.0
        j["params"]["bias_weight_b"] = 0

    # scenario 2
    scenario_2 = copy.deepcopy(original_specs)
    bias_weight_bs = [0, 1, 10]
    for i, j in enumerate(scenario_2):
        j["params"]["selectivity_threshold_theta"] = 0.5
        j["params"]["screening_noise_tau"] = 0.5
        j["params"]["initial_reputation"] = 1.0
        j["params"]["bias_weight_b"] = bias_weight_bs[i]

    scenario_3 = copy.deepcopy(original_specs)
    initial_reputations = [1, 100, 20]
    for i, j in enumerate(scenario_3):
        j["params"]["selectivity_threshold_theta"] = 0.5
        j["params"]["screening_noise_tau"] = 0.5
        j["params"]["initial_reputation"] = initial_reputations[i]
        j["params"]["bias_weight_b"] = 0

    for i, scenario in enumerate([scenario_1, scenario_2, scenario_3, original_specs]):
        for b_p in [5.0]:
            for g in [1, 1000]:
                subfolder = f'scenario_{i}_gamma_{g}_beta_p_{b_p}'
                no_economics_exp(g, b_p, scenario, subfolder)
