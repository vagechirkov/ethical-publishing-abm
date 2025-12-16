from datetime import datetime
from pathlib import Path
import pandas as pd
import mesa
import matplotlib.pylab as plt

from model import PublishingModel

from visualization_utils import *

def no_economics_exp(prestige_decay, prestige_social_multiplier):
    params = {
        "n_groups": 1000,
        "n_journals": 100,
        "enable_economics": False,
        "prestige_decay": prestige_decay, # 0.001,
        "beta_p": 0.5,
        "prestige_social_multiplier": prestige_social_multiplier,
    }
    max_steps = 1000

    result = mesa.batch_run(
        PublishingModel,
        number_processes=None,
        iterations=200,
        data_collection_period=1,
        parameters=params,
        max_steps=max_steps
    )

    df = pd.DataFrame(result)

    dir_name = Path(f'experiments/{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    Path.mkdir(dir_name, exist_ok=True, parents=True)

    # Prestige & reputation
    plot_dynamics(df)
    plt.savefig(dir_name / 'fig_1.png', dpi=300)
    plt.close()

    # Dynamics of Papers
    plot_category_dynamics(df)
    plt.savefig(dir_name / 'fig_2.png', dpi=300)
    plt.close()

    # Distributions Start vs End
    plot_start_end_distribution(df, max_steps=max_steps)
    plt.savefig(dir_name / 'fig_3.png', dpi=300)
    plt.close()

    # Gini Coefficients
    plot_gini(df)
    plt.savefig(dir_name / 'fig_4.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    for prest_decay in [0, 0.001, 0.01, 0.1]:
        for prest_social_multiplier in [0, 0.01, 0.1]:
            no_economics_exp(prest_decay, prest_social_multiplier)
