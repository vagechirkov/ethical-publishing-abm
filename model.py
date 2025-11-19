import mesa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def contribution(cur_prestige, top_pres, low_pres, gain=1.0, loss=-1.0):
    """
    Calculates the change in value based on current standing relative to quantiles.
    """
    if cur_prestige >= top_pres:
        return gain
    elif cur_prestige <= low_pres:
        return loss
    else:
        return 0.0


def acceptance_function(
    rng, journal_acc_rate, researcher_norm_prestige=0, journal_norm_reputation=0
):
    """
    Determines if a paper is accepted.
    """
    gap = researcher_norm_prestige - journal_norm_reputation

    # Steepness (k): Higher = stricter.
    k = 15

    #  Sigmoid Modifier [-0.5 to 0.5]
    sigmoid_modifier = (1 / (1 + np.exp(-k * gap))) - 0.5

    # Apply to base rate
    prob = journal_acc_rate + sigmoid_modifier

    # Strict Clamping
    return rng.uniform(0, 1) < max(0.001, min(prob, 0.999))


class JournalAgent(mesa.Agent):
    def __init__(self, model, is_oa, cost, ethics, reputation, acceptance_rate):
        super().__init__(model)
        self.is_oa = is_oa
        self.cost = cost
        self.ethics = ethics
        self.reputation = reputation
        self.acceptance_rate = acceptance_rate

        # Step-specific metrics
        self.revenue_this_step = 0
        self.papers_this_step = 0

    def step(self):
        """At the end of a step, reset per-step counters."""
        # Reputation decays by a factor (e.g., 0.5%) every step.
        decay_factor = 0.005
        self.reputation *= (1 - decay_factor)

        self.revenue_this_step = 0
        self.papers_this_step = 0


class ResearcherGroupAgent(mesa.Agent):
    def __init__(self, model, prestige, weight_prestige, weight_ethics):
        super().__init__(model)
        self.prestige = prestige
        self.weight_prestige = weight_prestige
        self.weight_ethics = weight_ethics

    def submit_paper(self):
        """Score journals, sort them, and attempt to publish a paper."""
        journals = self.model.cached_journal_list
        rep_scores = self.model.cached_journal_norm_reputations
        ethics_scores = self.model.cached_journal_ethics
        norm_prestige = self.prestige / self.model.all_group_prestiges.max()

        # Calculate weighted score
        journal_scores = (
                self.weight_prestige * rep_scores +
                self.weight_ethics * ethics_scores
        )

        # Sort journals by score in descending order
        sorted_indices = np.argsort(journal_scores)[::-1]

        # Submission loop
        for idx in sorted_indices:
            journal = journals[idx]
            norm_rep = rep_scores[idx]  # Get normalized reputation for this journal

            is_accepted = acceptance_function(
                self.rng,
                journal.acceptance_rate,
                norm_prestige,
                norm_rep
            )

            if is_accepted:
                # Update journal state
                journal.reputation += (self.model.weight_contribution *
                                       contribution(self.prestige,
                                                    self.model.group_quantile_90,
                                                    self.model.group_quantile_50,
                                                    gain=norm_prestige * 1,
                                                    loss=0.0))
                journal.revenue_this_step += journal.cost
                journal.papers_this_step += 1

                # Update this agent's state
                # Base Reward: The objective value of the journal
                base_reward = norm_rep * 1.0

                # The Multiplier: The social amplification of that reward
                # The more famous you are, the more you "squeeze out" of this success
                # Unbounded linear growth: self.prestige * 0.01
                # Diminishing returns: np.log1p(self.prestige) * 0.1
                # social_multiplier = self.prestige * 0.01
                social_multiplier = np.log1p(self.prestige) * 1.0

                # Total Gain
                total_gain = base_reward + (base_reward * social_multiplier)
                self.prestige += contribution(journal.reputation,
                                              self.model.journal_quantile_90,
                                              self.model.journal_quantile_50,
                                              gain=total_gain,
                                              loss=0.0)

                # Stop submission process for this step
                break

    def step(self):
        """The agent's action during a simulation step."""
        self.submit_paper()


class PublishingModel(mesa.Model):
    """The main model that runs the simulation."""

    def __init__(
        self,
        n_groups,
        n_journals,
        weight_contribution=0.1,
        ethics_weight_included=True,
        weight_prestige_max=0.1,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.n_groups = n_groups
        self.n_journals = n_journals
        self.weight_contribution = weight_contribution

        # Step-level caches
        self.group_quantile_90 = 0
        self.group_quantile_50 = 0
        self.journal_quantile_90 = 0
        self.journal_quantile_50 = 0
        self.cached_journal_list = []
        self.cached_journal_ethics = np.array([])
        self.cached_journal_norm_reputations = np.array([])
        self.cached_researcher_norm_prestiges = np.array([])

        # Create Journal Agents
        JournalAgent.create_agents(
            self,
            n_journals,
            is_oa=self.rng.choice([0, 1], size=n_journals),
            cost=self.rng.choice([50, 500, 5000], size=n_journals),
            ethics=self.rng.uniform(0, 1, size=n_journals),
            reputation=self.rng.exponential(scale=1 / 0.1, size=n_journals),
            acceptance_rate=self.rng.uniform(0, 1.0, size=n_journals),
        )

        # Create Researcher Group Agents
        if ethics_weight_included:
            weight_prestige = self.rng.uniform(0, weight_prestige_max, size=n_groups)
            weight_ethics = 1 - weight_prestige
        else:
            weight_prestige, weight_ethics = 1, 0

        ResearcherGroupAgent.create_agents(
            self,
            n_groups,
            prestige=self.rng.exponential(scale=1 / 0.01, size=n_groups),
            weight_prestige=weight_prestige,
            weight_ethics=weight_ethics,
        )

        # DataCollector
        agent_reporters = {
            "Type": lambda a: a.__class__.__name__,
            "Prestige": lambda a: getattr(a, "prestige", None),
            "Reputation": lambda a: getattr(a, "reputation", None),
            "NPapers": lambda a: getattr(a, "papers_this_step", None),
            "Profit": lambda a: getattr(a, "revenue_this_step", None),
            "Ethics": lambda a: getattr(a, "ethics", None),
            "Cost": lambda a: getattr(a, "cost", None),
            "OA": lambda a: getattr(a, "is_oa", None),
        }
        self.datacollector = mesa.DataCollector(agent_reporters=agent_reporters)

    @property
    def all_group_prestiges(self):
        """
        Helper property to get all current group prestiges.
        """
        return np.array([a.prestige for a in self.agents_by_type[ResearcherGroupAgent]])

    @property
    def all_journal_reputations(self):
        """
        Helper property to get all current journal reputations.
        """
        return np.array([a.reputation for a in self.agents_by_type[JournalAgent]])

    def _update_caches(self):
        # Researchers
        researcher_agents = self.agents_by_type[ResearcherGroupAgent]
        group_prestiges = np.array([r.prestige for r in researcher_agents])
        max_prestige = np.max(group_prestiges)

        # Journals
        journal_agents = self.agents_by_type[JournalAgent]
        self.cached_journal_list = journal_agents
        self.cached_journal_ethics = np.array([j.ethics for j in journal_agents])

        journal_reputations = np.array([j.reputation for j in journal_agents])
        max_rep = journal_reputations.max()

        # Normalization
        if max_rep == 0:
            self.cached_journal_norm_reputations = journal_reputations
        else:
            self.cached_journal_norm_reputations = journal_reputations / max_rep

        if max_prestige == 0:
            self.cached_researcher_norm_prestiges = group_prestiges
        else:
            self.cached_researcher_norm_prestiges = group_prestiges / max_prestige

        # Quantiles
        if group_prestiges.size > 0:
            self.group_quantile_90 = np.quantile(group_prestiges, 0.9)
            self.group_quantile_50 = np.quantile(group_prestiges, 0.5)
        else:
            self.group_quantile_90 = 0
            self.group_quantile_50 = 0

        if journal_reputations.size > 0:
            self.journal_quantile_90 = np.quantile(journal_reputations, 0.9)
            self.journal_quantile_50 = np.quantile(journal_reputations, 0.5)
        else:
            self.journal_quantile_90 = 0
            self.journal_quantile_50 = 0

    def step(self):
        self._update_caches()
        self.agents_by_type[JournalAgent].shuffle_do("step")
        self.agents_by_type[ResearcherGroupAgent].shuffle_do("step")
        self.datacollector.collect(self)


if __name__ == "__main__":
    print("Running ABM...")
    params = {
        "n_journals": 10,
        "n_groups": 100,
        "ethics_weight_included": False,
        "weight_contribution": 1,
        # "weight_prestige_max": [0.1, 0.9], # Compare low vs high prestige focus
    }
    max_steps = 1000
    result = mesa.batch_run(
        PublishingModel,
        number_processes=None,
        iterations=20,
        data_collection_period=1,
        parameters=params,
        max_steps=max_steps
    )

    df = pd.DataFrame(result)
    df['Ethics Group'] = np.where(df['Ethics'] < 0.5, 'Low Ethics', 'High Ethics')

    print("Plotting Dynamics...")

    # 1. Dynamics of Papers (Original)
    g1 = sns.relplot(
        data=df[df["Type"] == "JournalAgent"],
        x='Step', y='NPapers', hue='Ethics Group',  # col='weight_prestige_max',
        kind='line', estimator=np.sum, errorbar='ci',
        palette={'High Ethics': 'red', 'Low Ethics': 'blue'},
        height=4, aspect=1.5
    ).set(title="Papers Published Over Time")
    plt.tight_layout()
    plt.show()

    # 2. Dynamics of Mean Prestige/Reputation (Line Plots)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Researchers
    sns.lineplot(
        data=df[df["Type"] == "ResearcherGroupAgent"],
        x='Step', y='Prestige', #  hue='weight_prestige_max',
        estimator=np.median,
        errorbar=('pi', 80),
        ax=axes[0]
    )
    axes[0].set_title("Researcher Prestige Dynamics (median [10-90 percentile])")

    # Journals
    sns.lineplot(
        data=df[df["Type"] == "JournalAgent"],
        x='Step', y='Reputation',#  hue='weight_prestige_max',
        estimator=np.median,
        errorbar=('pi', 80),
        ax=axes[1]
    )
    axes[1].set_title("Journal Reputation Dynamics (median [10-90 percentile])")
    plt.show()

    steps_to_compare = [1, max_steps]
    df_dist = df[df['Step'].isin(steps_to_compare)].copy()
    df_dist['Time'] = df_dist['Step'].replace({1: 'Start', max_steps: 'End'})

    g3 = sns.displot(
        data=df_dist[df_dist["Type"] == "ResearcherGroupAgent"],
        x="Prestige", hue="Time",  # , col="weight_prestige_max",
        kind="hist", fill=True, common_norm=False, height=4, aspect=1.2,
    ).set(title="Researcher Prestige Distribution")
    plt.tight_layout()
    plt.show()

    g4 = sns.displot(
        data=df_dist[df_dist["Type"] == "JournalAgent"],
        x="Reputation", hue="Time",  # , col="weight_prestige_max",
        kind="hist", fill=True, common_norm=False, height=4, aspect=1.2
    ).set(title="Journal Reputation Distribution")
    plt.tight_layout()
    plt.show()