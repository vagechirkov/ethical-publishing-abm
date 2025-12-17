import mesa
import numpy as np
from scipy.special import expit, softmax

# Default configuration for journals if none is provided
DEFAULT_JOURNAL_SPECS = [
    {
        "type_label": "predatory",
        "ratio": 0.4,
        "params": {
            "selectivity_threshold_theta": -10.0,
            "screening_noise_tau": 0.5,
            "bias_weight_b": 0.0,
            "initial_reputation": 1,

            "apc_cost": 20,
            "reinvestment_rate": 0.0,
            "ethics_score": 0.1,

        }
    },
    {
        "type_label": "commercial",
        "ratio": 0.4,
        "params": {
            "selectivity_threshold_theta": ("uniform", 1.0, 3.0),
            "screening_noise_tau": 0.5,
            "bias_weight_b": 0.5,
            "initial_reputation": 40,

            "apc_cost": 30,
            "reinvestment_rate": 0.05,
            "ethics_score": 0.4,
        }
    },
    {
        "type_label": "society",
        "ratio": 0.2,
        "params": {
            "selectivity_threshold_theta": ("uniform", 1.0, 3.0),
            "screening_noise_tau": 0.5,
            "bias_weight_b": 0.1,
            "initial_reputation": 40,

            "apc_cost": 5,
            "reinvestment_rate": 0.8,
            "ethics_score": 0.9,
        }
    }
]

class JournalAgent(mesa.Agent):
    def __init__(
            self,
            model,
            selectivity_threshold_theta, # theta_j
            screening_noise_tau,         # tau_j
            bias_weight_b,               # b_j
            apc_cost,                    # C_j
            reinvestment_rate,           # rho_j
            ethics_score,                # E_j
            initial_reputation,          # R_j(0)
            reputation_decay=0.01,       # delta_R  # Added default for safety
            quality_to_reputation_alpha=0.1,  # alpha_R  # Added default for safety
            type_label=""
    ):
        super().__init__(model)

        self.type_label = type_label

        # Editorial Parameters
        self.theta = selectivity_threshold_theta
        self.tau = screening_noise_tau
        self.b_bias = bias_weight_b
        self.ethics = ethics_score

        # Economic Parameters
        self.apc = apc_cost
        self.reinvestment_rate = reinvestment_rate

        # State
        self.reputation = initial_reputation

        # Dynamics Parameters
        self.decay = reputation_decay
        self.alpha = quality_to_reputation_alpha

        # Step metrics
        self.papers_accepted = 0
        self.revenue = 0

    def evaluate_submission(self, paper_quality, author_norm_prestige):
        """
        Calculates acceptance probability based on biased signal and screening function.
        Ref: Equations (2) and (3)
        """
        # Biased Signal: q_hat = q + b * P_norm
        perceived_quality = paper_quality + (self.b_bias * author_norm_prestige)

        # Screening Probability: sigmoid((q_hat - theta) / tau)
        prob_accept = expit((perceived_quality - self.theta) / self.tau)
        return self.rng.random() < prob_accept

    def update_reputation(self, paper_quality):
        """
        Ref: Equations (11) and (12)
        R(t+1) = alpha * log(1 + max(q,0))
        """
        # Saturating function f_R(q)
        f_q = np.log(1 + max(paper_quality, 0))

        self.reputation += self.alpha * f_q

    def step(self):
        # Reputation decays by a factor every step
        # R(t+1) = (1-delta)*R(t)
        self.reputation *= (1 - self.decay)

        # Reset counters
        self.papers_accepted = 0
        self.revenue = 0


class ResearcherGroupAgent(mesa.Agent):
    def __init__(
            self,
            model,
            initial_prestige,      # P_i(0)
            initial_budget,        # B_i(0)
            baseline_quality_q0,   # q_0
            quality_noise_q,       # q
            quality_slope_kappa,   # kappa (prestige impact)
            budget_slope_lambda,   # lambda (budget impact)
            prestige_decay,        # delta_P
            prestige_social_multiplier,
            quality_to_prestige_beta, # beta_P
            weights_utility,       # [w_R, w_E, w_C]
            rationality_beta,      # beta (softmax temp)
            funding_params         # {G0, gamma}
    ):
        super().__init__(model)
        self.type_label = ""

        self.prestige = initial_prestige
        self.budget = initial_budget

        # Production Parameters
        self.q0 = baseline_quality_q0
        self.quality_noise = quality_noise_q
        self.kappa = quality_slope_kappa
        self.lamb = budget_slope_lambda

        # Dynamics Parameters
        self.decay = prestige_decay
        self.social_multiplier = prestige_social_multiplier
        self.beta_P = quality_to_prestige_beta

        # Decision Parameters
        self.w_R, self.w_E, self.w_C = weights_utility
        self.rationality_beta = rationality_beta

        # Economic Parameters
        self.G0 = funding_params['G0']
        self.gamma = funding_params['gamma']

        self.last_paper_quality = 0
        self.accepted_this_step = False

    def produce_manuscript(self):
        """
        Generates latent quality.
        Change: If economics is disabled, the budget term is ignored.
        """
        # 1. Prestige Effect
        prestige_effect = self.kappa * np.log(1 + self.prestige)

        # 2. Budget Effect (Conditional)
        if self.model.enable_economics:
            budget_effect = self.lamb * np.log(1 + max(0, self.budget))
        else:
            budget_effect = 0

        mu = self.q0 + prestige_effect + budget_effect
        return self.rng.normal(mu, self.quality_noise)

    def choose_target_journal(self):
        """
        Calculates Utility and uses Softmax to choose a journal.
        Ref: Equations (8) and (9)
        """
        journals = self.model.agents_by_type[JournalAgent]

        # Pre-calculation
        reputations = np.array([j.reputation for j in journals])
        ethics = np.array([j.ethics for j in journals])
        costs = np.array([j.apc for j in journals])

        # Normalize
        max_rep = np.max(reputations) if np.max(reputations) > 0 else 1.0
        norm_reputations = reputations / max_rep

        max_cost = np.max(costs) if np.max(costs) > 0 else 1.0
        norm_costs = costs / max_cost

        if self.model.enable_economics:
            # Full Utility: Reputation + Ethics - Cost
            utilities = (
                    self.w_R * norm_reputations +
                    self.w_E * ethics -
                    self.w_C * norm_costs
            )
            # Affordability check
            affordable_mask = (costs <= self.budget)
        else:
            # Pure Science Utility: Reputation + Ethics (Cost is ignored)
            utilities = (
                    self.w_R * norm_reputations +
                    self.w_E * ethics
            )
            # All journals are "affordable"
            affordable_mask = np.ones(len(journals), dtype=bool)

        if not np.any(affordable_mask):
            return None

        masked_utilities = np.where(affordable_mask, utilities, -np.inf)
        probs = softmax(self.rationality_beta * masked_utilities)

        return self.rng.choice(journals, p=probs)

    def update_prestige(self, paper_quality, journal_norm_reputation):
        """
        Ref: Equation (13)
        P(t+1) = (1-delta)P + beta_P * R_norm * f_P(q)
        """
        f_p = np.log(1 + max(paper_quality, 0))

        self.prestige = (
                (1 - self.decay) * self.prestige +
                self.beta_P * journal_norm_reputation * f_p +
                self.prestige * journal_norm_reputation * f_p * self.social_multiplier
        )

    def receive_funding(self, norm_prestige):
        """
        Ref: Equation (4) and (5)
        G_i(t) = G0 + gamma * P_norm
        """
        grant = self.G0 + self.gamma * norm_prestige
        self.budget += grant

    def step(self):
        # 1. Produce Paper
        q_it = self.produce_manuscript()
        self.last_paper_quality = q_it
        self.accepted_this_step = False

        # 2. Try to publish up to 5 times
        for _ in range(5):
            target_journal = self.choose_target_journal()

            # 3. Submit (if affordable target found)
            if target_journal:
                # We calculate normalized prestige for the journal's assessment
                max_p = self.model.max_prestige if self.model.max_prestige > 0 else 1.0
                norm_prestige = self.prestige / max_p

                # 4. Assessment
                accepted = target_journal.evaluate_submission(q_it, norm_prestige)

                if accepted:
                    self.accepted_this_step = True

                    if self.model.enable_economics:
                        self.budget -= target_journal.apc
                        target_journal.revenue += target_journal.apc
                        # Reinvestment logic
                        self.model.handle_reinvestment(target_journal.apc, target_journal.reinvestment_rate)

                    # Metrics (count papers regardless of money)
                    target_journal.papers_accepted += 1

                    # Updates
                    target_journal.update_reputation(q_it)

                    max_r = self.model.max_reputation if self.model.max_reputation > 0 else 1.0
                    norm_rep = target_journal.reputation / max_r
                    self.update_prestige(q_it, norm_rep)

                    # Stop trying if accepted
                    break

        if not self.accepted_this_step:
            self.prestige *= (1 - self.decay)


class PublishingModel(mesa.Model):
    def __init__(
            self,
            n_groups: int = 50,
            n_journals: int = 10,
            # Feature Toggle
            enable_economics: bool = True,

            # Researcher Params
            baseline_quality_q0: float = 1.0,
            quality_noise_q: float = 0.5,
            quality_slope_kappa: float = 0.5,
            budget_slope_lambda: float = 0.2,
            prestige_decay: float = 0.05,
            prestige_social_multiplier: float = 0.0,
            beta_p: float = 0.5,
            researcher_preferences: tuple = (1.0, 0.0, 0.0),
            g0: float = 100,
            gamma: float = 50,

            # Journal Params
            journal_setup: list = None,
            journal_reputation_decay: float = 0.01,
            journal_quality_to_reputation_alpha: float = 0.1,

            seed: float = None
    ) -> None:
        super().__init__(seed=seed)

        self.enable_economics = enable_economics

        self.n_groups = n_groups
        self.n_journals = n_journals
        self.reinvestment_pool = 0
        self.researcher_preferences = researcher_preferences

        self.max_prestige = 1.0
        self.max_reputation = 1.0

        if journal_setup is None:
            journal_setup = DEFAULT_JOURNAL_SPECS

        # Create Journals
        for config in journal_setup:
            count = int(config["ratio"] * n_journals)
            if count == 0 and n_journals > 0: continue # Safety check

            params = config["params"].copy()

            # Handle Distributions
            parsed_params = {}
            for key, val in params.items():
                if isinstance(val, tuple) and val[0] == "uniform":
                    parsed_params[key] = self.rng.uniform(val[1], val[2], count)
                else:
                    parsed_params[key] = val

            JournalAgent.create_agents(
                self,
                count,
                type_label=[config["type_label"]] * count,
                reputation_decay=journal_reputation_decay,
                quality_to_reputation_alpha=journal_quality_to_reputation_alpha,
                **parsed_params
            )

        # Create Researchers
        ResearcherGroupAgent.create_agents(
            self,
            n_groups,
            initial_prestige=self.rng.uniform(1, 100, n_groups),
            initial_budget=self.rng.uniform(1000, 10_000, n_groups),
            baseline_quality_q0=baseline_quality_q0,
            quality_slope_kappa=quality_slope_kappa,
            quality_noise_q=quality_noise_q,
            budget_slope_lambda=budget_slope_lambda,
            prestige_decay=prestige_decay,
            prestige_social_multiplier=prestige_social_multiplier,
            quality_to_prestige_beta=beta_p,
            weights_utility=[researcher_preferences for _ in range(n_groups)],
            rationality_beta=3.0,
            funding_params=[{'G0': g0, 'gamma': gamma} for _ in range(n_groups)]
        )

        # Data Collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "MeanPrestige": lambda m: np.mean(
                    [a.prestige for a in m.agents_by_type[ResearcherGroupAgent]]
                ),
                "MeanReputation": lambda m: np.mean(
                    [a.reputation for a in m.agents_by_type[JournalAgent]]
                ),
                "TotalBudget": lambda m: np.sum(
                    [a.budget for a in m.agents_by_type[ResearcherGroupAgent]]
                ),
                "Gini_Researchers": lambda m: self._compute_gini(
                    [a.prestige for a in m.agents_by_type[ResearcherGroupAgent]]
                ),
                "Gini_Journals": lambda m: self._compute_gini(
                    [a.reputation for a in m.agents_by_type[JournalAgent]]
                ),
                "SocietyShare": self.compute_society_share,
                "AvgQuality": self.compute_avg_quality,
            },
            agent_reporters={
                "Prestige": lambda a: getattr(a, "prestige", None),
                "Budget": lambda a: getattr(a, "budget", None),
                "Reputation": lambda a: getattr(a, "reputation", None),
                "Revenue": lambda a: getattr(a, "revenue", None),
                "Type": lambda a: a.__class__.__name__,
                "Category": lambda a: getattr(a, "type_label", None),
                "NPapers": lambda a: getattr(a, "papers_accepted", None),
                "ResearchQuality": lambda a: getattr(a, "last_paper_quality", None),
            },
        )

    def compute_society_share(self):
        """Calculates the percentage of total papers accepted by society journals this step."""
        journals = self.agents_by_type[JournalAgent]
        total = sum(j.papers_accepted for j in journals)
        if total == 0: return 0.0
        soc = sum(j.papers_accepted for j in journals if j.type_label == "society")
        return soc / total

    def compute_avg_quality(self):
        """Calculates mean quality of produced manuscripts."""
        researchers = self.agents_by_type[ResearcherGroupAgent]
        qualities = [r.last_paper_quality for r in researchers]
        return np.mean(qualities) if qualities else 0.0

    @staticmethod
    def _compute_gini(array):
        """Calculates the Gini Coefficient"""
        array = np.array(array)
        if np.sum(array) == 0:
            return 0.0
        array = np.where(array < 0, 0, array)
        sorted_array = np.sort(array)
        n = len(array)
        index = np.arange(1, n + 1)
        return ((2 * np.sum(index * sorted_array)) / (n * np.sum(sorted_array))) - ((n + 1) / n)

    def handle_reinvestment(self, amount, rate):
        """
        Takes a portion of APC and redistributes it to all groups.
        Ref: Equation (7)
        """
        if self.enable_economics:
            self.reinvestment_pool += amount * rate

    def step(self):
        # Update Globals
        researchers = self.agents_by_type[ResearcherGroupAgent]
        journals = self.agents_by_type[JournalAgent]

        self.max_prestige = max([r.prestige for r in researchers])
        self.max_reputation = max([j.reputation for j in journals])

        if self.enable_economics:
            share_per_group = self.reinvestment_pool / self.n_groups
            for r in researchers:
                norm_p = r.prestige / self.max_prestige if self.max_prestige > 0 else 0
                r.receive_funding(norm_p)
                r.budget += share_per_group
            self.reinvestment_pool = 0

        self.agents_by_type[ResearcherGroupAgent].shuffle_do("step")
        self.datacollector.collect(self)
        self.agents_by_type[JournalAgent].shuffle_do("step")