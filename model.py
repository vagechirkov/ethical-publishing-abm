import mesa
import numpy as np
from scipy.special import expit, softmax

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
            reputation_decay,            # delta_R
            quality_to_reputation_alpha, # alpha_R
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

        # Bernoulli trial
        return  self.rng.random() < prob_accept

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
        Generates latent quality q_{i,t}.
        Ref: Equation (1) and (10)
        mu = q0 + kappa*log(1+P) + lambda*log(1+B)
        """
        mu = (self.q0 +
              self.kappa * np.log(1 + self.prestige) +
              self.lamb * np.log(1 + max(0, self.budget)))

        # Draw from Normal distribution
        q_it = self.rng.normal(mu, self.quality_noise)
        return q_it

    def choose_target_journal(self):
        """
        Calculates Utility and uses Softmax to choose a journal.
        Ref: Equations (8) and (9)
        """
        journals = self.model.agents_by_type[JournalAgent]

        # Gather metrics for calculation
        reputations = np.array([j.reputation for j in journals])
        ethics = np.array([j.ethics for j in journals])
        costs = np.array([j.apc for j in journals])

        # Normalize metrics for the utility function (as per text descriptions)
        max_rep = np.max(reputations) if np.max(reputations) > 0 else 1.0
        norm_reputations = reputations / max_rep

        max_cost = np.max(costs) if np.max(costs) > 0 else 1.0
        norm_costs = costs / max_cost

        # Calculate Utility: U = w_R * R + w_E * E - w_C * C
        utilities = (
                self.w_R * norm_reputations +
                self.w_E * ethics -
                self.w_C * norm_costs
        )

        # Mask unaffordable journals (Prob = 0)
        # "Affordability is enforced by setting Pr=0 whenever B < C"
        affordable_mask = (costs <= self.budget)

        if not np.any(affordable_mask):
            return None # Cannot afford any journal

        # Apply Softmax to affordable utilities
        # We set unaffordable utilities to -infinity so exp(U) is 0
        masked_utilities = np.where(affordable_mask, utilities, -np.inf)

        probs = softmax(self.rationality_beta * masked_utilities)

        # Choose journal based on probs
        chosen_journal = self.rng.choice(journals, p=probs)
        return chosen_journal

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

        # 2. Choose Target
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

                # Transaction
                self.budget -= target_journal.apc
                target_journal.revenue += target_journal.apc
                target_journal.papers_accepted += 1

                # Reinvestment
                self.model.handle_reinvestment(target_journal.apc, target_journal.reinvestment_rate)

                # Updates
                target_journal.update_reputation(q_it)

                # Update Researcher Prestige
                # Need current normalized journal reputation
                max_r = self.model.max_reputation if self.model.max_reputation > 0 else 1.0
                norm_rep = target_journal.reputation / max_r
                self.update_prestige(q_it, norm_rep)

        # Prestige decay happens regardless of publication (implicit in eq 13 if second term is 0)
        # However, to be strict with Eq 13, it's an update rule *upon publication*.
        # if no publication, prestige just decays
        if not self.accepted_this_step:
            self.prestige *= (1 - self.decay)


class PublishingModel(mesa.Model):
    def __init__(
            self,
            n_groups: int = 50,
            n_journals: int = 10,

            # Researcher Params
            baseline_quality_q0: float = 1.0,
            quality_noise_q: float = 0.5,
            quality_slope_kappa: float = 0.5,
            budget_slope_lambda: float = 0.2, # budget_slope
            prestige_decay: float = 0.05,
            prestige_social_multiplier: float = 0.0,
            beta_p: float = 0.5,
            researcher_preferences: tuple = (1.0, 0.0, 0.0),  # weights: reputation, ethics, cost
            g0: float = 100,
            gamma: float = 50, # Funding Matthew effect

            # Journal Params (Distributions)
            journal_reputation_decay: float = 0.05,
            alpha_r: float = 0.5,
            journal_category_ratio: tuple = (0.4, 0.4, 0.2),  # Predatory, Commercial, Society

            # Simulation
            seed: float = None
    ) -> None:
        super().__init__(seed=seed)
        assert n_journals % 10 == 0, "N Journals must be divisible by 10"

        self.n_groups = n_groups
        self.n_journals = n_journals
        self.reinvestment_pool = 0 # Temporary storage for step

        self.max_prestige = 1.0
        self.max_reputation = 1.0

        # Create Journals with heterogeneous types
        # Predatory (High Cost, Zero ethics, No screening),
        # Commercial (High Cost, Low ethics, Good screening),
        # Ethical/Society (Low Cost, High ethics, Good screening)
        for j_category, ratio in zip(['predatory', 'commercial', 'society'], journal_category_ratio):
            n_journals_in_category = int(ratio * n_journals)
            labels = [j_category for _ in range(n_journals_in_category)]
            if j_category == 'predatory':
                JournalAgent.create_agents(
                    self,
                    n_journals_in_category,
                    selectivity_threshold_theta=-10.0,  # Accepts almost anything
                    screening_noise_tau=0.5,
                    bias_weight_b=0,
                    apc_cost=20,
                    reinvestment_rate=0.0,  # No reinvestment
                    ethics_score=0.1,
                    initial_reputation=1,
                    reputation_decay=journal_reputation_decay,
                    quality_to_reputation_alpha=alpha_r,
                    type_label=labels,
                )
            elif j_category == 'commercial':
                JournalAgent.create_agents(
                    self,
                    n_journals_in_category,
                    selectivity_threshold_theta=self.rng.uniform(1, 3, n_journals_in_category),
                    screening_noise_tau=0.5,
                    bias_weight_b=0.5,  # Likes famous authors
                    apc_cost=30,
                    reinvestment_rate=0.05,  # Minimal reinvestment
                    ethics_score=0.4,
                    initial_reputation=40,
                    reputation_decay=journal_reputation_decay,
                    quality_to_reputation_alpha=alpha_r,
                    type_label=labels,
                )
            elif j_category == "society":
                JournalAgent.create_agents(
                    self,
                    n_journals_in_category,
                    selectivity_threshold_theta=self.rng.uniform(1, 3, n_journals_in_category),
                    screening_noise_tau=0.5,
                    bias_weight_b=0.1,  # Fairer
                    apc_cost=5,
                    reinvestment_rate=0.8,  # High reinvestment
                    ethics_score=0.9,
                    initial_reputation=40,
                    reputation_decay=journal_reputation_decay,
                    quality_to_reputation_alpha=alpha_r,
                    type_label=j_category,
                )
            else:
                raise NotImplementedError(f"{j_category} not implemented")

        # Create Researchers
        ResearcherGroupAgent.create_agents(
            self,
            n_groups,
            initial_prestige=self.rng.uniform(1, 100, n_groups),
            initial_budget=self.rng.uniform(1000, 10_000, n_groups), # Initial seed money
            baseline_quality_q0=baseline_quality_q0,
            quality_slope_kappa=quality_slope_kappa,
            quality_noise_q=quality_noise_q,
            budget_slope_lambda=budget_slope_lambda,
            prestige_decay=prestige_decay,
            prestige_social_multiplier=prestige_social_multiplier,
            quality_to_prestige_beta=beta_p,
            weights_utility=[researcher_preferences for _ in range(n_groups)],
            rationality_beta=3.0, # Reasonably rational
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

    @staticmethod
    def _compute_gini(array):
        """Calculates the Gini Coefficient for Researcher Prestige/Journal Reputation"""
        array = np.array(array)
        if np.sum(array) == 0:
            return 0.0
        # Ensure positive values for Gini
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
        reinvest_amount = amount * rate
        self.reinvestment_pool += reinvest_amount

    def step(self):
        # Update globals for normalization
        researchers = self.agents_by_type[ResearcherGroupAgent]
        journals = self.agents_by_type[JournalAgent]

        self.max_prestige = max([r.prestige for r in researchers])
        self.max_reputation = max([j.reputation for j in journals])

        # Funding Step (Begin of year/round)
        # Distribute Grants + Reinvestment from previous steps (simplified to immediate)
        # 1. Distribute Reinvestment from previous step (or calculate dynamically)
        # In this loop we accumulate reinvestment during agent steps, so we distribute
        # the pool from the *current* step at the *end*, or distribute *last* step's pool now.
        # Let's distribute now (assuming pool represents exogenous + reinvestment).

        share_per_group = self.reinvestment_pool / self.n_groups
        for r in researchers:
            # Grant (Eq 5)
            norm_p = r.prestige / self.max_prestige if self.max_prestige > 0 else 0
            r.receive_funding(norm_p)
            # Reinvestment (Eq 7)
            r.budget += share_per_group

        self.reinvestment_pool = 0 # Reset pool

        # 2. Agent Steps (Submission, Review, Updates)
        self.agents_by_type[ResearcherGroupAgent].shuffle_do("step")
        self.datacollector.collect(self)
        self.agents_by_type[JournalAgent].shuffle_do("step")


if __name__ == "__main__":
    m = PublishingModel()

    for i in range(19):
        m.step()
