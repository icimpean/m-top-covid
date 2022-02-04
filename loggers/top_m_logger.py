from loggers import LogEntry


class TopMLogger(LogEntry):
    """A log entry containing data per episode for a sampler."""
    def __init__(self):
        # Super call
        super(TopMLogger, self).__init__()
        # The fields of the entry type
        self.episode = "Episode"
        self.arm = "Sampled Arm"
        self.ranking = "Ranking"
        self.means = "Ranking Means"
        self.variances = "Variances"
        # self.np_variances = "numpy var"
        self.std_dev = "std_dev"
        # self.samples = "samples"
        # All the entry fields
        self.entry_fields = [self.episode, self.arm, self.ranking, self.means, self.variances, #self.np_variances,
                             self.std_dev]

    def create_entry(self, episode, arm, ranking, ranking_means, variances, #np_variances,
                     std_dev):
        """Method to create an entry for the log."""
        return {
            self.episode: episode,
            self.arm: arm,
            self.ranking: ranking,
            self.means: ranking_means,
            self.variances: variances,
            # self.np_variances: np_variances,
            self.std_dev: std_dev,
        }
