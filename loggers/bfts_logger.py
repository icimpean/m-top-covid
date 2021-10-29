from loggers import LogEntry


class BFTSLogger(LogEntry):
    """A log entry containing data per episode for a BFTS sampler."""
    def __init__(self):
        # Super call
        super(BFTSLogger, self).__init__()
        # The fields of the entry type
        self.episode = "Episode"
        self.arm = "Sampled Arm"
        self.ranking = "Ranking"
        # All the entry fields
        self.entry_fields = [self.episode, self.arm, self.ranking, ]

    def create_entry(self, episode, arm, ranking):
        """Method to create an entry for the log."""
        return {
            self.episode: episode,
            self.arm: arm,
            self.ranking: ranking,
        }
