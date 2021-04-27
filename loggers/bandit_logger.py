from loggers import LogEntry


class BanditLogger(LogEntry):
    """A log entry containing data per episode for a bandit."""
    def __init__(self):
        # Super call
        super(BanditLogger, self).__init__()
        # The fields of the entry type
        self.episode = "Episode"
        self.arm = "Arm"
        self.reward = "Reward"
        self.time = "Time"
        # All the entry fields
        self.entry_fields = [self.episode, self.arm, self.reward, self.time, ]

    def create_entry(self, episode, arm, reward, time):
        """Method to create an entry for the log."""
        return {
            self.episode: episode,
            self.arm: arm,
            self.reward: reward,
            self.time: time,
        }
