import matplotlib.pyplot as plt


class Visualisation(object):
    """The visualisation for experiments."""

    def __init__(self):
        # Some parameters to produce readable graphs when more data is being used
        self._max_bars = 15
        self._centering_y = 0.001
        self._default_color = "blue"

    @staticmethod
    def _plot_text(title, x_label, y_label, legend=None, x_ticks=None, y_ticks=None):
        # Add title and axis labels
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # Add axis ticks
        if x_ticks is not None:
            plt.xticks(*x_ticks)
        if y_ticks is not None:
            plt.yticks(*y_ticks)
        # Add legend
        if legend is not None:
            plt.legend(legend)

    def _center_y_lim(self, min_y, max_y):
        # Center the graph around the rewards to plot
        y_lim = (max_y + min_y) / 2 * self._centering_y
        return min_y - y_lim, max_y + y_lim

    @staticmethod
    def _get_color(idx, colors):
        if colors is None:
            return None
        elif isinstance(colors, (list, tuple)):
            return colors[idx % len(colors)]
        else:
            return colors

    def _bar_plot(self, values, enum=True, get_y=lambda d: d, colors=None):
        # If there's a lot of bars to plot, make the graph wider to fit them all
        w = 0.8
        if len(values) > self._max_bars:
            fig = plt.gcf()
            width, height = fig.get_size_inches()
            new_width = width * len(values) // self._max_bars * 1.5
            fig.set_size_inches(new_width, height)
            w = new_width / len(values) * 1.5
        print(w)
        data = enumerate(values) if enum else values
        for idx, d in data:
            plt.bar(idx, get_y(d), color=self._get_color(idx, colors), width=w)

    @staticmethod
    def _show_save_close(show=True, save_file=None):
        if save_file is not None:
            plt.savefig(save_file, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
