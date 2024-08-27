import multiprocessing as mp

from .plotter import Plotter, Plotting


class PlotManager:
    def __init__(self, **kwargs):
        self.visualization = kwargs.get('visualization', None)
        self.compared_file = kwargs.get('compared_file', None)
        self.plotter = None
        self.plot_pipe = None
        self.plot_process = None
        self.plot_initialized = False
        self.initialize_plot()

    def initialize_plot(self):
        if self.plot_initialized:
            return
        mp.freeze_support()
        if self.visualization == 'parallel':
            self.plotter = Plotter(compared_file=self.compared_file)
            self.plot_pipe, plotter_pipe = mp.Pipe()
            self.plot_process = mp.Process(
                target=self.plotter, args=(plotter_pipe,), daemon=True)
            self.plot_process.start()
        elif self.visualization == 'non-parallel':
            self.plotter = Plotter(compared_file=self.compared_file)
            self.plotter()
        else:
            assert self.visualization is None
        self.plot_initialized = True

    def plot(self, **kwargs):
        if self.visualization is not None:
            plotting = Plotting()
            plotting.data = kwargs.get('data', None)
            plotting.model = kwargs.get('model', None)
            evolution = kwargs.get('evolution', None)
            if evolution is not None:
                plotting.time = evolution['evolved_time']
                plotting.scores = evolution['evolved_scores']
                plotting.iterations = evolution['evolved_iterations']
            plotting.modeler_name = self.__class__.__name__
            plotting.current_iteration = kwargs.get('iteration', None)
            plotting.env_id = kwargs.get('env_id', 0)
            plotting.best_score = kwargs.get('best_score', 0)
            plotting.model_color = kwargs.get('model_color', None)
            plotting.model_image = kwargs.get('model_image', None)
            if self.visualization == 'parallel':
                self.plot_pipe.send(plotting)
            elif self.visualization == 'non-parallel':
                self.plotter.plot(plotting)
            else:
                assert False

    def close(self):
        if self.plot_process is not None:
            self.plot_process.terminate()


