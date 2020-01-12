import manager
import tkinter as tk
import sys

STRATEGIES = [
    ("Sarsamax", "Sarsamax"),
    ("ExpectedSarsa", "Expected_Sarsa"),
    ("Vanilla DQN", "Vanilla_DQN"),
    ("DQN with Exp.Replay", "DQN_Exp_Replay"),
    ("Double DQN", "Double_DQN"),
    ("DQN with Prioritized Exp.Replay", "DQN_Prioritized_Exp_Replay"),
]

ENVIRONMENTS = [
    ('OpenAI Gym', 'OpenAI Gym'),
    ('Custom', 'Custom')
]

DEF_OUTPUT = sys.stdout


class IORedirector:
    """A general class for redirecting I/O to this Text widget."""
    def __init__(self, text_area):
        self.text_area = text_area


class StdoutRedirector(IORedirector):
    """A class for redirecting stdout to this Text widget."""

    def write(self, str):
        self.text_area.insert("end", str)

    def flush(self):
        pass


class Application(tk.Frame):
    """The class responsible for GUI and learning configs"""

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.row_indexer = 0

    def create_widgets(self):
        """
        Layout with all entries, labels and text
        """
        self.learning_hist_tb = tk.Text(self.master)
        self.learning_hist_tb.grid(row=self.row_indexer, columnspan=3)

        sys.stdout = StdoutRedirector(self.learning_hist_tb)

        self.scrollbar = tk.Scrollbar(self.master, orient="vertical", command=self.learning_hist_tb.yview)
        self.scrollbar.grid(row=self.row_indexer, column=1, sticky="nse")
        self.learning_hist_tb.configure(yscrollcommand=self.scrollbar.set)

        self.row_indexer += 1

        self.strategy_label = tk.Label(text="Choose training strategy:")
        self.strategy_label.grid(row=self.row_indexer, column=0, sticky="w")

        self.env_label = tk.Label(text="Choose environment:")
        self.env_label.grid(row=self.row_indexer, column=1, sticky="w")

        self.row_indexer = 9

        self.epoch_entry = tk.Entry(textvariable=self.epochs, width=10)
        self.epoch_label = tk.Label(text="Enter no. of epochs:")

        self.epoch_label.grid(row=self.row_indexer, column=0, sticky="w")
        self.epoch_entry.grid(row=self.row_indexer, column=1, sticky="w")

        self.row_indexer += 1

        self.env_size_entry = tk.Entry(textvariable=self.env_size, width=10)
        self.env_size_label = tk.Label(text="Maze size: WxW")

        self.env_size_label.grid(row=self.row_indexer, column=0, sticky="w")
        self.env_size_entry.grid(row=self.row_indexer, column=1, sticky="w")

        self.row_indexer += 1

        self.rocket = tk.PhotoImage(file=r"assets/rct_science.png")
        self.restart = tk.PhotoImage(file=r"assets/restart.png")

        self.start_btn = tk.Button(root, text='Start training', compound=tk.RIGHT, image=self.rocket,
                                   command=lambda: self.start(self.strategy.get(), self.epochs.get(),
                                                              self.environment.get(), self.env_size.get()))
        self.start_btn.grid(row=self.row_indexer, column=0, sticky="w")

        self.restart_btn = tk.Button(root, text='Restart', compound=tk.RIGHT, image=self.restart,
                                     command=self.quit)
        self.restart_btn.grid(row=self.row_indexer, column=1, sticky="w")

    def init_vars(self):
        self.strategy = tk.StringVar()
        self.strategy.set("Sarsamax")

        self.environment = tk.StringVar()
        self.environment.set("Custom")

        self.epochs = tk.IntVar()
        self.epochs.set(200)
        self.env_size = tk.IntVar()
        self.env_size.set(7)

    def create_options(self):
        """
        Add radios with strategy/environment to layout
        """
        self.row_indexer = 2
        for text, strategy in STRATEGIES:
            tk.Radiobutton(root, text=text, variable=self.strategy, value=strategy).grid(row=self.row_indexer,
                                                                                         column=0, sticky="w")
            self.row_indexer += 1

        self.row_indexer = 2
        for text, env in ENVIRONMENTS:
            tk.Radiobutton(root, text=text, variable=self.environment, value=env).grid(row=self.row_indexer,
                                                                                       column=1, sticky="w")
            self.row_indexer += 1

    def start(self, strategy, epochs, env, env_size):
        """
        Run learning on Start button press
        """
        if self.strategy.get() and self.epochs.get() and self.environment.get() and self.env_size_entry.get():
            if self.epochs.get() == 0 or self.env_size_entry.get() == 0:
                print('Mandatory data is missing')
                self.env_size_entry.focus_set()
            else:
                manager.main(strategy, epochs, env, (env_size, env_size))
        else:
            print('Mandatory data is missing')
            self.env_size_entry.focus_set()

    def quit(self):
        self.learning_hist_tb.delete('1.0', tk.END)
        self.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("GUI v.1.2")
    root.columnconfigure(1, weight=1)
    root.rowconfigure(1, weight=1)

    app = Application(root)
    app.init_vars()
    app.create_widgets()
    app.create_options()
    app.mainloop()
    sys.stdout = DEF_OUTPUT
