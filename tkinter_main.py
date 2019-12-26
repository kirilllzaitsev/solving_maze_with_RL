from tkinter import *
import manager
from tkinter import ttk

def start(strategy, epochs, env, env_size):
    if strat.get() and ep.get() and environment.get() and env_size_entry.get():
        manager.main(strategy, epochs, env, (env_size, env_size))
    else:
        print('Mandatory data is missing')
        env_size_entry.focus_set()

root = Tk()
root.wm_title("GUI v1.1")

textbox = Text(root)
textbox.grid(row=0, columnspan=3)

scrollbar = ttk.Scrollbar(root, orient="vertical", command=textbox.yview)
scrollbar.grid(row=0, column=1, sticky="nse")

textbox.configure(yscrollcommand=scrollbar.set)

TOPPINGS = [
	("Sarsamax", "Sarsamax"),
    ("ExpectedSarsa", "ExpectedSarsa"),
    ("DQN", "DQN"),
    ("DQN with Exp.Replay", "DQN with Exp.Replay"),
]

ENVS = [
    ('OpenAI Gym', 'OpenAI Gym'),
    ('Custom', 'Custom')
]

strat = StringVar()
strat.set("Sarsamax")



strat_label = Label(text="Choose training strategy:")
strat_label.grid(row=1, column=0, sticky="w")

i = 2
for text, topping in TOPPINGS:
    Radiobutton(root, text=text, variable=strat, value=topping).grid(row=i, column=0, sticky="w")
    i += 1

environment = StringVar()
environment.set("Custom")

env_label = Label(text="Choose environment:")
env_label.grid(row=1, column=1, sticky="w")

k=2
for text, env in ENVS:
    Radiobutton(root, text=text, variable=environment, value=env).grid(row=k, column=1, sticky="w")
    k += 1

ep = IntVar()
ep_entry = Entry(textvariable=ep, width=10)
ep_label = Label(text="Enter no. of epochs:")

ep_label.grid(row=i, column=0, sticky="w")
ep_entry.grid(row=i, column=1, sticky="w")

i += 1

env_size = IntVar()
env_size_entry = Entry(textvariable=env_size, width=10)
env_size_label = Label(text="Maze size: WxW")

env_size_label.grid(row=i, column=0, sticky="w")
env_size_entry.grid(row=i, column=1, sticky="w")

i += 1
photo = PhotoImage(file=r"assets/rct_science.png")
button1 = Button(root, text='Start training', compound = RIGHT, image = photo, command=lambda: start(strat.get(),ep.get(), environment.get(), env_size.get()))
button1.grid(row=i, sticky="w")



def redirector(inputStr):

    textbox.insert(INSERT, inputStr)

sys.stdout.write = redirector

root.columnconfigure(1, weight=1)
root.rowconfigure(1, weight=1)
root.mainloop()

