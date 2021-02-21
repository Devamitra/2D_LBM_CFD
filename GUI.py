import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation
import numpy as np
from PIL import Image
import json
import LBM


def donothing():
    print("works")


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("1280x720")
        theme = ttk.Style()
        theme.theme_use("clam")

        self.root.title("2D BLM CFD Simulation")
        self.init_menu()
        self.init_stats()

        run_button = tk.Button(self.root, text="RUN", command=self.run)
        run_button.place(x=750, y=10)



        self.root.mainloop()

        self.img_path = None
        self.file_path = None

        self.matrix = None

    def init_menu(self):
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New", command=self.open_img)
        file_menu.add_command(label="Open", command=self.load_file)
        file_menu.add_command(label="Close", command=donothing)

        file_menu.add_command(label="Export", command=donothing)

        file_menu.add_separator()

        file_menu.add_command(label="Close", command=donothing)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="Undo", command=donothing)

        edit_menu.add_separator()

        edit_menu.add_command(label="Cut", command=donothing)
        edit_menu.add_command(label="Copy", command=donothing)
        edit_menu.add_command(label="Paste", command=donothing)
        edit_menu.add_command(label="Delete", command=donothing)
        edit_menu.add_command(label="Select All", command=donothing)

        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_command(label="Theme", command=donothing)
        view_menu.add_command(label="graph type", command=donothing)

        menu_bar.add_cascade(label="View", menu=view_menu)

        setting_menu = tk.Menu(menu_bar, tearoff=0)
        setting_menu.add_command(label="save format", command=donothing)
        setting_menu.add_command(label="working dir", command=donothing)

        menu_bar.add_cascade(label="Setting", menu=setting_menu)

        self.root.config(menu=menu_bar)

    def init_stats(self):

        stats = tk.Frame(self.root, height=200, width=500)
        stats.place(x=20, y=510)

        dimension = tk.StringVar(stats, value="0x0")
        dimension_label = tk.Label(stats, text="Dimension :")
        dimension_value_label = tk.Label(stats, textvariable=dimension)
        dimension_label.grid(row=0, column=0, padx=5, pady=5)
        dimension_value_label.grid(row=0, column=1, padx=5, pady=5)

        viscosity_label = tk.Label(stats, text="Viscosity :")
        viscosity_value = tk.StringVar(stats, value=0)
        viscosity_entry = tk.Entry(stats, textvariable=viscosity_value, width=4, justify="center")
        viscosity_label.grid(row=1, column=0, padx=5, pady=5)
        viscosity_entry.grid(row=1, column=1, padx=5, pady=5)

        tao = tk.StringVar(stats, value=0)
        tao_label = tk.Label(stats, text="Relaxation time :")
        tao_entry = tk.Entry(stats, textvariable=tao, width=4, justify="center")
        tao_label.grid(row=2, column=0, padx=5, pady=5)
        tao_entry.grid(row=2, column=1, padx=5, pady=5)

        density = tk.StringVar(stats, value=0)
        density_label = tk.Label(stats, text="Density :")
        density_entry = tk.Entry(stats, textvariable=density, width=4, justify="center")
        density_label.grid(row=3, column=0, padx=5, pady=5)
        density_entry.grid(row=3, column=1, padx=5, pady=5)

        apply_button = tk.Button(stats, text="Apply", command=donothing)
        apply_button.grid(row=3, column=3, padx=400)

    def img2shape(self):
        img = Image.open(self.img_path)
        img = np.array(img)

        matrix = np.zeros((len(img[0]), len(img)))

        for y in range(len(img)):
            for x in range(len(img[0])):
                if img[y, x].all() == 0:
                    matrix[x, y] = True

        self.matrix = matrix

    def open_img(self):
        self.img_path = filedialog.askopenfilename(
            initialdir="~/Pictures",
            title="Select an image",
            filetypes=(("PNG", "*.png*"),
                       ("JPEG", "*.jpg*"),
                       ("all files", "*.*")))

        self.img2shape()
        # self.graph(self.matrix)

    def graph(self, values, returnVal=False):
        fig = plt.Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        Q = ax.imshow(np.transpose(values), cmap="Greys")
        graph = FigureCanvasTkAgg(fig, master=self.root)
        graph.draw()

        if returnVal:
            return Q, graph, fig, ax

        graph.get_tk_widget().grid(row=0, column=1)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            initialdir="~/Documents",
            title="Select the simulation file to load",
            filetypes=(("JSON", "*.json*"),
                       ("all files", "*.*")))

        with open(file_path, "r") as file:
            data = file.read()

        velocity_dict = json.loads(data)

        meta = velocity_dict[:1]
        velocity_dict = velocity_dict[1:]
        velocity = np.array(velocity_dict[0]["0"])
        velocity = velocity[:, :, 0]

        fig = plt.Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        q = ax.imshow(np.transpose(velocity))
        graph = FigureCanvasTkAgg(fig, master=self.root)
        graph.draw()
        graph.get_tk_widget().grid()

        anim = animation.FuncAnimation(fig, self.load_file_update, frames=len(velocity_dict), fargs=(q, velocity_dict), interval=50, blit=False)
        tk.mainloop()

    def load_file_update(self, frame, Q, velocity_dict):
        velocity = np.array(velocity_dict[frame][str(frame)])
        Q.set_data(np.transpose(np.sqrt(velocity[:, :, 0]**2 + velocity[:, :, 1]**2)))
        return Q


    def run(self):
        sim = LBM.Lattice(*self.matrix.shape)
        sim.set_obstacle(self.matrix)
        sim.set_tao(3)

        fig = plt.Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        heatmap = ax.imshow(np.transpose(self.matrix))
        graph = FigureCanvasTkAgg(fig, master=self.root)
        graph.draw()
        graph.get_tk_widget().grid()

        anim = animation.FuncAnimation(fig, self.update_run, fargs=(heatmap, sim), interval=50, blit=False)
        tk.mainloop()

    def update_run(self, num, q, sim):
        sim.simulate(1)
        q.set_data(np.transpose(np.sqrt(sim.bulk_velocity[:, :, 0]**2 + sim.bulk_velocity[:, :, 1]**2)))
        return q


if __name__ == "__main__":
    a = GUI()