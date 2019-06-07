import pandas as pd
import numpy as np
import ast
import re
import folium
import os
import matplotlib.colors as mcol
import matplotlib.cm as cm

class ColorBar:

     def __init__(self, vmin = 0, vmax = 1, colors = ("black", "white")):
         self.vmin = vmin
         self.vmax = vmax
         self.colors = colors

     def to_color(self, num):
         cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", self.colors)

         cnorm = mcol.Normalize(vmin=self.vmin, vmax=self.vmax)

         cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
         cpick.set_array([])
         color_tp = cpick.to_rgba(num, bytes=True)[:3]
         return "#{0:02x}{1:02x}{2:02x}".format(*color_tp)

filenames = list(map(lambda x : x.replace(".csv", ""), os.listdir("output")))
if not os.path.exists("map"):
    os.makedirs("map")

for filename in filenames:
    print(filename)
    dataset = pd.read_csv(f"output/{filename}.csv")

    dataset["LON"], dataset["LAT"] = dataset["xy"].str.split(",").str
    del dataset['xy']
    del dataset["Unnamed: 0"]

    values = np.array([ast.literal_eval(re.sub(" +", ",", list_string)) for list_string in dataset["prob"].values])
    threshold = np.sort(np.mean(values, axis=0))[-2]
    meanings_mask = np.mean(values, axis=0) >= threshold
    masked_values = values[:, meanings_mask]
    masked_values = masked_values / np.sum(masked_values, axis=1, keepdims=True)
    print(f"Max = {np.max(masked_values, axis=0)}")
    print(f"Mean = {np.mean(masked_values, axis=0)}")
    print(f"Min = {np.min(masked_values, axis=0)}")

    world_map = folium.Map(location=[0,0], tiles="cartodbpositron", zoom_start=2)
    cb = ColorBar(vmin=np.min(masked_values, axis=0)[0], vmax=np.max(masked_values, axis=0)[0],
                  colors=("r", "b"))
    for (_, row), prob in zip(dataset.iterrows(), masked_values):
        c = cb.to_color(prob[0])
        folium.Circle(
            location=(row["LAT"], row["LON"]),
            popup=row["text"],
            radius=10000,
            color=c,
            fill=True,
            fill_color=c
        ).add_to(world_map)

    world_map.save(f'map/{filename}.html')
