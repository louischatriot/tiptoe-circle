import matplotlib.pyplot as plt


def draw_circle(x, y, r):
    global plt
    global axes
    Drawing_colored_circle = plt.Circle((x, y), r)
    axes.add_artist(Drawing_colored_circle)


figure, axes = plt.subplots()
draw_circle(.3, .5, .2)
 
axes.set_aspect(1)
plt.show()





