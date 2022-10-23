import matplotlib.pyplot as plt
from math import sqrt, acos, asin, pi, cos, sin

L = 0
R = 3
B = 0
T = 3

dot_size = max(R - L, T - B) / 100



figure, axes = plt.subplots()
axes.set_aspect(1)
axes.set_xlim(0, 3)
axes.set_ylim(0, 3)


def draw_circle(x, y, r=0, color='#ffdd00'):
    c = plt.Circle((x, y), r, color=color)
    axes.add_artist(c)

def draw_dot(x, y):
    draw_circle(x, y, dot_size, '#000000')

def draw_infinite_line(x1, y1, x2, y2):
    axes.axline((x1, y1), (x2, y2))

def get_grid_change_angle(x1, y1, x2, y2):
    u = (x2 - x1, y2 - y1)
    un = sqrt(u[0] ** 2 + u[1] ** 2)
    u = (u[0] / un, u[1] / un)

    cos_theta = u[0]

    theta = acos(cos_theta)
    if u[1] < 0:
        theta = 2 * pi - theta

    return theta

# Relative to x axis
# Case where the two circles do not overlap, the two outer tangents
def get_non_overlap_outer_alpha(x1, y1, x2, y2):
    theta = get_grid_change_angle(x1, y1, x2, y2)
    d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    alpha = pi / 2 - asin((r1 - r2) / d)
    return alpha




x1 = 0.5
y1 = 1.5
r1 = 0.2


x2 = 2
y2 = 0.55
r2 = 1



# o1 is origin for largest circle
if r2 > r1:
    swp = (x1, y1, r1)
    x1, y1, r1 = x2, y2, r2
    x2, y2, r2 = swp


draw_circle(x1, y1, r1)
draw_circle(x2, y2, r2)


theta = get_grid_change_angle(x1, y1, x2, y2)
alpha = get_non_overlap_outer_alpha(x1, y1, x2, y1)
alpha1 = theta + alpha
alpha2 = theta - alpha


mx1 = x1 + r1 * cos(alpha1)
my1 = y1 + r1 * sin(alpha1)

mx2 = x2 + r2 * cos(alpha1)
my2 = y2 + r2 * sin(alpha1)


draw_dot(mx1, my1)
draw_dot(mx2, my2)

draw_infinite_line(mx1, my1, mx2, my2)



mx1 = x1 + r1 * cos(alpha2)
my1 = y1 + r1 * sin(alpha2)

mx2 = x2 + r2 * cos(alpha2)
my2 = y2 + r2 * sin(alpha2)


draw_dot(mx1, my1)
draw_dot(mx2, my2)

draw_infinite_line(mx1, my1, mx2, my2)








plt.show()

