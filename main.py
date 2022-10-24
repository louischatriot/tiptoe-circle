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

def get_non_overlap_outer_tangents(x1, y1, r1, x2, y2, r2):
    res = []

    theta = get_grid_change_angle(x1, y1, x2, y2)
    d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    alpha = pi / 2 - asin((r1 - r2) / d)
    alpha1 = theta + alpha
    alpha2 = theta - alpha

    mx1 = x1 + r1 * cos(alpha1)
    my1 = y1 + r1 * sin(alpha1)
    mx2 = x2 + r2 * cos(alpha1)
    my2 = y2 + r2 * sin(alpha1)
    res.append([(mx1, my1), (mx2, my2)])


    mx1 = x1 + r1 * cos(alpha2)
    my1 = y1 + r1 * sin(alpha2)
    mx2 = x2 + r2 * cos(alpha2)
    my2 = y2 + r2 * sin(alpha2)
    res.append([(mx1, my1), (mx2, my2)])

    return res

def get_non_overlap_inner_tangents(x1, y1, r1, x2, y2, r2):
    res = []

    theta = get_grid_change_angle(x1, y1, x2, y2)
    d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    alpha = pi / 2 - asin((r1 + r2) / d)

    mx1 = x1 + r1 * cos(theta + alpha)
    my1 = y1 + r1 * sin(theta + alpha)
    mx2 = x2 + r2 * cos(pi + theta + alpha)
    my2 = y2 + r2 * sin(pi + theta + alpha)
    res.append([(mx1, my1), (mx2, my2)])

    mx1 = x1 + r1 * cos(2 * pi + theta - alpha)
    my1 = y1 + r1 * sin(2 * pi + theta - alpha)
    mx2 = x2 + r2 * cos(pi + theta - alpha)
    my2 = y2 + r2 * sin(pi + theta - alpha)
    res.append([(mx1, my1), (mx2, my2)])

    return res

# Orthogonally project (x0, y0) on the line between (x1, y1) and (x2, y2)
def orthogonal_projection(x1, y1, x2, y2, x0, y0):
    xh = ((y2 - y1) * (x2 - x1) * (y0 - y1) + (y2 - y1) ** 2 * x1 + (x2 - x1) ** 2 * x0) / ((y2 - y1) ** 2 + (x2 - x1) ** 2)
    yh = (y2 - y1) * (xh - x1) / (x2 - x1) + y1
    return (xh, yh)

# Returns True if the circle with center 0 and radius r intersects the segments between points 1 and 2 (NOT the infinite line)
def circle_segment_intersect(x1, y1, x2, y2, x0, y0, r):
    xh, yh = orthogonal_projection(x1, y1, x2, y2, x0, y0)

    # Circle too far from the line
    if (yh - y0) ** 2 + (xh - x0) ** 2 > r ** 2:
        return False

    # Dot product of H1 and H2 vectors ; if negative it means H is between 1 and 2 (and circle intersects segment)
    dp = (x1 - xh) * (x2 - xh) + (y1 - yh) * (y2 - yh)
    if dp <= 0:
        return True

    # H outside segment so checking if radius is large enough to intersect
    dh1 = (x1 - xh) ** 2 + (y1 - yh) ** 2
    dh2 = (x2 - xh) ** 2 + (y2 - yh) ** 2
    if min(dh1, dh2) + (xh - x0) ** 2 + (yh - y0) ** 2 < r ** 2:
        return True
    else:
        return False

def draw_tangent(t):
    mx1, my1 = t[0]
    mx2, my2 = t[1]
    draw_line_and_dots(mx1, my1, mx2, my2)

def draw_line_and_dots(x1, y1, x2, y2):
    draw_dot(x1, y1)
    draw_dot(x2, y2)
    draw_infinite_line(x1, y1, x2, y2)





x1 = 0.5
y1 = 1.8
r1 = 0.5


x2 = 2
y2 = 1.2
r2 = 0.2



# o1 is origin for largest circle
if r2 > r1:
    swp = (x1, y1, r1)
    x1, y1, r1 = x2, y2, r2
    x2, y2, r2 = swp



# x0 = 0.2
# y0 = 1.5
# r0 = 0.42

# draw_circle(x0, y0, r0)
# draw_line_and_dots(x1, y1, x2, y2)

# print(circle_segment_intersect(x1, y1, x2, y2, x0, y0, r0))








draw_circle(x1, y1, r1)
draw_circle(x2, y2, r2)


for t in get_non_overlap_inner_tangents(x1, y1, r1, x2, y2, r2):
    draw_tangent(t)

for t in get_non_overlap_outer_tangents(x1, y1, r1, x2, y2, r2):
    draw_tangent(t)





plt.show()

