import matplotlib.pyplot as plt
from math import sqrt, acos, asin, pi, cos, sin
import numpy


from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float

    def draw(self):
        draw_dot(self.x, self.y)

class Circle(NamedTuple):
    ctr: Point
    r:   float

    def draw(self):
        draw_circle(self.ctr.x, self.ctr.y, self.r)

class CheckPoint(NamedTuple):
    circle: Circle
    angle: float


dot_size = 1/40
figure, axes = plt.subplots()
axes.set_aspect(1)


def normalize_angle(a):
    while a < 0:
        a += 2 * pi

    while a > 2 * pi:
        a -= 2 * pi

    return a

def draw_infinite_line(x1, y1, x2, y2):
    axes.axline((x1, y1), (x2, y2))

def draw_line_and_dots(x1, y1, x2, y2):
    draw_dot(x1, y1)
    draw_dot(x2, y2)
    draw_infinite_line(x1, y1, x2, y2)

def draw_tangent(t):
    mx1, my1 = t[0]
    mx2, my2 = t[1]
    draw_line_and_dots(mx1, my1, mx2, my2)

def draw_arc(x0, y0, r, angle_start, angle_stop, color = 'gray'):
    angles = numpy.linspace(angle_start, angle_stop, 100)

    xs = numpy.cos(angles)
    ys = numpy.sin(angles)

    xs = [x0 + r * x for x in xs]
    ys = [y0 + r * y for y in ys]

    plt.plot(xs, ys, color = color)

def draw_segment(p1, p2, color = 'gray'):
    x1, y1 = p1
    x2, y2 = p2
    xs = numpy.linspace(x1, x2, 100)
    ys = numpy.linspace(y1, y2, 100)

    plt.plot(xs, ys, color = color)

def draw_circle(x, y, r, color='#ffdd00', dot=False):
    c = plt.Circle((x, y), r, color=color)
    axes.add_artist(c)
    if not dot:
        draw_arc(x, y, r, 0, 2 * pi)

def draw_dot(x, y):
    draw_circle(x, y, dot_size, '#000000', True)

def get_grid_change_angle(x1, y1, x2, y2):
    u = (x2 - x1, y2 - y1)
    un = sqrt(u[0] ** 2 + u[1] ** 2)
    u = (u[0] / un, u[1] / un)

    cos_theta = u[0]

    theta = acos(cos_theta)
    if u[1] < 0:
        theta = 2 * pi - theta

    return theta

def get_outer_tangents_checkpoints(c1, c2):
    (x1, y1), r1 = c1
    (x2, y2), r2 = c2

    # Small circle fully inside the big circle
    d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if d + min(r1, r2) <= max(r1, r2):
        return []

    theta = get_grid_change_angle(x1, y1, x2, y2)
    d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    alpha = pi / 2 - asin((r1 - r2) / d)
    alpha1 = theta + alpha
    alpha2 = theta - alpha

    return [[CheckPoint(c1, alpha1), CheckPoint(c2, alpha1)], [CheckPoint(c1, alpha2), CheckPoint(c2, alpha2)]]

def get_inner_tangents_checkpoints(c1, c2):
    (x1, y1), r1 = c1
    (x2, y2), r2 = c2

    # Circles overlap
    d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if d <= r1 + r2:
        return []

    theta = get_grid_change_angle(x1, y1, x2, y2)
    d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    alpha = pi / 2 - asin((r1 + r2) / d)

    return [[CheckPoint(c1, theta + alpha), CheckPoint(c2, pi + theta + alpha)], [CheckPoint(c1, 2 * pi + theta - alpha), CheckPoint(c2, pi + theta - alpha)]]

def get_tangents_checkpoints(c1, c2):
    return get_inner_tangents_checkpoints(c1, c2) + get_outer_tangents_checkpoints(c1, c2)

def point_from_checkpoint(cp):
    x = cp.circle.ctr.x + cp.circle.r * cos(cp.angle)
    y = cp.circle.ctr.y + cp.circle.r * sin(cp.angle)
    return (x, y)

def get_tangent_from_checkpoint_couple(cpc):
    return [point_from_checkpoint(cp) for cp in cpc]

def get_tangents_from_checkpoints(cpl):
    return [get_tangent_from_checkpoint_couple(cps) for cps in cpl]

# Orthogonally project (x0, y0) on the line between (x1, y1) and (x2, y2)
def orthogonal_projection(x1, y1, x2, y2, x0, y0):
    if x1 == x2:
        return (x1, y0)
    else:
        xh = ((y2 - y1) * (x2 - x1) * (y0 - y1) + (y2 - y1) ** 2 * x1 + (x2 - x1) ** 2 * x0) / ((y2 - y1) ** 2 + (x2 - x1) ** 2)
        yh = (y2 - y1) * (xh - x1) / (x2 - x1) + y1
        return (xh, yh)

# Returns True if the circle with center 0 and radius r intersects the segments between points 1 and 2 (NOT the infinite line)
def circle_segment_intersect(cp1, cp2, c):
    x1, y1 = point_from_checkpoint(cp1)
    x2, y2 = point_from_checkpoint(cp2)
    (x0, y0), r = c

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

# Intersect arc defined as a part of circle 1
# Two angles are returned
def circle_circle_intersect_arc(x1, y1, r1, x2, y2, r2):
    d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Circles far away
    if d >= r1 + r2:
        return None

    # Small circle in large circle
    if d < min(r1, r2):
        return None

    h1 = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    alpha = acos(h1 / r1)
    theta = get_grid_change_angle(x1, y1, x2, y2)
    return (theta - alpha, theta + alpha)







x1 = 1
y1 = 1.4
r1 = 0.8


x2 = 2.4
y2 = 1.2
r2 = 0.35


# c1 = Circle(Point(x1, y1), r1)
# c2 = Circle(Point(x2, y2), r2)

# draw_circle(x1, y1, r1)
# draw_circle(x2, y2, r2)




# for t in get_tangents_from_checkpoints(get_tangents_checkpoints(c1, c2)):
    # a, b = t
    # draw_segment(a, b)
    # print(t)





# o1 is origin for largest circle
# if r2 > r1:
    # swp = (x1, y1, r1)
    # x1, y1, r1 = x2, y2, r2
    # x2, y2, r2 = swp



# x0 = 0.2
# y0 = 1.5
# r0 = 0.42

# draw_circle(x0, y0, r0)
# draw_line_and_dots(x1, y1, x2, y2)

# print(circle_segment_intersect(x1, y1, x2, y2, x0, y0, r0))


a, b = Point(-3, 1), Point(4.25, 0)
c = [Circle(Point(0,0), 2.5), Circle(Point(1.5,2), 0.5), Circle(Point(3.5,1), 1), Circle(Point(3.5,-1.7), 1.2)]


xmin = min(a[0], b[0])
xmax = max(a[0], b[0])
ymin = min(a[1], b[1])
ymax = max(a[1], b[1])

circles = c
for c in circles:
    xmin = min(xmin, c.ctr[0] - c.r)
    xmax = max(xmax, c.ctr[0] + c.r)
    ymin = min(ymin, c.ctr[1] - c.r)
    ymax = max(ymax, c.ctr[1] + c.r)

h = xmax - xmin
xmin -= h / 10
xmax += h / 10
v = ymax - ymin
ymin -= v / 10
ymax += v / 10

axes.set_xlim(xmin, xmax)
axes.set_ylim(ymin, ymax)

dot_size = max(xmax - xmin, ymax - ymin) / 200


a.draw()
b.draw()

for c in circles:
    c.draw()


a = Circle(a, 0)
b = Circle(b, 0)


circles.append(a)
circles.append(b)


circle_checkpoints = {}
edges = {}

t = []

for c in circles:
    for cc in circles:
        if c == cc:
            continue

        cps = get_tangents_checkpoints(c, cc)

        for cpc in cps:
            cp1, cp2 = cpc
            if not any(circle_segment_intersect(cp1, cp2, ccc) for ccc in circles if ccc != c and ccc != cc):




                draw_segment(point_from_checkpoint(cp1), point_from_checkpoint(cp2))








# draw_circle(x1, y1, r1)
# draw_circle(x2, y2, r2)


# draw_dot(x1, y1)
# draw_dot(x2, y2)


# for t in get_tangents(x1, y1, r1, x2, y2, r2):
    # draw_tangent(t)

# res = circle_circle_intersect_arc(x1, y1, r1, x2, y2, r2)

# a, b = res

# draw_arc(x1, y1, r1, a, b)
# draw_arc(x1, y1, r1, 0, 2*pi)








plt.show()

