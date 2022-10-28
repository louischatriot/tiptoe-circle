import matplotlib.pyplot as plt

dot_size = 1/40
figure, axes = plt.subplots()
axes.set_aspect(1)





import time
class Timer():
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def time(self, message = ''):
        duration = time.time() - self.start
        print(f"{message} ===> Duration: {duration}")
        self.reset()

t = Timer()





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
    # if angle_stop < angle_start:
        # angle_stop += 2 * pi

    angles = numpy.linspace(angle_start, angle_stop, 100)

    xs = numpy.cos(angles)
    ys = numpy.sin(angles)

    xs = [x0 + r * x for x in xs]
    ys = [y0 + r * y for y in ys]

    plt.plot(xs, ys, color = color)

def draw_arc_between_checkpoints(cp1, cp2):
    if cp1.circle != cp2.circle:
        raise ValueError("Checkpoints need to be on the same circle")

    draw_arc(cp1.circle.ctr.x, cp1.circle.ctr.y, cp1.circle.r, cp1.angle, cp2.angle, color='red')

def draw_segment(p1, p2, color = 'red'):
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

def draw_dot(x, y=0):
    if type(x) != int and type(x) != float:
        x, y = x

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
    alpha1 = normalize_angle(theta + alpha)
    alpha2 = normalize_angle(theta - alpha)

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

    return [[CheckPoint(c1, normalize_angle(theta + alpha)), CheckPoint(c2, normalize_angle(pi + theta + alpha))], [CheckPoint(c1, normalize_angle(2 * pi + theta - alpha)), CheckPoint(c2, normalize_angle(pi + theta - alpha))]]

def get_tangents_checkpoints(c1, c2):
    if c1.r == 0 or c2.r == 0:
        return get_outer_tangents_checkpoints(c1, c2)
    else:
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
def circle_circle_intersect_arc(c1, c2):
    (x1, y1), r1 = c1
    (x2, y2), r2 = c2

    d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Circles far away
    if d >= r1 + r2:
        return None

    # Small circle in large circle
    if d + min(r1, r2) <=  max(r1, r2):
        return None

    h1 = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    alpha = acos(h1 / r1)
    theta = get_grid_change_angle(x1, y1, x2, y2)
    return (normalize_angle(theta - alpha), normalize_angle(theta + alpha))

# Bad pattern from a performance perspective, redoing calculations many times
def circle_checkpoint_couple_intersect(cp1, cp2, c):
    if cp1.circle != cp2.circle:
        raise ValueError("Both checkpoints need to be on the same circle")

    intersect = circle_circle_intersect_arc(cp1.circle, c)

    if intersect is None:
        return False

    forbidden_l, forbidden_u = intersect
    if forbidden_u < forbidden_l:
        forbidden_u += 2 * pi

    al = cp1.angle
    au = cp2.angle
    if au < al:
        au += 2 * pi

    return forbidden_l < al < forbidden_u or forbidden_l < au < forbidden_u or (al < forbidden_l and au > forbidden_u)

# Distance between checkpoints
def distance(cp1, cp2):
    if cp1.circle == cp2.circle:
        a1 = cp1.angle
        a2 = cp2.angle
        if a2 == a1:
            return 0

        if (a2 < a1):
            a2 += 2 * pi

        return cp1.circle.r * (a2 - a1)

        return 2 * pi * cp1.circle.r / (a2 - a1)
    else:
        x1, y1 = point_from_checkpoint(cp1)
        x2, y2 = point_from_checkpoint(cp2)
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def shortest_path_length(a, b, circles):
    # circles = c
    circle_checkpoints = {}
    edges = {}

    for c in circles:
        circle_checkpoints[c] = []
        edges[c] = []

    # Tangents between circles
    for c in circles:
        for cc in circles:
            if c == cc:
                continue

            cps = get_tangents_checkpoints(c, cc)

            for cp1, cp2 in cps:
                if not any(circle_segment_intersect(cp1, cp2, ccc) for ccc in circles if ccc != c and ccc != cc):
                    circle_checkpoints[cp1.circle].append(cp1)
                    circle_checkpoints[cp2.circle].append(cp2)

                    if cp1 not in edges:
                        edges[cp1] = []

                    if cp2 not in edges:
                        edges[cp2] = []

                    edges[cp1].append((distance(cp1, cp2), cp2))
                    edges[cp2].append((distance(cp1, cp2), cp1))

                    # draw_segment(point_from_checkpoint(cp1), point_from_checkpoint(cp2))

    # Tangents from start to circles and from circles to end
    ca = Circle(a, 0)
    cpa = CheckPoint(ca, 0)
    edges[cpa] = []

    cb = Circle(b, 0)
    cpb = CheckPoint(cb, 0)

    for c in circles:
        cps = get_tangents_checkpoints(ca, c)

        for cp1, cp2 in cps:
            if not any(circle_segment_intersect(cp1, cp2, cc) for cc in circles if cc != c):
                cp = cp2 if cp1.circle == ca else cp1
                circle_checkpoints[cp.circle].append(cp)
                edges[cpa].append((distance(cpa, cp), cp))

                # draw_segment(point_from_checkpoint(cpa), point_from_checkpoint(cp))


        cps = get_tangents_checkpoints(cb, c)
        for cp1, cp2 in cps:
            if not any(circle_segment_intersect(cp1, cp2, cc) for cc in circles if cc != c):
                cp = cp2 if cp1.circle == cb else cp1
                circle_checkpoints[cp.circle].append(cp)

                if cp not in edges:
                    edges[cp] = []

                edges[cp].append((distance(cpb, cp), cpb))

                # draw_segment(point_from_checkpoint(cpb), point_from_checkpoint(cp))

    # From start to finish
    if not any(circle_segment_intersect(cpa, cpb, c) for c in circles):
        edges[cpa].append((distance(cpa, cpb), cpb))

    # Add all arcs
    for c in circles:
        circle_checkpoints[c] = sorted(circle_checkpoints[c], key = lambda cp: cp.angle)

        for i in range(0, len(circle_checkpoints[c])):
            cp1 = circle_checkpoints[c][i]
            cp2 = circle_checkpoints[c][i+1 if i+1 < len(circle_checkpoints[c]) else 0]

            # TODO: Many nil arcs that should be found and killed

            if not any(circle_checkpoint_couple_intersect(cp1, cp2, cc) for cc in circles if cc != c):
                if cp1 not in edges:
                    edges[cp1] = []

                edges[cp1].append((distance(cp1, cp2), cp2))

                if cp2 not in edges:
                    edges[cp2] = []

                edges[cp2].append((distance(cp1, cp2), cp1))

            # draw_arc_between_checkpoints(cp1, cp2)

    # Djikstra the shit out of this graph
    done = {}
    done[cpa] = (0, [cpa])

    checkpoints = edges.keys()

    while True:
        best_next = None
        best_path = None

        min_d = 999999999   # Ugly but oh well

        for cp, v in done.items():
            path_distance, path = v

            if cp not in edges:
                continue

            for edge_distance, cpn in edges[cp]:
                if cpn in done:
                    continue

                if path_distance + edge_distance < min_d:
                    min_d = path_distance + edge_distance
                    best_next = cpn
                    best_path = path

        if best_next is None:
            return -1

        done[best_next] = (min_d, best_path + [best_next])

        if best_next == cpb:
            return done[best_next]
            # return min_d










a, b = Point(-3, 1), Point(4.25, 0)
circles = [Circle(Point(0,0), 2.5), Circle(Point(1.5,2), 0.5), Circle(Point(3.5,1), 1), Circle(Point(3.5,-1.7), 1.2)]




# a = Point(x=3, y=0)
# b = Point(x=0, y=4)
# circles = [Circle(Point(0, 0), 1.0)]




# a = Point(x=-2.0000994759611785, y=1.2499553579837084)
# b = Point(x=3.4664484127424657, y=2.7098305765539408)
# circles = [Circle(ctr=Point(x=-1.8726982572115958, y=-1.4505507214926183), r=1.2798830554122105), Circle(ctr=Point(x=4.4052389659918845, y=1.7868544603697956), r=1.1577861715806648), Circle(ctr=Point(x=-1.381234957370907, y=-4.853022729512304), r=1.3571559524396433), Circle(ctr=Point(x=-2.7130564232356846, y=-0.778407237958163), r=1.1078567777993156), Circle(ctr=Point(x=-1.0767320054583251, y=-2.872875838074833), r=1.5301451867213471), Circle(ctr=Point(x=-1.080321369227022, y=-4.562847905326635), r=0.7488734856946393), Circle(ctr=Point(x=4.296537891495973, y=-3.3940289937891066), r=1.056444676569663), Circle(ctr=Point(x=-1.3830888480879366, y=-3.860056472476572), r=1.703816102654673), Circle(ctr=Point(x=-0.6714646681211889, y=0.42927465168759227), r=0.8180872394470498), Circle(ctr=Point(x=-4.1457892511971295, y=4.12491548107937), r=1.1966896192403509), Circle(ctr=Point(x=2.8962924727238715, y=-3.5006185178644955), r=1.2227254134370014), Circle(ctr=Point(x=4.152973096352071, y=-0.4719136538915336), r=1.0627032480435445), Circle(ctr=Point(x=-1.0597760300152004, y=2.6609927020035684), r=1.2917233243351802), Circle(ctr=Point(x=4.492078812327236, y=-0.5311520001851022), r=0.8302998779574409), Circle(ctr=Point(x=1.4652533293701708, y=1.8539306777529418), r=1.171349666523747), Circle(ctr=Point(x=-1.4384943176992238, y=-2.7891610632650554), r=1.3649599430384114), Circle(ctr=Point(x=2.292890746612102, y=1.8301984830759466), r=1.063330560713075), Circle(ctr=Point(x=1.824695912655443, y=-2.114908972289413), r=1.3729172248626127), Circle(ctr=Point(x=1.4497884386219084, y=-4.048559733200818), r=0.5520619102520867), Circle(ctr=Point(x=2.5848811003379524, y=-4.424960787873715), r=1.0701453331159427), Circle(ctr=Point(x=4.131050885189325, y=-1.2276008003391325), r=0.9422143053030595), Circle(ctr=Point(x=-1.2675193906761706, y=-1.8731833971105516), r=0.8241877684602513), Circle(ctr=Point(x=-2.737150730099529, y=2.465134698431939), r=1.073507726821117), Circle(ctr=Point(x=4.898688911926001, y=4.481217071879655), r=1.0780039766104892), Circle(ctr=Point(x=4.031039339024574, y=-2.3442237288691103), r=1.1480808550259098), Circle(ctr=Point(x=3.9417006471194327, y=-3.4770649229176342), r=0.8556844745529815), Circle(ctr=Point(x=1.256994630675763, y=-0.2018730971030891), r=0.6452041073935106), Circle(ctr=Point(x=4.397483908105642, y=-2.3720119544304907), r=1.3364206559723242), Circle(ctr=Point(x=-2.8887587622739375, y=-4.161823575850576), r=0.8661632916191593), Circle(ctr=Point(x=-3.8396280794404447, y=4.662293146830052), r=0.8674290808616205), Circle(ctr=Point(x=3.8026424567215145, y=0.12027687625959516), r=1.2097883184673264), Circle(ctr=Point(x=-1.8230717140249908, y=-1.1450118408538401), r=1.286289065121673)]


# a, b = Point(0,-7), Point(8,8)
# circles = []




# a, b = Point(-3.5,0.1), Point(3.5,0.0)
# r = 2.01
# c = [Circle(Point(0,0), 1), Circle(Point(r,0), 1), Circle(Point(r*0.5, r*sqrt(3)/2), 1), Circle(Point(-r*0.5, r*sqrt(3)/2), 1),
     # Circle(Point(-r, 0), 1), Circle(Point(r*0.5, -r*sqrt(3)/2), 1), Circle(Point(-r*0.5, -r*sqrt(3)/2), 1)]


# a, b = Point(0,0), Point(20,20)
# c = [Circle(Point(4,0), 3), Circle(Point(-4,0), 3), Circle(Point(0,4), 3), Circle(Point(0,-4), 3)]



xmin = min(a[0], b[0])
xmax = max(a[0], b[0])
ymin = min(a[1], b[1])
ymax = max(a[1], b[1])

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






t.reset()

length, path = shortest_path_length(a, b, circles)

t.time("Calculation done")

print(length)







for i in range(0, len(path) - 1):
    cp1 = path[i]
    cp2 = path[i+1]

    if cp1.circle != cp2.circle:
        draw_segment(point_from_checkpoint(cp1), point_from_checkpoint(cp2))
    else:
        draw_arc_between_checkpoints(cp1, cp2)


for cp in path:
    draw_dot(point_from_checkpoint(cp))








plt.show()

