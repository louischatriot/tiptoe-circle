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





import itertools
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

    def draw(self):
        # So ugly
        x, y = point_from_checkpoint(self)
        Point(x, y).draw()

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

def draw_arc(x0, y0, r, angle_start, angle_stop, color = 'gray', minimal_arc=False):
    if minimal_arc:
        if angle_stop < angle_start:
            swp = angle_start
            angle_start = angle_stop
            angle_stop = swp

        if angle_stop - angle_start > pi:
            angle_start += 2 * pi

    _angles = numpy.linspace(angle_start, angle_stop, 100)
    xs = numpy.cos(_angles)
    ys = numpy.sin(_angles)
    xs = [x0 + r * x for x in xs]
    ys = [y0 + r * y for y in ys]

    plt.plot(xs, ys, color = color)

def draw_arc_between_checkpoints(cp1, cp2):
    if cp1.circle != cp2.circle:
        raise ValueError("Checkpoints need to be on the same circle")

    draw_arc(cp1.circle.ctr.x, cp1.circle.ctr.y, cp1.circle.r, cp1.angle, cp2.angle, color='red', minimal_arc=True)

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
def circle_segment_intersect(x1, y1, x2, y2, c):
    # x1, y1 = point_from_checkpoint(cp1)
    # x2, y2 = point_from_checkpoint(cp2)

    # x1, y1 = point_from_checkpoint(cp1)
    # x2, y2 = point_from_checkpoint(cp2)
    (x0, y0), r = c



    # x2 -= x1
    # y2 -= y1

    # x0 -= x1
    # y0 -= y1

    # x1 = 0
    # y1 = 0

    # theta = get_grid_change_angle(x1, y1, x2, y2)

    # x2 = 0
    # y2 = sqrt(x2 ** 2 + y2 ** 2)

    # # do = sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    # c = cos(theta)
    # s = sin(theta)

    # x0p = x0 * c + y0 * s
    # y0p = y0 * c - x0 * s

    # x0 = x0p
    # y0 = y0p


    # xh = 0
    # yh = y0


    x2 -= x1
    y2 -= y1

    x0 -= x1
    y0 -= y1


    if 0 == x2:
        xh = 0
        yh = y0
        # return (x1, y0)
    else:
        xh = (y2 * x2 * y0 + x2 ** 2 * x0) / (y2 ** 2 + x2 ** 2)
        yh = y2 * xh / x2
        # return (xh, yh)



    # xh, yh = orthogonal_projection(x1, y1, x2, y2, x0, y0)

    # Circle too far from the line
    if (yh - y0) * (yh - y0) + (xh - x0) * (xh - x0) > r * r:
        return False

    x2 -= xh
    y2 -= yh


    # Dot product of H1 and H2 vectors ; if negative it means H is between 1 and 2 (and circle intersects segment)
    dp = xh * x2 + yh * y2
    if dp >= 0:
        return True
    # return True

    # H outside segment so checking if radius is large enough to intersect
    dh1 = xh ** 2 + yh ** 2
    dh2 = x2 ** 2 + y2 ** 2
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


# cache = {}


# Bad pattern from a performance perspective, redoing calculations many times
def circle_checkpoint_couple_intersect(cp1, cp2, c):
    if cp1.circle != cp2.circle:
        raise ValueError("Both checkpoints need to be on the same circle")

    # key = str(cp1.circle) + str(c)

    # if key in cache:
        # intersect = cache[key]
    # else:
        # intersect = circle_circle_intersect_arc(cp1.circle, c)
        # cache[key] = intersect

    intersect = circle_circle_intersect_arc(cp1.circle, c)


    if intersect is None:
        return False

    forbidden_l, forbidden_u = intersect
    if forbidden_u < forbidden_l:
        forbiddens = [(forbidden_l, 2 * pi), (0, forbidden_u)]
    else:
        forbiddens = [(forbidden_l, forbidden_u)]

    al = cp1.angle
    au = cp2.angle

    if au < al:
        arcs = [(al, 2 * pi), (0, au)]
    else:
        arcs = [(al, au)]

    for f, a in itertools.product(forbiddens, arcs):
        if min(a) <= f[1] and max(a) >= f[0]:
            return True

    return False

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
    NC = len(circles)
    circle_checkpoints = {}
    edges = {}

    tt = Timer()
    tt.reset()

    # circle_checkpoints = [[] for i in range(0, NC)]


    for c in circles:
        circle_checkpoints[c] = []
        edges[c] = []

    tt.time("Prep")

    # Tangents between circles

    for i in range(0, NC):
        for j in range(i+1, NC):
            c = circles[i]
            cc = circles[j]

            cps = get_tangents_checkpoints(c, cc)

            for cp1, cp2 in cps:
                x1, y1 = point_from_checkpoint(cp1)
                x2, y2 = point_from_checkpoint(cp2)

                if not any(circle_segment_intersect(x1, y1, x2, y2, ccc) for ccc in circles if ccc != c and ccc != cc):
                    circle_checkpoints[cp1.circle].append(cp1)
                    circle_checkpoints[cp2.circle].append(cp2)

                    if cp1 not in edges:
                        edges[cp1] = []

                    if cp2 not in edges:
                        edges[cp2] = []

                    edges[cp1].append((distance(cp1, cp2), cp2))
                    edges[cp2].append((distance(cp1, cp2), cp1))

                    # draw_segment(point_from_checkpoint(cp1), point_from_checkpoint(cp2))

    tt.time("Tangents between circles")

    # Tangents from start to circles and from circles to end
    ca = Circle(a, 0)
    cpa = CheckPoint(ca, 0)
    edges[cpa] = []

    cb = Circle(b, 0)
    cpb = CheckPoint(cb, 0)

    for c in circles:
        cps = get_tangents_checkpoints(ca, c)

        for cp1, cp2 in cps:
            x1, y1 = point_from_checkpoint(cp1)
            x2, y2 = point_from_checkpoint(cp2)

            if not any(circle_segment_intersect(x1, y1, x2, y2, cc) for cc in circles if cc != c):
                cp = cp2 if cp1.circle == ca else cp1
                circle_checkpoints[cp.circle].append(cp)
                edges[cpa].append((distance(cpa, cp), cp))

                # draw_segment(point_from_checkpoint(cpa), point_from_checkpoint(cp))


        cps = get_tangents_checkpoints(cb, c)
        for cp1, cp2 in cps:
            x1, y1 = point_from_checkpoint(cp1)
            x2, y2 = point_from_checkpoint(cp2)

            if not any(circle_segment_intersect(x1, y1, x2, y2, cc) for cc in circles if cc != c):
                cp = cp2 if cp1.circle == cb else cp1
                circle_checkpoints[cp.circle].append(cp)

                if cp not in edges:
                    edges[cp] = []

                edges[cp].append((distance(cpb, cp), cpb))

                # draw_segment(point_from_checkpoint(cpb), point_from_checkpoint(cp))

    # From start to finish
    x1, y1 = point_from_checkpoint(cpa)
    x2, y2 = point_from_checkpoint(cpb)

    if not any(circle_segment_intersect(x1, y1, x2, y2, c) for c in circles):
        edges[cpa].append((distance(cpa, cpb), cpb))

    tt.time("Tangents with start or finish")


    # Add all arcs
    for c in circles:
        circle_checkpoints[c] = sorted(circle_checkpoints[c], key = lambda cp: cp.angle)

        for i in range(0, len(circle_checkpoints[c])):
            cp1 = circle_checkpoints[c][i]
            cp2 = circle_checkpoints[c][i+1 if i+1 < len(circle_checkpoints[c]) else 0]

            if not any(circle_checkpoint_couple_intersect(cp1, cp2, cc) for cc in circles if cc != c):
                if cp1 not in edges:
                    edges[cp1] = []

                edges[cp1].append((distance(cp1, cp2), cp2))

                if cp2 not in edges:
                    edges[cp2] = []

                edges[cp2].append((distance(cp1, cp2), cp1))

            # draw_arc_between_checkpoints(cp1, cp2)

    tt.time("Adding arcs")

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
            tt.time("Djikstra done")

            return done[best_next]
            # return min_d

def draw_path(path):
    for i in range(0, len(path) - 1):
        cp1 = path[i]
        cp2 = path[i+1]

        if cp1.circle != cp2.circle:
            draw_segment(point_from_checkpoint(cp1), point_from_checkpoint(cp2))
        else:
            draw_arc_between_checkpoints(cp1, cp2)

    for cp in path:
        draw_dot(point_from_checkpoint(cp))









a, b = Point(-3, 1), Point(4.25, 0)
circles = [Circle(Point(0,0), 2.5), Circle(Point(1.5,2), 0.5), Circle(Point(3.5,1), 1), Circle(Point(3.5,-1.7), 1.2)]




# a = Point(x=3, y=0)
# b = Point(x=0, y=4)
# circles = [Circle(Point(0, 0), 1.0)]




a = Point(x=-2.0000994759611785, y=1.2499553579837084)
b = Point(x=3.4664484127424657, y=2.7098305765539408)
circles = [Circle(ctr=Point(x=-1.8726982572115958, y=-1.4505507214926183), r=1.2798830554122105), Circle(ctr=Point(x=4.4052389659918845, y=1.7868544603697956), r=1.1577861715806648), Circle(ctr=Point(x=-1.381234957370907, y=-4.853022729512304), r=1.3571559524396433), Circle(ctr=Point(x=-2.7130564232356846, y=-0.778407237958163), r=1.1078567777993156), Circle(ctr=Point(x=-1.0767320054583251, y=-2.872875838074833), r=1.5301451867213471), Circle(ctr=Point(x=-1.080321369227022, y=-4.562847905326635), r=0.7488734856946393), Circle(ctr=Point(x=4.296537891495973, y=-3.3940289937891066), r=1.056444676569663), Circle(ctr=Point(x=-1.3830888480879366, y=-3.860056472476572), r=1.703816102654673), Circle(ctr=Point(x=-0.6714646681211889, y=0.42927465168759227), r=0.8180872394470498), Circle(ctr=Point(x=-4.1457892511971295, y=4.12491548107937), r=1.1966896192403509), Circle(ctr=Point(x=2.8962924727238715, y=-3.5006185178644955), r=1.2227254134370014), Circle(ctr=Point(x=4.152973096352071, y=-0.4719136538915336), r=1.0627032480435445), Circle(ctr=Point(x=-1.0597760300152004, y=2.6609927020035684), r=1.2917233243351802), Circle(ctr=Point(x=4.492078812327236, y=-0.5311520001851022), r=0.8302998779574409), Circle(ctr=Point(x=1.4652533293701708, y=1.8539306777529418), r=1.171349666523747), Circle(ctr=Point(x=-1.4384943176992238, y=-2.7891610632650554), r=1.3649599430384114), Circle(ctr=Point(x=2.292890746612102, y=1.8301984830759466), r=1.063330560713075), Circle(ctr=Point(x=1.824695912655443, y=-2.114908972289413), r=1.3729172248626127), Circle(ctr=Point(x=1.4497884386219084, y=-4.048559733200818), r=0.5520619102520867), Circle(ctr=Point(x=2.5848811003379524, y=-4.424960787873715), r=1.0701453331159427), Circle(ctr=Point(x=4.131050885189325, y=-1.2276008003391325), r=0.9422143053030595), Circle(ctr=Point(x=-1.2675193906761706, y=-1.8731833971105516), r=0.8241877684602513), Circle(ctr=Point(x=-2.737150730099529, y=2.465134698431939), r=1.073507726821117), Circle(ctr=Point(x=4.898688911926001, y=4.481217071879655), r=1.0780039766104892), Circle(ctr=Point(x=4.031039339024574, y=-2.3442237288691103), r=1.1480808550259098), Circle(ctr=Point(x=3.9417006471194327, y=-3.4770649229176342), r=0.8556844745529815), Circle(ctr=Point(x=1.256994630675763, y=-0.2018730971030891), r=0.6452041073935106), Circle(ctr=Point(x=4.397483908105642, y=-2.3720119544304907), r=1.3364206559723242), Circle(ctr=Point(x=-2.8887587622739375, y=-4.161823575850576), r=0.8661632916191593), Circle(ctr=Point(x=-3.8396280794404447, y=4.662293146830052), r=0.8674290808616205), Circle(ctr=Point(x=3.8026424567215145, y=0.12027687625959516), r=1.2097883184673264), Circle(ctr=Point(x=-1.8230717140249908, y=-1.1450118408538401), r=1.286289065121673)]


# a, b = Point(0,-7), Point(8,8)
# circles = []




a, b = Point(-3.5,0.1), Point(3.5,0.0)
r = 2.01
circles = [Circle(Point(0,0), 1), Circle(Point(r,0), 1), Circle(Point(r*0.5, r*sqrt(3)/2), 1), Circle(Point(-r*0.5, r*sqrt(3)/2), 1),
     Circle(Point(-r, 0), 1), Circle(Point(r*0.5, -r*sqrt(3)/2), 1), Circle(Point(-r*0.5, -r*sqrt(3)/2), 1)]


# a, b = Point(0,0), Point(20,20)
# c = [Circle(Point(4,0), 3), Circle(Point(-4,0), 3), Circle(Point(0,4), 3), Circle(Point(0,-4), 3)]


a = Point(x=1, y=1)
b = Point(x=5, y=5)
circles = [Circle(ctr=Point(x=0, y=0), r=0.16287464636843652), Circle(ctr=Point(x=0, y=1), r=0.35239859672728924), Circle(ctr=Point(x=0, y=2), r=0.5364255122607574), Circle(ctr=Point(x=0, y=3), r=0.43006224010605365), Circle(ctr=Point(x=0, y=4), r=0.3106004946632311), Circle(ctr=Point(x=0, y=5), r=0.5266889514634385), Circle(ctr=Point(x=0, y=6), r=0.5684803681215271), Circle(ctr=Point(x=0, y=7), r=0.5833063065307215), Circle(ctr=Point(x=1, y=0), r=0.19940062786918133), Circle(ctr=Point(x=1, y=2), r=0.1901628721738234), Circle(ctr=Point(x=1, y=3), r=0.4963121007895097), Circle(ctr=Point(x=1, y=4), r=0.7945029408903792), Circle(ctr=Point(x=1, y=5), r=0.25107551633846015), Circle(ctr=Point(x=1, y=6), r=0.7169563776114956), Circle(ctr=Point(x=1, y=7), r=0.5250582014909014), Circle(ctr=Point(x=2, y=0), r=0.1930718991206959), Circle(ctr=Point(x=2, y=1), r=0.4261120012728497), Circle(ctr=Point(x=2, y=2), r=0.2375767939025536), Circle(ctr=Point(x=2, y=3), r=0.5907849639421329), Circle(ctr=Point(x=2, y=4), r=0.3803132777335122), Circle(ctr=Point(x=2, y=5), r=0.4899552673799917), Circle(ctr=Point(x=2, y=6), r=0.5319813678273931), Circle(ctr=Point(x=2, y=7), r=0.38447430075611916), Circle(ctr=Point(x=3, y=0), r=0.29231034584809096), Circle(ctr=Point(x=3, y=1), r=0.392366753029637), Circle(ctr=Point(x=3, y=2), r=0.5005855676019564), Circle(ctr=Point(x=3, y=3), r=0.6281589973485097), Circle(ctr=Point(x=3, y=4), r=0.4360745647223666), Circle(ctr=Point(x=3, y=5), r=0.675492997909896), Circle(ctr=Point(x=3, y=6), r=0.3913260711589828), Circle(ctr=Point(x=3, y=7), r=0.49383140334393827), Circle(ctr=Point(x=4, y=0), r=0.2520577947841957), Circle(ctr=Point(x=4, y=1), r=0.5762704281834885), Circle(ctr=Point(x=4, y=2), r=0.3109034419292584), Circle(ctr=Point(x=4, y=3), r=0.605750561482273), Circle(ctr=Point(x=4, y=4), r=0.4061566901626065), Circle(ctr=Point(x=4, y=5), r=0.6387910791439936), Circle(ctr=Point(x=4, y=6), r=0.3983294921228662), Circle(ctr=Point(x=4, y=7), r=0.5167727740248665), Circle(ctr=Point(x=5, y=0), r=0.3643342807190493), Circle(ctr=Point(x=5, y=1), r=0.30225425537209955), Circle(ctr=Point(x=5, y=2), r=0.5889043335104361), Circle(ctr=Point(x=5, y=3), r=0.6993306066608056), Circle(ctr=Point(x=5, y=4), r=0.652983741578646), Circle(ctr=Point(x=5, y=6), r=0.5801291283918545), Circle(ctr=Point(x=5, y=7), r=0.14674811570439486), Circle(ctr=Point(x=6, y=0), r=0.6416853299131616), Circle(ctr=Point(x=6, y=1), r=0.15815935016144067), Circle(ctr=Point(x=6, y=2), r=0.49615327182691543), Circle(ctr=Point(x=6, y=3), r=0.29401245194021614), Circle(ctr=Point(x=6, y=4), r=0.6900777980452403), Circle(ctr=Point(x=6, y=5), r=0.566799630713649), Circle(ctr=Point(x=6, y=6), r=0.3927921340567991), Circle(ctr=Point(x=6, y=7), r=0.5148965323576703), Circle(ctr=Point(x=7, y=0), r=0.12740276546683163), Circle(ctr=Point(x=7, y=1), r=0.503067686338909), Circle(ctr=Point(x=7, y=2), r=0.47778444837313144), Circle(ctr=Point(x=7, y=3), r=0.26259292478207497), Circle(ctr=Point(x=7, y=4), r=0.3290316406404599), Circle(ctr=Point(x=7, y=5), r=0.46293371852952986), Circle(ctr=Point(x=7, y=6), r=0.5243985806358978), Circle(ctr=Point(x=7, y=7), r=0.5472880630055442)]






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



# za = 0
# x0 = 0.5
# y0 = -3
# r00 = 3
# x00 = x0 + r00 * cos(za)
# y00 = y0 + r00 * sin(za)



# c1 = Circle(ctr=Point(x=x0, y=y0), r=3)
# c2 = Circle(ctr=Point(x=x00, y=y00), r=2)



# cp1 = CheckPoint(circle=c1, angle=3)
# cp2 = CheckPoint(circle=c1, angle=6)

# c1 = circles[0]
# c2 = circles[1]


# cp1 = CheckPoint(circle=c1, angle=1.5707963267948966)
# cp2 = CheckPoint(circle=c1, angle=2.1607840633667426)




# cp1 = CheckPoint(circle=c2, angle=0.28379410920832804)
# cp2 = CheckPoint(circle=c2, angle=0.371834264977422)
# cp3 = CheckPoint(circle=c2, angle=1.3326621236922218)
# cp4 = CheckPoint(circle=c2, angle=1.5707963267948966)




# c1.draw()
# c2.draw()

# cp1.draw()
# cp2.draw()


# res = circle_checkpoint_couple_intersect(cp1, cp2, c1)

# print(res)





a.draw()
b.draw()

for c in circles:
    c.draw()

t.reset()

res = shortest_path_length(a, b, circles)
# print(res)

t.time("Calculation done")

if res == -1:
    print("NOTHING FOUND")
else:
    length, path = res
    print(length)
    draw_path(path)












plt.show()

