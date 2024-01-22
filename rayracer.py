import cmath
import random
import kandinsky 

WIDTH, HEIGHT = 320, 240

TOTAL_SAMPLES = 5
NUMBER_OF_REFLECTIONS = 10

class HitRecord:
    def __init__(self, shape=None, position=None, ray=None):

        self.shape = shape
        self.position = position
        self.distance = 10**10

        if ray and shape and position:
            self.distance = (ray.position.sub(position)).mod()

    def has_shape(self):
        return self.shape != None

class Material:
    def __init__(self, colour, attenuation, fuzz):
        self.colour = colour
        self.attenuation = attenuation
        self.fuzz = fuzz

class Ray:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction

    def get_at(self, v):
        return self.position.add(self.direction.mul(v))

    def trace(self, maxDepth, world):
        return self.trace_scene(0, maxDepth, world)

    def trace_scene(self, depth, max_depth, world):
        if depth == max_depth:
            return Vector(255, 255, 255)
        else:
            hit = self.get_closest_intersection(world)
            if hit.has_shape():
                reflected_ray = hit.shape.scatter(hit.position, self.direction, world)

                shape_material = hit.shape.material
                attenuation = shape_material.attenuation

                colour = hit.shape.material.colour.mul(attenuation)
                reflection_colour = reflected_ray.trace_scene(depth + 1, max_depth, world)

                return colour.add(reflection_colour.mul(1 - attenuation))

            else:
                return Vector(255, 255, 255)

    def get_closest_intersection(self, scene):
        closest = HitRecord()

        for shape in scene.shapes:
            new = shape.get_intersection(self)

            if convert_to_real(new.distance) < convert_to_real(closest.distance):
                closest = new

        return closest

class Scene:
    def __init__(self, shapes, random):
        self.shapes = shapes
        self.random = random

class Sphere:
    def __init__(self, pos, r, m):
        self.position = pos
        self.radius = r
        self.material = m

    def get_colour(self):
        return self.material.colour

    def get_intersection(self, ray):
        dif = ray.position.sub(self.position)
        x  = dif.dot(ray.direction)
        y  = dif.dot(dif) - self.radius ** 2

        d = convert_to_real(x * x - y)

        if (d > 0):
            xy = cmath.sqrt(d)

            root1 = convert_to_real(-x - xy)

            if (root1 >= 0):
                position = ray.get_at(root1)
                return HitRecord(self, position, ray)

            root2 = convert_to_real(-x + xy)

            if (root2 >= 0):
                position = ray.get_at(root2)
                return HitRecord(self, position, ray)

        return HitRecord()

    def scatter(self, position, vec, world):
        normal = position.sub(self.position)
        normal = normal.unit()

        if self.material.fuzz == 0:
            reflected = vec.specular(normal)
        else:
            reflected = vec.fuzzed_vector(normal, self.material.fuzz, world)

        return Ray(self.position.add(normal.mul(self.radius * 1.0001)), reflected)
    
class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def add(self, v):
        return Vector(self.x + v.x, self.y + v.y, self.z + v.z)

    def sub(self, v):
        return Vector(self.x - v.x, self.y - v.y, self.z - v.z)

    def mul(self, v):
        return Vector(self.x * v, self.y * v, self.z * v)
    
    def rmul(self, v):
        return Vector(self.x * v, self.y * v, self.z * v)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def distance_squared(self):
        return self.x ** 2 + self.y ** 2 + self.z ** 2
    
    def mod(self):
        return cmath.sqrt(self.distance_squared())
    
    def unit(self):
        return self.mul(1 / self.mod())

    def to_colour(self):
        return (self.x, self.y, self.z)

    def specular(self, normal):
        vec = self
        vec = vec.sub(normal.mul(convert_to_real(vec.dot(normal))).mul(2))
        vec = vec.unit()
        return vec

    def random_vector(self, world):
        return Vector(world.random.random()-0.5, world.random.random()-0.5, world.random.random()-0.5).unit()

    def fuzzed(self, normal, fuzz, world):
        specular = self.specular(normal)

        random = normal.sub(self.random_vector(world).mul(0.999))
        random = random.unit()

        reflected = (random.mul(fuzz)).add(specular.mul(1 - fuzz))
        reflected = reflected.unit()

        return reflected

    def __str__(self):
        return "x={} y={} z={}".format(self.x, self.y, self.z)

shapes = []
shapes.append(Sphere(Vector(-1, 0, 1), 0.4, Material(Vector(255, 0, 0), 0.5, 0.1)))
shapes.append(Sphere(Vector(0, 0, 3), 0.4, Material(Vector(0, 0, 255), 0.5, 0)))
shapes.append(Sphere(Vector(0, -0.8, 1), 0.4, Material(Vector(0, 255, 0), 0.5, 0)))
shapes.append(Sphere(Vector(1, 0, 1), 0.4, Material(Vector(0, 0, 255), 0.5, 0)))

random.seed(0)

world = Scene(shapes, random)

def convert_to_real(num):
    if isinstance(num, float): return num
    if isinstance(num, int): return num

    num = str(num)[1:-1]
    split = num.rsplit("+", 1)
    if len(split) == 1: split = num.rsplit("-", 1)
    if split[0][-1] == "e": split = split[0].rsplit("+", 1)
    if split[0][-1] == "e": split = split[0].rsplit("-", 1)
    return eval(split[0])

def generate_ray(x, y, world):
    left = Vector(1, 0 ,0)
    right = Vector(0, -1, 0)
    forward = Vector(0, 0, 1)
    origin = Vector(0, 0, 0)

    plane_distance = 1

    U = (x - WIDTH/2)/WIDTH*2
    V = (y - HEIGHT/2)/HEIGHT*2

    offsetX = (1 / WIDTH) * world.random.random() - (1 / (WIDTH * 2))
    offsetY = (1 / HEIGHT) * world.random.random() - (1 / (HEIGHT * 2))

    direction = forward.mul(plane_distance).add((left.mul(U + offsetX)).mul(WIDTH/HEIGHT)).add(right.mul(V + offsetY))
    direction = direction.unit()

    return Ray(origin, direction)

def draw(sample_num):
    for x in range(WIDTH):
        for y in range(HEIGHT):
            ray = generate_ray(x, y, world)
            ray_colour = ray.trace(NUMBER_OF_REFLECTIONS, world)

            if sample_num == 0:
                kandinsky.set_pixel(x, y, ray_colour.to_colour())

            else:
                pixel_colour = kandinsky.get_pixel(x, y)
                r, g, b = pixel_colour
                pixel_colour_vector = Vector(r, g, b)
                pixel_colour_vector = pixel_colour_vector.mul(sample_num)
                pixel_colour = (pixel_colour_vector.add(ray_colour)).mul(1/(sample_num + 1))

                kandinsky.set_pixel(x, y, pixel_colour.to_colour())

for i in range(TOTAL_SAMPLES):
    draw(i)