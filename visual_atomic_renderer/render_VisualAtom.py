import math
from PIL import Image, ImageDraw
import random
import noise
import os
import argparse

########################################################################################
def conf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", default="./VisualAtom_dataset",type=str, help="path to image file save directory")
    parser.add_argument("--numof_classes", default=1000, type=int, help="Visual Atom category number")
    parser.add_argument("--numof_instances", default=1000, type=int, help="Visual Atom instance number")
    parser.add_argument("--start_class", default=0, type=int, help="Visual Atom start class number")
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--numof_thread", default=1, type=int, help="")
    parser.add_argument("--thread_num", default=0, type=int, help="")
    # Category parameter setting
    parser.add_argument("--vertex_num_max", default=1000, type=int, help="")
    parser.add_argument("--vertex_num_min", default=200, type=int, help="")
    parser.add_argument("--perlin_min", default=2, type=int, help="")
    parser.add_argument("--line_width", default=0.01, type=float, help="")
    parser.add_argument("--radius_min", default=0, type=int, help="")
    parser.add_argument("--line_num_min", default=1, type=int, help="")
    parser.add_argument("--line_num_max", default=200, type=int, help="")
    parser.add_argument("--oval_rate", default=2, type=int, help="")
    parser.add_argument("--start_pos", default=512, type=int, help="")
    parser.add_argument("--nami_1_min", default=0, type=int, help="")
    parser.add_argument("--nami_2_min", default=0, type=int, help="")
    parser.add_argument("--nami_1_max", default=20, type=int, help="")
    parser.add_argument("--nami_2_max", default=20, type=int, help="")
    # Display on screen
    parser.add_argument("--display", action='store_true', help="Display the generated images")
    args = parser.parse_args()
    return args

args = conf()

vertex_x = []
vertex_y = []
point_x = []
point_y = []
Noise_x = []
Noise_y = []
im = []
vertex_number = 3
nami1 = 0
nami2 = 0

random.seed(args.thread_num + 1)

class_per_thread = args.numof_classes / args.numof_thread
cat_start = args.thread_num * int(class_per_thread) + args.start_class
cat_finish = cat_start + int(class_per_thread)

for cat in range(int(cat_start), int(cat_finish)):

    if not os.path.exists(os.path.join(args.save_root, "%05d" % cat)):
        os.makedirs(os.path.join(args.save_root, "%05d" % cat))

    vertex_number = random.randint(args.vertex_num_min, args.vertex_num_max)
    line_draw_num = random.randint(args.line_num_min, args.line_num_max)
    perlin_noise_coefficient = random.uniform(args.perlin_min, (args.perlin_min + 4))
    line_width = random.uniform(0.0, args.line_width)
    start_rad = random.randint(args.radius_min, args.radius_min + 50)
    nami1 = random.randint(args.nami_1_min, args.nami_1_max)
    while True:
        nami2 = random.randint(args.nami_2_min, args.nami_2_max)
        if (nami1 != nami2):
            break
        elif (nami1 == 0):
            break

    oval_rate_x = random.uniform(1, args.oval_rate)
    oval_rate_y = random.uniform(1, args.oval_rate)
    start_pos_h = (args.image_size + random.randint(-1 * args.start_pos, args.start_pos)) / 2
    start_pos_w = (args.image_size + random.randint(-1 * args.start_pos, args.start_pos)) / 2

    vertex_x.clear()
    vertex_y.clear()
    im.clear()

    for k2 in range(args.numof_instances):

        im.append(Image.new('RGB', (args.image_size, args.image_size), (0, 0, 0)))
        draw = ImageDraw.Draw(im[k2])
        angle = (math.pi * 2) / vertex_number
    
        for vertex in range(vertex_number):
            vertex_x.append(math.cos(angle * vertex) * start_rad * oval_rate_x + start_pos_w)
            vertex_y.append(math.sin(angle * vertex) * start_rad * oval_rate_y + start_pos_h)
    
        vertex_x.append(vertex_x[0])
        vertex_y.append(vertex_y[0])

        for line_draw in range(line_draw_num):
            gray = random.randint(0, 255)

            Noise_x.clear()
            Noise_y.clear()
            for vertex in range(vertex_number):
                Noise_x.append(random.uniform(0 , 10000))
                Noise_x[vertex] = noise.pnoise1(Noise_x[vertex]) * perlin_noise_coefficient - perlin_noise_coefficient - 0.5*math.sin(angle * vertex * nami1) - 0.5*math.sin(angle * vertex * nami2)

            for vertex in range(vertex_number):
                Noise_y.append(random.uniform(0 , 10000))
                Noise_y[vertex] = noise.pnoise1(Noise_y[vertex]) * perlin_noise_coefficient - perlin_noise_coefficient - 0.5*math.sin(angle * vertex * nami1) - 0.5*math.sin(angle * vertex * nami2)

            for vertex in range(vertex_number):
                vertex_x[vertex] -= math.cos(angle * vertex) * (Noise_x[vertex] - line_width)
                vertex_y[vertex] -= math.sin(angle * vertex) * (Noise_y[vertex] - line_width)

            vertex_x[vertex_number] = vertex_x[0]
            vertex_y[vertex_number] = vertex_y[0]

            for i in range(vertex_number):
                draw.line((vertex_x[i], vertex_y[i], vertex_x[i + 1], vertex_y[i + 1]) , fill = (gray, gray, gray), width = 1)

        if not args.display:
            im[k2].save(args.save_root + "/%05d/vertex_%04d_instance_%04d.png" % (cat, vertex_number, k2), quality = 95)
        else:
            im[k2].show()
        
        vertex_x.clear()
        vertex_y.clear()

        start_pos_h = (args.image_size + random.randint(-1 * args.start_pos, args.start_pos)) / 2
        start_pos_w = (args.image_size + random.randint(-1 * args.start_pos, args.start_pos)) / 2

    print('Gerated Category:' + str(cat))