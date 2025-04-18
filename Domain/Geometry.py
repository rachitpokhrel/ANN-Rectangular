import Domain.Domains as dms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random


def draw_Domains():
    s1_vertices, s1_faces = dms.dms[0].drawRect()
    s2_vertices, s2_faces = dms.dms[1].drawRect()
    s3_vertices, s3_faces = dms.dms[2].drawRect()
    tumor1_vertices, tumor1_faces = dms.dms[3].drawRect()
    tumor2_vertices, tumor2_faces = dms.dms[4].drawRect()
    gold_vertices, gold_faces = dms.dms[5].drawRect()

    # Combine all vertices and faces
    vertices = s1_vertices + s2_vertices + s3_vertices + tumor1_vertices + tumor2_vertices + gold_vertices
    faces = s1_faces + s2_faces + s3_faces + tumor1_faces + tumor2_faces + gold_faces
    # Draw the combined geometry

    color='skyblue'
    alpha=0.3
    edgecolor='k'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors=edgecolor, linewidths=1, alpha=alpha))

    # Set limits and labels
    length = 1.00
    width = 1.00
    height = 1.00
    x0, y0, z0 = 0.0, 0.0, 0.0
    max_dim = max(length, width, height)
    ax.set_xlim([x0 - 0.1, x0 + max_dim + 0.1])
    ax.set_ylim([y0 - 0.1, y0 + max_dim + 0.1])
    ax.set_zlim([z0 - 0.1, z0 + max_dim + 0.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])

    plt.show()

def draw_Interfaces():
    interface1 = interface1.interface
    interface2 = interface2.interface
    interface3 = interface3.interface
    interface4 = interface4.interface
    interface5 = interface5.interface
    interface6 = interface6.interface
    interface7 = interface7.interface
    interface8 = interface8.interface
    interface9 = interface9.interface
    interface10 = interface10.interface
    interface11 = interface11.interface
    interface12 = interface12.interface
    interface13 = interface13.interface
    interface14 = interface14.interface
    interface15 = interface15.interface
    interface16 = interface16.interface

    # Combine all interfaces
    interfaces = [interface1 , interface2 , interface3 , interface4 , interface5 , interface6 , interface7 , interface8 , interface9 , interface10 , interface11 , interface12 , interface13 , interface14 , interface15, interface16]
    
    # Draw the combined geometry
    number_of_colors = 16
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    #color='red'
    alpha=0.3
    edgecolor='k'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #for i in range(number_of_colors):
    ax.add_collection3d(Poly3DCollection(interfaces, facecolors=colors, edgecolors=edgecolor, linewidths=1, alpha=alpha))

    # Set limits and labels
    length = 1.00
    width = 1.00
    height = 1.00
    x0, y0, z0 = 0.0, 0.0, 0.0
    max_dim = max(length, width, height)
    ax.set_xlim([x0 - 0.1, x0 + max_dim + 0.1])
    ax.set_ylim([y0 - 0.1, y0 + max_dim + 0.1])
    ax.set_zlim([z0 - 0.1, z0 + max_dim + 0.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])

    plt.show()