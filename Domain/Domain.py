

class Domain:

    def __init__(self, x0, y0, z0, length, width, height):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.length = length
        self.width = width
        self.height = height




    def vertices(self):
        # Vertices of the cube
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0
        length = self.length
        width = self.width
        height = self.height
        x1, y1, z1 = x0 + length, y0 + width, z0 + height

        vertices = [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1]
        ]
        return vertices

    def bottom_face(self):
        # Bottom face of the cube
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0
        length = self.length
        width = self.width
        height = self.height
        x1, y1, z1 = x0 + length, y0 + width, z0 + height

        bottom_face = [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0]
        ]
        return bottom_face

    def top_face(self):
        # Top face of the cube
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0
        length = self.length
        width = self.width
        height = self.height
        x1, y1, z1 = x0 + length, y0 + width, z0 + height

        top_face = [
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1]
        ]
        return top_face

    def front_face(self):
        # Front face of the cube
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0
        length = self.length
        width = self.width
        height = self.height
        x1, y1, z1 = x0 + length, y0 + width, z0 + height

        front_face = [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0]
        ]
        return front_face

    def back_face(self):
        # Back face of the cube
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0
        length = self.length
        width = self.width
        height = self.height
        x1, y1, z1 = x0 + length, y0 + width, z0 + height

        back_face = [
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1]
        ]
        return back_face

    def left_face(self):
        # Left face of the cube
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0
        length = self.length
        width = self.width
        height = self.height
        x1, y1, z1 = x0 + length, y0 + width, z0 + height

        left_face = [
            [x0, y0, z0],
            [x0, y1, z0],
            [x0, y1, z1],
            [x0, y0, z1]
        ]
        return left_face

    def right_face(self):
        # Right face of the cube
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0
        length = self.length
        width = self.width
        height = self.height
        x1, y1, z1 = x0 + length, y0 + width, z0 + height

        right_face = [
            [x1, y0, z0],
            [x1, y1, z0],
            [x1, y1, z1],
            [x1, y0, z1]
        ]
        return right_face


    def drawRect(self):
        # Draw the cube using matplotlib
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0
        length = self.length
        width = self.width
        height = self.height
        x1, y1, z1 = x0 + length, y0 + width, z0 + height

        # Vertices of the cube
        vertices = [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1]
        ]

        # Define the 6 faces of the cube
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # left
        ]

        return vertices, faces


    
