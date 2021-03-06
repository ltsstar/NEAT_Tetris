import tkinter
from tkinter import Canvas, Label, Tk, StringVar

from random import choice
from collections import Counter
import time


class Game():
    WIDTH = 200
    HEIGHT = 400

    def start(self, continous_evaluation, net, graphical = False):
        '''Starts the game.

        Creates a window, a canvas, and a first shape. Binds the event handler.
        Then starts a GUI timer of ms interval self.speed and starts the GUI main
        loop.

        '''
        # TODO start() needs to be refactored so that the creation of the
        # window, label, and canvas are independent from setting them to
        # defaults and starting the game.
        #
        # There should also be a way for the user to restart and pause
        # the game if he or she wishes.
        #
        # It's a little weird that level is based only on time and that
        # as a result it increases faster and faster. Wouldn't it make
        # more sense for level to be a result of completed lines?
        self.continous_evaluation = continous_evaluation
        self.net = net
        self.graphical = graphical
        self.level = 1
        self.score = 0
        self.speed = 10
        self.counter = 0
        self.create_new_game = True

        if graphical:
            self.start_graphical()

        self.matrix = Matrix()
        self.gg = False
        self.created_shapes = 0


            #self.root.bind("<Key>", self.handle_events)
        if graphical:
            self.graphical_timer()
            self.root.mainloop()
        else:
            self.timer()

    def start_graphical(self):
        self.root = Tk()
        self.root.title("Tetris")

        self.status_var = StringVar()
        self.status_var.set("Level: 1, Score: 0")
        self.status = Label(self.root,
                            textvariable=self.status_var,
                            font=("Helvetica", 10, "bold"))
        self.status.pack()

        self.canvas = Canvas(
            self.root,
            width=Game.WIDTH,
            height=Game.HEIGHT)
        self.canvas.pack()

    def timer(self):
        if self.create_new_game:
            self.matrix.clear()
            self.matrix.create_shape()
            self.create_new_game = False

        if self.matrix.can_fall():
            self.matrix.fall()
        else:
            self.matrix.set_fixed()
            deleted_lines = self.matrix.delete_lines()
            self.score += deleted_lines * 500
            self.matrix.create_shape()
            if not self.matrix.can_fall():
                self.gg = True
                self.score += self.matrix.losing_value()
                return 0

        #time.sleep(self.speed / 1000)
        self.continous_evaluation(self, self.net)
        self.timer()

    def graphical_timer(self):
        '''Every self.speed ms, attempt to cause the current_shape to fall().

        If fall() returns False, create a new shape and check if it can fall.
        If it can't, then the game is over.

        '''
        if self.create_new_game == True:
            self.current_shape = Shape(self.canvas)
            self.create_new_game = False

        if not self.current_shape.fall():
            lines = self.remove_complete_lines()
            if lines:
                self.score += 1000
                self.status_var.set("Level: %d, Score: %d" %
                                    (self.level, self.score))

            self.current_shape = Shape(self.canvas)
            self.created_shapes += 1
            if self.is_game_over():
                # TODO This is a problem. You rely on the timer method to
                # create a new game rather than creating it here. As a
                # result, there is an intermittent error where the user
                # event keypress Down eventually causes can_move_box
                # to throw an IndexError, since the current shape has
                # no boxes. Instead, you need to cleanly start a new
                # game. I think refactoring start() might help a lot
                # here.
                #
                # Furthermore, starting a new game currently doesn't reset
                # the levels. You should place all your starting constants
                # in the same place so it's clear what needs to be reset
                # when.
                #self.create_new_game = True
                #self.game_over()
                self.gg = True

            self.counter += 1
            if self.counter == 5:
                self.level += 1
                self.speed -= 0
                self.counter = 0
                self.status_var.set("Level: %d, Score: %d" %
                                    (self.level, self.score))

        if not self.gg:
            self.matrix.mirror(self.canvas, self.current_shape)
            self.root.after(self.speed, self.timer)
        self.continous_evaluation(self, self.net)

    def handle_events(self, event):
        '''Handle all user events.'''
        if event.keysym == "Left": self.current_shape.move(-1, 0)
        if event.keysym == "Right": self.current_shape.move(1, 0)
        if event.keysym == "Down": self.current_shape.move(0, 1)
        if event.keysym == "Up": self.current_shape.rotate()
        self.matrix.mirror(self.canvas, self.current_shape)

    def is_game_over(self):
        '''Check if a newly created shape is able to fall.

        If it can't fall, then the game is over.

        '''
        for box in self.current_shape.boxes:
            if not self.current_shape.can_move_box(box, 0, 1):
                return True
        return False

    def remove_complete_lines(self):
        shape_boxes_coords = [self.canvas.coords(box)[3] for box
                              in self.current_shape.boxes]
        all_boxes = self.canvas.find_all()
        all_boxes_coords = {k: v for k, v in
                            zip(all_boxes, [self.canvas.coords(box)[3]
                                            for box in all_boxes])}
        lines_to_check = set(shape_boxes_coords)
        boxes_to_check = dict((k, v) for k, v in all_boxes_coords.items()
                              if any(v == line for line in lines_to_check))
        counter = Counter()
        for box in boxes_to_check.values(): counter[box] += 1
        complete_lines = [k for k, v in counter.items()
                          if v == (Game.WIDTH / Shape.BOX_SIZE)]

        if not complete_lines: return False

        for k, v in boxes_to_check.items():
            if v in complete_lines:
                self.canvas.delete(k)
                del all_boxes_coords[k]

        # TODO Would be cooler if the line flashed or something
        for (box, coords) in all_boxes_coords.items():
            for line in complete_lines:
                if coords < line:
                    self.canvas.move(box, 0, Shape.BOX_SIZE)
        return len(complete_lines)

    def game_over(self):
        self.canvas.delete(tkinter.ALL)
        #tkMessageBox.showinfo(
        #    "Game Over",
        #    "You scored %d points." % self.score)

class Matrix:
    MAX_WIDTH = 10
    MAX_HEIGHT = 20
    SHAPES = (
        ("yellow", (0, 0), (1, 0), (0, 1), (1, 1)),  # square
        ("lightblue", (0, 0), (1, 0), (2, 0), (3, 0)),  # line
        ("orange", (2, 0), (0, 1), (1, 1), (2, 1)),  # right el
        ("blue", (0, 0), (0, 1), (1, 1), (2, 1)),  # left el
        ("green", (0, 1), (1, 1), (1, 0), (2, 0)),  # right wedge
        ("red", (0, 0), (1, 0), (1, 1), (2, 1)),  # left wedge
        ("purple", (1, 0), (0, 1), (1, 1), (2, 1)),  # symmetrical wedge
    )

    def __init__(self):
        '''
        none -> 0
        fixed shape -> 1
        current shapae -> 2
        '''
        self.matrix = self.create()
        self.current_shape = []

    def create(self):
        matrix = []
        for i in range(Matrix.MAX_HEIGHT):
            matrix.append(Matrix.MAX_WIDTH * [0])
        return matrix

    def mirror(self, canvas, shape):
        self.clear()
        self.get_static_from_boxes(canvas)
        self.get_current_shape(canvas, shape)

    def clear(self):
        self.matrix = None
        self.current_shape = []
        self.matrix = self.create()

    def get_static_from_boxes(self, canvas):
        boxes = canvas.find_all()
        for box in boxes:
            cords = [int(x / Shape.BOX_SIZE) for x in canvas.coords(box)]
            self.matrix[cords[1]][cords[0]] = 1

    def get_current_shape(self, canvas, shape):
        boxes = shape.boxes
        for box in boxes:
            cords = [int(x / Shape.BOX_SIZE) for x in canvas.coords(box)]
            self.matrix[cords[1]][cords[0]] = 2
            self.current_shape.append((cords[1], cords[0]))

    def create_shape(self):
        middle = int(Matrix.MAX_WIDTH / 2)
        shape = choice(Matrix.SHAPES)
        self.current_shape = []
        for box in shape[1:]:
            self.matrix[box[1]][box[0] + middle] = 2
            self.current_shape.append((box[1], box[0] + middle))

    def can_move(self, right, down):
        for row_index, row in enumerate(self.matrix):
            for column_index, column in enumerate(row):
                if column == 2:
                    check_row = row_index + down
                    check_column = column_index + right
                    if check_row >= Matrix.MAX_HEIGHT or check_row < 0 or \
                        check_column >= Matrix.MAX_WIDTH or check_column < 0:
                        return False
                    if self.matrix[check_row][check_column] == 1:
                        return False
        return True

    def can_fall(self):
        return self.can_move(0, 1)

    def move(self, right, down):
        to_move = []
        for row_index, row in enumerate(self.matrix):
            for column_index, column in enumerate(row):
                if column == 2:
                    to_move.append((row_index, column_index))
                    self.matrix[row_index][column_index] = 0

        self.current_shape = []
        for i in to_move:
            self.matrix[i[0] + down][i[1] + right] = 2
            self.current_shape.append((i[0] + down, i[1] + right))

    def fall(self):
        if self.can_fall():
            return self.move(0, 1)

    def left(self):
        if self.can_move(-1, 0):
            return self.move(-1, 0)

    def right(self):
        if self.can_move(1, 0):
            return self.move(1, 0)

    def can_rotate(self):
        '''
        box[0] represents y
        box[1] represents x
        !!!
        '''
        pivot = self.current_shape[2]
        for box in self.current_shape:
            x_diff = box[1] - pivot[1]
            y_diff = box[0] - pivot[0]
            x_move = -x_diff - y_diff
            y_move = x_diff - y_diff
            new_pos = (box[0] + y_move, box[1] + x_move)
            if new_pos[0] >= Matrix.MAX_HEIGHT or new_pos[0] < 0 or new_pos[1] >= Matrix.MAX_WIDTH or new_pos[1] < 0:
                return False
            if self.matrix[new_pos[0]][new_pos[1]] == 1:
                return False
        return True

    def rotate(self):
        '''
        box[0] represents y
        box[1] represents x
        !!!
        '''
        if not self.can_rotate():
            return False

        pivot = self.current_shape[2]
        to_rotate = []
        for box in self.current_shape:
            x_diff = box[1] - pivot[1]
            y_diff = box[0] - pivot[0]
            x_move = -x_diff - y_diff
            y_move = x_diff - y_diff
            new_pos = (box[0] + y_move, box[1] + x_move)
            to_rotate.append(new_pos)
            self.matrix[box[0]][box[1]] = 0

        self.current_shape = []
        for i in to_rotate:
            self.matrix[i[0]][i[1]] = 2
            self.current_shape.append((i[0], i[1]))

    def set_fixed(self):
        for row_index, row in enumerate(self.matrix):
            for column_index, column in enumerate(row):
                if column == 2:
                    self.matrix[row_index][column_index] = 1

        self.current_shape = []

    def delete_lines(self):
        deletes = 0
        for row_index, row in enumerate(self.matrix):
            if row.count(1) == 10:
                self.delete_line(row_index)
                deletes += 1
        return deletes

    def delete_line(self, row_index):
        for i in range(row_index, 0, -1):
            self.matrix[i] = self.matrix[i - 1]
        self.matrix[0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def losing_value(self):
        value = 0
        for row_index, row in enumerate(self.matrix):
            value += (row.count(1) + row.count(2)) * 10
        return value


class Shape:
    '''Defines a tetris shape.'''
    BOX_SIZE = 20
    # START_POINT relies on screwy integer arithmetic to approximate the middle
    # of the canvas while remaining correctly on the grid.
    START_POINT = Game.WIDTH / 2 / BOX_SIZE * BOX_SIZE - BOX_SIZE
    SHAPES = (
        ("yellow", (0, 0), (1, 0), (0, 1), (1, 1)),  # square
        ("lightblue", (0, 0), (1, 0), (2, 0), (3, 0)),  # line
        ("orange", (2, 0), (0, 1), (1, 1), (2, 1)),  # right el
        ("blue", (0, 0), (0, 1), (1, 1), (2, 1)),  # left el
        ("green", (0, 1), (1, 1), (1, 0), (2, 0)),  # right wedge
        ("red", (0, 0), (1, 0), (1, 1), (2, 1)),  # left wedge
        ("purple", (1, 0), (0, 1), (1, 1), (2, 1)),  # symmetrical wedge
    )

    def __init__(self, canvas):
        '''Create a shape.

        Select a random shape from the SHAPES tuple. Then, for each point
        in the shape definition given in the SHAPES tuple, create a
        rectangle of size BOX_SIZE. Save the integer references to these
        rectangles in the self.boxes list.

        Args:
        canvas - the parent canvas on which the shape appears

        '''
        self.boxes = []  # the squares drawn by canvas.create_rectangle()
        self.shape = choice(Shape.SHAPES)  # a random shape
        self.color = self.shape[0]
        self.canvas = canvas

        for point in self.shape[1:]:
            box = canvas.create_rectangle(
                point[0] * Shape.BOX_SIZE + Shape.START_POINT,
                point[1] * Shape.BOX_SIZE,
                point[0] * Shape.BOX_SIZE + Shape.BOX_SIZE + Shape.START_POINT,
                point[1] * Shape.BOX_SIZE + Shape.BOX_SIZE,
                fill=self.color)
            self.boxes.append(box)

    def move(self, x, y):
        '''Moves this shape (x, y) boxes.'''
        if not self.can_move_shape(x, y):
            return False
        else:
            for box in self.boxes:
                self.canvas.move(box, x * Shape.BOX_SIZE, y * Shape.BOX_SIZE)
            return True

    def fall(self):
        '''Moves this shape one box-length down.'''
        if not self.can_move_shape(0, 1):
            return False
        else:
            for box in self.boxes:
                self.canvas.move(box, 0 * Shape.BOX_SIZE, 1 * Shape.BOX_SIZE)
            return True

    def rotate(self):
        '''Rotates the shape clockwise.'''
        boxes = self.boxes[:]
        pivot = boxes.pop(2)

        def get_move_coords(box):
            '''Return (x, y) boxes needed to rotate a box around the pivot.'''
            box_coords = self.canvas.coords(box)
            pivot_coords = self.canvas.coords(pivot)
            x_diff = box_coords[0] - pivot_coords[0]
            y_diff = box_coords[1] - pivot_coords[1]
            x_move = (- x_diff - y_diff) / self.BOX_SIZE
            y_move = (x_diff - y_diff) / self.BOX_SIZE
            return x_move, y_move

        # Check if shape can legally move
        for box in boxes:
            x_move, y_move = get_move_coords(box)
            if not self.can_move_box(box, x_move, y_move):
                return False

        # Move shape
        for box in boxes:
            x_move, y_move = get_move_coords(box)
            self.canvas.move(box,
                             x_move * self.BOX_SIZE,
                             y_move * self.BOX_SIZE)

        return True

    def can_move_box(self, box, x, y):
        '''Check if box can move (x, y) boxes.'''
        x = x * Shape.BOX_SIZE
        y = y * Shape.BOX_SIZE
        coords = self.canvas.coords(box)

        # Returns False if moving the box would overrun the screen
        if coords[3] + y > Game.HEIGHT: return False
        if coords[0] + x < 0: return False
        if coords[2] + x > Game.WIDTH: return False

        # Returns False if moving box (x, y) would overlap another box
        overlap = set(self.canvas.find_overlapping(
            (coords[0] + coords[2]) / 2 + x,
            (coords[1] + coords[3]) / 2 + y,
            (coords[0] + coords[2]) / 2 + x,
            (coords[1] + coords[3]) / 2 + y
        ))
        other_items = set(self.canvas.find_all()) - set(self.boxes)
        if overlap & other_items: return False

        return True

    def can_move_shape(self, x, y):
        '''Check if the shape can move (x, y) boxes.'''
        for box in self.boxes:
            if not self.can_move_box(box, x, y): return False
        return True


if __name__ == "__main__":
    game = Game()
    game.start(None, None)