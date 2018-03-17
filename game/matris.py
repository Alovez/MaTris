#!/usr/bin/env python
import pygame
from pygame import Rect, Surface
import random
import os
import kezmenu

from tetrominoes import list_of_tetrominoes
from tetrominoes import rotate

from scores import load_score, write_score
from datetime import datetime

class BrokenMatrixException(Exception):
    pass

logger_count = {}

def logger(message):
    now = datetime.now().strftime('%H:%M:%S')
    logger_count[message] = logger_count.get(message, 0) + 1
    print now + ' |INFO| NO.%s %s' % (logger_count.get(message, 0), message)

BGCOLOR = (15, 15, 20)
BORDERCOLOR = (140, 140, 140)

BLOCKSIZE = 30
BORDERWIDTH = 10

MATRIS_OFFSET = 20

WIDTH = 700
HEIGHT = 20*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2

MATRIX_WIDTH = 10
MATRIX_HEIGHT = 22
VISIBLE_MATRIX_HEIGHT = MATRIX_HEIGHT - 2


class Matris(object):
    def __init__(self, size=(MATRIX_WIDTH, MATRIX_HEIGHT), blocksize=BLOCKSIZE):
        self.size = {'width': size[0], 'height': size[1]}
        self.blocksize = blocksize
        self.surface = Surface((self.size['width']  * self.blocksize,
                                (self.size['height']-2) * self.blocksize))


        self.matrix = dict()
        for y in range(self.size['height']):
            for x in range(self.size['width']):
                self.matrix[(y,x)] = None


        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.set_tetrominoes()
        self.tetromino_rotation = 0
        self.downwards_timer = 0
        self.base_downwards_speed = 0.4 # Move down every 400 ms

        self.movement_keys = {'left': 0, 'right': 0}
        self.movement_keys_speed = 0.05
        self.movement_keys_timer = (-self.movement_keys_speed)*2

        self.level = 1
        self.score = 0
        self.lines = 0

        self.combo = 1 # Combo will increase when you clear lines with several tetrominos in a row
        
        self.paused = False
        self.gameover = False

        self.highscore = load_score()
        self.played_highscorebeaten_sound = False

    def set_tetrominoes(self):
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        self.shadow_block = self.block(self.current_tetromino.color, shadow=True)

    
    def hard_drop(self):
        amount = 0
        while self.request_movement('down'):
            amount += 1

        self.lock_tetromino()
        self.score += 10*amount

    def update(self, timepassed, action):

        if action == 'p':
            self.surface.fill((0,0,0))
            self.paused = not self.paused
        elif action == 'quit':
            self.prepare_and_execute_gameover(playsound=False)
            return True
        if self.paused:
            return

        if action == 'w':
            self.hard_drop()
        elif action == 'j':
            self.request_rotation()
        elif action == 'a':
            self.request_movement('left')
            self.movement_keys['left'] = 1
        elif action == 'd':
            self.request_movement('right')
            self.movement_keys['right'] = 1
        #
        # elif unpressed(pygame.K_LEFT) or unpressed(pygame.K_a):
        #     self.movement_keys['left'] = 0
        #     self.movement_keys_timer = (-self.movement_keys_speed)*2
        # elif unpressed(pygame.K_RIGHT) or unpressed(pygame.K_d):
        #     self.movement_keys['right'] = 0
        #     self.movement_keys_timer = (-self.movement_keys_speed)*2




        self.downwards_speed = self.base_downwards_speed ** (1 + self.level/10.)

        self.downwards_timer += timepassed
        downwards_speed = self.downwards_speed*0.10 if any([pygame.key.get_pressed()[pygame.K_DOWN],
                                                            pygame.key.get_pressed()[pygame.K_s]]) else self.downwards_speed
        if self.downwards_timer > downwards_speed:
            if not self.request_movement('down'):
                if self.lock_tetromino() == False:
                    self.prepare_and_execute_gameover()
                    return
                    # Under normal circumstances, gameover should happen below, when the BrokenMatrixException occurs.
                    # Basically, when writing this code 5 years ago, I must have assumed that self.lock_tetromino could
                    # not be called more than once in self.update. Actually, a hard drop and a "natural" drop can happen
                    # at the same time. This previously resulted in an extremely rare bug. It took me hours staring at
                    # this code to understand what was going on. Be safe!

            self.downwards_timer %= downwards_speed


        if any(self.movement_keys.values()):
            self.movement_keys_timer += timepassed
        # if self.movement_keys_timer > self.movement_keys_speed:
        #     self.request_movement('right' if self.movement_keys['right'] else 'left')
        #     self.movement_keys_timer %= self.movement_keys_speed

        with_shadow = self.place_shadow()

        try:
            with_tetromino = self.blend(self.rotated(), allow_failure=False, matrix=with_shadow)
        except BrokenMatrixException:
            self.prepare_and_execute_gameover()
            return

        for y in range(self.size['height']):
            for x in range(self.size['width']):

                #                                       I hide the 2 first rows by drawing them outside of the surface
                block_location = Rect(x*self.blocksize, (y*self.blocksize - 2*self.blocksize), self.blocksize, self.blocksize)
                if with_tetromino[(y,x)] is None:
                    self.surface.fill(BGCOLOR, block_location)
                else:
                    if with_tetromino[(y,x)][0] == 'shadow':
                        self.surface.fill(BGCOLOR, block_location)
                    
                    self.surface.blit(with_tetromino[(y,x)][1], block_location)
        return False
                    
    def prepare_and_execute_gameover(self, playsound=True):
        write_score(self.score)
        self.gameover = True

    def place_shadow(self):
        posY, posX = self.tetromino_position
        while self.blend(position=(posY, posX)):
            posY += 1

        position = (posY-1, posX)

        return self.blend(position=position, block=self.shadow_block, shadow=True) or self.matrix
        # If the blend isn't successful just return the old matrix. The blend will fail later in self.update, it's game over.

    def fits_in_matrix(self, shape, position):
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if self.matrix.get((y, x), False) is False and shape[y-posY][x-posX]: # outside matrix
                    return False

        return position
                    

    def request_rotation(self):
        rotation = (self.tetromino_rotation + 1) % 4
        shape = self.rotated(rotation)

        y, x = self.tetromino_position

        position = (self.fits_in_matrix(shape, (y, x)) or
                    self.fits_in_matrix(shape, (y, x+1)) or
                    self.fits_in_matrix(shape, (y, x-1)) or
                    self.fits_in_matrix(shape, (y, x+2)) or
                    self.fits_in_matrix(shape, (y, x-2)))
        # ^ Thats how wall-kick is implemented

        if position and self.blend(shape, position):
            self.tetromino_rotation = rotation
            self.tetromino_position = position
            return self.tetromino_rotation
        else:
            return False
            
    def request_movement(self, direction):
        posY, posX = self.tetromino_position
        if direction == 'left' and self.blend(position=(posY, posX-1)):
            logger('Enter Left')
            logger('Before X: %s, Y: %s' % (posX, posY))
            self.tetromino_position = (posY, posX-1)
            logger('After X: %s, Y: %s' % self.tetromino_position)

            return self.tetromino_position
        elif direction == 'right' and self.blend(position=(posY, posX+1)):
            self.tetromino_position = (posY, posX+1)
            return self.tetromino_position
        elif direction == 'up' and self.blend(position=(posY-1, posX)):
            self.tetromino_position = (posY-1, posX)
            return self.tetromino_position
        elif direction == 'down' and self.blend(position=(posY+1, posX)):
            self.tetromino_position = (posY+1, posX)
            return self.tetromino_position
        else:
            return False

    def rotated(self, rotation=None):
        if rotation is None:
            rotation = self.tetromino_rotation
        return rotate(self.current_tetromino.shape, rotation)

    def block(self, color, shadow=False):
        colors = {'blue':   (27, 34, 224),
                  'yellow': (225, 242, 41),
                  'pink':   (242, 41, 195),
                  'green':  (22, 181, 64),
                  'red':    (204, 22, 22),
                  'orange': (245, 144, 12),
                  'cyan':   (10, 255, 226)}


        if shadow:
            end = [40] # end is the alpha value
        else:
            end = [] # Adding this to the end will not change the array, thus no alpha value

        border = Surface((self.blocksize, self.blocksize), pygame.SRCALPHA, 32)
        border.fill(list(map(lambda c: c*0.5, colors[color])) + end)

        borderwidth = 2

        box = Surface((self.blocksize-borderwidth*2, self.blocksize-borderwidth*2), pygame.SRCALPHA, 32)
        boxarr = pygame.PixelArray(box)
        for x in range(len(boxarr)):
            for y in range(len(boxarr)):
                boxarr[x][y] = tuple(list(map(lambda c: min(255, int(c*random.uniform(0.8, 1.2))), colors[color])) + end) 

        del boxarr # deleting boxarr or else the box surface will be 'locked' or something like that and won't blit.
        border.blit(box, Rect(borderwidth, borderwidth, 0, 0))


        return border

    """
    When the block drop to bottom, lock it from receiving any option. 
    """
    def lock_tetromino(self):
        self.matrix = self.blend()
        if not self.matrix:
            return False # Extremely rarely happens

        lines_cleared = self.remove_lines()
        self.lines += lines_cleared

        if lines_cleared:
            self.score += 100 * (lines_cleared**2) * self.combo

            if not self.played_highscorebeaten_sound and self.score > self.highscore:
                self.played_highscorebeaten_sound = True

        if self.lines >= self.level*10:
            self.level += 1

        self.combo = self.combo + 1 if lines_cleared else 1

        self.set_tetrominoes()
        return True

    # When the whole line is full, remove the line.
    def remove_lines(self):
        lines = []
        for y in range(self.size['height']):
            line = (y, [])
            for x in range(self.size['width']):
                if self.matrix[(y,x)]:
                    line[1].append(x)
            if len(line[1]) == self.size['width']:
                lines.append(y)

        for line in sorted(lines):
            for x in range(self.size['width']):
                self.matrix[(line,x)] = None
            for y in range(0, line+1)[::-1]:
                for x in range(self.size['width']):
                    self.matrix[(y,x)] = self.matrix.get((y-1,x), None)

        return len(lines)

    def blend(self, shape=None, position=None, matrix=None, block=None, allow_failure=True, shadow=False):
        # logger('Enter blend')
        if shape is None:
            shape = self.rotated()
        if position is None:
            position = self.tetromino_position

        copy = dict(self.matrix if matrix is None else matrix)
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if (copy.get((y, x), False) is False and shape[y-posY][x-posX] # shape is outside the matrix
                    or # coordinate is occupied by something else which isn't a shadow
                    copy.get((y,x)) and shape[y-posY][x-posX] and copy[(y,x)][0] != 'shadow'): 
                    if allow_failure:
                        return False
                    else:
                        raise BrokenMatrixException("Tried to blend a broken matrix. This should mean game over, if you see this it is certainly a bug. (or you are developing)")
                elif shape[y-posY][x-posX] and not shadow:
                    copy[(y,x)] = ('block', self.tetromino_block if block is None else block)
                elif shape[y-posY][x-posX] and shadow:
                    copy[(y,x)] = ('shadow', block)

        return copy

    def construct_surface_of_next_tetromino(self):
        shape = self.next_tetromino.shape
        surf = Surface((len(shape)*self.blocksize, len(shape)*self.blocksize), pygame.SRCALPHA, 32)

        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.next_tetromino.color), (x*self.blocksize, y*self.blocksize))
        return surf

class Game(object):
    def __init__(self):
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("MaTris")
        self.done = False
        self.screen = screen
        self.init_background()
        self.matris = Matris()
        self.fill_screen()
        self.step('e')
        self.n_actions = 4
        self.n_features = 3

    def init_background(self):
        self.background = Surface(self.screen.get_size())
        self.background.blit(construct_nightmare(self.background.get_size()), (0, 0))

    def fill_screen(self):
        self.matris_border = Surface((MATRIX_WIDTH*BLOCKSIZE+BORDERWIDTH*2, VISIBLE_MATRIX_HEIGHT*BLOCKSIZE+BORDERWIDTH*2))
        self.matris_border.fill(BORDERCOLOR)



    def step(self, action):
        self.matris.update(1, action)
        if self.matris.gameover:
            self.done = True
            return self.matris.score, self.matris.lines, self.matris.combo

        tricky_centerx = WIDTH - (WIDTH - (MATRIS_OFFSET + BLOCKSIZE * MATRIX_WIDTH + BORDERWIDTH * 2)) / 2

        self.background.blit(self.matris_border, (MATRIS_OFFSET, MATRIS_OFFSET))
        self.background.blit(self.matris.surface, (MATRIS_OFFSET + BORDERWIDTH, MATRIS_OFFSET + BORDERWIDTH))

        self.nextts = self.next_tetromino_surf(self.matris.surface_of_next_tetromino)
        self.background.blit(self.nextts, self.nextts.get_rect(top=MATRIS_OFFSET, centerx=tricky_centerx))

        self.infos = self.info_surf()
        self.background.blit(self.infos, self.infos.get_rect(bottom=HEIGHT - MATRIS_OFFSET, centerx=tricky_centerx))

        self.screen.blit(self.background, (0, 0))

        pygame.display.flip()
        return self.matris.score, self.matris.lines, self.matris.combo

    def info_surf(self):

        textcolor = (255, 255, 255)
        font = pygame.font.Font(None, 30)
        width = (WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2)) - MATRIS_OFFSET*2

        def renderpair(text, val):
            text = font.render(text, True, textcolor)
            val = font.render(str(val), True, textcolor)

            surf = Surface((width, text.get_rect().height + BORDERWIDTH*2), pygame.SRCALPHA, 32)

            surf.blit(text, text.get_rect(top=BORDERWIDTH+10, left=BORDERWIDTH+10))
            surf.blit(val, val.get_rect(top=BORDERWIDTH+10, right=width-(BORDERWIDTH+10)))
            return surf

        scoresurf = renderpair("Score", self.matris.score)
        levelsurf = renderpair("Level", self.matris.level)
        linessurf = renderpair("Lines", self.matris.lines)
        combosurf = renderpair("Combo", "x{}".format(self.matris.combo))

        height = 20 + (levelsurf.get_rect().height + 
                       scoresurf.get_rect().height +
                       linessurf.get_rect().height + 
                       combosurf.get_rect().height )

        area = Surface((width, height))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, width-BORDERWIDTH*2, height-BORDERWIDTH*2))

        area.blit(levelsurf, (0,0))
        area.blit(scoresurf, (0, levelsurf.get_rect().height))
        area.blit(linessurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height))
        area.blit(combosurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height))

        return area

    def next_tetromino_surf(self, tetromino_surf):
        area = Surface((BLOCKSIZE*5, BLOCKSIZE*5))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, BLOCKSIZE*5-BORDERWIDTH*2, BLOCKSIZE*5-BORDERWIDTH*2))

        areasize = area.get_size()[0]
        tetromino_surf_size = tetromino_surf.get_size()[0]
        # ^^ I'm assuming width and height are the same

        center = areasize/2 - tetromino_surf_size/2
        area.blit(tetromino_surf, (center, center))

        return area

    def construct_highscoresurf(self):
        font = pygame.font.Font(None, 50)
        highscore = load_score()
        text = "Highscore: {}".format(highscore)
        return font.render(text, True, (255,255,255))

    def reset(self):
        self.__init__()

    def destroy(self):
        pygame.quit()

    def get_matrix_state(self):
        matrix_list = []
        for y in range(MATRIX_HEIGHT):
            x_matrix = []
            for x in range(MATRIX_WIDTH):
                if self.matris.matrix[(y, x)] is not None:
                    x_matrix.append(1)
                else:
                    x_matrix.append(0)
            matrix_list.append(x_matrix)
        for i in range(len(matrix_list)):
            print matrix_list[i]

def construct_nightmare(size):
    surf = Surface(size)

    boxsize = 8
    bordersize = 1
    vals = '1235' # only the lower values, for darker colors and greater fear
    arr = pygame.PixelArray(surf)
    for x in range(0, len(arr), boxsize):
        for y in range(0, len(arr[x]), boxsize):

            color = int(''.join([random.choice(vals) + random.choice(vals) for _ in range(3)]), 16)

            for LX in range(x, x+(boxsize - bordersize)):
                for LY in range(y, y+(boxsize - bordersize)):
                    if LX < len(arr) and LY < len(arr[x]):
                        arr[LX][LY] = color
    del arr
    return surf


if __name__ == '__main__':
    env = Game()
    while True:
        action = raw_input("action: ")
        env.get_matrix_state()
        logger('Score: %s\tLines: %s\tCombo:%s' % env.step(action))
        if env.done:
            start_new = raw_input('New Game(y/n)?')
            if start_new == 'y':
                env = Game()
            else:
                pygame.quit()
                break
