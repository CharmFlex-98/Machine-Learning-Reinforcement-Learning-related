import pygame
import PIL.Image as Image
import turtle
import torchvision.transforms as transform
from collections import namedtuple

Dim = namedtuple('Dim', ('x', 'y', 'w', 'h'))
pil_tensor = transform.ToTensor()
pygame.init()


class my_game:
    def __init__(self, w, h, colour):
        self.window = pygame.display.set_mode((w, h))
        pygame.display.set_caption('learning_game')
        self.clock = pygame.time.Clock()
        self.colour = colour
        self.window_w = w
        self.window_h = h

        self.agent = Dim(w / 4, h * 0.8, 20, 20)
        self.wall = Dim(w * 0.8, h / 5, 80, 80)
        self.wall2 = Dim(w * 0.8, h - h / 5, 80, 80)
        self.wall_is_up = False
        self.food = Dim(self.wall.x + 30, self.window_h / 2 - 10, 20, 20)
        self.reward = 0

    def get_control(self):
        state = ''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    state = 'up'
                elif event.key == pygame.K_DOWN:
                    state = 'down'
                elif event.key == pygame.K_LEFT:
                    state = 'left'
                elif event.key == pygame.K_RIGHT:
                    state = 'right'
            else:
                pass

        return state

    def agent_move(self, velocity_x, velocity_y):
        x, y, w, h = self.agent
        state = self.get_control()
        if state == '':
            pass
        elif state == 'down':
            y += velocity_y
        elif state == 'up':
            y -= velocity_y
        elif state == 'left':
            x -= velocity_x
        elif state == 'right':
            x += velocity_x

        self.agent = Dim(x, y, w, h)

    def wall_move(self, velocity_y):
        # wall
        x, y, w, h = self.wall
        if y >= self.window_h * 0.45 - 80:
            y -= velocity_y
            self.wall_is_up = True
        elif y <= 0:
            y += velocity_y
            self.wall_is_up = False
        else:
            if self.wall_is_up:
                y -= velocity_y
            else:
                y += velocity_y

        self.wall = Dim(x, y, w, h)

        # wall2
        x2, y2, w2, h2 = self.wall2
        y2 = self.window_h - y - 80
        self.wall2 = Dim(x2, y2, w2, h2)

    def check_collision(self):
        if self.food.x-20 <= self.agent.x <= self.food.x + 20 and self.food.y -20 <= self.agent.y <= self.food.y + 20:
            self.reward += 1
            print('reward plus one!')
            self.agent = Dim(self.window_w / 4, self.window_h * 0.8, 20, 20)

        if (self.wall.x-20<=self.agent.x<=self.wall.x+80 and self.wall.y-20<=self.agent.y<=self.wall.y+80) or \
                (self.wall2.x-20<=self.agent.x<=self.wall2.x+80 and self.wall2.y-20<=self.agent.y<=self.wall2.x+80):
            self.reward += -1
            print('reward minus one!')
            self.agent = Dim(self.window_w / 4, self.window_h * 0.8, 20, 20)

        else:
            pass

    def update(self):
        self.window.fill(self.colour)
        pygame.draw.rect(self.window, 'red', (self.agent.x, self.agent.y, self.agent.w, self.agent.h), 0)
        pygame.draw.rect(self.window, 'blue', (self.wall.x, self.wall.y, self.wall.w, self.wall.h), 0)
        pygame.draw.rect(self.window, 'blue', (self.wall2.x, self.wall2.y, self.wall2.w, self.wall2.h), 0)
        pygame.draw.rect(self.window, 'green', (self.food.x, self.food.y, self.food.w, self.food.h), 0)
        self.agent_move(10, 10)
        self.check_collision()
        self.wall_move(3)
        pygame.display.flip()

        self.clock.tick(60)


if __name__ == '__main__':
    game = my_game(500, 500, 'yellow')
    while True:
        game.update()
