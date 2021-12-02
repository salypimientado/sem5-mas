#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# La clase `Model` se hace cargo de los atributos a nivel del modelo, maneja los agentes. 
# Cada modelo puede contener múltiples agentes y todos ellos son instancias de la clase `Agent`.
from mesa import Agent, Model

from mesa.space import MultiGrid

from mesa.batchrunner import BatchRunnerMP
# Con `SimultaneousActivation` hacemos que todos los agentes se activen de manera simultanea.
from mesa.time import SimultaneousActivation, RandomActivation, BaseScheduler

# Vamos a hacer uso de `DataCollector` para obtener el grid completo cada paso (o generación) y lo usaremos para graficarlo.
from mesa.datacollection import DataCollector

'''
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128
'''

# Definimos los siguientes paquetes para manejar valores númericos.
import numpy as np
import pandas as pd

import itertools
import random
from enum import Enum
from collections import OrderedDict

from flask import Flask

# Definimos otros paquetes que vamos a usar para medir el tiempo de ejecución de nuestro algoritmo.
import time
import datetime


# In[ ]:


def get_grid(model):
    dimensions = model.grid.width, model.grid.height
    grid = np.zeros(dimensions)
    for x, line in enumerate(grid):
        for y, _ in enumerate(line):
            # amount of dirt agents in cell
            car = len(list(filter(lambda agent: isinstance(agent,carAgent), model.grid.iter_neighbors((x,y), False, True, 0)))) 
            #print(f'{x=} {y=} {dirt=}')
            # len of dirt agents in cell
            lights = list(filter(lambda agent: isinstance(agent,trafficLight), model.grid.iter_neighbors((x,y), False, True, 0)))
            light_status = lights[0].state if lights else (4 if car else 5)
            #print(f'{vacuums=}')
            if not car:
                if light_status == 0:
                    grid[x][y] = 0
                elif light_status == 1:
                    grid[x][y] = 5
                elif light_status == 2 or light_status == 3:
                    grid[x][y] = 2
                elif light_status == 5:
                    grid[x][y] = 8
            else:
                if light_status != 4:
                    if light_status == 0:
                        grid[x][y] = 4
                    elif light_status == 1:
                        grid[x][y] = 5
                    elif light_status == 2 or light_status == 3:
                        grid[x][y] = 1
                else:
                    grid[x][y] = 7
    return grid

def get_other_lights(id, model):
    return [agent for agent in model.traffic_lights_schedule.agents if agent.id != id]


# In[ ]:


class direction():
    UP = -1, 0
    DOWN = 1, 0
    LEFT = 0, -1
    RIGHT = 0, 1
    lst = [UP,DOWN,RIGHT,LEFT]
    
class states():
    RED = 0
    YELLOW = 1
    GREEN = 2
    SECONDTICK = 3
    
def state_to_string(state):
    if state == states.RED:
        return "RED"
    elif state == states.YELLOW:
        return "YELLOW"
    else:
        return "GREEN"

def direction_to_string(orientation):
    if orientation is direction.UP:
        return "UP"
    elif orientation is direction.DOWN:
        return "DOWN"
    elif orientation is direction.LEFT:
        return "LEFT"
    else:
        return "RIGHT"

def coords_to_pos(coords):
    return {"x":coords[1],"y":11 - coords[0]}

def rotation_status(car):
    if car.turn_direction:
        return {"rotate_direction": direction_to_string(car.turn_direction), "turning_now": car.steps_until_turn == 0}
    else:
        return {"rotate_direction": direction_to_string(car.direction), "turning_now": False}

def turn_direction(orientation, steps_to_turn):
    #steps_to_turn = random.choice([0,1,2])
    if orientation is direction.UP:
        if steps_to_turn == 1:
            return direction.RIGHT
        elif steps_to_turn == 2:
            return direction.LEFT
        else:
            return None
    elif orientation is direction.DOWN:
        if steps_to_turn == 1:
            return direction.LEFT
        elif steps_to_turn == 2:
            return direction.RIGHT
        else:
            return None
    elif orientation is direction.LEFT:
        if steps_to_turn == 1:
            return direction.UP
        elif steps_to_turn == 2:
            return direction.DOWN
        else:
            return None
    else:
        if steps_to_turn == 1:
            return direction.DOWN
        elif steps_to_turn == 2:
            return direction.UP
        else:
            return None


# In[ ]:


class carAgent(Agent):
    def __init__(self, unique_id, coords, direction, model):
        super().__init__(unique_id,model)
        self.id = unique_id
        self.coords = coords
        self.model = model
        self.direction = direction
        self.turn_factor = random.choice([0,1,2])
        self.turn_direction = None
        self.steps_until_turn = None
        self.next_coords = coords
        self.done = False
    
    def step(self):
        under_light = any([isinstance(agent,trafficLight) for agent in self.model.grid.iter_neighbors(self.coords, False, True, 0)])
        red_light = any([(isinstance(agent,trafficLight) and agent.state == states.RED)
                          for agent in self.model.grid.iter_neighbors(self.coords, False, True, 0)])
        will_crash = any([isinstance(agent,carAgent) and agent.coords == (self.coords[0] + self.direction[0], self.coords[1] + self.direction[1])                                             and agent.direction == self.direction for agent in self.model.grid.iter_neighbors(self.coords, False, True, 1)])
        canMove = not red_light and not will_crash
        if canMove:
            if self.turn_direction:
                if self.steps_until_turn == 0:
                    self.direction = self.turn_direction
                    self.turn_direction = None
                    self.steps_until_turn = None
                else:
                    self.steps_until_turn -= 1
            if under_light:
                turn_to = turn_direction(self.direction, self.turn_factor)
                if turn_to:
                    self.steps_until_turn = self.turn_factor - 1
                    self.turn_direction = turn_to
            self.next_coords = self.coords[0] + self.direction[0], self.coords[1] + self.direction[1]
            if self.model.grid.out_of_bounds(self.next_coords):
                if not self.done:
                    self.model.kill_agents.append(self)
                self.done = True
                self.model.instructions.append({"id":self.id, "method":"destroy"})
            else:
                if self.steps_until_turn == 0:
                    self.model.instructions.append({"id":self.id, "method": f'advance-rotate', "params":direction_to_string(self.turn_direction)})
                else:
                    self.model.instructions.append({"id":self.id, "method": "advance"})
                
        
    def advance(self):
        if not self.done:
            self.coords = self.next_coords
            self.model.grid.move_agent(self,self.coords)

# scheduler goes through each traffic light randomly
# if light has car below it, set all other lights to red
# next step does nothing
# next step sets all lights to yellow before anything happens
class trafficLight(Agent):
    def __init__(self, unique_id, coords, model):
        super().__init__(unique_id,model)
        self.id = unique_id
        self.coords = coords
        self.model = model
        self.state = states.YELLOW
        self.blocked_steps = 0
        self.reset_state = False
        self.changed_color = False
    

    def setRed(self):
        self.state = states.RED
        self.model.instructions.append({"id":self.id, "method": f'set-light', "params":'RED'})
        self.blocked_steps = 3
    
    def unblock(self):
        self.blocked_steps = self.blocked_steps - 1
        if self.blocked_steps == 0:
            self.reset_state = True
    
    def step(self):
        isGreen = self.state == states.GREEN
        if self.blocked_steps > 0 and not isGreen:
            return
        if self.reset_state:
            self.state = states.YELLOW
            self.changed_color = True
            self.reset_state = False
        car = any([isinstance(agent, carAgent) for agent in self.model.grid.iter_neighbors(self.coords, False, True, 0)])
        if car:
            for light in get_other_lights(self.id, self.model):
                light.setRed()
            
            if not self.state == states.GREEN:
                self.changed_color = True
            self.state = states.GREEN
            self.blocked_steps = 3
        if self.changed_color is True:
            if self.state is states.GREEN:
                self.model.instructions.append({"id":self.id, "method": f'set-light', "params":'GREEN'})
            else:      
                self.model.instructions.append({"id":self.id, "method": f'set-light', "params":'YELLOW'})
            


# In[ ]:


class TrafficScheduler(BaseScheduler):
    def __init__(self, model):
        self.model = model
        self.steps = 0
        self.time = 0
        self._agents=OrderedDict()
        
    def step(self):
        v = list(self._agents.values())
        random_lights = random.sample(v, len(v))
        for agent in random_lights:
            agent.step()
        for agent in self._agents.values():
            agent.unblock()
        self.steps += 1
        self.time += 1 
        
class SpawnPoints:
    UP = (10,5)
    DOWN = (1,4)
    LEFT = (6,0)
    RIGHT = (5,9)
    
    @staticmethod
    def to_string(pos):
        if pos is SpawnPoints.UP:
            return "UP"
        elif pos is SpawnPoints.DOWN:
            return "DOWN"
        elif pos is SpawnPoints.LEFT:
            return "LEFT"
        else:
            return "RIGHT"
        

class trafficSimulation(Model):
    def __init__(self, spawn_speed):
        self.spawn_speed = spawn_speed
        self.counter = 0
        self.traffic_lights_schedule = TrafficScheduler(self)
        self.cars_schedule = SimultaneousActivation(self)
        self.grid = MultiGrid(11,10,False)
        self.id = 0
        self.spawnpoints = [SpawnPoints.UP, SpawnPoints.DOWN, SpawnPoints.LEFT, SpawnPoints.RIGHT]
        self.kill_agents = []
        self.instructions = []
        
        self.datacollector = DataCollector(
            model_reporters={"Grid": get_grid})
        
        traffic_light_coords = [(7,5),(4,4),(6,3),(5,6)]

        for coord in traffic_light_coords:
            light = trafficLight(self.id, coord, self)
            self.id = self.id + 1
            self.grid.place_agent(light,coord)
            self.traffic_lights_schedule.add(light)
            
    def get_data(self):
        '''
        traffic_lights = [{"coords":coords_to_pos(agent.coords), "state": state_to_string(agent.state)} for agent in get_other_lights(-1,self)]
        cars = [{"id": agent.id, "direction":direction_to_string(agent.direction),"coords": coords_to_pos(agent.coords),"turn_status": rotation_status(agent)}for agent in self.cars_schedule.agents]
        #nt(traffic_lights, cars)
        
        return traffic_lights, cars
        '''
        res = [x for x in self.instructions]
        self.instructions = []
        return res
    
    def step(self):
        self.datacollector.collect(self)
        self.traffic_lights_schedule.step()
        self.cars_schedule.step()
        
        if self.counter == self.spawn_speed-1 and random.random() > 0.5:
            idx, (orientation, coords) = random.choice([(index, x) for index, x in enumerate(zip(direction.lst,self.spawnpoints))])
            anyCar = any([isinstance(agent, carAgent) for agent in self.grid.iter_neighbors(coords, False, True, 0)])
            if not anyCar:
                self.instructions.append({"id":self.id, "method":f'spawn-car', "params":SpawnPoints.to_string(coords)})
                car = carAgent(self.id, coords, orientation, self)
                self.grid.place_agent(car,coords)
                self.cars_schedule.add(car)
                self.id = self.id + 1
        self.counter = (self.counter + 1) % self.spawn_speed
        
        
        for x in self.kill_agents:
            self.grid.remove_agent(x)
            self.cars_schedule.remove(x)
            self.kill_agents.remove(x)


# In[ ]:


m = trafficSimulation(1)

'''
steps = 300
for i in range(steps):
    m.step()
    m.get_data()

df = m.datacollector.get_model_vars_dataframe()

cmap = plt.cm.Pastel1
fig, axs = plt.subplots(figsize=(11,11))
axs.set_xticks([])
axs.set_yticks([])
patch = plt.imshow(df.iloc[0][0], cmap=cmap, vmin=-0.1)

def animate(i):
    patch.set_data(df.iloc[i][0])
    
anim = animation.FuncAnimation(fig, animate, frames=steps)
'''


# In[ ]:


#anim


# In[ ]:


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/")
def get_next_version():
    m.step()
    instructions = m.get_data()
    return {"instructions":instructions}

