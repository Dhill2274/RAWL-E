from .dqn.dqn_agent import DQNAgent
from .ethics_module import EthicsModule
from .norms_module import NormsModule

class HarvestAgent(DQNAgent):
    def __init__(self,unique_id,model,agent_type,min_width,max_width,min_height,max_height,n_features,training,epsilon,shared_replay_buffer=None):
        self.actions = ["north", "east", "south", "west", "throw", "eat"]
        #dqn agent class handles learning and action selection
        super().__init__(unique_id,model,agent_type,self.actions,n_features,training,epsilon,shared_replay_buffer=shared_replay_buffer)
        self.health = 0.8
        self.berries = 0
        self.berries_consumed = 0
        self.berries_thrown = 0
        self.days_survived = 0
        self.max_berries = 0
        self.max_width = max_width
        self.min_width = min_width
        self.width = max_width - min_width + 1
        self.max_height = max_height
        self.min_height = min_height
        self.height = max_height - min_height + 1
        self.health_decay = 0.1
        self.days_left_to_live = self.health/0.1
        self.berry_health_payoff = 0.6
        self.low_health_threshold = 0.6
        self.type = agent_type
        self.norm_module = NormsModule(model,self.unique_id)
        self.norm_clipping_frequency = 10
        if agent_type == "rawlsian":
            self.rewards = self.get_rawlsian_rewards()
            self.ethics_module = EthicsModule(model,self.unique_id,self.rewards["shaped_reward"])
        else:
            self.rewards = self.get_baseline_rewards()
        self.off_grid = False
        self.current_action = None
        
    def execute_action(self, action):
        done = False
        self.current_action = action
        if self.model.write_norms:
            self.norm_module.update_norm_age()
            antecedent = self.norm_module.get_antecedent(self.health, self.berries)
        if self.type == "rawlsian":
            if self.berries > 0:
                have_berries = True
            else:
                have_berries = False
            min_days_left, min_agents, self_in_min = self.ethics_module.get_social_welfare()
        reward, x, y = self.move(action)
        self.model.grid.move_agent(self, (x,y))
        reward += self.forage((x,y))
        next_state = self.model.observe(self)
        if self.type == "rawlsian":
            reward += self.ethics_module.maximin(min_days_left, min_agents, self_in_min, have_berries)
        done, reward = self.update_attributes(reward)
        if self.model.write_norms:
            self.norm_module.update_norm(antecedent, self.actions[action], reward)
            if self.model.day % self.norm_clipping_frequency == 0:
                self.norm_module.clip_norm_base()
        return reward, next_state, done
    
    def move(self, action):
        x, y = self.pos
        reward = 0
        #action 0
        if self.actions[action] == "north":
            if (y + 1) < self.max_height:
                y += 1
            else:
                reward = self.rewards["crash"]
        #action 1
        elif self.actions[action] == "east":
            if (x + 1) < self.max_width:
                x += 1
            else:
                reward = self.rewards["crash"]
        #action 2
        elif self.actions[action] == "south":
            if (y - 1) >= self.min_height:
                y -= 1
            else:
                reward = self.rewards["crash"]
        #action 3
        elif self.actions[action] == "west":
            if (x - 1) >= self.min_width:
                x -= 1
            else:
                reward = self.rewards["crash"]
        #action 4
        elif self.actions[action] == "throw":
            reward = self.throw()
        #action 5
        elif self.actions[action] == "eat":
            reward = self.eat()
        return reward, x, y

    def forage(self, cell):
        #check if there is a berry at current location
        location = self.model.grid.iter_cell_list_contents(cell)
        for a in location:
            if a.type == "berry":
                if self.training or (not self.training and a.allocated_agent_id == self.unique_id):
                    self.berries += 1
                    self.model.spawn_berry(a)
                    return self.rewards["forage"]
        return self.rewards["empty_forage"]
    
    def throw(self):
        if self.berries <= 0:
            return self.rewards["no_berries"]
        #have to have a minimum amount of health to throw
        if self.health < self.low_health_threshold:
            return self.rewards["insufficient_health"]
        benefactor = self.choose_benefactor()
        if not benefactor:
            return self.rewards["no_benefactor"]
        assert(benefactor.type != "berry")
        benefactor.health += self.berry_health_payoff 
        benefactor.berries_consumed += 1
        benefactor.days_left_to_live = benefactor.get_days_left_to_live()
        self.berries -= 1
        self.berries_thrown += 1
        return self.rewards["throw"]
    
    def choose_benefactor(self):
        benefactor = [a for a in self.model.living_agents if a.unique_id != self.unique_id]
        if len(benefactor) > 0:
            return benefactor[0]
        else:
            return False
    
    def eat(self):
        if self.berries > 0:
            self.health += self.berry_health_payoff
            self.berries -= 1
            self.berries_consumed += 1
            return self.rewards["eat"]
        else:
            return self.rewards["no_berries"]
    
    def get_baseline_rewards(self):
        rewards = {"crash": -0.2,
                   "no_berries": -0.2,
                   "no_benefactor": -0.2,
                   "insufficient_health": -0.2,
                   "empty_forage": 0,
                   "throw": 0.5,
                   "forage": 1,
                   "eat": 1
                   }
        return rewards
    
    def get_rawlsian_rewards(self):
        rewards = {"crash": -0.1,
                   "no_berries": -0.1,
                   "no_benefactor": -0.1,
                   "insufficient_health": -0.1,
                   "empty_forage": 0,
                   "shaped_reward": 0.4,
                   "throw": 0.5,
                   "forage": 0.8,
                   "eat": 0.8
                   }
        return rewards

    def get_days_left_to_live(self):
        health = self.health
        health += self.berry_health_payoff * self.berries
        days_left_to_live = health / self.health_decay
        if days_left_to_live < 0:
            return 0
        return days_left_to_live
    
    def update_attributes(self, reward):
        done = False
        self.health -= self.health_decay
        self.days_left_to_live = self.get_days_left_to_live()
        if len(self.model.living_agents) < self.model.num_agents:
            reward -= 1
            self.days_survived = self.model.day
            done = True
        if self.health <= 0:
            #environment class checks for dead agents to remove at the end of each step
            done = True
            self.days_survived = self.model.day
            self.health = 0
            reward -= 1
        if self.model.day == self.model.max_days - 1:
            reward += 1
        if self.berries > self.max_berries:
            self.max_berries = self.berries
        return done, reward