from src.harvest_exception import UnrecognisedPrinciple
import numpy as np

class EthicsModule():
    """
    Ethics Module (Algorithm 1) evaluates societal well-being before and after acting and generates a self-directed sanction
    Instance variables:
        sanction -- amount of reward to return to agent
        current_principle -- normative ethics principle
        society_well_being -- list of well-being for each living agent
        measure_of_well_being -- metric to evaluate well-being before and after acting (minimum experience)
        number_of_minimums -- number of agents which have minimum experience
    """
    def __init__(self,sanction):
        self.sanction = sanction
        self.current_principle = None
        self.society_well_being = None
        self.measure_of_well_being = None
        self.number_of_minimums = None
    
    def update_social_welfare(self, principle, society_well_being):
        """
        Updates social welfare before agent acts: measure of well-being and number of minimums (Algorithm 1 Line 1)
        """
        self._calculate_social_welfare(principle, society_well_being)
    
    def get_sanction(self, society_well_being):
        """
        Obtain sanction from principle comparing current society well-being with previous well-being (Algorithm 1 Lines 3-8)
        """
        if self.current_principle == "maximin":
            return self._maximin_sanction(self.measure_of_well_being, self.number_of_minimums, society_well_being)
    
    def _calculate_social_welfare(self, principle, society_well_being):
        self.current_principle = principle
        if principle == "maximin":
            self.measure_of_well_being, self.number_of_minimums = self._maximin_welfare(society_well_being)
        else:
            raise UnrecognisedPrinciple(principle)

    def _maximin_welfare(self, society_well_being):
        min_value = min(society_well_being)
        num_mins = np.count_nonzero(society_well_being==min_value)
        return min_value, num_mins
        
    def _maximin_sanction(self, previous_min, number_of_previous_mins, society_well_being):
        current_min, current_number_of_current_mins = self._maximin_welfare(society_well_being)
        current_number_of_previous_mins = np.count_nonzero(society_well_being==previous_min)

        # Differences for partial penalty/reward
        min_diff = current_min - previous_min
        count_diff = number_of_previous_mins - current_number_of_previous_mins
        beta = 0.05
        alpha = 0.04

        #if the global min has been made better, pos reward
        if current_min > previous_min:
            return self.sanction + (alpha * min_diff)
        #if the global min has been made worse, neg reward
        elif current_min < previous_min:
            return -self.sanction + (alpha * min_diff)
        #if the global min has not changed, but there are fewer instances of it, pos reward
        elif current_number_of_previous_mins < number_of_previous_mins and current_min == previous_min:
            return self.sanction + (beta * count_diff)
        #if the global min has not changed, and there are more or same number of instances of it, neg reward
        elif current_number_of_previous_mins > number_of_previous_mins and current_min == previous_min:
            return -self.sanction + (beta * count_diff)
        return 0