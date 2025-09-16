import pymunk
import numpy as np

class Simulator:
    """
    A simple 2D physics simulator using Pymunk.
    THIS VERSION CONTAINS THE CRITICAL BUG FIX.
    """
    def __init__(self, params=None):
        if params is None:
            params = {}

        self.space = pymunk.Space()
        self.space.gravity = (0, -1000)

        # Create floor
        floor_body = self.space.static_body
        floor_shape = pymunk.Segment(floor_body, (-5000, 50), (5000, 50), 5)
        floor_shape.friction = 1.0
        self.space.add(floor_shape)

        # Create the dynamic object
        mass = params.get('mass', 10.0)
        moment = pymunk.moment_for_circle(mass, 0, 20)
        body = pymunk.Body(mass, moment)
        body.position = (0, 300)

        shape = pymunk.Circle(body, 20)
        shape.elasticity = params.get('elasticity', 0.95)
        shape.friction = params.get('friction', 0.7)
        
        # *** THE CRITICAL BUG FIX IS HERE ***
        # We must add both the body and the shape to the space.
        self.space.add(body, shape)

        self.body = body

    def run_simulation_for_trajectory(self, steps, impulses):
        """
        Runs the simulation for a given number of steps, applying impulses at
        specific times, and returns the full trajectory.
        """
        trajectory = []
        for step in range(steps):
            # Apply any scheduled impulses
            for t_impulse, impulse_vector in impulses:
                if step == t_impulse:
                    # Apply impulse to the center of the body
                    self.body.apply_impulse_at_local_point(impulse_vector)

            self.space.step(1 / 60.0)
            trajectory.append(self.body.position)

        return trajectory

