from src.agents import Team43Agent, RandomAgent
from src.bw4t.BW4TWorld import BW4TWorld
from src.bw4t import Statistics

from setuptools.sandbox import save_path   # type: ignore

from src.agents import Team36Agent


"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [
        {'name': 'us', 'botclass': Team36Agent, 'settings': {'slowdown': 1, 'colorblind' : True}},
        {'name': 'us', 'botclass': RandomAgent, 'settings': {'slowdown': 1, 'shape_blind' : True}}
        ]

    print("Started world...")
    world=BW4TWorld(agents).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))
    
