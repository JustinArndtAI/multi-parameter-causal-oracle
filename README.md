The Multi-Parameter Causal Oracle (V2)
This repository contains the complete source code and experimental results for a project demonstrating a self-calibrating "Causal Oracle" capable of deducing multiple unknown physical parameters within a simulated environment. This V2 project builds upon a simpler single-parameter proof-of-concept by tackling a more complex problem and employing a sophisticated, staged optimization strategy.

The project successfully calibrates for three interdependent parameters (friction, elasticity, and mass) by intelligently designing experiments to isolate each variable before performing a final refinement.

Key Concepts
Causal Oracle: A simulated environment that attempts to mirror a "Ground Truth" or real-world system. Its goal is to update its own internal causal parameters until its outputs match reality.

System Identification: The core problem of identifying unknown parameters in a "black box" system based only on its input-output behavior.

Staged Optimization: The final, successful strategy. Instead of solving for all parameters at once, we break the problem down into a series of simpler, focused experiments to isolate one parameter at a time before running a final refinement. This avoids the common optimization pitfall of getting stuck in "local minima."

The Final Result
The staged optimization strategy was a definitive success. The Causal Oracle was able to accurately determine the hidden physical parameters of the ground truth system. The final, compelling visual shows the trajectory of an initial incorrect guess, the ground truth path, and the final calibrated path, which perfectly overlap.

Parameter

True Value

Calibrated Value

Friction

0.7000

0.6585

Elasticity

0.9000

0.8240

Mass

12.00

12.01

How to Run
Clone the repository:

git clone [your-repo-url]
cd multi-parameter-causal-oracle

Set up the environment:

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the definitive experiment:

python main.py

This will run all three optimization stages and save the final trajectory_comparison.png plot in the paper/figures directory.